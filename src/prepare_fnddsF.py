"""prepare the FNDDS nutrient for RAG"""
import argparse
import json
from pathlib import Path

from utils import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    config = load_config(parser.parse_args().config)
    output = Path(config["workspace"]) / "fndds"
    output.mkdir(parents=True, exist_ok=False)

    import faiss
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    source = Path(config["paths"]["fndds_dir"])
    tables = {}
    for name, filename in (
        ("foods", "2021-2023 FNDDS At A Glance - Foods and Beverages.xlsx"),
        ("nutrients", "2021-2023 FNDDS At A Glance - FNDDS Nutrient Values.xlsx"),
        ("portions", "2021-2023 FNDDS At A Glance - Portions and Weights.xlsx"),
    ):
        tables[name] = pd.read_excel(source / filename, header=1)
        tables[name].columns = tables[name].columns.str.strip().str.replace("\n", " ")

    foods = pd.DataFrame()
    raw = tables["foods"]
    for key, term in (("food_code", "food code"), ("main_desc", "main food description"),
                      ("additional_desc", "additional food description"), ("wweia_category", "wweia category description")):
        col = next((col for col in raw.columns if term in col.lower()), None)
        if key == "food_code":
            foods[key] = raw[col].astype(str).str.strip()
        else:
            foods[key] = raw[col].fillna("").astype(str) if col else ""
    raw = tables["nutrients"]
    nutrients = pd.DataFrame({"food_code": raw[next(col for col in raw.columns if "food code" in col.lower())].astype(str).str.strip()})
    for key, term in (("energy_kcal_100g", "Energy (kcal)"), ("protein_g_100g", "Protein (g)"),
                      ("fat_g_100g", "Total Fat (g)"), ("carbs_g_100g", "Carbohydrate (g)")):
        col = next((col for col in raw.columns if term.lower() in col.lower()), None)
        nutrients[key] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0) if col else 0.0
    canonical = foods.merge(nutrients, on="food_code", how="inner").sort_values("food_code").reset_index(drop=True)
    canonical.to_csv(output / "fndds_canonical.csv", index=False)
    canonical.to_parquet(output / "fndds_canonical.parquet", index=False)

    raw = tables["portions"]
    cols = {}
    for key, terms in (("code", ["food code"]), ("portion", ["portion description"]),
                       ("gram", ["gram weight", "portion weight"]), ("desc", ["main food description"])):
        cols[key] = next((col for term in terms for col in raw.columns if term in col.lower()), None)
    total_rows = len(raw)
    raw = raw.dropna(subset=[cols["code"], cols["portion"]]).copy()
    raw = raw[raw[cols["portion"]].astype(str).str.strip() != ""]
    raw[cols["gram"]] = pd.to_numeric(raw[cols["gram"]], errors="coerce")
    raw = raw.dropna(subset=[cols["gram"]])
    raw = raw[raw[cols["gram"]] > 0]
    lookup = {}
    for _, row in raw.iterrows():
        try:
            code = str(int(float(row[cols["code"]])))
        except (ValueError, TypeError):
            continue
        if code not in lookup:
            lookup[code] = {"desc": str(row[cols["desc"]]).strip()
                           if cols["desc"] and pd.notna(row[cols["desc"]]) else "", "portions": []}
        lookup[code]["portions"].append({"portion": str(row[cols["portion"]]).strip(), "gram": round(float(row[cols["gram"]]), 2)})
    for entry in lookup.values():
        entry["portions"].sort(key=lambda portion: portion["gram"])
    (output / "portions_lookup.json").write_text(json.dumps(lookup, ensure_ascii=False, indent=2), encoding="utf-8")

    codes = set(canonical["food_code"].astype(str).str.strip())
    descriptions = dict(zip(canonical["food_code"], canonical["main_desc"]))
    missing = [{"food_code": code, "desc": descriptions.get(code, "")} for code in sorted(codes - set(lookup))]
    counts = [len(entry["portions"]) for entry in lookup.values()]
    weights = [portion["gram"] for entry in lookup.values() for portion in entry["portions"]]
    report = {"total_rows_in_excel": total_rows, "rows_after_filtering": len(raw),
              "unique_food_codes_with_portions": len(lookup), "unique_food_codes_in_canonical_csv": len(codes),
              "coverage_rate": round(len(codes & set(lookup)) / len(codes), 4) if codes else 0.0,
              "food_codes_without_portions_count": len(missing), "food_codes_without_portions": missing,
              "portions_per_food_code": {"min": int(np.min(counts)), "max": int(np.max(counts)),
                                         "mean": round(float(np.mean(counts)), 2), "median": round(float(np.median(counts)), 1)},
              "gram_weight_stats": {"min": round(float(np.min(weights)), 2), "max": round(float(np.max(weights)), 2),
                                    "mean": round(float(np.mean(weights)), 2), "median": round(float(np.median(weights)), 2)}}
    samples = {}
    for code in ["75109000", "11100000", "42100100", *sorted(lookup)]:
        if len(samples) == 5:
            break
        if code in lookup and code not in samples:
            samples[code] = {"desc": lookup[code]["desc"], "portions": [
                f"{p['portion']} = {p['gram']}g" for p in lookup[code]["portions"]]}
    report["sample_entries"] = samples
    (output / "portions_coverage_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    canonical = pd.read_csv(output / "fndds_canonical.csv")
    canonical["additional_desc"] = canonical["additional_desc"].fillna("")
    canonical["wweia_category"] = canonical["wweia_category"].fillna("")
    texts, metadata = [], []
    for _, row in canonical.iterrows():
        parts = [row["main_desc"]]
        for key in ("additional_desc", "wweia_category"):
            if row[key]:
                parts.append(str(row[key]))
        texts.append(" | ".join(parts))
        metadata.append({**{key: str(row[key]) for key in ("food_code", "main_desc", "additional_desc", "wweia_category")},
                         **{key: float(row[key]) for key in ("energy_kcal_100g", "protein_g_100g", "fat_g_100g", "carbs_g_100g")}})
    embeddings = np.array(SentenceTransformer(
        config["rag"]["embedding_model"], trust_remote_code=True, device=config["rag"]["embedding_device"]
    ).encode([f"passage: {text}" for text in texts], batch_size=128,
             show_progress_bar=True, normalize_embeddings=True), dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    index_dir = output / "index"
    index_dir.mkdir()
    faiss.write_index(index, str(index_dir / "faiss.index"))
    np.save(index_dir / "embeddings.npy", embeddings)
    for name, rows in (("faiss_meta", metadata), ("doc_texts", [{"text": text} for text in texts])):
        (index_dir / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "embedding_model": config["rag"]["embedding_model"], "input_directory": str(source),
        "encoding": "normalized passage: documents / query: queries",
    }, indent=2), encoding="utf-8")
    print(f"FNDDS ready: {output}")
