"""Base model and SFT model with RAG or SFT+RAG,"""
import argparse
import json
import re
from contextlib import ExitStack
from pathlib import Path

from utils import (add_eval_arguments, evaluate, has_nutrition, load_config,
                   load_samples, make_prediction_record, save_evaluation, start_run)
from models import VisionModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_eval_arguments(parser)
    parser.add_argument("--source", choices=("base", "sft"), default="base")
    parser.add_argument("--adapter")
    modes = ("code_selection", "refinement", "vision_code_selection", "vision_refinement", "vision_portions")
    parser.add_argument("--modes", choices=modes, nargs="+", default=list(modes))
    parser.add_argument("--baseline-run", help="A matching completed image-only evaluation")
    args = parser.parse_args()
   
    if (args.source == "sft") != bool(args.adapter):
        parser.error("source=sft requires --adapter; source=base must not specify one")
    config = load_config(args.config)
    samples = load_samples(config, args.view, args.split, args.datasets, args.limit_dishes)
    output = start_run(args.output, config, samples, paradigm="rag", model=args.model,
                       view=args.view, arguments=vars(args))
    model = VisionModel(config, args.model, args.adapter)
    if args.baseline_run:
        by_id = {row["sample_id"]: row for row in (json.loads(line) for line in
            (Path(args.baseline_run) / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
        baseline = [by_id[sample["sample_id"]] for sample in samples]
        (output / "baseline").mkdir()
        (output / "baseline/predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in baseline), encoding="utf-8")
        save_evaluation(output / "baseline", baseline)
    else:
        baseline = evaluate(model, config, args.model, samples, output / "baseline")

    from PIL import Image
    from rag_support.retriever import FNDDSRetriever
    from rag_support.code_selection import select_codes
    from rag_support.refinement import refine_nutrition
    fndds = Path(config["workspace"]) / "fndds"
    retriever = FNDDSRetriever(fndds / "index", config["rag"]["embedding_model"], config["rag"]["embedding_device"])
    portions = json.loads((fndds / "portions_lookup.json").read_text(encoding="utf-8"))
    expansion_prompt = (Path(__file__).resolve().parent / "prompts/query_expansion.txt").read_text(encoding="utf-8").rstrip("\n")
    results = {mode: [] for mode in args.modes}
    with ExitStack() as stack:
        handles = {}
        for mode in args.modes:
            (output / mode).mkdir()
            handles[mode] = stack.enter_context((output / mode / "predictions.jsonl").open("w", encoding="utf-8"))
        retrieval_log = stack.enter_context((output / "retrieval.jsonl").open("w", encoding="utf-8"))
        for index, (sample, base_record) in enumerate(zip(samples, baseline), 1):
            base = {**(base_record["predicted"] or {}), "foods_text": base_record["foods_text"]}
            candidates, retrieval = [], {}
            foods = base["foods_text"]
            if not has_nutrition(base):
                retrieval["fallback"] = "base_parse_failure"
            elif not foods or foods.lower() in ("unknown", "none"):
                retrieval["fallback"] = "unknown_foods"
            else:
                queries = [foods]
                if config["rag"]["query_expansion"]:
                    queries += [query for query in (
                        re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                        for line in model.text(expansion_prompt.format(description=foods)).strip().split("\n")
                        if line.strip()) if query][:5]
                seen = {}
                for group in retriever.search_batch(queries, k=config["rag"]["retrieval_k"]):
                    for document in group:
                        code = document.metadata["food_code"]
                        if code not in seen or document.score > seen[code].score:
                            seen[code] = document
                candidates = sorted(seen.values(), key=lambda document: document.score, reverse=True)
                retrieval = {"queries": queries, "candidates": [
                    {"text": doc.page_content, "score": doc.score, **doc.metadata} for doc in candidates]}
                if not candidates:
                    retrieval["fallback"] = "no_candidates"
            retrieval_log.write(json.dumps({"sample_id": sample["sample_id"], **retrieval}, ensure_ascii=False) + "\n")
            retrieval_log.flush()
            with Image.open(sample["image_path"]) as source:
                image = source.convert("RGB")
            try:
                for mode in args.modes:
                    prediction, details = dict(base), {"mode": mode}
                    if not candidates:
                        details["fallback"] = ("base_parse_failure" if retrieval.get("fallback") == "base_parse_failure"
                                               else "no_candidates_or_unknown_foods")
                    else:
                        vision = mode.startswith("vision_")
                        selections, aggregate, raw = select_codes(
                            foods, base["total_mass_g"], candidates, model.image if vision else model.text,
                            image=image if vision else None,
                            portions_lookup=portions if mode == "vision_portions" else None)
                        details.update(selections=selections, fndds_aggregated=aggregate, code_selection_raw_response=raw)
                        if not selections:
                            details["fallback"] = "code_selection_empty"
                        else:
                            prediction = {**base, **aggregate}
                            if mode in ("refinement", "vision_refinement"):
                                refined, raw = refine_nutrition(
                                    base, selections, aggregate, model.image if vision else model.text,
                                    image=image if vision else None)
                                details.update(refined=refined, refinement_raw_response=raw)
                                prediction = {**refined, "foods_text": foods}
                    record = make_prediction_record(sample, prediction, base_record["raw_output"],
                                               metadata=details, retrieval_fallback=retrieval.get("fallback"))
                    results[mode].append(record)
                    handles[mode].write(json.dumps(record, ensure_ascii=False) + "\n")
                    handles[mode].flush()
            finally:
                image.close()
            print(f"RAG {index}/{len(samples)}", flush=True)
    for mode, records in results.items():
        save_evaluation(output / mode, records)
    (output / "completion.json").write_text(json.dumps({"status": "complete", "modes": args.modes}), encoding="utf-8")
