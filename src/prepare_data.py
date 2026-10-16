"""Prepare shared total labels, train/dev/test splits and camera manifests."""
import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

from utils import load_config


def resize_image(source, destination, config):
    """Resize to the configured long edge and save an RGB JPEG."""
    from PIL import Image
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as source_image:
        image = source_image.convert("RGB")
        image.thumbnail((config["data"]["image_long_edge"],) * 2, Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=config["data"]["jpeg_quality"])
    return str(destination.resolve())


def prepare_n5k_image(dish_id, camera, config):
    """Use an existing camera frame or decode one, then resize and save it."""
    source_dir = Path(config["paths"]["nutrition5k_imagery"]) / dish_id
    destination = Path(config["workspace"]) / "assets/nutrition5k" / dish_id / f"camera_{camera}.jpg"
    existing = source_dir / "frames_sampled60" / f"camera_{camera}_frame_001.jpeg"
    if existing.is_file():
        return resize_image(existing, destination, config), {"source": str(existing), "kind": "existing_frame_001"}
    video = next((source_dir / f"camera_{camera}{suffix}" for suffix in (".h264", ".mp4")
                  if (source_dir / f"camera_{camera}{suffix}").is_file()), None)
    if video is None:
        return None, None
    destination.parent.mkdir(parents=True, exist_ok=True)
    decoded = destination.with_name(f"camera_{camera}.decoded.png")
    frame_index = int(config["data"]["video_frame_index"])
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                    "-vf", f"select=eq(n\\,{frame_index})", "-frames:v", "1", str(decoded)], check=True)
    output = resize_image(decoded, destination, config)
    decoded.unlink()
    return output, {"source": str(video), "kind": "decoded_video", "zero_based_frame": frame_index}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    config = load_config(parser.parse_args().config)
    paths, seed = config["paths"], int(config["seed"])
    cameras = ("A", "B", "C", "D")
    exclusions = frozenset(config["data"]["nutrition5k_excluded_ingredients"])
    data_dir = Path(config["workspace"]) / "data"
    data_dir.mkdir(parents=True, exist_ok=False)
    report = {key: 0 for key in ("n5k_invalid_totals", "n5k_no_images", "acetada_invalid_totals", "acetada_no_image")}

    hf_path = paths.get("nutrition5k_hf_metadata")
    hf = [json.loads(line) for line in Path(hf_path).read_text(encoding="utf-8").splitlines() if line.strip()] if hf_path else None
    allowed = {str(row["id"]) for row in hf} if hf is not None else None
    n5k, removed_counts = {}, Counter()
    n5k_split_ids = set()
    targets = ("total_kcal", "total_mass_g", "total_fat_g", "total_carbs_g", "total_protein_g")
    removed_totals = {key: 0.0 for key in targets}
    with Path(paths["nutrition5k_metadata"]).open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            dish_id = row[0]
            if allowed is not None and dish_id not in allowed:
                continue
            # The original split uses metadata IDs and image directories,
            # before checking nutrition labels or individual camera frames.
            if (Path(paths["nutrition5k_imagery"]) / dish_id).is_dir():
                n5k_split_ids.add(dish_id)
            if len(row) < 6 or not dish_id.startswith("dish_"):
                continue
            try:
                nutrition = {key: float(value) for key, value in zip(targets, row[1:6])}
            except ValueError:
                report["n5k_invalid_totals"] += 1
                continue
            # Match the N5K loader: only calories and mass must be positive.
            if nutrition["total_kcal"] <= 0 or nutrition["total_mass_g"] <= 0:
                report["n5k_invalid_totals"] += 1
                continue
            views = []
            for camera in cameras:
                image_path, origin = prepare_n5k_image(dish_id, camera, config)
                if image_path:
                    views.append({"camera": camera, "image_path": image_path, "image_origin": origin})
            if not views:
                report["n5k_no_images"] += 1
                continue
            items = []
            for start in range(6, len(row) - 6, 7):
                _, name, mass, energy, fat, carbs, protein = row[start:start + 7]
                name = name.strip()
                if not name or name == "nan":
                    continue
                if name in exclusions:
                    removed_counts[name] += 1
                    for key, value in zip(targets, (energy, mass, fat, carbs, protein)):
                        try:
                            removed_totals[key] += float(value)
                        except ValueError:
                            pass
                    continue
                items.append({"name": name, "mass_g": float(mass) if mass else None})
            foods = {}
            for item in items:
                foods.setdefault(item["name"].lower(), item["name"])
            n5k[dish_id] = {"dataset": "nutrition5k", "dish_id": dish_id,
                            "foods": list(foods.values()),
                            "items": items, "views": views, **nutrition}
    report["n5k_excluded_item_occurrences"] = dict(removed_counts)
    report["n5k_excluded_item_nutrients_raw_sum"] = removed_totals

    acetada, acetada_participants = {}, set()
    columns = ("total_kcal", "total_portion_g", "total_fat_g", "total_carbs_g", "total_protein_g")
    table = pd.read_csv(paths["acetada_csv"], encoding="utf-8-sig")
    for index, row in table.iterrows():
        participant = str(int(row["participant_id"]))
        relative = row.get("before_filepath", "")
        relative = "" if pd.isna(relative) else str(relative)
        if not (Path(paths["acetada_images"]) / participant / Path(relative).name).is_file():
            report["acetada_no_image"] += 1
            continue
        # Keep image-bearing participants in the split even if their labels
        # are later rejected by the original ACETADA loading rules.
        acetada_participants.add(participant)
        if any(pd.isna(row.get(key, float("nan"))) for key in columns):
            report["acetada_invalid_totals"] += 1
            continue
        if any(row[key] <= 0 for key in columns):
            report["acetada_invalid_totals"] += 1
            continue
        nutrition = {key: float(row[column]) for key, column in zip(targets, columns)}
        parts = relative.split("/")
        if len(parts) < 3:
            report["acetada_no_image"] += 1
            continue
        source = Path(paths["acetada_images"]) / parts[1] / parts[2]
        if not source.is_file():
            report["acetada_no_image"] += 1
            continue
        destination = Path(config["workspace"]) / "assets/acetada" / participant / (source.stem + ".jpg")
        image_path = resize_image(source, destination, config)
        items = []
        count = row.get("food_item_count", float("nan"))
        for number in range(1, 1 + (0 if pd.isna(count) else int(count))):
            name = row.get(f"food_item_{number}_name")
            if pd.isna(name):
                continue
            mass = row.get(f"food_item_{number}_before_portion_g")
            items.append({"name": str(name).strip(), "mass_g": None if pd.isna(mass) else float(mass)})
        dish_id = f"acetada_{participant}_{index}"
        acetada[dish_id] = {"dataset": "acetada", "dish_id": dish_id, "participant_id": participant,
                           "row_index": index, "foods": [item["name"] for item in items],
                           "items": items, "views": [{"camera": None, "image_path": image_path,
                           "image_origin": {"source": str(source), "kind": "before_image"}}], **nutrition}

    # One split assignment, shared by every model, paradigm and camera view.
    ids = {}
    for split in ("train", "test"):
        ids[split] = [line.strip() for line in Path(paths[f"nutrition5k_{split}_ids"]).read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#")]
    valid_train = [dish for dish in ids["train"] if dish in n5k_split_ids]
    random.Random(seed).shuffle(valid_train)
    assignment = {dish: "dev" if i < len(valid_train) // 8 else "train"
                  for i, dish in enumerate(valid_train) if dish in n5k}
    assignment.update({dish: "test" for dish in ids["test"] if dish in n5k})
    participants = sorted(acetada_participants, key=int)
    random.Random(seed).shuffle(participants)
    train_end = int(len(participants) * .7)
    participant_split = {pid: "train" if i < train_end else "dev" if i < train_end + int(len(participants) * .1) else "test"
                         for i, pid in enumerate(participants)}
    assignment.update({key: participant_split[dish["participant_id"]] for key, dish in acetada.items()})
    (data_dir / "splits.json").write_text(json.dumps({"dish_split": assignment, "participant_split": participant_split}, indent=2), encoding="utf-8")

    all_dishes = {**n5k, **acetada}
    prepared = {view: {split: [] for split in ("train", "dev", "test")} for view in ("single", "all")}
    for dish_id in sorted(assignment):
        dish = all_dishes[dish_id]
        common = {key: value for key, value in dish.items() if key != "views"}
        common["split"] = assignment[dish_id]
        selected = dish["views"][0]
        if dish["dataset"] == "nutrition5k":
            order = list(cameras)
            random.Random(int(hashlib.md5((dish_id + str(seed)).encode()).hexdigest(), 16)).shuffle(order)
            by_camera = {view["camera"]: view for view in dish["views"]}
            selected = next(by_camera[camera] for camera in order if camera in by_camera)
        for view_name, views in (("single", [selected]), ("all", dish["views"])):
            for view in views:
                prepared[view_name][common["split"]].append({**common, **view,
                    "sample_id": dish_id + (f"_cam{view['camera']}" if view["camera"] else "")})
    for view, splits in prepared.items():
        (data_dir / view).mkdir()
        for split, records in splits.items():
            (data_dir / view / f"{split}.jsonl").write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")

    candidates = {dataset: {food for key, dish in all_dishes.items() if key in assignment
                           and dish["dataset"] == dataset
                           and (config["data"]["candidate_scope"] == "all" or assignment[key] == "train")
                           for food in dish["foods"]} for dataset in ("nutrition5k", "acetada")}
    if config["data"]["candidate_scope"] == "all":
        candidates["acetada"] = {str(value).strip() for key in table.columns
                                 if key.startswith("food_item_") and key.endswith("_name")
                                 for value in table[key].dropna() if str(value).strip()}
        if hf is not None:
            candidates["nutrition5k"] = {item["name"].strip() for row in hf for item in row["ingredients"]
                                         if item["name"].strip()} - exclusions
    candidates = {dataset: sorted(names) for dataset, names in candidates.items()}
    report["n5k_without_official_split"] = sorted(set(n5k) - set(assignment))
    report["counts"] = {view: {split: dict(Counter(row["dataset"] for row in rows))
                               for split, rows in splits.items()} for view, splits in prepared.items()}
    report["candidate_counts"] = {dataset: len(names) for dataset, names in candidates.items()}
    manifest = {"seed": config["seed"], "data_config": config["data"], "inputs": paths,
                "labels": "ACETADA total_* columns; Nutrition5k dish-level totals",
                "single_camera_rule": "MD5(dish_id + str(seed)), shuffle A/B/C/D, first available",
                "split_rule": "Nutrition5k official test + 1/8 train for dev; ACETADA participant 70/10/20"}
    for name, value in (("candidates", candidates), ("preparation_report", report), ("manifest", manifest)):
        (data_dir / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared data: {data_dir}")
    print(report["counts"])
