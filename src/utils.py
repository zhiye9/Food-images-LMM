import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

def load_config(path):
    import yaml
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for container, key in [(config, "workspace"), *[(config["paths"], key) for key in config["paths"]],
                           *[(entry, "path") for entry in config["models"].values()],
                           (config["rag"], "embedding_model")]:
        if container[key] is not None:
            container[key] = str((path.parent / Path(os.path.expandvars(str(container[key]))).expanduser()).resolve())
    config["config_file"] = str(path)
    return config


def load_samples(config, view="single", split="test", datasets=("nutrition5k", "acetada"), limit_dishes=None):
    """Read the prepared split; the optional limit keeps all views of each dish."""
    seen = {dataset: set() for dataset in datasets}
    samples = []
    for line in (Path(config["workspace"]) / "data" / view / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        dataset, dish = record["dataset"], record["dish_id"]
        if dataset not in seen:
            continue
        if limit_dishes is not None and dish not in seen[dataset] and len(seen[dataset]) >= limit_dishes:
            continue
        seen[dataset].add(dish)
        samples.append(record)
    return samples


def start_run(path, config, samples, **metadata):
    """Create a fresh experiment directory and record its settings."""
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    (path / "run.json").write_text(json.dumps({
        "config": config, "sample_ids": [s["sample_id"] for s in samples],
        "metric_definition": "mean of five per-target wMAPE percentages", **metadata,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def has_nutrition(prediction):
    """Use the original SFT/RAG rule: all five values must be parsed."""
    return prediction is not None and all(prediction.get(key) is not None for key in (
        "total_kcal", "total_mass_g", "total_fat_g", "total_carbs_g", "total_protein_g"))


def parse_nutrition(text):
    """Parse the six-line nutrition response using the original SFT patterns."""
    text = text.replace("*", "")
    result = {}
    for key, label in (("total_kcal", "Energy (kcal)"), ("total_mass_g", "Total mass (g)"),
                       ("total_fat_g", "Fat (g)"), ("total_carbs_g", "Carbs (g)"),
                       ("total_protein_g", "Protein (g)")):
        result[key] = None
        match = re.search(re.escape(label) + r":\s*([\d.]+)", text, re.IGNORECASE)
        if match:
            try:
                result[key] = float(match.group(1))
            except ValueError:
                pass
    foods = re.search(r"Foods:\s*(.+)", text, re.IGNORECASE)
    result["foods_text"] = foods.group(1).strip() if foods else ""
    return result

def _resolve_arithmetic(text: str) -> str:
    """
    Resolve inline arithmetic expressions in model output.

    Models sometimes show their work in the summary section, e.g.:
      Total calories: 140 + 160 + 220 + 160 + 0 = 680 kcal
      Fat: (6.5 + 2 + 8 + 15 + 0) = 31.5 g
    This replaces such expressions so only the final value remains:
      Total calories: 680 kcal
      Fat: 31.5 g
    """
    return re.sub(
        r'(:\s*)\(?\d+\.?\d*(?:\s*\+\s*\d+\.?\d*)+\)?\s*=\s*(\d+\.?\d*)',
        r'\1\2',
        text,
    )


def parse_total_nutrition(output: str) -> Optional[Dict[str, float]]:
    """
    Extract total nutrition values from model output.

    Returns:
        Dict with keys: calories, mass, fat, carbs, protein.
        None if minimum requirement (calories) not met.
    """
    result = {
        'calories': None,
        'mass': None,
        'fat': None,
        'carbs': None,
        'protein': None,
    }

    try:
        json_match = re.search(r'\{[^{}]*\}', output, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            for target, possible in {
                'calories': ['calories', 'total_calories', 'kcal', 'energy',
                             'energy_kcal', 'total_calories_kcal', 'calorie'],
                'mass': ['mass', 'total_mass', 'weight', 'grams',
                         'total_weight', 'portion', 'total_mass_grams'],
                'fat': ['fat', 'fat_grams', 'total_fat', 'fats', 'fat_g'],
                'carbs': ['carbs', 'carbohydrates', 'total_carb', 'carb',
                          'carbs_g', 'carbohydrate', 'carbohydrates_grams'],
                'protein': ['protein', 'protein_grams', 'total_protein',
                            'proteins', 'protein_g'],
            }.items():
                for pk in possible:
                    if pk in data and data[pk] is not None:
                        try:
                            result[target] = float(data[pk])
                            break
                        except (ValueError, TypeError):
                            continue
            if result['calories'] is not None:
                return result
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    output_lower = re.sub(r'\*{1,3}', '', output).lower()

    summary_section = output_lower
    for marker in ['total nutrition summary', 'stage 3', 'stage 4',
                   'step 2', 'step 3',
                   'total mass', 'total calories']:
        idx = output_lower.rfind(marker)
        if idx >= 0:
            summary_section = output_lower[idx:]
            break
    summary_section = _resolve_arithmetic(summary_section)

    approx_prefix = r'(?:approximately|approx\.?|about|around|roughly|estimated|~|≈)?\s*'
    patterns = {
        'calories': [
            r'total\s*calories?\s*:\s*' + approx_prefix + r'(\d+\.?\d*)',
            r'calories?\s*:\s*' + approx_prefix + r'(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*kcal',
        ],
        'mass': [
            r'total\s*mass\s*:\s*' + approx_prefix + r'(\d+\.?\d*)',
            r'(?:mass|weight|portion)\s*:\s*' + approx_prefix + r'(\d+\.?\d*)',
        ],
        'fat': [
            r'fat\s*:\s*' + approx_prefix + r'(\d+\.?\d*)',
        ],
        'carbs': [
            r'carbohydrates?\s*:\s*' + approx_prefix + r'(\d+\.?\d*)',
            r'carbs?\s*:\s*' + approx_prefix + r'(\d+\.?\d*)',
        ],
        'protein': [
            r'protein\s*:\s*' + approx_prefix + r'(\d+\.?\d*)',
        ],
    }

    for key, pattern_list in patterns.items():
        if result[key] is not None:
            continue
        for pattern in pattern_list:
            match = re.search(pattern, summary_section)
            if match:
                result[key] = float(match.group(1))
                break

    # fallback
    if result['calories'] is None:
        output_lower_resolved = _resolve_arithmetic(output_lower)
        for key, pattern_list in patterns.items():
            if result[key] is not None:
                continue
            for pattern in pattern_list:
                match = re.search(pattern, output_lower_resolved)
                if match:
                    result[key] = float(match.group(1))
                    break

    if result['calories'] is not None:
        return result
    return None


def make_prediction_record(sample, predicted, raw_output, **extra):
    targets = ("total_kcal", "total_mass_g", "total_fat_g", "total_carbs_g", "total_protein_g")
    return {
        **{key: sample[key] for key in ("sample_id", "dish_id", "dataset", "split", "camera", "image_path")},
        "ground_truth": {key: sample[key] for key in targets},
        "predicted": {key: predicted.get(key) for key in targets} if predicted is not None else None,
        "foods_text": predicted.get("foods_text", "") if predicted else "",
        "parse_success": has_nutrition(predicted), "raw_output": raw_output, **extra,
    }


def save_evaluation(path, records):
    import numpy as np
    dishes = defaultdict(list)
    for record in records:
        dishes[(record["dataset"], record["dish_id"])].append(record)
    averaged = []
    targets = ("total_kcal", "total_mass_g", "total_fat_g", "total_carbs_g", "total_protein_g")
    for group in dishes.values():
        good = [r for r in group if has_nutrition(r["predicted"])]
        averaged.append({"dataset": group[0]["dataset"], "ground_truth": group[0]["ground_truth"],
                         "predicted": {key: sum(r["predicted"][key] for r in good) / len(good)
                                       for key in targets} if good else None})
    result = {"definition": "100 * sum(abs(pred-gt)) / sum(abs(gt)); mean over all five targets",
              "targets": list(targets),
              "failure_policy": "require five parsed predictions; report coverage; no zero imputation",
              "per_camera": {}, "per_dish": {}}
    for name in ("all", "nutrition5k", "acetada"):
        for level, source in (("per_camera", records), ("per_dish", averaged)):
            selected = source if name == "all" else [r for r in source if r["dataset"] == name]
            good = [record for record in selected if has_nutrition(record["predicted"])]
            metrics = {"n_total": len(selected), "n_valid": len(good),
                       "parse_rate_pct": 100 * len(good) / len(selected) if selected else 0.0}
            values = []
            for key in targets:
                truth = np.array([r["ground_truth"][key] for r in good], dtype=float)
                errors = np.abs(np.array([r["predicted"][key] for r in good], dtype=float) - truth)
                denominator = float(np.abs(truth).sum())
                wmape = float(100 * errors.sum() / denominator) if denominator > 0 else None
                metrics[key] = {"wMAPE": wmape, "MAE": float(errors.mean()) if good else None,
                                "n_valid": len(good), "sum_abs_ground_truth": denominator}
                values.append(wmape)
            metrics["mean_wMAPE"] = float(np.mean(values)) if all(v is not None for v in values) else None
            metrics["mean_wMAPE_5targets"] = metrics["mean_wMAPE"]
            result[level][name] = metrics
    path = Path(path)
    (path / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (path / "completion.json").write_text(json.dumps({"status": "complete", "n_predictions": len(records)}), encoding="utf-8")


def add_eval_arguments(parser):
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=("qwen3-vl-4b", "qwen3-vl-8b", "gemma3-4b", "gemma3-12b"), required=True)
    parser.add_argument("--view", choices=("single", "all"), default="single")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    datasets = ("nutrition5k", "acetada")
    parser.add_argument("--datasets", choices=datasets, nargs="+", default=list(datasets))
    parser.add_argument("--limit-dishes", type=int, help="First N frozen dishes per dataset, all their selected views")
    parser.add_argument("--output", required=True)


def evaluate(model, config, model_name, samples, output, prompts=None, benchmark=False,
             max_new_tokens=None):
    """Shared frozen-split evaluation for zero-shot, SFT and the RAG baseline."""
    from PIL import Image
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    batch_size = int(config["models"][model_name]["inference_batch_size"])
    prompts = prompts if prompts is not None else [
        (Path(__file__).resolve().parent / "prompts/image_only.txt").read_text(encoding="utf-8").rstrip("\n")
    ] * len(samples)
    records = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            images = []
            try:
                for sample in batch:
                    with Image.open(sample["image_path"]) as image:
                        images.append(image.convert("RGB"))
                batch_prompts = prompts[start:start + batch_size]
                for sample, prompt, text in zip(batch, batch_prompts, model.generate(batch_prompts, images, max_new_tokens)):
                    if benchmark:
                        nutrition = parse_total_nutrition(text) or {}
                        prediction = {target: nutrition.get(key) for target, key in (
                            ("total_kcal", "calories"), ("total_mass_g", "mass"), ("total_fat_g", "fat"),
                            ("total_carbs_g", "carbs"), ("total_protein_g", "protein"))}
                    else:
                        prediction = parse_nutrition(text)
                    record = make_prediction_record(sample, prediction, text, prompt=prompt)
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            finally:
                for image in images:
                    image.close()
            print(f"{output.name}: {min(start + batch_size, len(samples))}/{len(samples)}", flush=True)
    save_evaluation(output, records)
    return records
