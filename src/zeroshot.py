"""Zero-shot benchmarking"""
import argparse
import json
from pathlib import Path

import yaml

from utils import add_eval_arguments, evaluate, load_config, load_samples, start_run
from models import VisionModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_eval_arguments(parser)
    parser.add_argument("--protocol", choices=("image_only", "five_modes"), default="image_only")
    modes = ("items_with_true_weights", "items_names_only", "total_mass_only",
             "candidate_items_with_total_mass", "candidate_items_names_only")
    parser.add_argument("--modes", choices=modes, nargs="+", default=list(modes))
    parser.add_argument("--prompt-types", choices=("direct_estimation", "itemized_analysis"),
                        nargs="+", default=["direct_estimation", "itemized_analysis"])
    args = parser.parse_args()
    config = load_config(args.config)
    samples = load_samples(config, args.view, args.split, args.datasets, args.limit_dishes)
    output = start_run(args.output, config, samples, paradigm="zeroshot",
                       model=args.model, view=args.view, arguments=vars(args))
    model = VisionModel(config, args.model)
    if args.protocol == "image_only":
        evaluate(model, config, args.model, samples, output)
        raise SystemExit

    templates = yaml.safe_load((Path(__file__).resolve().parent / "prompts/benchmark.yaml").read_text(encoding="utf-8"))
    candidates = json.loads((Path(config["workspace"]) / "data/candidates.json").read_text(encoding="utf-8"))
    for mode in args.modes:
        for style in args.prompt_types:
            settings = templates[mode][style]
            prompts = []
            for sample in samples:
                prompt = settings["template"]
                if mode == "items_with_true_weights":
                    prompt = prompt.replace("{item_list_with_weights}",
                        "\n".join(f"- {item['name']}: {item['mass_g']:.1f} g" for item in sample["items"]))
                elif mode == "items_names_only":
                    prompt = prompt.replace("{item_names_list}", "\n".join(f"- {name}" for name in sample["foods"]))
                elif mode in ("total_mass_only", "candidate_items_with_total_mass"):
                    prompt = prompt.replace("{total_mass_g}", f"{sample['total_mass_g']:.1f}")
                if mode in ("candidate_items_with_total_mass", "candidate_items_names_only") and candidates[sample["dataset"]]:
                    prompt = prompt.replace("{candidate_items_list}", ", ".join(candidates[sample["dataset"]]))
                prompts.append(prompt)
            evaluate(model, config, args.model, samples, output / mode / style, prompts,
                     benchmark=True, max_new_tokens={**templates.get("defaults", {}).get("generation_config", {}),
                         **settings.get("generation_config", {})}.get("max_new_tokens", 1200 if style == "itemized_analysis" else 500))
    (output / "completion.json").write_text(json.dumps({"status": "complete", "modes": args.modes,
                                                        "prompt_types": args.prompt_types}), encoding="utf-8")
