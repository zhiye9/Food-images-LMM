"""Train LoRA SFT"""
import argparse
import json
from collections import Counter
from pathlib import Path

from utils import load_config, load_samples, start_run
from models import load_processor


class Collator:
    def __init__(self, processor, family, max_length):
        self.processor, self.max_length = processor, max_length
        self.marker = processor.tokenizer.encode(
            "<|im_start|>assistant\n" if family == "qwen3_vl" else "<start_of_turn>model\n",
            add_special_tokens=False)
        self.prompt = (Path(__file__).resolve().parent / "prompts/image_only.txt").read_text(encoding="utf-8").rstrip("\n")

    def __call__(self, batch):
        from PIL import Image
        images, messages = [], []
        try:
            for sample in batch:
                with Image.open(sample["image_path"]) as image:
                    images.append(image.convert("RGB"))
                messages.append([
                    {"role": "user", "content": [{"type": "image", "image": images[-1]},
                                                  {"type": "text", "text": self.prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text":
                        f"Foods: {', '.join(sample['foods']) or 'unknown'}\nTotal mass (g): {sample['total_mass_g']:.1f}\n"
                        f"Energy (kcal): {sample['total_kcal']:.1f}\nFat (g): {sample['total_fat_g']:.1f}\n"
                        f"Carbs (g): {sample['total_carbs_g']:.1f}\nProtein (g): {sample['total_protein_g']:.1f}"}]}])

            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt", padding=True,
                truncation=self.max_length is not None, max_length=self.max_length,
                add_special_tokens=False, add_generation_prompt=False, enable_thinking=False)
        finally:
            for image in images:
                image.close()
        labels = inputs["input_ids"].clone()
        for index, row in enumerate(labels.tolist()):
            labels[index, :next((i + len(self.marker) for i in range(len(row) - len(self.marker) + 1)
                                if row[i:i + len(self.marker)] == self.marker), len(row))] = -100
        labels[inputs["input_ids"] == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=("qwen3-vl-4b", "qwen3-vl-8b", "gemma3-4b", "gemma3-12b"), required=True)
    parser.add_argument("--view", choices=("single", "all"), default="single")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train_samples = load_samples(config, args.view, "train")
    dev_samples = load_samples(config, args.view, "dev")
    from accelerate import PartialState
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import WeightedRandomSampler
    from transformers import AutoModelForImageTextToText, EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

    state = PartialState()
    output = Path(args.output).expanduser().resolve()
    if state.is_main_process:
        start_run(output, config, train_samples, paradigm="sft_training", model=args.model, view=args.view,
                  dev_sample_ids=[s["sample_id"] for s in dev_samples], arguments=vars(args))
    state.wait_for_everyone()
    set_seed(int(config["seed"]))
    entry, settings = config["models"][args.model], config["training"]
    processor = load_processor(config, args.model, training=True)

    model = AutoModelForImageTextToText.from_pretrained(
        entry["path"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=settings["lora_r"], lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"], bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_weights = None
    if settings["balance_datasets"]:
        counts = Counter(s["dataset"] for s in train_samples)
        train_weights = [counts["nutrition5k"] / counts["acetada"] if s["dataset"] == "acetada" else 1.
                         for s in train_samples]

    class WeightedTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            if train_weights is not None:
                return WeightedRandomSampler(train_weights, len(self.train_dataset), replacement=True)
            return super()._get_train_sampler(train_dataset)

    trainer = WeightedTrainer(
        model=model, args=TrainingArguments(
            output_dir=str(output / "checkpoints"), num_train_epochs=settings["epochs"],
            per_device_train_batch_size=entry["train_batch_size"],
            per_device_eval_batch_size=entry["train_batch_size"],
            gradient_accumulation_steps=entry["gradient_accumulation_steps"],
            learning_rate=settings["learning_rate"], warmup_ratio=settings["warmup_ratio"],
            weight_decay=.01, max_grad_norm=1., lr_scheduler_type="cosine", optim="adamw_torch",
            bf16=True, tf32=True, logging_steps=5, eval_strategy="steps", save_strategy="steps",
            eval_steps=settings["eval_steps"], save_steps=settings["eval_steps"], save_total_limit=3,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            report_to="none", dataloader_num_workers=0, dataloader_pin_memory=False,
            remove_unused_columns=False, ddp_find_unused_parameters=False, label_names=["labels"],
            seed=int(config["seed"]), data_seed=int(config["seed"])),
        train_dataset=train_samples, eval_dataset=dev_samples,
        data_collator=Collator(processor, entry["family"], settings["max_length"]),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=settings["early_stopping_patience"])])
    trainer.train()

    trainer.save_model(str(output / "lora_adapter"))
    if state.is_main_process:
        processor.save_pretrained(str(output / "processor"))
        (output / "adapter_manifest.json").write_text(json.dumps({
            "model": args.model, "model_path": entry["path"], "view": args.view,
            "targets": ["total_kcal", "total_mass_g", "total_fat_g", "total_carbs_g", "total_protein_g"],
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_dev_loss": trainer.state.best_metric,
            "training_scope": "mixed Nutrition5k + ACETADA train; dev selection; no test use"}, indent=2), encoding="utf-8")
        (output / "completion.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    state.wait_for_everyone()
