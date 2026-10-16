"""Qwen3-VL / Gemma3 loading and processing across paradigms."""
def load_processor(config, model_name, training=False):
    from transformers import AutoProcessor
    entry = config["models"][model_name]
    kwargs = {"local_files_only": True}
    if entry["family"] == "qwen3_vl":
        kwargs.update({key: config["generation"][key] for key in ("min_pixels", "max_pixels")})
    processor = AutoProcessor.from_pretrained(entry["path"], **kwargs)
    processor.tokenizer.padding_side = "right" if training else "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


class VisionModel:
    def __init__(self, config, model_name, adapter=None):
        import torch
        from transformers import AutoModelForImageTextToText, set_seed
        set_seed(int(config["seed"]))
        self.config = config
        self.processor = load_processor(config, model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config["models"][model_name]["path"], torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True)
        self.model.config.use_cache = True
        self.model.config.text_config.use_cache = True
        self.model.to("cuda")
        self.model.eval()
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False,
                                                   local_files_only=True)
            self.model.eval()

    def generate(self, prompts, images=None, max_new_tokens=None):
        import torch
        inputs = self.processor.apply_chat_template(
            [[{"role": "user", "content":
               ([{"type": "image", "image": images[index]}] if images is not None else [])
               + [{"type": "text", "text": prompt}]}] for index, prompt in enumerate(prompts)],
            tokenize=True, return_dict=True, return_tensors="pt", padding=True,
            truncation=False, add_special_tokens=False, add_generation_prompt=True,
            enable_thinking=False).to(self.model.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=False, num_beams=1,
                max_new_tokens=max_new_tokens or self.config["generation"]["max_new_tokens"],
                pad_token_id=self.processor.tokenizer.pad_token_id)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)

    def text(self, prompt):
        return self.generate([prompt], max_new_tokens=self.config["generation"]["rag_max_new_tokens"])[0]

    def image(self, prompt, image):
        return self.generate([prompt], [image], self.config["generation"]["rag_max_new_tokens"])[0]
