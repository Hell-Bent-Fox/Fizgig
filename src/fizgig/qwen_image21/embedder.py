"""Qwen Image 2.1 text conditioning: Qwen3-VL-8B, last decoder layer BEFORE the final RMSNorm.

Follows the reference pipeline (diffusers pipeline_qwenimage21.py `_get_qwen_prompt_embeds`): the prompt goes into a
raw template string (not apply_chat_template - "the two tokenize differently and the checkpoint expects this one"),
left padding, `hidden_states[-1]` with the final norm neutralised by a forward hook, the system-turn tokens dropped,
an empty prompt becomes " ". Text-to-image only for now (no reference images).

Weights: the ComfyUI single file (bare `model.` / `model.visual.` keys, converted by the Krea 2 loader's converter)
or official shards. Tokenizer/processor: the Qwen-Image-2.1 `processor/` files (they differ from the Qwen3-VL-4B copy
bundled for Krea 2).
"""
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

SYS_PROMPT = "Comprehend and analyze the provided prompt."
TEMPLATE_T2I = (f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n"
                "<|im_start|>assistant\n")


def _text_encoder_config(config_path=None) -> dict:
    here = os.path.join(os.path.dirname(__file__), "qwen3vl_8b_config.json")
    with open(config_path or here, encoding="utf-8") as f:
        return json.load(f)


class Qwen21TextEncoder:
    def __init__(self, model_path, tokenizer_dir, device="cuda", dtype=torch.bfloat16, config_path=None):
        from accelerate import init_empty_weights
        from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

        from fizgig.krea2.embedder import _convert_comfyui_qwen3vl_state_dict
        from fizgig.krea2.safetensors_utils import load_split_weights

        self.device, self.dtype = torch.device(device), dtype
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        config = Qwen3VLConfig.from_dict(_text_encoder_config(config_path))
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration._from_config(config)
        sd = _convert_comfyui_qwen3vl_state_dict(load_split_weights(model_path, device="cpu", dtype=dtype))
        info = model.load_state_dict(sd, strict=False, assign=True)
        if info.unexpected_keys or info.missing_keys:
            raise RuntimeError(f"Qwen3-VL-8B checkpoint mismatch: missing={info.missing_keys[:8]}, "
                               f"unexpected={info.unexpected_keys[:8]}")
        self.model = model.to(self.device, dtype).eval().requires_grad_(False)
        # Number of leading system-turn tokens to drop (the reference derives it from the tokenized system message).
        sys_ids = self.tokenizer(f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n", add_special_tokens=False).input_ids
        self.drop_idx = len(sys_ids)

    @torch.no_grad()
    def encode(self, prompts):
        """-> list of [L_i, 4096] tensors (variable length, padding removed), in the model dtype, on CPU."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        texts = [TEMPLATE_T2I.format(p if p else " ") for p in prompts]
        tok = self.tokenizer(texts, padding=True, padding_side="left", return_tensors="pt").to(self.device)
        lm = getattr(self.model.model, "language_model", self.model.model)
        # hidden_states[-1] must be the last decoder layer BEFORE the final RMSNorm (what the DiT was trained on).
        # transformers 4.x already returns that; the hook makes it hold on 5.x too.
        handle = lm.norm.register_forward_hook(lambda m, args, out: args[0])
        try:
            out = self.model(input_ids=tok.input_ids, attention_mask=tok.attention_mask, output_hidden_states=True)
        finally:
            handle.remove()
        hs = out.hidden_states[-1]
        res = []
        for h, m in zip(hs, tok.attention_mask.bool()):
            res.append(h[m][self.drop_idx:].to("cpu"))
        return res

    def unload(self):
        self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()
