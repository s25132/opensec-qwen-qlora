from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import MODEL_NAME


def make_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_tokenizer(
    model_name: str = MODEL_NAME,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name: str = MODEL_NAME,
    for_training: bool = False,
) -> PreTrainedModel:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA nie jest dostępna. Ten projekt zakłada trening/inference 4-bit na GPU."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=make_quantization_config(),
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if for_training:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
        model.config.use_cache = False
    else:
        model.eval()

    return model


def make_lora_config(rank: int = 8) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def attach_adapter(model: PreTrainedModel, adapter_path: str) -> PeftModel:
    adapted = PeftModel.from_pretrained(model, adapter_path)
    adapted.eval()
    return adapted
