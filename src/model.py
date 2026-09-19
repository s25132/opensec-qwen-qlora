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
    # Kwantyzacja NF4 zapisuje wagi w 4 bitach, ograniczając zużycie pamięci GPU.
    # Podwójna kwantyzacja zmniejsza też pamięć potrzebną na parametry kwantyzacji.
    # Same obliczenia są wykonywane w float16.
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
    # Jeśli tokenizer nie ma tokenu dopełnienia, wykorzystujemy token końca tekstu.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Krótsze sekwencje w partii będą dopełniane z prawej strony.
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name: str = MODEL_NAME,
    for_training: bool = False,
) -> PreTrainedModel:
    # Projekt wymaga GPU z obsługą CUDA do pracy z modelem w 4 bitach.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA nie jest dostępna. Ten projekt zakłada trening/inference 4-bit na GPU."
        )

    # Wczytujemy skwantyzowane wagi z automatycznym rozmieszczeniem warstw.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=make_quantization_config(),
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if for_training:
        # Przygotowujemy model do uczenia adapterów przy zamrożonych wagach bazowych.
        # Checkpointing oszczędza pamięć kosztem ponownego obliczania aktywacji.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
        # Wyłączamy pamięć podręczną generowania na czas treningu z checkpointingiem.
        model.config.use_cache = False
    else:
        # Tryb ewaluacji wyłącza m.in. dropout podczas wnioskowania.
        model.eval()

    return model


def make_lora_config(rank: int = 8) -> LoraConfig:
    # Rank określa rozmiar adapterów, a lora_alpha skaluje ich wpływ na wynik.
    # Adaptery dodajemy do projekcji zapytań, kluczy, wartości i wyjścia uwagi.
    return LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Adaptery będą dodawane do wszystkich projekcji uwagi. 
    )


def attach_adapter(model: PreTrainedModel, adapter_path: str) -> PeftModel:
    # Dołączamy zapisany adapter do modelu bazowego i ustawiamy tryb ewaluacji.
    adapted = PeftModel.from_pretrained(model, adapter_path)
    adapted.eval()
    return adapted
