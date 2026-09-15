from __future__ import annotations

import re

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import LABELS


def normalize_prediction(output: str) -> str:
    normalized = output.strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z_]", "", normalized)

    if normalized in LABELS:
        return normalized

    # Akceptujemy nazwę klasy wewnątrz krótkiej odpowiedzi, ale wynik taki
    # nadal warto analizować w predictions.csv.
    matches = [label for label in LABELS if label in normalized]
    return matches[0] if len(matches) == 1 else "invalid"


def render_prompt(
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
    event_text: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": event_text},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback dla wersji tokenizera bez argumentu enable_thinking.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def classify_event(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
    event_text: str,
    max_input_length: int = 256,
) -> tuple[str, str]:
    prompt = render_prompt(tokenizer, system_prompt, event_text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return normalize_prediction(raw_output), raw_output
