from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from .config import DATASET_NAME, EVAL_CONFIG, LABELS, TRAIN_CONFIG


def load_chat_dataset() -> DatasetDict:
    return load_dataset(DATASET_NAME, TRAIN_CONFIG)


def load_classification_dataset() -> DatasetDict:
    return load_dataset(DATASET_NAME, EVAL_CONFIG)


def get_system_prompt(chat_dataset: DatasetDict) -> str:
    messages = chat_dataset["train"][0]["messages"]
    system_messages = [m["content"] for m in messages if m["role"] == "system"]
    if len(system_messages) != 1:
        raise ValueError("Oczekiwano dokładnie jednej wiadomości systemowej.")
    return system_messages[0]


def validate_chat_example(example: dict[str, Any]) -> None:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Rekord musi zawierać trzy wiadomości.")

    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"Nieprawidłowa kolejność ról: {roles}")

    label = messages[2].get("content", "").strip().lower()
    if label not in LABELS:
        raise ValueError(f"Nieznana etykieta: {label}")


def select_smoke_data(
    dataset: DatasetDict,
    train_size: int = 350,
    validation_size: int = 70,
) -> tuple[Dataset, Dataset]:
    train_size = min(train_size, len(dataset["train"]))
    validation_size = min(validation_size, len(dataset["validation"]))

    train = dataset["train"].shuffle(seed=42).select(range(train_size))
    validation = (
        dataset["validation"].shuffle(seed=42).select(range(validation_size))
    )
    return train, validation


def extract_test_pair(example: dict[str, Any]) -> tuple[str, str]:
    """Extract input and target from the non-chat edge_trainer representation."""
    text = str(example["text"]).strip()
    target = str(example["target"]).strip().lower()
    if target not in LABELS:
        raise ValueError(f"Nieznana etykieta testowa: {target}")
    return text, target
