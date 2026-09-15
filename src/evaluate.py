from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import LABELS, MODEL_NAME
from .data import (
    extract_test_pair,
    get_system_prompt,
    load_chat_dataset,
    load_classification_dataset,
)
from .inference import classify_event
from .model import attach_adapter, load_base_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ewaluacja klasyfikatora OpenSec")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chat_dataset = load_chat_dataset()
    system_prompt = get_system_prompt(chat_dataset)
    eval_dataset = load_classification_dataset()["test"]
    if args.limit is not None:
        eval_dataset = eval_dataset.select(range(min(args.limit, len(eval_dataset))))

    tokenizer_source = args.adapter or args.model
    tokenizer = load_tokenizer(tokenizer_source)
    model = load_base_model(args.model, for_training=False)
    if args.adapter:
        model = attach_adapter(model, args.adapter)

    rows: list[dict[str, str]] = []
    for index, example in enumerate(eval_dataset, start=1):
        text, expected = extract_test_pair(example)
        predicted, raw_output = classify_event(
            model,
            tokenizer,
            system_prompt,
            text,
            max_input_length=args.max_length,
        )
        rows.append(
            {
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "raw_output": raw_output,
            }
        )
        if index % 25 == 0 or index == len(eval_dataset):
            print(f"Przetworzono {index}/{len(eval_dataset)}")

    frame = pd.DataFrame(rows)
    expected_values = frame["expected"].tolist()
    predicted_values = frame["predicted"].tolist()
    invalid_count = predicted_values.count("invalid")

    metrics = {
        "model": args.model,
        "adapter": args.adapter,
        "examples": len(frame),
        "accuracy": accuracy_score(expected_values, predicted_values),
        "macro_f1": f1_score(
            expected_values,
            predicted_values,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "invalid_outputs": invalid_count,
        "invalid_rate": invalid_count / len(frame) if len(frame) else 0.0,
    }

    report = classification_report(
        expected_values,
        predicted_values,
        labels=LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(expected_values, predicted_values, labels=LABELS)

    frame.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (args.output_dir / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.xlabel("Predykcja")
    plt.ylabel("Prawidłowa klasa")
    plt.title("OpenSec Triage — macierz pomyłek")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print("\n", report)


if __name__ == "__main__":
    main()
