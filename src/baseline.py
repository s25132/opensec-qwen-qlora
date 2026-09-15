from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from .config import LABELS
from .data import extract_test_pair, load_classification_dataset


def unpack(split) -> tuple[list[str], list[str]]:
    pairs = [extract_test_pair(example) for example in split]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def main() -> None:
    output_dir = Path("outputs/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_classification_dataset()

    train_texts, train_labels = unpack(dataset["train"])
    test_texts, test_labels = unpack(dataset["test"])

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=2_000, random_state=42),
            ),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)

    metrics = {
        "model": "TF-IDF + LogisticRegression",
        "examples": len(test_labels),
        "accuracy": accuracy_score(test_labels, predictions),
        "macro_f1": f1_score(
            test_labels,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
    }
    report = classification_report(
        test_labels, predictions, labels=LABELS, zero_division=0
    )
    matrix = confusion_matrix(test_labels, predictions, labels=LABELS)

    pd.DataFrame(
        {
            "text": test_texts,
            "expected": test_labels,
            "predicted": predictions,
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.xlabel("Predykcja")
    plt.ylabel("Prawidłowa klasa")
    plt.title("TF-IDF baseline — macierz pomyłek")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print("\n", report)


if __name__ == "__main__":
    main()
