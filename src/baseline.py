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
    # Rozdziel przykłady na dwie listy: teksty wejściowe i odpowiadające im klasy.
    pairs = [extract_test_pair(example) for example in split]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def main() -> None:
    output_dir = Path("outputs/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_classification_dataset()

    # Model uczy się na zbiorze treningowym; zbiór testowy służy tylko do oceny.
    train_texts, train_labels = unpack(dataset["train"])
    test_texts, test_labels = unpack(dataset["test"])

    # Baseline to prosty punkt odniesienia: TF-IDF + regresja logistyczna.
    # Pipeline łączy zamianę tekstu na cechy liczbowe z klasyfikatorem.
    pipeline = Pipeline(
        [
            (
                "tfidf",
                # TF-IDF nadaje większą wagę słowom charakterystycznym dla tekstu,
                # a mniejszą takim, które występują w wielu dokumentach.
                TfidfVectorizer(
                    # Uwzględnij pojedyncze słowa i pary kolejnych słów.
                    ngram_range=(1, 2),
                    # Pomiń cechy występujące w mniej niż dwóch dokumentach.
                    min_df=2,
                    # Ogranicz słownik do 50 000 cech, aby zmniejszyć zużycie pamięci.
                    max_features=50_000,
                    # Użyj 1 + log(liczby wystąpień), aby ograniczyć wpływ powtórzeń.
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                # Klasyfikator uczy się wag cech pozwalających rozróżniać klasy.
                # max_iter to limit iteracji optymalizatora, a nie liczba epok;
                # uczenie może zakończyć się wcześniej po osiągnięciu zbieżności.
                LogisticRegression(max_iter=2_000, random_state=42),
            ),
        ]
    )
    # Właściwe trenowanie odbywa się w fit(): najpierw TF-IDF buduje słownik
    # i oblicza wagi IDF wyłącznie na tekstach treningowych, a następnie
    # przekształca je w macierz cech. Regresja logistyczna dopasowuje swoje wagi
    # do tej macierzy i znanych etykiet, minimalizując regularyzowaną stratę.
    # Dzięki temu informacje ze zbioru testowego nie trafiają do procesu uczenia.
    pipeline.fit(train_texts, train_labels)

    # Podczas predykcji używane są już wyuczone słownik, IDF i wagi klasyfikatora.
    # predict() przetwarza nowe teksty i wybiera ich klasy bez ponownego uczenia.
    predictions = pipeline.predict(test_texts)

    # Accuracy to odsetek poprawnych odpowiedzi, a macro F1 to średnia F1
    # po klasach z LABELS, w której każda klasa ma taką samą wagę.
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
    # Raport pokazuje precision, recall i F1 dla poszczególnych klas.
    # zero_division=0 ustawia wynik na 0, gdy dana miara wymaga dzielenia przez 0.
    report = classification_report(
        test_labels, predictions, labels=LABELS, zero_division=0
    )
    # Wiersze macierzy oznaczają klasy prawdziwe, a kolumny klasy przewidziane.
    matrix = confusion_matrix(test_labels, predictions, labels=LABELS)

    # Zapisz predykcje i wyniki oceny, aby można było przeanalizować błędy modelu.
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

    # Na przekątnej są poprawne odpowiedzi; pozostałe pola pokazują pomyłki.
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
