from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from trl import SFTConfig, SFTTrainer

from .config import DEFAULT_OUTPUT_DIR, MODEL_NAME
from .data import load_chat_dataset, select_smoke_data, validate_chat_example
from .model import load_base_model, load_tokenizer, make_lora_config


def parse_args() -> argparse.Namespace:
    # Parametry uruchomienia pozwalają zmieniać trening bez edytowania kodu.
    parser = argparse.ArgumentParser(description="QLoRA SFT Qwen3 na OpenSec Triage")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Trening modelu w kwantyzacji 4-bitowej zakłada dostęp do GPU z CUDA.
    if not torch.cuda.is_available():
        raise SystemExit("CUDA nie jest dostępna. Uruchom scripts_check_gpu.py.")

    output_dir = args.output_dir
    epochs = args.epochs
    if args.smoke_test:
        # Krótka próba działania: jedna epoka, mniejszy zbiór i osobny katalog.
        output_dir = Path("outputs/qwen3-opensec-smoke")
        epochs = 1.0

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_chat_dataset()
    # Sprawdź strukturę pierwszej rozmowy i poprawność etykiety odpowiedzi.
    validate_chat_example(dataset["train"][0])

    if args.smoke_test:
        train_dataset, validation_dataset = select_smoke_data(dataset)
    else:
        # Zbiór treningowy służy do uczenia, a walidacyjny do oceny w jego trakcie.
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    # Tokenizer zamienia tekst na identyfikatory tokenów zrozumiałe dla modelu.
    tokenizer = load_tokenizer(args.model)
    # QLoRA łączy zamrożony model bazowy w 4 bitach z uczonymi adapterami LoRA.
    # Funkcja przygotowuje model bazowy do treningu z kwantyzacją.
    model = load_base_model(args.model, for_training=True)
    # r określa rangę adapterów: większa wartość zwiększa liczbę uczonych wag.
    lora_config = make_lora_config(args.lora_r)

    # SFT to nadzorowane dostrajanie modelu językowego na przykładach rozmów.
    training_config = SFTConfig(
        output_dir=str(output_dir),
        # Epoka oznacza jedno przejście przez zbiór treningowy.
        num_train_epochs=epochs,
        # Współczynnik uczenia steruje wielkością aktualizacji wag adapterów.
        learning_rate=args.learning_rate,
        # Liczba przykładów przetwarzanych jednocześnie na urządzeniu.
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        # Zbieraj gradienty z kilku partii przed aktualizacją wag. Przy jednym
        # GPU domyślne wartości dają efektywną partię 1 * 8 = 8 przykładów.
        gradient_accumulation_steps=args.gradient_accumulation,
        # Limit długości sekwencji w tokenach; dłuższe przykłady są przycinane.
        max_length=args.max_length,
        # Nie łącz kilku przykładów w jedną sekwencję treningową.
        packing=False,
        # Mieszana precyzja FP16 ogranicza pamięć części obliczeń treningowych.
        fp16=True,
        bf16=False,
        # Oszczędzaj VRAM, odtwarzając część aktywacji podczas liczenia gradientów
        # zamiast przechowywać je wszystkie; kosztem są dodatkowe obliczenia.
        gradient_checkpointing=True,
        # AdamW z 8-bitowymi stanami i stronicowaniem ogranicza zużycie pamięci.
        optim="paged_adamw_8bit",
        # Kroki oznaczają aktualizacje optymalizatora, po akumulacji gradientów.
        logging_steps=10,
        # Regularnie mierz stratę na walidacji bez aktualizowania wag.
        eval_strategy="steps",
        eval_steps=100 if not args.smoke_test else 25,
        # Zapisuj checkpointy w tym samym rytmie co pomiary walidacyjne.
        save_strategy="steps",
        save_steps=100 if not args.smoke_test else 25,
        save_total_limit=2,
        # Na końcu wczytaj checkpoint z najniższą zmierzoną stratą walidacyjną.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Wyłącz zewnętrzne integracje do raportowania wyników.
        report_to="none",
        # Ustal ziarno losowości, aby ułatwić odtwarzanie eksperymentu.
        seed=42,
    )

    # Trainer przygotowuje dane z tokenizerem, dołącza adaptery według
    # peft_config i obsługuje pętlę treningową, walidację oraz checkpointy.
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Przechowuj tylko uczone parametry w FP32, aby uniknąć problemów z obsługą
    # gradientów FP16. Zamrożone wagi bazowe pozostają w dotychczasowym formacie;
    # mieszana precyzja nadal pozwala wykonywać część obliczeń w FP16.
    for parameter in trainer.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)

    # Pokaż, jak małą część wszystkich parametrów faktycznie dostrajamy.
    trainer.model.print_trainable_parameters()
    # Właściwy trening: model przewiduje kolejne tokeny, trainer oblicza stratę
    # i gradienty, a optymalizator aktualizuje uczone wagi po ich akumulacji.
    # Walidacja i zapisy odbywają się zgodnie z harmonogramem z SFTConfig.
    result = trainer.train()

    # Zapisz adapter wybrany po treningu oraz tokenizer. Do późniejszego użycia
    # adaptera potrzebny jest także model bazowy wskazany przez args.model.
    adapter_dir = output_dir / "final_adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Zapisz parametry eksperymentu i średnią stratę treningową; train_loss
    # nie jest accuracy ani stratą walidacyjną najlepszego checkpointu.
    summary = {
        "model": args.model,
        "smoke_test": args.smoke_test,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "epochs": epochs,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "train_loss": result.training_loss,
        "adapter_path": str(adapter_dir),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
