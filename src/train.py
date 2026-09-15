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
    if not torch.cuda.is_available():
        raise SystemExit("CUDA nie jest dostępna. Uruchom scripts_check_gpu.py.")

    output_dir = args.output_dir
    epochs = args.epochs
    if args.smoke_test:
        output_dir = Path("outputs/qwen3-opensec-smoke")
        epochs = 1.0

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_chat_dataset()
    validate_chat_example(dataset["train"][0])

    if args.smoke_test:
        train_dataset, validation_dataset = select_smoke_data(dataset)
    else:
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, for_training=True)
    lora_config = make_lora_config(args.lora_r)

    training_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length,
        packing=False,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100 if not args.smoke_test else 25,
        save_strategy="steps",
        save_steps=100 if not args.smoke_test else 25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Na niektórych konfiguracjach Windows + CUDA adaptery LoRA
    # powstają jako BF16. GradScaler FP16 nie potrafi wtedy
    # przeskalować ich gradientów, dlatego wymuszamy FP32
    # tylko dla niewielkich, trenowanych parametrów LoRA.
    for parameter in trainer.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)

    trainer.model.print_trainable_parameters()
    result = trainer.train()

    adapter_dir = output_dir / "final_adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

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
