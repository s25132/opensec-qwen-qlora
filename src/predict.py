from __future__ import annotations

import argparse

from .config import MODEL_NAME
from .data import get_system_prompt, load_chat_dataset
from .inference import classify_event
from .model import attach_adapter, load_base_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Klasyfikacja pojedynczego alertu")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_chat_dataset()
    system_prompt = get_system_prompt(dataset)

    tokenizer = load_tokenizer(args.adapter or args.model)
    model = load_base_model(args.model, for_training=False)
    if args.adapter:
        model = attach_adapter(model, args.adapter)

    if args.text:
        prediction, raw = classify_event(
            model, tokenizer, system_prompt, args.text, args.max_length
        )
        print("Klasa:", prediction)
        print("Surowe wyjście:", raw)
        return

    print("Wpisz alert bezpieczeństwa. Pusty wiersz kończy program.")
    while True:
        try:
            text = input("\nAlert> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        prediction, raw = classify_event(
            model, tokenizer, system_prompt, text, args.max_length
        )
        print("Klasa:", prediction)
        if prediction == "invalid":
            print("Surowe wyjście:", raw)


if __name__ == "__main__":
    main()
