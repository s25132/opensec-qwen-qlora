from collections import Counter

from .data import get_system_prompt, load_chat_dataset, validate_chat_example


def main() -> None:
    dataset = load_chat_dataset()
    print(dataset)

    for split_name, split in dataset.items():
        counts = Counter(
            row["messages"][2]["content"].strip().lower() for row in split
        )
        print(f"\n{split_name}: {len(split)}")
        for label, count in sorted(counts.items()):
            print(f"  {label:22} {count}")

    validate_chat_example(dataset["train"][0])
    print("\nInstrukcja systemowa:\n", get_system_prompt(dataset))
    print("\nPierwszy rekord:")
    for message in dataset["train"][0]["messages"]:
        print(f"\n[{message['role']}]\n{message['content']}")


if __name__ == "__main__":
    main()
