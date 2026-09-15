import pytest

from src.data import extract_test_pair, validate_chat_example
from src.inference import normalize_prediction


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("malicious", "malicious"),
        (" Suspicious\n", "suspicious"),
        ("expected admin", "expected_admin"),
        ("The disposition is benign.", "benign"),
        ("I cannot decide.", "invalid"),
    ],
)
def test_normalize_prediction(raw, expected):
    assert normalize_prediction(raw) == expected


def test_validate_chat_example_accepts_valid_record():
    validate_chat_example(
        {
            "messages": [
                {"role": "system", "content": "Classify."},
                {"role": "user", "content": "An event."},
                {"role": "assistant", "content": "unknown"},
            ]
        }
    )


def test_validate_chat_example_rejects_unknown_label():
    with pytest.raises(ValueError):
        validate_chat_example(
            {
                "messages": [
                    {"role": "system", "content": "Classify."},
                    {"role": "user", "content": "An event."},
                    {"role": "assistant", "content": "critical"},
                ]
            }
        )


def test_extract_test_pair():
    assert extract_test_pair({"text": " Alert ", "target": " Benign "}) == (
        "Alert",
        "benign",
    )
