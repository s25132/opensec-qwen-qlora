from pathlib import Path


MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = "tegridydev/opensec-triage"
TRAIN_CONFIG = "edge_trainer_chat"
EVAL_CONFIG = "edge_trainer"

LABELS = [
    "malicious",
    "suspicious",
    "benign",
    "expected_admin",
    "authorised_testing",
    "misconfiguration",
    "unknown",
]

DEFAULT_OUTPUT_DIR = Path("outputs/qwen3-opensec")
