from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from defapi.training.config import QwenQloraConfig
from defapi.training.prompts import format_with_chat_template


REQUIRED_FIELDS = {"instruction", "input", "output"}
EVAL_FILE_CANDIDATES = ("eval.jsonl", "validation.jsonl", "test.jsonl")


def load_and_format_dataset(config: QwenQloraConfig, tokenizer: Any) -> DatasetDict:
    dataset = load_defapi_dataset(config)
    return format_dataset(dataset, tokenizer)


def load_defapi_dataset(config: QwenQloraConfig) -> DatasetDict:
    train_path, eval_path = resolve_dataset_files(config)

    if eval_path:
        dataset = load_dataset("json", data_files={"train": str(train_path), "eval": str(eval_path)})
    else:
        loaded = load_dataset("json", data_files={"train": str(train_path)})
        if len(loaded["train"]) < 2:
            raise ValueError("At least two JSONL rows are required when eval split is created automatically.")
        dataset = loaded["train"].train_test_split(test_size=config.eval_split_size, seed=config.seed)
        dataset["eval"] = dataset.pop("test")

    validate_dataset_fields(dataset["train"], "train")
    validate_dataset_fields(dataset["eval"], "eval")
    return dataset


def resolve_dataset_files(config: QwenQloraConfig) -> tuple[Path, Path | None]:
    dataset_path = config.dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if dataset_path.is_dir():
        train_path = dataset_path / "train.jsonl"
        eval_path = config.eval_dataset_path or first_existing_eval_file(dataset_path)
    else:
        train_path = dataset_path
        eval_path = config.eval_dataset_path

    validate_jsonl_file(train_path, "Training")
    if eval_path is not None:
        validate_jsonl_file(eval_path, "Evaluation")
    return train_path, eval_path


def first_existing_eval_file(dataset_dir: Path) -> Path | None:
    for filename in EVAL_FILE_CANDIDATES:
        candidate = dataset_dir / filename
        if candidate.exists():
            return candidate
    return None


def validate_jsonl_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} JSONL file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} dataset path is not a file: {path}")
    if path.suffix != ".jsonl":
        raise ValueError(f"{label} dataset must be a .jsonl file: {path}")


def validate_dataset_fields(dataset: Dataset, split_name: str) -> None:
    missing = sorted(REQUIRED_FIELDS - set(dataset.column_names))
    if missing:
        raise ValueError(f"{split_name} dataset is missing required field(s): {', '.join(missing)}")

    bad_rows = [
        index
        for index, row in enumerate(dataset)
        if any(not isinstance(row.get(field), str) or not row[field].strip() for field in REQUIRED_FIELDS)
    ]
    if bad_rows:
        preview = ", ".join(str(index) for index in bad_rows[:5])
        raise ValueError(f"{split_name} dataset has empty/non-string required fields at row(s): {preview}")


def format_dataset(dataset: DatasetDict, tokenizer: Any) -> DatasetDict:
    def format_row(example: dict[str, Any]) -> dict[str, str]:
        return {"text": format_with_chat_template(example, tokenizer)}

    return dataset.map(
        format_row,
        remove_columns=dataset["train"].column_names,
        desc="Formatting DefAPI samples with Qwen chat template",
    )
