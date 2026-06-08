from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from defapi.training.config import QwenQloraConfig
from defapi.training.prompts import format_with_chat_template


REQUIRED_FIELDS = {"instruction", "input", "output"}
CROSSVUL_FIELDS = {"cwe_id", "cwe_description", "language", "vulnerable_code", "fixed_code"}
EVAL_FILE_CANDIDATES = ("eval.jsonl", "validation.jsonl", "test.jsonl")


def load_and_format_dataset(config: QwenQloraConfig, tokenizer: Any) -> DatasetDict:
    dataset = load_defapi_dataset(config)
    return format_dataset(dataset, tokenizer)


def load_defapi_dataset(config: QwenQloraConfig) -> DatasetDict:
    if config.dataset_path is None:
        return load_hf_dataset(config)
    return load_jsonl_dataset(config)


def load_hf_dataset(config: QwenQloraConfig) -> DatasetDict:
    if not config.dataset_name:
        raise ValueError("dataset_path를 사용하지 않을 때는 dataset_name이 필요합니다.")

    loaded = load_dataset(config.dataset_name)
    dataset = split_hf_dataset(loaded, config)
    return normalize_dataset(dataset)


def split_hf_dataset(dataset: DatasetDict, config: QwenQloraConfig) -> DatasetDict:
    if config.dataset_split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(
            f"설정한 train split이 존재하지 않습니다: {config.dataset_split}. 사용 가능한 split: {available}"
        )

    if config.eval_dataset_split:
        if config.eval_dataset_split not in dataset:
            available = ", ".join(dataset.keys())
            raise ValueError(
                f"설정한 eval split이 존재하지 않습니다: {config.eval_dataset_split}. 사용 가능한 split: {available}"
            )
        return DatasetDict({"train": dataset[config.dataset_split], "eval": dataset[config.eval_dataset_split]})
    if "eval" in dataset:
        return DatasetDict({"train": dataset[config.dataset_split], "eval": dataset["eval"]})
    if "validation" in dataset:
        return DatasetDict({"train": dataset[config.dataset_split], "eval": dataset["validation"]})
    if "test" in dataset:
        return DatasetDict({"train": dataset[config.dataset_split], "eval": dataset["test"]})

    train_dataset = dataset[config.dataset_split]
    if len(train_dataset) < 2:
        raise ValueError("eval split을 자동 생성하려면 데이터가 최소 2개 이상 필요합니다.")
    print(
        "eval split이 없어 자동으로 train/eval을 분리합니다. "
        f"대상 split='{config.dataset_split}', eval 비율={config.eval_split_size}."
    )
    split = train_dataset.train_test_split(
        test_size=config.eval_split_size,
        seed=config.seed,
        shuffle=True,
    )
    return DatasetDict({"train": split["train"], "eval": split["test"]})


def normalize_dataset(dataset: DatasetDict) -> DatasetDict:
    train_columns = set(dataset["train"].column_names)
    if REQUIRED_FIELDS <= train_columns:
        validate_dataset_fields(dataset["train"], "train")
        validate_dataset_fields(dataset["eval"], "eval")
        return dataset

    if CROSSVUL_FIELDS <= train_columns:
        normalized = dataset.map(
            crossvul_to_defapi_sample,
            remove_columns=dataset["train"].column_names,
            desc="CrossVul 데이터를 DefAPI 학습 샘플로 변환 중",
        )
        validate_dataset_fields(normalized["train"], "train")
        validate_dataset_fields(normalized["eval"], "eval")
        return normalized

    missing_defapi = sorted(REQUIRED_FIELDS - train_columns)
    missing_crossvul = sorted(CROSSVUL_FIELDS - train_columns)
    raise ValueError(
        "지원하지 않는 데이터셋 스키마입니다. "
        f"누락된 DefAPI 필드: {', '.join(missing_defapi)}. "
        f"누락된 CrossVul 필드: {', '.join(missing_crossvul)}."
    )


def crossvul_to_defapi_sample(row: dict[str, Any]) -> dict[str, str]:
    language = str(row["language"]).strip() or "text"
    cwe_id = str(row["cwe_id"]).strip()
    cwe_description = str(row["cwe_description"]).strip()
    vulnerable_code = str(row["vulnerable_code"]).strip()
    fixed_code = str(row["fixed_code"]).strip()

    return {
        "instruction": "다음 코드의 보안 취약점을 분석하고 안전한 코드로 수정하라.",
        "input": (
            f"CWE: {cwe_id}\n"
            f"설명: {cwe_description}\n"
            f"언어: {language}\n\n"
            f"취약 코드:\n{vulnerable_code}"
        ),
        "output": (
            f"취약점 설명: {cwe_id} - {cwe_description}\n\n"
            f"안전한 수정 코드:\n```{language}\n{fixed_code}\n```\n\n"
            "수정 이유: 취약 코드에서 CWE 설명에 해당하는 위험한 동작을 제거하거나 완화하도록 "
            "검증된 수정 코드로 변경했다.\n\n"
            "추가 주의사항: 실제 서비스 적용 전에는 프로젝트별 입력 검증, 권한 제어, 예외 처리, "
            "보안 테스트와 회귀 테스트를 함께 수행해야 한다."
        ),
    }


def load_jsonl_dataset(config: QwenQloraConfig) -> DatasetDict:
    train_path, eval_path = resolve_dataset_files(config)

    if eval_path:
        dataset = load_dataset("json", data_files={"train": str(train_path), "eval": str(eval_path)})
    else:
        loaded = load_dataset("json", data_files={"train": str(train_path)})
        if len(loaded["train"]) < 2:
            raise ValueError("eval split을 자동 생성하려면 JSONL row가 최소 2개 이상 필요합니다.")
        print(
            "로컬 eval JSONL이 없어 train JSONL을 train/eval로 자동 분리합니다. "
            f"eval 비율={config.eval_split_size}."
        )
        dataset = loaded["train"].train_test_split(
            test_size=config.eval_split_size,
            seed=config.seed,
            shuffle=True,
        )
        dataset["eval"] = dataset.pop("test")

    validate_dataset_fields(dataset["train"], "train")
    validate_dataset_fields(dataset["eval"], "eval")
    return dataset


def resolve_dataset_files(config: QwenQloraConfig) -> tuple[Path, Path | None]:
    dataset_path = config.dataset_path
    if dataset_path is None:
        raise ValueError("로컬 JSONL 데이터셋을 사용하려면 dataset_path가 필요합니다.")
    if not dataset_path.exists():
        raise FileNotFoundError(f"데이터셋 경로가 존재하지 않습니다: {dataset_path}")

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
        raise FileNotFoundError(f"{label} JSONL 파일이 존재하지 않습니다: {path}")
    if not path.is_file():
        raise ValueError(f"{label} 데이터셋 경로가 파일이 아닙니다: {path}")
    if path.suffix != ".jsonl":
        raise ValueError(f"{label} 데이터셋은 .jsonl 파일이어야 합니다: {path}")


def validate_dataset_fields(dataset: Dataset, split_name: str) -> None:
    missing = sorted(REQUIRED_FIELDS - set(dataset.column_names))
    if missing:
        raise ValueError(f"{split_name} 데이터셋에 필수 필드가 없습니다: {', '.join(missing)}")

    bad_rows = [
        index
        for index, row in enumerate(dataset)
        if any(not isinstance(row.get(field), str) or not row[field].strip() for field in REQUIRED_FIELDS)
    ]
    if bad_rows:
        preview = ", ".join(str(index) for index in bad_rows[:5])
        raise ValueError(f"{split_name} 데이터셋에 비어 있거나 문자열이 아닌 필드가 있습니다. row: {preview}")


def format_dataset(dataset: DatasetDict, tokenizer: Any) -> DatasetDict:
    def format_row(example: dict[str, Any]) -> dict[str, str]:
        return {"text": format_with_chat_template(example, tokenizer)}

    return dataset.map(
        format_row,
        remove_columns=dataset["train"].column_names,
        desc="DefAPI 샘플을 Qwen chat template 형식으로 변환 중",
    )
