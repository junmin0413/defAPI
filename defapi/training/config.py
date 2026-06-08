from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class QwenQloraConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct"
    dataset_name: str | None = "hitoshura25/crossvul"
    dataset_split: str = "train"
    eval_dataset_split: str | None = None
    dataset_path: Path | None = None
    eval_dataset_path: Path | None = None
    output_dir: Path = Path("/workspace/checkpoints/defapi-qwen2.5-coder-14b-qlora")

    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    learning_rate: float = 2e-4
    num_train_epochs: float = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "cosine"

    wandb_project: str = "defapi-finetuning"
    wandb_run_name: str = "qwen2.5-coder-14b-qlora"
    report_to: str = "wandb"

    seed: int = 42
    eval_split_size: float = 0.1
    resume_from_checkpoint: str | None = None
    device_map: str = "auto"
    trust_remote_code: bool = True
    packing: bool = False


PATH_FIELDS = {"dataset_path", "eval_dataset_path", "output_dir"}
CONFIG_ALIASES = {
    "per_device_train_batch_size": "batch_size",
    "train_path": "dataset_path",
    "eval_path": "eval_dataset_path",
    "hf_dataset_name": "dataset_name",
    "train_split": "dataset_split",
    "eval_split": "eval_dataset_split",
}


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"설정 파일이 존재하지 않습니다: {path}")

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("YAML 설정을 읽으려면 PyYAML이 필요합니다. requirements-finetune.txt를 설치하세요.") from exc

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"설정 파일은 key-value 객체 형태여야 합니다: {path}")
    return data


def create_config(raw_config: dict[str, Any] | None = None) -> QwenQloraConfig:
    raw = normalize_config(raw_config or {})
    return QwenQloraConfig(**raw)


def normalize_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(raw_config)

    for old_key, new_key in CONFIG_ALIASES.items():
        if old_key in raw and new_key not in raw:
            raw[new_key] = raw.pop(old_key)

    valid_fields = {field.name for field in fields(QwenQloraConfig)}
    unknown = sorted(set(raw) - valid_fields)
    if unknown:
        raise ValueError(f"알 수 없는 설정 키입니다: {', '.join(unknown)}")

    for key in PATH_FIELDS:
        if raw.get(key) is not None:
            raw[key] = Path(raw[key])
    return raw


def apply_overrides(config: QwenQloraConfig, overrides: dict[str, Any]) -> QwenQloraConfig:
    valid_fields = {field.name for field in fields(QwenQloraConfig)}
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in valid_fields:
            raise ValueError(f"알 수 없는 CLI 설정 override입니다: {key}")
        setattr(config, key, Path(value) if key in PATH_FIELDS else value)
    return config


def validate_config(config: QwenQloraConfig) -> None:
    if not config.dataset_name and config.dataset_path is None:
        raise ValueError("Hugging Face dataset_name 또는 로컬 JSONL dataset_path 중 하나는 반드시 설정해야 합니다.")
    if config.fp16 and config.bf16:
        raise ValueError("fp16과 bf16은 동시에 켤 수 없습니다. 둘 중 하나만 사용하세요.")
    if not 0 < config.eval_split_size < 1:
        raise ValueError("eval_split_size는 0보다 크고 1보다 작아야 합니다.")
    if config.batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps는 1 이상이어야 합니다.")
    if config.max_seq_length < 128:
        raise ValueError("max_seq_length가 너무 작습니다. 최소 128 이상으로 설정하세요.")
