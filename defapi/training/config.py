from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class QwenQloraConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct"
    dataset_name: str | None = "hitoshura25/crossvul"
    dataset_split: str = "train"
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
}


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load YAML configs. Install requirements-finetune.txt.") from exc

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping/object: {path}")
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
        raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")

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
            raise ValueError(f"Unknown config override: {key}")
        setattr(config, key, Path(value) if key in PATH_FIELDS else value)
    return config


def validate_config(config: QwenQloraConfig) -> None:
    if not config.dataset_name and config.dataset_path is None:
        raise ValueError("Set either dataset_name for Hugging Face datasets or dataset_path for local JSONL.")
    if config.fp16 and config.bf16:
        raise ValueError("Only one of fp16 or bf16 can be enabled.")
    if not 0 < config.eval_split_size < 1:
        raise ValueError("eval_split_size must be between 0 and 1.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")
    if config.max_seq_length < 128:
        raise ValueError("max_seq_length is unexpectedly small; use at least 128.")
