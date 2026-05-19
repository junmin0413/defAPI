from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import torch
from peft import AdaLoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from defapi.training.config import FineTuneConfig


def report_to_disabled(report_to: str) -> bool:
    return report_to.lower() in {"", "none", "disabled", "false", "off"}


def configure_wandb(config: FineTuneConfig, project: str) -> None:
    if report_to_disabled(config.report_to):
        return

    os.environ.setdefault("WANDB_PROJECT", project)
    os.environ.setdefault("WANDB_LOG_MODEL", "true")
    os.environ.setdefault("WANDB_WATCH", "false")


def finish_wandb(report_to: str) -> None:
    if report_to_disabled(report_to):
        return

    try:
        import wandb
    except ModuleNotFoundError:
        return

    wandb.finish()


def torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return dtype


def validate_existing_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} dataset file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} dataset path is not a file: {path}")


def validate_runtime_config(config: FineTuneConfig, *, validate_cuda: bool = True) -> None:
    torch_dtype(config.bnb_4bit_compute_dtype)

    if config.fp16 and config.bf16:
        raise ValueError("Only one of fp16 and bf16 can be enabled.")

    if config.use_4bit:
        if validate_cuda and not torch.cuda.is_available():
            raise RuntimeError("4bit training requires a CUDA-capable NVIDIA GPU. Use --no-4bit for CPU/MPS.")
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("4bit training requires bitsandbytes to be installed.")


def validate_sft_inputs(config: FineTuneConfig) -> None:
    validate_existing_file(config.train_path, "train")
    validate_existing_file(config.test_path, "test")
    validate_runtime_config(config)


def validate_dataset_columns(dataset: Any, required_columns: set[str] | None = None) -> None:
    required = required_columns or {"prompt", "completion"}
    train_columns = set(dataset["train"].column_names)
    missing = sorted(required - train_columns)
    if missing:
        raise ValueError(f"Training dataset is missing required columns: {', '.join(missing)}")

    test_columns = set(dataset["test"].column_names)
    missing = sorted(required - test_columns)
    if missing:
        raise ValueError(f"Evaluation dataset is missing required columns: {', '.join(missing)}")


def create_tokenizer(config: FineTuneConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def create_bnb_config(config: FineTuneConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch_dtype(config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )


def create_model(config: FineTuneConfig, device_map: str | dict = "auto"):
    kwargs: dict[str, Any] = {"device_map": device_map, "trust_remote_code": True}
    if config.use_4bit:
        kwargs["quantization_config"] = create_bnb_config(config)

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **kwargs)
    if config.gradient_checkpointing:
        model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1
    if config.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs=config.gradient_checkpointing_kwargs,
        )
    return model


def create_adalora_config(config: FineTuneConfig) -> AdaLoraConfig:
    return AdaLoraConfig(
        r=config.adalora_target_r,
        init_r=config.adalora_init_r,
        target_r=config.adalora_target_r,
        lora_alpha=config.adalora_alpha,
        lora_dropout=config.adalora_dropout,
        tinit=config.adalora_tinit,
        tfinal=config.adalora_tfinal,
        deltaT=config.adalora_delta_t,
        beta1=config.adalora_beta1,
        beta2=config.adalora_beta2,
        orth_reg_weight=config.adalora_orth_reg_weight,
        total_step=config.adalora_total_step,
        bias="none",
        task_type="CAUSAL_LM",
    )
