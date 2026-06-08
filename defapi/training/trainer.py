from __future__ import annotations

import inspect
import os
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from transformers import TrainingArguments, set_seed

from defapi.training.config import QwenQloraConfig, validate_config
from defapi.training.data import load_and_format_dataset
from defapi.training.modeling import (
    create_lora_config,
    create_model,
    create_tokenizer,
    save_final_adapter,
)


def train(config: QwenQloraConfig) -> None:
    validate_config(config)
    set_seed(config.seed)
    configure_wandb(config)

    tokenizer = create_tokenizer(config)
    dataset = load_and_format_dataset(config, tokenizer)
    model = create_model(config)

    trainer = create_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        peft_config=create_lora_config(config),
        training_args=create_training_arguments(config),
        config=config,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    save_final_adapter(trainer, tokenizer, config.output_dir)
    finish_wandb(config)


def create_training_arguments(config: QwenQloraConfig) -> TrainingArguments:
    args_cls = get_sft_config_class()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "run_name": config.wandb_run_name,
        "report_to": config.report_to,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "optim": config.optim,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "gradient_checkpointing": config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "logging_steps": config.logging_steps,
        "save_total_limit": config.save_total_limit,
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "lr_scheduler_type": config.lr_scheduler_type,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "dataset_text_field": "text",
        "max_length": config.max_seq_length,
        "max_seq_length": config.max_seq_length,
        "packing": config.packing,
    }

    kwargs[evaluation_strategy_arg(args_cls)] = "steps"
    return args_cls(**supported_kwargs(args_cls, kwargs))


def create_sft_trainer(
    *,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    peft_config: Any,
    training_args: TrainingArguments,
    config: QwenQloraConfig,
) -> Any:
    from trl import SFTTrainer

    kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["eval"],
        "peft_config": peft_config,
        "args": training_args,
        "packing": config.packing,
        "dataset_text_field": "text",
        "max_seq_length": config.max_seq_length,
    }

    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer

    return SFTTrainer(**supported_kwargs(SFTTrainer, kwargs))


def get_sft_config_class() -> type:
    try:
        from trl import SFTConfig
    except ImportError:
        return TrainingArguments
    return SFTConfig


def evaluation_strategy_arg(args_cls: type) -> str:
    return "eval_strategy" if "eval_strategy" in inspect.signature(args_cls.__init__).parameters else "evaluation_strategy"


def supported_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(cls.__init__).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs

    supported = set(parameters)
    if is_dataclass(cls):
        supported.update(field.name for field in fields(cls))
    return {key: value for key, value in kwargs.items() if key in supported}


def wandb_enabled(config: QwenQloraConfig) -> bool:
    return config.report_to.lower() == "wandb"


def configure_wandb(config: QwenQloraConfig) -> None:
    if not wandb_enabled(config):
        return

    os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
    os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")
    os.environ.setdefault("WANDB_WATCH", "false")

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError("report_to='wandb'일 때는 wandb가 필요합니다. requirements-finetune.txt를 설치하세요.") from exc

    if wandb.run is None:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=wandb_config(config),
        )


def wandb_config(config: QwenQloraConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def finish_wandb(config: QwenQloraConfig) -> None:
    if not wandb_enabled(config):
        return

    import wandb

    wandb.finish()
