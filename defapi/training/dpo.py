from __future__ import annotations

import os
from typing import TYPE_CHECKING

from datasets import load_dataset
from transformers import TrainingArguments
from trl import DPOTrainer

from defapi.training.common import (
    configure_wandb as configure_wandb_for_config,
    create_adalora_config,
    create_model,
    create_tokenizer,
    validate_runtime_config,
)
from defapi.training.config import FineTuneConfig

if TYPE_CHECKING:
    from datasets import DatasetDict


def configure_wandb(project: str = "defAPI-dpo") -> None:
    os.environ.setdefault("WANDB_PROJECT", project)
    os.environ.setdefault("WANDB_LOG_MODEL", "true")
    os.environ.setdefault("WANDB_WATCH", "false")


def load_dpo_dataset() -> "DatasetDict":
    return load_dataset("CyberNative/Code_Vulnerability_Security_DPO")


def create_training_args(config: FineTuneConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(config.output_dir),
        report_to=config.report_to,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        dataloader_num_workers=config.dataloader_num_workers,
        gradient_checkpointing_kwargs=config.gradient_checkpointing_kwargs,
        remove_unused_columns=False,
    )


def create_dpo_trainer(config: FineTuneConfig, device_map: str | dict = "auto") -> DPOTrainer:
    validate_runtime_config(config)
    configure_wandb_for_config(config, project="defAPI-dpo")

    dataset = load_dpo_dataset()
    tokenizer = create_tokenizer(config)
    model = create_model(config, device_map=device_map)

    return DPOTrainer(
        model=model,
        ref_model=None,
        beta=0.1,
        args=create_training_args(config),
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        tokenizer=tokenizer,
        peft_config=create_adalora_config(config),
        max_length=config.max_seq_length,
        max_prompt_length=512,
    )
