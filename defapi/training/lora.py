from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import load_dataset
from transformers import TrainingArguments

from defapi.training.common import (
    configure_wandb,
    create_adalora_config,
    create_model,
    create_tokenizer,
    validate_dataset_columns,
    validate_runtime_config,
)
from defapi.training.config import FineTuneConfig

if TYPE_CHECKING:
    from trl import SFTTrainer

DATASET_NAME = "hitoshura25/crossvul"


def load_lora_dataset():
    return load_dataset(DATASET_NAME)


def tokenize_dataset(dataset: Any, tokenizer: Any, config: FineTuneConfig):
    validate_dataset_columns(dataset)

    def preprocess_fn(example):
        return tokenizer(
            format_sft_text(example["prompt"], example["completion"]),
            truncation=True,
            max_length=config.max_seq_length,
        )

    return dataset.map(
        preprocess_fn,
        remove_columns=dataset["train"].column_names,
        num_proc=config.dataset_num_proc,
    )


def format_sft_text(prompt: str, completion: str) -> str:
    return (
        "### Fix the vulnerability in this code.\n\n"
        f"### Vulnerability Code:\n{prompt.strip()}\n\n"
        f"### Clean Code:\n{completion.strip()}"
    )


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
        group_by_length=config.group_by_length,
        lr_scheduler_type=config.lr_scheduler_type,
        dataloader_num_workers=config.dataloader_num_workers,
        gradient_checkpointing_kwargs=config.gradient_checkpointing_kwargs,
    )


def create_lora_trainer(config: FineTuneConfig, device_map: str | dict = "auto") -> "SFTTrainer":
    from trl import SFTTrainer

    validate_runtime_config(config)
    configure_wandb(config, project="defAPI")

    raw_dataset = load_lora_dataset()
    tokenizer = create_tokenizer(config)
    dataset = tokenize_dataset(raw_dataset, tokenizer, config)

    model = create_model(config, device_map=device_map)
    train_dataset = dataset["train"].shuffle(seed=config.seed)
    eval_dataset = dataset["test"].shuffle(seed=config.seed)

    return SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=create_adalora_config(config),
        args=create_training_args(config),
        packing=config.packing,
    )
