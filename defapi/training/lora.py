from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
from datasets import load_dataset
from peft import AdaLoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from defapi.training.config import FineTuneConfig

if TYPE_CHECKING:
    from trl import SFTTrainer


def configure_wandb(project: str = "defAPI", log_model: str = "true", watch: str = "false") -> None:
    os.environ.setdefault("WANDB_PROJECT", project)
    os.environ.setdefault("WANDB_LOG_MODEL", log_model)
    os.environ.setdefault("WANDB_WATCH", watch)


def load_json_datasets(config: FineTuneConfig):
    return load_dataset(
        "json",
        data_files={
            "train": str(config.train_path),
            "test": str(config.test_path),
        },
    )


def create_tokenizer(config: FineTuneConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_dataset(dataset: Any, tokenizer: Any, config: FineTuneConfig):
    def preprocess_fn(example):
        text = example["prompt"] + example["completion"]
        return tokenizer(text, truncation=True, max_length=config.max_seq_length)

    return dataset.map(preprocess_fn, remove_columns=dataset["train"].column_names)


def create_bnb_config(config: FineTuneConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )


def create_model(config: FineTuneConfig, device_map: str | dict = "auto"):
    kwargs = {"device_map": device_map}
    if config.use_4bit:
        kwargs["quantization_config"] = create_bnb_config(config)

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **kwargs)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
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
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=False,
        bf16=False,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        group_by_length=config.group_by_length,
        lr_scheduler_type=config.lr_scheduler_type,
    )


def create_lora_trainer(config: FineTuneConfig, device_map: str | dict = "auto") -> "SFTTrainer":
    from trl import SFTTrainer

    configure_wandb()
    dataset = load_json_datasets(config)
    tokenizer = create_tokenizer(config)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, config)

    return SFTTrainer(
        model=create_model(config, device_map=device_map),
        train_dataset=tokenized_dataset["train"].shuffle(seed=config.seed),
        eval_dataset=tokenized_dataset["test"].shuffle(seed=config.seed),
        peft_config=create_adalora_config(config),
        args=create_training_args(config),
        packing=config.packing,
    )
