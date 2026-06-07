from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from defapi.training.config import QwenQloraConfig


LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def create_tokenizer(config: QwenQloraConfig) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def create_model(config: QwenQloraConfig) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=create_quantization_config(config),
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
    )
    if config.gradient_checkpointing:
        model.config.use_cache = False
    return prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def create_quantization_config(config: QwenQloraConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def create_lora_config(config: QwenQloraConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )


def save_final_adapter(trainer: Any, tokenizer: Any, output_dir: Path) -> None:
    final_dir = output_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"최종 LoRA adapter를 저장했습니다: {final_dir}")
