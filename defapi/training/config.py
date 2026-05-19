from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FineTuneConfig:
    base_model: str = "deepseek-ai/deepseek-coder-1.3b-instruct"
    train_path: Path = PROJECT_ROOT / "dataset" / "ft_data.jsonl"
    test_path: Path = PROJECT_ROOT / "dataset" / "test_data.jsonl"
    output_dir: Path = PROJECT_ROOT / "results"
    new_model: Path = PROJECT_ROOT / "deepseek-coder-1.3b-instruct-adalora-gbsw"
    max_seq_length: int = 1024
    adalora_target_r: int = 8
    adalora_init_r: int = 32
    adalora_alpha: int = 8
    adalora_dropout: float = 0.06
    adalora_tinit: int = 0
    adalora_tfinal: int = 0
    adalora_delta_t: int = 1
    adalora_beta1: float = 0.85
    adalora_beta2: float = 0.85
    adalora_orth_reg_weight: float = 0.5
    adalora_total_step: int | None = None
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.6
    learning_rate: float = 2e-6
    weight_decay: float = 0.001
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    max_steps: int = -1
    warmup_ratio: float = 0.03
    group_by_length: bool = True
    save_steps: int = 0
    logging_steps: int = 500
    packing: bool = False
    report_to: str = "wandb"
    seed: int = 42
