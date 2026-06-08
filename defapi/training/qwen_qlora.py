from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from defapi.training.config import (
    QwenQloraConfig,
    apply_overrides,
    create_config,
    load_yaml_config,
)


class KoreanArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"인자 오류: {translate_argparse_error(message)}\n")


def translate_argparse_error(message: str) -> str:
    if message.startswith("unrecognized arguments:"):
        return message.replace("unrecognized arguments:", "알 수 없는 인자:", 1)
    if message.startswith("argument ") and "invalid choice:" in message:
        return message.replace("invalid choice:", "허용되지 않는 값:", 1)
    if message.startswith("argument ") and "expected one argument" in message:
        return message.replace("expected one argument", "값이 하나 필요합니다", 1)
    if "the following arguments are required:" in message:
        return message.replace("the following arguments are required:", "다음 필수 인자가 없습니다:", 1)
    return message


def parse_args() -> argparse.Namespace:
    parser = KoreanArgumentParser(
        add_help=False,
        description="DefAPI용 Qwen2.5-Coder-14B-Instruct QLoRA 학습을 실행합니다."
    )
    parser.add_argument("-h", "--help", action="help", help="도움말 출력 후 종료")
    parser.add_argument("--config", type=Path, help="YAML 설정 파일 경로")
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-split")
    parser.add_argument("--eval-dataset-split")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--eval-dataset-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--device-map")
    parser.add_argument("--report-to", choices=["wandb", "none"], help="W&B를 끄려면 'none'을 사용하세요.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> QwenQloraConfig:
    config = create_config(load_yaml_config(args.config) if args.config else {})
    apply_overrides(config, cli_overrides(args))

    if args.fp16:
        config.fp16 = True
        config.bf16 = False
    if args.no_bf16:
        config.bf16 = False
    if args.no_gradient_checkpointing:
        config.gradient_checkpointing = False
    return config


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "eval_dataset_split": args.eval_dataset_split,
        "dataset_path": args.dataset_path,
        "eval_dataset_path": args.eval_dataset_path,
        "output_dir": args.output_dir,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "device_map": args.device_map,
        "report_to": args.report_to,
    }


def main() -> None:
    try:
        config = config_from_args(parse_args())

        from defapi.training.trainer import train

        train(config)
    except Exception as exc:
        print(f"학습 실행 중 오류가 발생했습니다: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
