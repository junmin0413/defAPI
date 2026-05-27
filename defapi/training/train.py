import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defapi.training import FineTuneConfig
from defapi.training.common import finish_wandb
from defapi.training.lora import create_lora_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=FineTuneConfig.base_model)
    parser.add_argument("--train-path", type=Path, default=FineTuneConfig.train_path)
    parser.add_argument("--test-path", type=Path, default=FineTuneConfig.test_path)
    parser.add_argument("--output-dir", type=Path, default=FineTuneConfig.output_dir)
    parser.add_argument("--new-model", type=Path, default=FineTuneConfig.new_model)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-seq-length", type=int, default=FineTuneConfig.max_seq_length)
    parser.add_argument("--epochs", type=int, default=FineTuneConfig.num_train_epochs)
    parser.add_argument("--learning-rate", type=float, default=FineTuneConfig.learning_rate)
    parser.add_argument("--batch-size", type=int, default=FineTuneConfig.per_device_train_batch_size)
    parser.add_argument("--grad-accum", type=int, default=FineTuneConfig.gradient_accumulation_steps)
    parser.add_argument("--report-to", default=FineTuneConfig.report_to)
    parser.add_argument("--seed", type=int, default=FineTuneConfig.seed)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> FineTuneConfig:
    return FineTuneConfig(
        base_model=args.base_model,
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        new_model=args.new_model,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        report_to=args.report_to,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        use_4bit=not args.no_4bit,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    trainer = create_lora_trainer(config, device_map=args.device_map)
    trainer.train()
    trainer.model.save_pretrained(config.new_model)
    finish_wandb(config.report_to)


if __name__ == "__main__":
    main()
