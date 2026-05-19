import argparse

import wandb

from defapi.training import FineTuneConfig
from defapi.training.lora import create_lora_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FineTuneConfig(use_4bit=not args.no_4bit)
    trainer = create_lora_trainer(config, device_map=args.device_map)
    trainer.train()
    trainer.model.save_pretrained(config.new_model)
    wandb.finish()


if __name__ == "__main__":
    main()
