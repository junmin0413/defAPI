from .dpo import create_dpo_trainer
from .lora import create_lora_trainer
from .config import FineTuneConfig

def create_trainer(config: FineTuneConfig, device_map: str | dict = "auto"):
    if config.method == "lora":
        return create_lora_trainer(config, device_map)
    elif config.method == "dpo":
        return create_dpo_trainer(config, device_map)
    else:
        raise ValueError(f"Unsupported fine-tuning method: {config.method}")
    
    
def training(config: FineTuneConfig, device_map:str | dict = "auto"):
    trainer = create_trainer(config, device_map)
    trainer.train()