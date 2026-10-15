from dataclasses import dataclass

@dataclass
class LoRAConfig:
    # LoRA
    r: int = 16                    # rank
    lora_alpha: int = 32           # scaling
    lora_dropout: float = 0.05
    target_modules: list = None    # ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Training
    num_epochs: int = 1
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # эффективный батч = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_length: int = 1024
    
    # Precision
    fp16: bool = True
    bf16: bool = False             # Turing не поддерживает bf16
    
    # Optim
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    
    # Logging
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    output_dir: str = "checkpoints/lora-bitext"