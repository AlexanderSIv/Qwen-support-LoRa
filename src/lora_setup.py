from peft import LoraConfig, get_peft_model

def setup_lora(model, config):

    #Навешивает LoRA-адаптер на модель.
    #Работает и для LoRA (fp16 base), и для QLoRA (4-bit base).

    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model