from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from config import LoRAConfig
from model_loader import load_base_model
from lora_setup import setup_lora
from prepare_multiturn_data import build_multiturn_examples, tokenize_multiturn
from prepare_data import load_bitext, split_data


def main():
    config = LoRAConfig()
    USE_QLORA = True
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

     # Загрузка Bitext
    df = load_bitext("D:/Admin/LLM/Stage 1/qwen-support-lora/data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
    train_df, val_df, _ = split_data(df, seed=42)
    train_examples = build_multiturn_examples(
        train_df, min_turns=1, max_turns=5, seed=42, target_size=len(train_df)
    )
    val_examples = build_multiturn_examples(
        val_df, min_turns=1, max_turns=5, seed=43, target_size=len(val_df) // 2
    )
    train_ds = tokenize_multiturn(train_examples, tokenizer, max_length=config.max_length)
    val_ds = tokenize_multiturn(val_examples, tokenizer, max_length=config.max_length)

    # train_ds, val_ds, _ = load_and_prepare(tokenizer, config)
    
    model = load_base_model("Qwen/Qwen2.5-0.5B-Instruct", use_qlora=USE_QLORA)
    model = setup_lora(model, config)
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=0.03,  
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        bf16=config.bf16,
        optim=config.optim,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps", 
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        gradient_checkpointing=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()