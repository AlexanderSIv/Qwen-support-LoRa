import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Respond concisely, professionally, and directly address the user's request."
)

def load_bitext(path):
    # HuggingFace: bitext/Bitext-customer-support-llm-chatbot-training-dataset
    df = pd.read_csv(path)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Intents: {df['intent'].nunique()}")
    return df

def format_chat(example, tokenizer, max_length):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    
    tokenized = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }

def split_data(df, val_size=0.1, test_size=0.1, seed=42):
    # Стратификация по intent
    train_df, temp_df = train_test_split(df, test_size=val_size+test_size, stratify=df['intent'], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(val_size+test_size), stratify=temp_df['intent'], random_state=seed)
    return train_df, val_df, test_df

def load_and_prepare(
    tokenizer: PreTrainedTokenizer,
    config,
    csv_path: str = "D:/Admin/LLM/Stage 1/qwen-support-lora/data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
):

    # Основная функция: загрузка → разбиение → форматирование → Dataset.
    # Возвращает (train_dataset, val_dataset, test_df).

    df = load_bitext(csv_path)
    train_df, val_df, test_df = split_data(df)
    
    # Убираем дубликаты и пустые строки
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split.dropna(subset=["instruction", "response"], inplace=True)
        split.drop_duplicates(subset=["instruction"], inplace=True)
        print(f"{name}: {len(split)} rows after cleaning")
    
    # В HF Dataset
    train_ds = Dataset.from_pandas(train_df[["instruction", "response"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[["instruction", "response"]].reset_index(drop=True))
    
    # Форматирование с токенизацией
    fmt = lambda ex: format_chat(ex, tokenizer, config.max_length)
    train_ds = train_ds.map(fmt, remove_columns=train_ds.column_names, desc="Tokenizing train")
    val_ds = val_ds.map(fmt, remove_columns=val_ds.column_names, desc="Tokenizing val")
    
    
    
    print(f"Final: train={len(train_ds)}, val={len(val_ds)}")
    return train_ds, val_ds, test_df