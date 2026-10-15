import random
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Respond concisely, professionally, and directly address the user's request."
)

MIN_TURNS = 1
MAX_TURNS = 5


def build_multiturn_examples(df, min_turns=1, max_turns=5, seed=42, target_size=None):
    """
    Генерирует multi-turn диалоги из Bitext.
    
    Алгоритм:
    1. Группируем Q&A по intent (чтобы диалоги были тематически связными).
    2. Для каждого диалога случайно выбираем n_turns in [min_turns, max_turns].
    3. Берём n_turns случайных Q&A из группы.
    4. Склеиваем в messages: system + [user, assistant] * n_turns.
    
    Args:
        df: исходный DataFrame с колонками instruction, response, intent
        min_turns, max_turns: диапазон числа turn'ов в диалоге
        seed: для воспроизводимости
        target_size: желаемое число диалогов (None = max возможное)
    
    Returns:
        список dict-ов вида {"messages": [...]}
    """
    rng = random.Random(seed)
    examples = []
    
    # Группируем по intent
    groups = {intent: group.reset_index(drop=True) for intent, group in df.groupby("intent")}
    
    # Определяем, сколько диалогов можем сгенерировать
    # Грубо: сумма len(group) // avg_turns по всем группам
    avg_turns = (min_turns + max_turns) / 2
    max_possible = sum(int(len(g) / avg_turns) for g in groups.values())
    n_dialogues = target_size if target_size else max_possible
    
    print(f"Generating {n_dialogues} multi-turn dialogues...")
    print(f"Groups (intents): {len(groups)}")
    
    for _ in range(n_dialogues):
        # Случайный intent
        intent = rng.choice(list(groups.keys()))
        group = groups[intent]
        
        # Случайное число turn'ов в диапазоне
        n_turns = rng.randint(min_turns, max_turns)
        
        # Если в группе меньше примеров, чем нужно turn'ов — уменьшаем
        n_turns = min(n_turns, len(group))
        if n_turns < min_turns:
            continue
        
        # Случайные n_turns Q&A из группы (без повторений)
        indices = rng.sample(range(len(group)), n_turns)
        chunk = group.iloc[indices]
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for _, row in chunk.iterrows():
            messages.append({"role": "user", "content": row["instruction"]})
            messages.append({"role": "assistant", "content": row["response"]})
        
        examples.append({"messages": messages})
    
    # Статистика по числу turn'ов
    turn_counts = [len(ex["messages"]) // 2 for ex in examples]  # делим на 2, т.к. user+assistant
    from collections import Counter
    print(f"Turn distribution: {dict(sorted(Counter(turn_counts).items()))}")
    print(f"Total dialogues: {len(examples)}")
    print(f"Total messages: {sum(len(ex['messages']) for ex in examples)}")
    
    return examples


def tokenize_multiturn(examples, tokenizer, max_length=1024):
    """
    Токенизирует multi-turn диалоги.
    
    Для каждого диалога:
    - apply_chat_template на полный список messages
    - truncation по max_length (обрезаем справа, чтобы сохранить начало диалога)
    """
    def _fmt(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )
        return enc
    
    ds = Dataset.from_list(examples)
    ds = ds.map(_fmt, remove_columns=["messages"], desc="Tokenizing")
    return ds


if __name__ == "__main__":
    from prepare_data import load_bitext, split_data
    
    # Загружаем Bitext и делаем те же сплиты, что и раньше
    df = load_bitext("D:/Admin/LLM/Stage 1/qwen-support-lora/data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
    train_df, val_df, test_df = split_data(df, seed=42)
    
    # Генерируем multi-turn
    train_examples = build_multiturn_examples(
        train_df, min_turns=1, max_turns=5, seed=42, target_size=len(train_df)
    )
    val_examples = build_multiturn_examples(
        val_df, min_turns=1, max_turns=5, seed=43, target_size=len(val_df) // 2
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_ds = tokenize_multiturn(train_examples, tokenizer, max_length=1024)
    val_ds = tokenize_multiturn(val_examples, tokenizer, max_length=1024)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Sanity check
    sample = train_ds[0]
    print("\n--- SAMPLE ---")
    print(f"Length: {len(sample['input_ids'])}")
    print(tokenizer.decode(sample["input_ids"][:500]))