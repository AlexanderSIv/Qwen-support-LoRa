# src/evaluate.py
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
from prepare_data import load_and_prepare
from config import LoRAConfig
from batch_inference import generate_batch

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Respond concisely, professionally, and directly address the user's request."
)

def load_base(model_name: str, device: str = "cuda"):
    #Base Qwen без LoRA. Используется как baseline.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    return model


def load_lora(base_model_name: str, adapter_path: str, device: str = "cuda"):
    #Base Qwen + LoRA adapter.
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@torch.inference_mode()
def compute_perplexity(model, tokenizer, texts, max_length=512, batch_size=4):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="PPL"):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=max_length,
        ).to(model.device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        # Освобождаем всё явно
        del outputs, loss, labels, input_ids, attention_mask, enc
        torch.cuda.empty_cache()   # возвращает зарезервированное в пул

    avg_loss = total_loss / total_tokens
    return {"loss": avg_loss, "ppl": np.exp(avg_loss)}

def format_for_ppl(example, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def compute_rouge_l(predictions, references):
    #predictions: список сгенерированных ответов (str)
    #references: список эталонных ответов (str)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score["rougeL"].fmeasure)
    return {
        "rouge_l_mean": float(np.mean(scores)),
        "rouge_l_std": float(np.std(scores)),
        "rouge_l_scores": scores,
    }

@torch.no_grad()
def generate_response(model, tokenizer, instruction, max_new_tokens=128):
    #Greedy-генерация для детерминированного evaluation.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy — детерминированно
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

@torch.no_grad()
def compute_token_accuracy(model, tokenizer, texts, max_length=512, batch_size=8):
    """
    Считает долю правильно предсказанных токенов в teacher-forcing режиме.
    """
    model.eval()
    correct = 0
    total = 0
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Token Acc"):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)
        
        # Сдвиг: предсказание для позиции t — это logits[t], а правильный токен — input_ids[t+1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        preds = shift_logits.argmax(dim=-1)  # (B, T-1)
        correct += ((preds == shift_labels) & (shift_mask == 1)).sum().item()
        total += shift_mask.sum().item()
    
    return {"token_accuracy": correct / total}

def run_full_evaluation(
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    lora_path: str = "checkpoints/lora-bitext",
    test_csv: str = "data/processed/test.csv",
    output_path: str = "results/evaluation.json",
):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer = load_tokenizer(base_model_name)
    config = LoRAConfig()
    tokenizer = load_tokenizer(base_model_name)
    _, _, test_df = load_and_prepare(tokenizer, config)
    print(f"Test set size: {len(test_df)}")
    
    # === 1. Perplexity на test set ===
    print("\n=== Perplexity ===")
    test_texts = [
        format_for_ppl(row, tokenizer) 
        for _, row in test_df.iterrows()
    ]
    
    base_model = load_base(base_model_name)
    base_ppl = compute_perplexity(base_model, tokenizer, test_texts)
    print(f"Base PPL: {base_ppl['ppl']:.3f}, loss: {base_ppl['loss']:.4f}")
    del base_model
    torch.cuda.empty_cache()
    
    lora_model = load_lora(base_model_name, lora_path)
    lora_ppl = compute_perplexity(lora_model, tokenizer, test_texts)
    print(f"LoRA PPL: {lora_ppl['ppl']:.3f}, loss: {lora_ppl['loss']:.4f}")

    del lora_model
    torch.cuda.empty_cache()
    
    # === 2. Генерация ответов на subset test set (первые 200, чтобы не ждать долго) ===
    print("\n=== Generating responses (200 samples) ===")
    base_model = load_base(base_model_name)
    eval_subset = test_df.head(200).reset_index(drop=True)
    instructions = eval_subset["instruction"].tolist()
    references = eval_subset["response"].tolist()

    batch_size = 8
    base_responses = generate_batch(base_model, tokenizer, instructions, batch_size=batch_size)
    del base_model
    torch.cuda.empty_cache()

    lora_model = load_lora(base_model_name, lora_path)
    lora_responses = generate_batch(lora_model, tokenizer, instructions, batch_size=batch_size)
    del lora_model
    torch.cuda.empty_cache()
    
    # === 3. ROUGE-L ===
    print("\n=== ROUGE-L ===")
    base_rouge = compute_rouge_l(base_responses, references)
    lora_rouge = compute_rouge_l(lora_responses, references)
    print(f"Base ROUGE-L: {base_rouge['rouge_l_mean']:.3f}")
    print(f"LoRA ROUGE-L: {lora_rouge['rouge_l_mean']:.3f}")
    
    
    # === 6. Сохранение ===
    results = {
        "base": {
            "ppl": base_ppl,
            "rouge_l": base_rouge["rouge_l_mean"],
        },
        "lora": {
            "ppl": lora_ppl,
            "rouge_l": lora_rouge["rouge_l_mean"],
        },
        "samples": [
            {
                "instruction": inst,
                "reference": ref,
                "base_response": base_r,
                "lora_response": lora_r,
            }
            for inst, ref, base_r, lora_r in zip(
                instructions[:20], references[:20], base_responses[:20], lora_responses[:20]
            )
        ],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # === 7. Сводная таблица ===
    # print("\n=== Summary ===")
    # print(f"{'Metric':<25} {'Base':>10} {'LoRA':>10} {'Δ':>10}")
    # print("-" * 60)
    # for key in ["ppl", "rouge_l"]:
    #     b = results["base"][key]
    #     l = results["lora"][key]
    #     delta = l - b if key != "ppl" else b - l  # для PPL меньше = лучше
    #     print(f"{key:<25} {b:>10.3f} {l:>10.3f} {delta:>+10.3f}")
    
    return results

def generate_readme_table(results):
    lines = ["| Metric | Base Qwen | Qwen + LoRA | Δ |", "|--------|-----------|-------------|---|"]
    for key in ["ppl", "rouge_l"]:
        b = results["base"][key]
        l = results["lora"][key]
        if isinstance(b, dict):
            b = b["ppl"]
            l = l["ppl"]
        delta = l - b if key != "ppl" else b - l
        arrow = "↑" if (delta > 0 and key != "ppl") or (delta < 0 and key == "ppl") else "↓"
        lines.append(f"| {key} | {b:.3f} | {l:.3f} | {delta:+.3f} {arrow} |")
    return "\n".join(lines)

if __name__ == "__main__":
    results = run_full_evaluation(
        base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        lora_path="checkpoints/lora-bitext",
        test_csv="qwen-support-lora/data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
        output_path="results/evaluation.json",
    )
    
    # Сгенерировать README-таблицу
    table = generate_readme_table(results)
    with open("results/readme_table.md", "w", encoding="utf-8") as f:
        f.write(table)
    
    print("\nDone. Results saved to results/evaluation.json")