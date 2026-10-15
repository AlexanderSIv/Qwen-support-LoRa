import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Respond concisely, professionally, and directly address the user's request."
)


def build_prompt(tokenizer, instruction):
    #Форматирует один промпт в chat template Qwen.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    instructions,
    max_new_tokens=128,
    batch_size=8,
):

    model.eval()
    

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"    # left padding для decoder-only
    
    all_responses = []
    
    try:
        for i in tqdm(range(0, len(instructions), batch_size), desc="Generating"):
            batch_inst = instructions[i:i + batch_size]
            prompts = [build_prompt(tokenizer, inst) for inst in batch_inst]
            
            # Токенизация с паддингом до макс. длины в батче
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=False,  # chat template уже содержит спецтокены
            ).to(model.device)
            
            input_len = inputs["input_ids"].shape[1]
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            
            # Обрезаем промптную часть. Для left padding это просто срез [input_len:]
            for j in range(outputs.size(0)):
                generated = outputs[j, input_len:]
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                all_responses.append(text)
            
            del inputs, outputs
            torch.cuda.empty_cache()
    
    finally:
        # Возвращаем исходный padding_side, чтобы не сломать другие вызовы
        tokenizer.padding_side = original_padding_side
    
    return all_responses