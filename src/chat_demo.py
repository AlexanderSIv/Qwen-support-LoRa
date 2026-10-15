import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Respond concisely, professionally, and directly address the user's request."
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "checkpoints/lora-bitext"


def load_model(use_lora: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    if use_lora:
        model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def chat_once(model, tokenizer, user_message, max_new_tokens=128):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    print("Loading Qwen + LoRA...")
    model, tokenizer = load_model(use_lora=True)
    print("Ready. Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            break
        
        response = chat_once(model, tokenizer, user_input)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()