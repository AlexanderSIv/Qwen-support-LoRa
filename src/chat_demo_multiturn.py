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
def generate_reply(model, tokenizer, messages, max_new_tokens=128):
    #messages: список dict с role/content, БЕЗ system prompt.
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    prompt = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=True
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
    print("Ready. Commands: /clear (reset history), /exit\n")
    
    history = []  # список {"role": "user"/"assistant", "content": ...}
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q", "/exit"}:
            break
        if user_input.lower() == "/clear":
            history = []
            print("[history cleared]\n")
            continue
        
        history.append({"role": "user", "content": user_input})
        
        # Обрезаем историю, чтобы не превысить контекст
        # Оставляем последние N пар
        MAX_TURNS = 6
        if len(history) > MAX_TURNS * 2:
            history = history[-MAX_TURNS * 2:]
        
        response = generate_reply(model, tokenizer, history)
        history.append({"role": "assistant", "content": response})
        
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()