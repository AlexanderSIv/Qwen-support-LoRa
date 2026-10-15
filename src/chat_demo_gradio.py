import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Respond concisely, professionally, and directly address the user's request."
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "checkpoints/lora-bitext"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()


@torch.inference_mode()
def chat_fn(message, history):
    #Gradio передаёт history в формате [[user1, bot1], [user2, bot2], ...].
    #Конвертируем в messages для Qwen.

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        content = item.get("content", "")
        # Если content — список, извлекаем текст
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            content = " ".join(text_parts)
        
        messages.append({"role": item.get("role"), "content": content})
    messages.append({"role": "user", "content": message})
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Qwen2.5-0.5B Support Bot (LoRA)",
    description="Customer support assistant fine-tuned with QLoRA on Bitext dataset.",
    examples=[
        "I want to cancel my order #12345",
        "How do I track my refund?",
        "Can I change my shipping address?",
        "My payment failed, what should I do?",
    ],
)

if __name__ == "__main__":
    demo.launch(share=False)  # share=True даст публичную ссылку