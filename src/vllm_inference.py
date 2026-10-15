from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Respond concisely, professionally, and directly address the user's request."
)


def build_prompts(instructions, tokenizer_path="Qwen/Qwen2.5-0.5B-Instruct"):
    #Форматирует список инструкций в chat template Qwen.
    #vLLM работает с сырыми строками, не токенами.

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    prompts = []
    for inst in instructions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": inst},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
    return prompts


def generate_with_vllm(
    instructions,
    base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    lora_path=None,
    max_new_tokens=128,
    temperature=0.0,
):

    # Инициализация движка
    llm = LLM(
        model=base_model_name,
        enable_lora=lora_path is not None,
        max_lora_rank=16,           # под r=16
        max_loras=1,                # один адаптер за раз
        dtype="float16",            # Turing
        gpu_memory_utilization=0.85,
        max_model_len=1024,
    )
    
    # Параметры сэмплинга
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop=["<|im_end|>"],        # останавливаемся на конце хода ассистента
    )
    
    # Подготовка промптов
    prompts = build_prompts(instructions)
    
    # LoRA request (если адаптер указан)
    lora_request = None
    if lora_path is not None:
        lora_request = LoRARequest(
            lora_name="support-lora",
            lora_int_id=1,
            lora_path=lora_path,
        )
    
    # Генерация
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
    )
    
    # Извлечение текста
    responses = [out.outputs[0].text.strip() for out in outputs]
    return responses


# === Интеграция в evaluate.py ===
def generate_base_vllm(instructions):
    return generate_with_vllm(instructions, lora_path=None)

def generate_lora_vllm(instructions, lora_path="checkpoints/lora-bitext"):
    return generate_with_vllm(instructions, lora_path=lora_path)