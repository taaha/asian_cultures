from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from PIL import Image
import requests
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

asian_cooking_ds = load_dataset("darthPanda/cvqa_edit5")

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Pixtral-12B-2409-bnb-4bit",
    # "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)


FastVisionModel.for_inference(model) # Enable for inference!

output_data = []

for index, row in tqdm(enumerate(asian_cooking_ds["test"]), total=len(asian_cooking_ds["test"])):
    instruction = """Which asian culture does this image best represent?
You must choose from options below. Answer in one word only.
- Bengali
- Javanese
- Tamil
- Filipino
- Minangkabau
- Sundanese
- Korean
- Indonesian
- Chinese
- Marathi
- Telugu
- Mongolian
- Urdu
- Malay
- Hindi
- Sinhala
- Japanese

Answer:
"""

    image = row["image"]

    messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    output = model.generate(**inputs, max_new_tokens = 30,
                    use_cache = True, temperature = 1.5, min_p = 0.1)

    input_len = inputs["input_ids"].shape[-1]

    output = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    print(output)

    output_data.append({
        "ID": row["ID"],
        "Culture": row["Subset"],
        "Category": row["Category"],
        "output": output,
    })


df = pd.DataFrame(output_data)

df.to_csv("results//pixtral_culture_v0.csv")