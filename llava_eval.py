from datasets import load_dataset
from huggingface_hub import login
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
from transformers import BitsAndBytesConfig
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

asian_cooking_ds = load_dataset("darthPanda/cvqa_edit5")

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_4bit=True,
    use_flash_attention_2=True
).to(0)

processor = AutoProcessor.from_pretrained(model_id)


output_data = []

for index, row in tqdm(enumerate(asian_cooking_ds["test"]), total=len(asian_cooking_ds["test"])):
    prompt = """Which asian culture does this image best represent?
You must choose from options below.
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

    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    raw_image = row["image"]
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    input_len = inputs["input_ids"].shape[-1]

    output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    output = processor.decode(output[0][input_len:], skip_special_tokens=True)
    print(output)

    output_data.append({
        "ID": row["ID"],
        "Culture": row["Subset"],
        "Category": row["Category"],
        "output": output
    })
    # break


df = pd.DataFrame(output_data)

df.to_csv("results//llava_culture_v1.csv")
