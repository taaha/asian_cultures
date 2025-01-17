from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import requests
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

asian_cooking_ds = load_dataset("darthPanda/cvqa_edit5")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)



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
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": instruction},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    output = output_text[0]
    
    print(output)

    output_data.append({
        "ID": row["ID"],
        "Culture": row["Subset"],
        "Category": row["Category"],
        "output": output,
    })


df = pd.DataFrame(output_data)

df.to_csv("results//qwen_culture_v2.csv")
