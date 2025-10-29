from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

model_name = "Qwen/Qwen3-VL-2B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model and PROCESSOR (not tokenizer!)
print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
    device_map=None,
    trust_remote_code=True
).to(device)
model.eval()
print(f"Model loaded on {device}\n")

# ============================================
# A) OCR Test on text.png
# ============================================
print("=" * 50)
print("A) OCR TEST")
print("=" * 50)

image_ocr = Image.open("text.png")

messages_ocr = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_ocr},
            {"type": "text", "text": "Extract all text from this image."}
        ]
    }
]

# Use processor.apply_chat_template (not tokenizer!)
inputs_ocr = processor.apply_chat_template(
    messages_ocr,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    out_ocr = model.generate(**inputs_ocr, max_new_tokens=200)

# Decode only the generated tokens
generated_ids = [
    out_ids[len(in_ids):] 
    for in_ids, out_ids in zip(inputs_ocr['input_ids'], out_ocr)
]
print("OCR Result:")
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
print()

# ============================================
# B) Image Caption Test on image.png
# ============================================
print("=" * 50)
print("B) IMAGE CAPTION TEST")
print("=" * 50)

image_caption = Image.open("image.png")

messages_caption = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_caption},
            {"type": "text", "text": "Describe this image in one sentence."}
        ]
    }
]

inputs_caption = processor.apply_chat_template(
    messages_caption,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    out_caption = model.generate(**inputs_caption, max_new_tokens=100)

generated_ids = [
    out_ids[len(in_ids):] 
    for in_ids, out_ids in zip(inputs_caption['input_ids'], out_caption)
]
print("Caption Result:")
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
print()

# ============================================
# C) Text-Only Sanity Check
# ============================================
print("=" * 50)
print("C) TEXT-ONLY SANITY CHECK")
print("=" * 50)

messages_text = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the capital of France?"}
        ]
    }
]

inputs_text = processor.apply_chat_template(
    messages_text,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    out_text = model.generate(**inputs_text, max_new_tokens=50)

generated_ids = [
    out_ids[len(in_ids):] 
    for in_ids, out_ids in zip(inputs_text['input_ids'], out_text)
]
print("Text-Only Result:")
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
print()

print("=" * 50)
print("All tests completed!")
print("=" * 50)