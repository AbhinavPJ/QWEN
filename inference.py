from transformers import AutoModelForVision2Seq, AutoTokenizer
from PIL import Image
import torch
model_name = "Qwen/Qwen3-VL-2B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
    device_map=None,
    trust_remote_code=True
).to(device)
model.eval()
image = Image.open("image.png") 
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in one sentence."}
        ]
    }
]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)
inputs = tokenizer(text, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(out[0], skip_special_tokens=True))
