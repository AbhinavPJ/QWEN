from transformers import AutoModelForVision2Seq, AutoTokenizer
from PIL import Image
import torch

model_name = "Qwen/Qwen3-VL-2B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
    device_map=None,
    trust_remote_code=True
).to(device)
model.eval()

# 🖼️ Load your image
image = Image.open("image.png")  # Replace with your image file path

# 💬 Prepare a multimodal message (image + text)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in one sentence."}
        ]
    }
]

# Tokenize for multimodal input
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)
inputs = tokenizer(text, return_tensors="pt").to(device)

# Run generation
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100)

# Decode the model output
print(tokenizer.decode(out[0], skip_special_tokens=True))
