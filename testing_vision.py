import requests
url = "https://abhinavpj-qwen-tryingout.hf.space/inference"
image_path = "image.png" 
prompt_text = "What is in this image?"
files = {
    'image': (image_path, open(image_path, 'rb'), 'image/png')
}
payload = {
    'prompt': prompt_text
}
response = requests.post(url, files=files, data=payload)
response.raise_for_status()
data = response.json()
if data.get('success'):
    print(f"Result: {data.get('result')}")
