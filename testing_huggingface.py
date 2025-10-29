import requests
url = "https://abhinavpj-qwen-tryingout.hf.space/text-only" 
payload = {'prompt': 'What is the capital of France?'}
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    print(response.json())

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")