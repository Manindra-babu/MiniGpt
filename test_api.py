import requests

api_url = "https://mani-359-custom-llm.hf.space/v1/completions"

headers = {
    "Authorization": "Bearer sk-custom-llm-token",
    "Content-Type": "application/json"
}

data = {
    "prompt": "Question: What is the meaning of life?\nAnswer:",
    "max_tokens": 100,
    "temperature": 0.8
}

response = requests.post(api_url, headers=headers, json=data)

if response.status_code == 200:
    print(response.json()["choices"][0]["text"])
else:
    print("Error:", response.text)