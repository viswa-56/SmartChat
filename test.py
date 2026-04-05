import os
import requests
from dotenv import load_dotenv
load_dotenv()


API_URL = "https://router.huggingface.co/together/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}",
}
print(os.environ['HUGGINGFACE_API_KEY'])
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

response = query({
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "model": "google/gemma-2-27b-it"
})

print(response["choices"][0]["message"])