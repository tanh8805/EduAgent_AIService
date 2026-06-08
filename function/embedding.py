import os
import requests
from dotenv import load_dotenv

load_dotenv()

ROUTER_URI = os.getenv("ROUTER_URI")
ROUTER_KEY = os.getenv("ROUTER_KEY")

def embedding(text):
    payload = {
        "input": text,
        "model": "text-embedding-004"
    }

    headers = {
        "Authorization": f"Bearer {ROUTER_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(ROUTER_URI, json=payload, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return data["data"][0]["embedding"]
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print("Error:", e)
        return None
