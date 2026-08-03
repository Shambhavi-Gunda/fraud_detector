"""
Run this once to see which Gemini models your API key actually has access to.
"""
import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name)