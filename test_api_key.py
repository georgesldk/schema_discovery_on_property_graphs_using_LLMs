import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Use the 2026 stable flagship model
model = genai.GenerativeModel('gemini-2.5-flash') 

print("Sending text request to Gemini 2.5...")

try:
    response = model.generate_content("What is the best way to represent a database schema in JSON in one sentence?")
    print("-" * 30)
    print(response.text)
    print("-" * 30)
except Exception as e:
    print(f"Error: {e}")