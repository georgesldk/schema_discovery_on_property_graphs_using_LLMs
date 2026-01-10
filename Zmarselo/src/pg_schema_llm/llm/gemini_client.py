import os
import json
import google.generativeai as genai
from dotenv import load_dotenv


def call_gemini_api(prompt):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return response.text
    except Exception as e:
        print(f"API Error: {e}")
        return None


def extract_json(text):
    try:
        return json.loads(
            text.strip()
                .replace("```json", "")
                .replace("```", "")
        )
    except Exception:
        return None
