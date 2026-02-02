import os
import json
import google.generativeai as genai
from dotenv import load_dotenv


def call_gemini_api(prompt):
    """
    Invoke the Gemini LLM API with a structured schema inference prompt.

    This function loads API credentials from environment variables,
    configures the Gemini client, and submits a prompt requesting a
    JSON-formatted response. It serves as a thin abstraction layer
    between the inference pipeline and the external LLM service.

    Args:
        prompt (str): Fully constructed prompt for schema inference.

    Returns:
        Optional[str]: Raw text response from the LLM, or None if the
        request fails.
    """
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
    """
    Extract and parse a JSON object from an LLM response.

    This function cleans common Markdown code-fence wrappers and
    attempts to deserialize the remaining content into a JSON object.
    It provides a defensive parsing layer for model outputs that are
    expected, but not guaranteed, to be valid JSON.

    Args:
        text (str): Raw text response from the LLM.

    Returns:
        Optional[dict]: Parsed JSON object if successful, otherwise None.
    """
    try:
        return json.loads(
            text.strip()
                .replace("```json", "")
                .replace("```", "")
        )
    except Exception:
        return None
