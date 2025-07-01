from models.gemma import run_gemma_inference  # ✅ fixed import
import json
import re

def analyze_sentiment(text: str) -> dict:
    """
    Uses Gemma 3n to classify emotional tone from disaster-related input.
    Returns a dictionary with sentiment, tone, and escalation level.
    """
    prompt = f"""
You are a disaster response assistant. Analyze the emotional tone of the following message and classify it into:
- sentiment: Calm, Concerned, Anxious, Panic
- tone: Descriptive, Urgent, Frantic, Helpless
- escalation: Low, Moderate, High, Critical

Message:
\"\"\"{text}\"\"\"

Respond in JSON format:
{{"sentiment": "...", "tone": "...", "escalation": "..."}}
"""
    try:
        response = run_gemma_inference(prompt)
        parsed = parse_gemma_json_response(response)
        return parsed
    except Exception as e:
        return {
            "sentiment": "Unknown",
            "tone": "Unknown",
            "escalation": "Unknown",
            "error": str(e)
        }

def parse_gemma_json_response(text: str) -> dict:
    """
    Attempts to parse Gemma response into structured JSON.
    Falls back to regex extraction if the format is invalid.
    """
    try:
        json_start = text.find('{')
        json_str = text[json_start:]
        return json.loads(json_str)
    except Exception:
        fallback = {
            "sentiment": "Unknown",
            "tone": "Unknown",
            "escalation": "Unknown"
        }
        for field in fallback:
            match = re.search(fr"{field}\s*:\s*(\w+)", text, re.IGNORECASE)
            if match:
                fallback[field] = match.group(1)
        return fallback