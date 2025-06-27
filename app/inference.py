# app/inference.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
import re

MODEL_NAME = "google/gemma-1.1-2b-it"  # Replace with 3n when available

def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        print("[ERROR] Could not load model:", e)
        return None, None

def parse_response(raw_output: str):
    """
    Extracts structured fields from the model's raw response.
    Expected format:
      Priority: Critical
      Detected: Structural Collapse
      Action: Dispatch rescue team
    """
    result = {
        "priority": "Unknown",
        "damage_type": "Unclear",
        "suggestion": "No recommendation available."
    }

    try:
        # Normalize whitespace and punctuation
        lines = raw_output.strip().splitlines()

        for line in lines:
            if "priority" in line.lower():
                result["priority"] = line.split(":")[-1].strip()
            elif "detected" in line.lower():
                result["damage_type"] = line.split(":")[-1].strip()
            elif "action" in line.lower() or "suggest" in line.lower():
                result["suggestion"] = line.split(":")[-1].strip()

    except Exception as e:
        print(f"[WARN] Could not parse response: {e}")

    return result

def run_text_inference(prompt, tokenizer, model, max_new_tokens=128):
    if not tokenizer or not model:
        return {
            "priority": "Unavailable",
            "damage_type": "Unavailable",
            "suggestion": "Model failed to load."
        }

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model.cuda()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return parse_response(raw_output)

def run_image_inference(image_path):
    try:
        img = Image.open(image_path).convert("RGB")

        # Placeholder logic — replace with vision + Gemma 3n pipeline
        return {
            "priority": "Medium",
            "damage_type": "Collapsed Structure",
            "suggestion": "Mark zone and request evacuation team"
        }

    except Exception as e:
        return {
            "priority": "Unavailable",
            "damage_type": "Image error",
            "suggestion": f"Could not process image: {str(e)}"
        }

def run_disaster_analysis(input_data):
    tokenizer, model = load_model()

    if input_data["type"] == "text":
        return run_text_inference(input_data["content"], tokenizer, model)
    elif input_data["type"] == "image":
        return run_image_inference(input_data["content"])
    else:
        return {
            "priority": "Unknown",
            "damage_type": "Unsupported input type",
            "suggestion": "Please submit text or an image."
        }
