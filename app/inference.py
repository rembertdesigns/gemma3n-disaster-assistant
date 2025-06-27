# app/inference.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch

# Set model path or Hugging Face repo (adjust to your model when Gemma 3n is released)
MODEL_NAME = "google/gemma-1.1-2b-it"  # Placeholder — replace with actual 3n model when available

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

def run_text_inference(prompt, tokenizer, model, max_new_tokens=128):
    if not tokenizer or not model:
        return {"error": "Model not loaded."}

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model.cuda()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"type": "text", "output": result.strip()}

def run_image_inference(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        # Placeholder: replace with ViT + Gemma 3n vision pipeline when available
        return {
            "type": "image",
            "damage_detected": "Collapsed Structure",
            "confidence": 0.87,
            "suggested_action": "Mark zone and request evacuation team"
        }
    except Exception as e:
        return {"error": f"Could not process image: {str(e)}"}

def run_disaster_analysis(input_data):
    tokenizer, model = load_model()

    if input_data['type'] == 'text':
        return run_text_inference(input_data['content'], tokenizer, model)
    elif input_data['type'] == 'image':
        return run_image_inference(input_data['content'])
    else:
        return {"error": "Unsupported input type"}
