import os

def preprocess_input(raw_input):
    """
    Detects the type of input (text, image, or future: voice) and returns
    a standardized dictionary for downstream inference.
    """
    if os.path.isfile(raw_input):
        ext = os.path.splitext(raw_input)[-1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            return {
                "type": "image",
                "content": raw_input
            }
        elif ext in ['.wav', '.mp3', '.m4a']:
            # Future: voice support
            return {
                "type": "voice",
                "content": raw_input
            }
        else:
            return {
                "type": "text",
                "content": f"Unrecognized file extension: {ext}"
            }
    else:
        return {
            "type": "text",
            "content": raw_input
        }
