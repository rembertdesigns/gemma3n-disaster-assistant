import os
import whisper

def preprocess_input(raw_input):
    """
    Detects input type: text, image, or voice file.
    If voice: transcribes to text using Whisper.
    """
    if os.path.isfile(raw_input):
        ext = os.path.splitext(raw_input)[-1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            return {
                "type": "image",
                "content": raw_input
            }
        
        elif ext in ['.wav', '.mp3', '.m4a']:
            print("🎤 Transcribing voice input...")
            try:
                model = whisper.load_model("base")
                result = model.transcribe(raw_input)
                return {
                    "type": "text",
                    "content": result['text']
                }
            except Exception as e:
                return {
                    "type": "text",
                    "content": f"[ERROR] Could not transcribe: {e}"
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
