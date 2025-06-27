import whisper
import tempfile
import os

# Load model once
model = whisper.load_model("base")  # You can use "tiny", "base", "small", "medium", "large"

def transcribe_audio(file_path):
    try:
        result = model.transcribe(file_path)
        return {
            "text": result["text"],
            "language": result["language"],
            "segments": result.get("segments", [])
        }
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}