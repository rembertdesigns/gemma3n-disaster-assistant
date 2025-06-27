# app/audio_transcription.py

import whisper
import os

# Load the Whisper model once
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

# Define hazard keyword mapping
HAZARD_KEYWORDS = {
    "scream": "Possible distress call detected",
    "help": "Distress signal detected",
    "fire": "Possible fire hazard detected",
    "gunshot": "Gunfire-like sound referenced",
    "sirens": "Emergency response nearby",
    "glass": "Sound of glass breaking detected"
}

def transcribe_audio(file_path: str) -> dict:
    """
    Transcribes an audio file and detects potential hazard keywords.
    Returns a dictionary with the transcription, language, segments (if any), and detected hazards.
    """
    try:
        result = model.transcribe(file_path)
        text = result["text"].lower()

        hazards = []
        for keyword, alert in HAZARD_KEYWORDS.items():
            if keyword in text:
                hazards.append(alert)

        return {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", []),
            "hazards": hazards
        }

    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}