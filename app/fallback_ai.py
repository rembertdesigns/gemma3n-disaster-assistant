# app/fallback_ai.py - Fallback implementations for missing AI modules

import json
import random
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

class MockGemmaProcessor:
    """Mock Gemma 3n processor for when the real one isn't available"""
    
    def __init__(self, mode="balanced"):
        self.mode = mode
        self.model = True
        self.device = "CPU"
        self.config = {"model_name": "gemma-3n-4b", "context_window": 128000}
    
    def analyze_multimodal_emergency(self, text=None, image_data=None, audio_data=None, context=None):
        """Simulate multimodal emergency analysis"""
        # Simulate processing time
        time.sleep(0.1)
        
        severity_score = 5.0
        confidence = 0.8
        
        # Adjust based on inputs
        if text and any(word in text.lower() for word in ["fire", "critical", "emergency", "help"]):
            severity_score += 2.0
            confidence += 0.1
        
        if image_data:
            severity_score += 1.0
            confidence += 0.05
        
        if audio_data:
            severity_score += 1.5
            confidence += 0.05
        
        emergency_types = ["fire", "medical", "accident", "weather", "infrastructure"]
        primary_type = "fire" if text and "fire" in text.lower() else random.choice(emergency_types)
        
        return {
            "severity": {
                "overall_score": min(10.0, severity_score),
                "confidence": min(1.0, confidence),
                "reasoning": f"Analysis based on {self.mode} mode processing"
            },
            "emergency_type": {
                "primary": primary_type,
                "secondary": [random.choice(emergency_types)],
                "confidence": confidence
            },
            "immediate_risks": [
                {"risk": "Structural damage", "probability": 0.7, "impact": 8},
                {"risk": "Personal injury", "probability": 0.6, "impact": 7}
            ],
            "priority_actions": [
                {"action": "Dispatch emergency services", "priority": 1, "timeline": "immediate"},
                {"action": "Secure area perimeter", "priority": 2, "timeline": "5 minutes"}
            ],
            "resource_requirements": {
                "personnel": {"first_responders": 4, "medical": 2},
                "equipment": ["emergency_vehicle", "medical_kit"],
                "estimated_response_time": "10 minutes"
            },
            "device_performance": {
                "inference_speed": 0.15,
                "cpu_usage": 45.0,
                "memory_usage": 60.0
            }
        }

class MockVoiceProcessor:
    """Mock voice emergency processor"""
    
    def process_emergency_call(self, audio_path, context=None):
        """Simulate voice emergency processing"""
        time.sleep(0.2)  # Simulate processing
        
        urgency_keywords = ["help", "fire", "emergency", "critical", "urgent"]
        
        # Simulate transcript based on filename or generate generic
        transcript = "There's an emergency situation requiring immediate assistance."
        
        if "fire" in audio_path.lower():
            transcript = "There's a fire in the building. We need help immediately."
        elif "medical" in audio_path.lower():
            transcript = "Medical emergency. Person is unconscious and not breathing."
        
        urgency = "critical" if any(word in transcript.lower() for word in urgency_keywords) else "medium"
        
        return {
            "transcript": transcript,
            "confidence": 0.85,
            "overall_urgency": urgency,
            "emotional_state": {
                "primary_emotion": "urgent",
                "stress_level": 0.7 if urgency == "critical" else 0.4,
                "caller_state": "distressed"
            },
            "hazards_detected": ["fire", "smoke"] if "fire" in transcript else ["injury"],
            "location_info": {"addresses": ["Emergency location"]},
            "audio_duration": 30,
            "severity_indicators": [8 if urgency == "critical" else 5],
            "gemma_analysis": {
                "emergency_type": "fire" if "fire" in transcript else "medical",
                "confidence": 0.8
            }
        }

class MockAIOptimizer:
    """Mock AI optimizer for device performance"""
    
    def __init__(self):
        self.device_caps = {
            "cpu_cores": 4, "memory_gb": 8, "gpu_available": False, "gpu_memory_gb": 0
        }
        self.current_config = type('Config', (), {
            'model_variant': 'gemma-3n-4b',
            'context_window': 64000,
            'precision': 'fp16',
            'optimization_level': 'balanced',
            'batch_size': 1
        })()
    
    def optimize_for_device(self, level):
        """Optimize AI settings for device"""
        config = type('Config', (), {
            'model_variant': 'gemma-3n-2b' if level == "emergency" else 'gemma-3n-4b',
            'context_window': 32000 if level == "emergency" else 64000,
            'precision': 'fp16',
            'optimization_level': level,
            'batch_size': 1
        })()
        return config
    
    def monitor_performance(self):
        """Monitor system performance"""
        try:
            import psutil
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
        except:
            cpu_usage = random.uniform(30, 70)
            memory_usage = random.uniform(40, 80)
        
        return type('Performance', (), {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "gpu_usage": 0,
            "battery_level": 80,
            "inference_speed": random.uniform(8, 15),
            "temperature": random.uniform(35, 55),
            "timestamp": datetime.utcnow()
        })()

# Create global instances
gemma_processor = MockGemmaProcessor()
voice_processor = MockVoiceProcessor()
ai_optimizer = MockAIOptimizer()

# Fallback functions
def analyze_voice_emergency(transcript: str, audio_features: dict, emotional_state: dict) -> dict:
    """Analyze voice emergency with mock context understanding"""
    
    result = gemma_processor.analyze_multimodal_emergency(text=transcript)
    
    voice_analysis = {
        "urgency": _determine_urgency_from_analysis(result),
        "emergency_type": result.get("emergency_type", {}).get("primary", "Unknown"),
        "location": _extract_location_mentions(transcript),
        "confidence": result.get("severity", {}).get("confidence", 0.5),
        "response": result.get("priority_actions", [])
    }
    
    return voice_analysis

def _determine_urgency_from_analysis(analysis: dict) -> str:
    """Determine urgency level from analysis results"""
    severity = analysis.get("severity", {}).get("overall_score", 5)
    
    if severity >= 8:
        return "critical"
    elif severity >= 6:
        return "high"
    elif severity >= 4:
        return "medium"
    else:
        return "low"

def _extract_location_mentions(text: str) -> str:
    """Extract location mentions from text"""
    import re
    
    location_patterns = [
        r'\b(?:at|on|near|in)\s+([A-Z][a-zA-Z\s]+(?:Street|St|Avenue|Ave|Road|Rd))\b',
        r'\b([A-Z][a-zA-Z\s]+(?:Hospital|School|Mall|Center|Building))\b'
    ]
    
    locations = []
    for pattern in location_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        locations.extend(matches)
    
    return locations[0] if locations else "Location not specified"

def detect_hazards(image_data):
    """Mock hazard detection"""
    hazards = ["fire", "smoke", "debris"]
    return random.sample(hazards, random.randint(1, 2))

def transcribe_audio(audio_path):
    """Mock audio transcription"""
    return {
        "transcript": "Emergency situation detected from audio analysis",
        "confidence": 0.8
    }

def analyze_sentiment(text):
    """Mock sentiment analysis"""
    if any(word in text.lower() for word in ["urgent", "emergency", "critical"]):
        return {"sentiment": "urgent", "tone": "concerned", "escalation": "high"}
    return {"sentiment": "neutral", "tone": "descriptive", "escalation": "low"}

def generate_report_pdf(data):
    """Mock PDF generation"""
    return f"mock_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

def generate_map_preview_data(lat, lon):
    """Mock map preview data"""
    return {
        "success": True,
        "coordinates": {"latitude": lat, "longitude": lon},
        "emergency_resources": [
            {"name": "General Hospital", "type": "hospital", "distance_km": 2.3, "estimated_time": "8 min"},
            {"name": "Fire Station 1", "type": "fire_station", "distance_km": 1.1, "estimated_time": "4 min"}
        ]
    }
