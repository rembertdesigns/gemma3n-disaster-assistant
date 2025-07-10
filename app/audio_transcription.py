# app/audio_transcription.py - Enhanced for Gemma 3n Voice Emergency Processing

import whisper
import librosa
import numpy as np
import torch
from transformers import pipeline, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from datetime import datetime
import scipy.signal
from scipy.io import wavfile
import tempfile
import os

logger = logging.getLogger(__name__)

# Enhanced hazard keyword mapping with context
ENHANCED_HAZARD_KEYWORDS = {
    # Immediate danger indicators
    "fire": {"alert": "Fire hazard detected", "severity": 9, "category": "fire_emergency"},
    "explosion": {"alert": "Explosion reported", "severity": 10, "category": "blast_emergency"},
    "gunshot": {"alert": "Gunfire detected", "severity": 9, "category": "violence"},
    "shooting": {"alert": "Active shooter situation", "severity": 10, "category": "violence"},
    
    # Medical emergencies
    "heart attack": {"alert": "Cardiac emergency", "severity": 9, "category": "medical"},
    "stroke": {"alert": "Stroke symptoms reported", "severity": 9, "category": "medical"},
    "overdose": {"alert": "Drug overdose", "severity": 8, "category": "medical"},
    "seizure": {"alert": "Seizure in progress", "severity": 7, "category": "medical"},
    
    # Environmental hazards
    "flood": {"alert": "Flooding conditions", "severity": 7, "category": "natural_disaster"},
    "tornado": {"alert": "Tornado warning", "severity": 10, "category": "weather"},
    "earthquake": {"alert": "Seismic activity", "severity": 8, "category": "natural_disaster"},
    "landslide": {"alert": "Ground instability", "severity": 8, "category": "natural_disaster"},
    
    # Infrastructure emergencies
    "gas leak": {"alert": "Gas leak detected", "severity": 8, "category": "hazmat"},
    "power lines": {"alert": "Electrical hazard", "severity": 7, "category": "utility"},
    "building collapse": {"alert": "Structural failure", "severity": 9, "category": "structural"},
    "bridge": {"alert": "Bridge emergency", "severity": 8, "category": "infrastructure"},
    
    # General distress
    "help": {"alert": "General distress call", "severity": 6, "category": "distress"},
    "emergency": {"alert": "Emergency situation", "severity": 7, "category": "general"},
    "trapped": {"alert": "Person trapped", "severity": 8, "category": "rescue"},
    "injured": {"alert": "Injury reported", "severity": 6, "category": "medical"}
}

# Emotional state indicators for voice analysis
EMOTIONAL_INDICATORS = {
    "panic": ["panicking", "can't breathe", "terrified", "scared"],
    "calm": ["okay", "fine", "under control", "stable"],
    "urgent": ["hurry", "quickly", "fast", "immediately", "urgent"],
    "confusion": ["don't know", "confused", "not sure", "unclear"],
    "pain": ["hurts", "painful", "agony", "suffering", "ache"]
}

class VoiceEmergencyProcessor:
    """Advanced voice emergency processing with Gemma 3n integration"""
    
    def __init__(self):
        self.whisper_model = None
        self.emotion_classifier = None
        self.speech_classifier = None
        self._load_models()
    
    def _load_models(self):
        """Load all required models for voice processing"""
        try:
            # Load Whisper for transcription
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
            
            # Load emotion classification model
            try:
                self.emotion_classifier = pipeline(
                    "audio-classification",
                    model="superb/wav2vec2-base-superb-er",
                    return_all_scores=True
                )
                logger.info("Emotion classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load emotion classifier: {e}")
                self.emotion_classifier = None
            
            # Load speech classification for stress detection
            try:
                self.speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                logger.info("Speech processor loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load speech processor: {e}")
                self.speech_processor = None
                
        except Exception as e:
            logger.error(f"Failed to load voice processing models: {e}")
    
    def process_emergency_call(self, audio_path: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Enhanced emergency call processing with comprehensive analysis
        
        Args:
            audio_path: Path to audio file
            context: Additional context (location, caller info, etc.)
            
        Returns:
            Comprehensive voice analysis results
        """
        
        if not self.whisper_model:
            return {"error": "Voice processing models not available"}
        
        try:
            # Step 1: Transcribe audio
            logger.info(f"Processing emergency call: {audio_path}")
            transcription_result = self._transcribe_with_metadata(audio_path)
            
            # Step 2: Extract audio features
            audio_features = self._extract_comprehensive_audio_features(audio_path)
            
            # Step 3: Analyze emotional state
            emotional_analysis = self._analyze_emotional_state(audio_path, transcription_result["text"])
            
            # Step 4: Detect hazards and keywords
            hazard_analysis = self._enhanced_hazard_detection(transcription_result["text"])
            
            # Step 5: Assess voice stress and urgency
            stress_analysis = self._analyze_voice_stress(audio_path, audio_features)
            
            # Step 6: Extract location and contact information
            location_info = self._extract_location_information(transcription_result["text"])
            
            # Step 7: Gemma 3n contextual analysis
            from .inference import analyze_voice_emergency
            
            gemma_analysis = analyze_voice_emergency(
                transcript=transcription_result["text"],
                audio_features=audio_features,
                emotional_state=emotional_analysis
            )
            
            # Compile comprehensive results
            comprehensive_result = {
                # Core transcription
                "transcript": transcription_result["text"],
                "language": transcription_result["language"],
                "confidence": transcription_result.get("confidence", 0.0),
                "segments": transcription_result.get("segments", []),
                
                # Audio analysis
                "audio_features": audio_features,
                "audio_quality": self._assess_audio_quality(audio_features),
                
                # Emotional and stress analysis
                "emotional_state": emotional_analysis,
                "stress_analysis": stress_analysis,
                "caller_state": self._determine_caller_state(emotional_analysis, stress_analysis),
                
                # Emergency detection
                "hazards_detected": hazard_analysis["hazards"],
                "emergency_keywords": hazard_analysis["keywords"],
                "severity_indicators": hazard_analysis["severity_indicators"],
                
                # Location and context
                "location_info": location_info,
                "context": context or {},
                
                # AI analysis
                "gemma_analysis": gemma_analysis,
                "overall_urgency": self._calculate_overall_urgency(hazard_analysis, emotional_analysis, stress_analysis),
                
                # Metadata
                "processing_timestamp": datetime.utcnow().isoformat(),
                "audio_duration": audio_features.get("duration", 0),
                "processing_time": None,  # Will be calculated
                
                # Recommendations
                "recommended_actions": self._generate_response_recommendations(
                    gemma_analysis, hazard_analysis, emotional_analysis
                ),
                "priority_level": self._determine_priority_level(hazard_analysis, stress_analysis),
                "dispatch_recommendations": self._generate_dispatch_recommendations(gemma_analysis, location_info)
            }
            
            logger.info(f"Emergency call processed successfully. Urgency: {comprehensive_result['overall_urgency']}")
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Emergency call processing failed: {e}")
            return {
                "error": f"Processing failed: {str(e)}",
                "transcript": "Processing failed",
                "overall_urgency": "unknown",
                "processing_timestamp": datetime.utcnow().isoformat()
            }
    
    def _transcribe_with_metadata(self, audio_path: str) -> Dict[str, Any]:
        """Enhanced transcription with metadata extraction"""
        
        result = self.whisper_model.transcribe(
            audio_path,
            verbose=True,
            word_timestamps=True,
            temperature=0.0  # More deterministic for emergency calls
        )
        
        # Calculate confidence metrics
        if "segments" in result:
            confidences = []
            for segment in result["segments"]:
                if "no_speech_prob" in segment:
                    confidences.append(1.0 - segment["no_speech_prob"])
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
        else:
            avg_confidence = 0.0
        
        return {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "confidence": avg_confidence,
            "segments": result.get("segments", []),
            "word_timestamps": self._extract_word_timestamps(result)
        }
    
    def _extract_comprehensive_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract comprehensive audio features for analysis"""
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            # Basic audio properties
            features = {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1,  # Librosa loads as mono by default
            }
            
            # Speech characteristics
            features.update({
                "speech_rate": self._calculate_speech_rate(y, sr),
                "volume_statistics": self._analyze_volume_patterns(y),
                "frequency_analysis": self._analyze_frequency_content(y, sr),
                "silence_analysis": self._analyze_silence_patterns(y, sr),
                "speech_clarity": self._assess_speech_clarity(y, sr)
            })
            
            # Voice stress indicators
            features.update({
                "fundamental_frequency": self._extract_f0_features(y, sr),
                "spectral_features": self._extract_spectral_features(y, sr),
                "prosodic_features": self._extract_prosodic_features(y, sr)
            })
            
            # Background noise analysis
            features.update({
                "background_noise": self._analyze_background_noise(y, sr),
                "environmental_sounds": self._detect_environmental_sounds(y, sr)
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            return {
                "duration": 0,
                "error": str(e),
                "features_available": False
            }
    
    def _analyze_emotional_state(self, audio_path: str, transcript: str) -> Dict[str, Any]:
        """Comprehensive emotional state analysis"""
        
        emotional_analysis = {
            "primary_emotion": "unknown",
            "confidence": 0.0,
            "emotional_indicators": [],
            "text_based_emotions": {},
            "audio_based_emotions": {},
            "stress_level": 0.0
        }
        
        try:
            # Text-based emotional analysis
            text_emotions = self._analyze_text_emotions(transcript)
            emotional_analysis["text_based_emotions"] = text_emotions
            
            # Audio-based emotional analysis
            if self.emotion_classifier:
                audio_emotions = self._analyze_audio_emotions(audio_path)
                emotional_analysis["audio_based_emotions"] = audio_emotions
                
                if audio_emotions:
                    emotional_analysis["primary_emotion"] = audio_emotions[0]["label"]
                    emotional_analysis["confidence"] = audio_emotions[0]["score"]
            
            # Combined emotional indicators
            emotional_analysis["emotional_indicators"] = self._detect_emotional_indicators(transcript)
            emotional_analysis["stress_level"] = self._calculate_stress_level(text_emotions, emotional_analysis["emotional_indicators"])
            
        except Exception as e:
            logger.error(f"Emotional analysis failed: {e}")
            emotional_analysis["error"] = str(e)
        
        return emotional_analysis
    
    def _enhanced_hazard_detection(self, transcript: str) -> Dict[str, Any]:
        """Enhanced hazard detection with context and severity assessment"""
        
        text_lower = transcript.lower()
        detected_hazards = []
        detected_keywords = []
        severity_indicators = []
        
        # Check for enhanced hazard keywords
        for keyword, info in ENHANCED_HAZARD_KEYWORDS.items():
            if keyword in text_lower:
                detected_hazards.append({
                    "keyword": keyword,
                    "alert": info["alert"],
                    "severity": info["severity"],
                    "category": info["category"],
                    "context": self._extract_keyword_context(transcript, keyword)
                })
                detected_keywords.append(keyword)
                severity_indicators.append(info["severity"])
        
        # Analyze severity indicators in context
        severity_phrases = self._detect_severity_phrases(transcript)
        
        # Calculate composite severity
        if severity_indicators:
            max_severity = max(severity_indicators)
            avg_severity = np.mean(severity_indicators)
        else:
            max_severity = 0
            avg_severity = 0
        
        return {
            "hazards": detected_hazards,
            "keywords": detected_keywords,
            "severity_indicators": severity_indicators,
            "severity_phrases": severity_phrases,
            "max_severity": max_severity,
            "average_severity": avg_severity,
            "hazard_categories": list(set([h["category"] for h in detected_hazards])),
            "immediate_danger": max_severity >= 8,
            "multiple_hazards": len(detected_hazards) > 1
        }
    
    def _analyze_voice_stress(self, audio_path: str, audio_features: Dict) -> Dict[str, Any]:
        """Analyze voice stress patterns"""
        
        try:
            stress_indicators = {
                "speech_rate_stress": 0.0,
                "pitch_variation_stress": 0.0,
                "volume_stress": 0.0,
                "voice_tremor": 0.0,
                "breathing_patterns": "unknown",
                "overall_stress_level": 0.0
            }
            
            # Analyze speech rate for stress
            speech_rate = audio_features.get("speech_rate", 0)
            if speech_rate > 200:  # Very fast speech
                stress_indicators["speech_rate_stress"] = 0.8
            elif speech_rate > 150:  # Fast speech
                stress_indicators["speech_rate_stress"] = 0.6
            elif speech_rate < 80:  # Very slow speech
                stress_indicators["speech_rate_stress"] = 0.7
            
            # Analyze pitch variation
            f0_features = audio_features.get("fundamental_frequency", {})
            if f0_features:
                pitch_std = f0_features.get("std", 0)
                if pitch_std > 50:  # High pitch variation
                    stress_indicators["pitch_variation_stress"] = 0.7
            
            # Analyze volume patterns
            volume_stats = audio_features.get("volume_statistics", {})
            if volume_stats:
                volume_range = volume_stats.get("range", 0)
                if volume_range > 30:  # High volume variation
                    stress_indicators["volume_stress"] = 0.6
            
            # Calculate overall stress level
            stress_values = [v for k, v in stress_indicators.items() if isinstance(v, (int, float))]
            stress_indicators["overall_stress_level"] = np.mean(stress_values) if stress_values else 0.0
            
            return stress_indicators
            
        except Exception as e:
            logger.error(f"Voice stress analysis failed: {e}")
            return {"error": str(e), "overall_stress_level": 0.0}
    
    def _extract_location_information(self, transcript: str) -> Dict[str, Any]:
        """Extract location information from transcript"""
        
        import re
        
        location_info = {
            "addresses": [],
            "landmarks": [],
            "intersections": [],
            "general_areas": [],
            "coordinate_mentions": []
        }
        
        # Address patterns
        address_patterns = [
            r'\b\d+\s+(?:[A-Z][a-zA-Z]*\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Circle|Cir|Place|Pl)\b',
            r'\b(?:on|at|near)\s+([A-Z][a-zA-Z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd))\b'
        ]
        
        for pattern in address_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            location_info["addresses"].extend(matches)
        
        # Landmark patterns
        landmark_patterns = [
            r'\b([A-Z][a-zA-Z\s]+(?:Hospital|School|Mall|Center|Building|Park|Church|Store|Station))\b',
            r'\b(?:at|near)\s+([A-Z][a-zA-Z\s]+(?:Hospital|School|Mall|Center))\b'
        ]
        
        for pattern in landmark_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            location_info["landmarks"].extend(matches)
        
        # Intersection patterns
        intersection_patterns = [
            r'\b([A-Z][a-zA-Z\s]+)\s+(?:and|&|\+)\s+([A-Z][a-zA-Z\s]+)\b',
            r'\bcorner\s+of\s+([A-Z][a-zA-Z\s]+)\s+(?:and|&)\s+([A-Z][a-zA-Z\s]+)\b'
        ]
        
        for pattern in intersection_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            location_info["intersections"].extend([f"{m[0]} and {m[1]}" for m in matches])
        
        # Clean and deduplicate results
        for key in location_info:
            if isinstance(location_info[key], list):
                location_info[key] = list(set([item.strip() for item in location_info[key] if item.strip()]))
        
        return location_info
    
    def _calculate_overall_urgency(self, hazard_analysis: Dict, emotional_analysis: Dict, stress_analysis: Dict) -> str:
        """Calculate overall urgency level from all analyses"""
        
        urgency_score = 0
        
        # Hazard-based urgency
        if hazard_analysis.get("immediate_danger", False):
            urgency_score += 4
        elif hazard_analysis.get("max_severity", 0) >= 6:
            urgency_score += 3
        elif hazard_analysis.get("max_severity", 0) >= 4:
            urgency_score += 2
        
        # Emotional state urgency
        stress_level = emotional_analysis.get("stress_level", 0)
        if stress_level >= 0.8:
            urgency_score += 3
        elif stress_level >= 0.6:
            urgency_score += 2
        elif stress_level >= 0.4:
            urgency_score += 1
        
        # Voice stress urgency
        voice_stress = stress_analysis.get("overall_stress_level", 0)
        if voice_stress >= 0.7:
            urgency_score += 2
        elif voice_stress >= 0.5:
            urgency_score += 1
        
        # Convert score to urgency level
        if urgency_score >= 7:
            return "critical"
        elif urgency_score >= 5:
            return "high"
        elif urgency_score >= 3:
            return "medium"
        elif urgency_score >= 1:
            return "low"
        else:
            return "routine"
    
    def _generate_response_recommendations(self, gemma_analysis: Dict, hazard_analysis: Dict, emotional_analysis: Dict) -> List[Dict]:
        """Generate specific response recommendations"""
        
        recommendations = []
        
        # Immediate safety recommendations
        if hazard_analysis.get("immediate_danger", False):
            recommendations.append({
                "priority": 1,
                "category": "immediate_safety",
                "action": "Dispatch emergency response immediately",
                "timeline": "0-5 minutes",
                "resources": ["police", "fire", "ems"]
            })
        
        # Hazard-specific recommendations
        for hazard in hazard_analysis.get("hazards", []):
            if hazard["category"] == "fire_emergency":
                recommendations.append({
                    "priority": 1,
                    "category": "fire_response",
                    "action": "Dispatch fire department with full equipment",
                    "timeline": "immediate",
                    "resources": ["fire_department", "ems"]
                })
            elif hazard["category"] == "medical":
                recommendations.append({
                    "priority": 1,
                    "category": "medical_response",
                    "action": "Dispatch advanced life support unit",
                    "timeline": "immediate",
                    "resources": ["paramedics", "als_unit"]
                })
        
        # Emotional support recommendations
        stress_level = emotional_analysis.get("stress_level", 0)
        if stress_level >= 0.7:
            recommendations.append({
                "priority": 2,
                "category": "caller_support",
                "action": "Maintain continuous contact for emotional support",
                "timeline": "ongoing",
                "resources": ["dispatcher", "crisis_counselor"]
            })
        
        # Gemma AI recommendations
        if gemma_analysis and "response" in gemma_analysis:
            for action in gemma_analysis["response"]:
                recommendations.append({
                    "priority": action.get("priority", 3),
                    "category": "ai_recommendation",
                    "action": action.get("action", "Unknown action"),
                    "timeline": action.get("timeline", "unknown"),
                    "resources": ["to_be_determined"]
                })
        
        return sorted(recommendations, key=lambda x: x["priority"])
    
    def _determine_priority_level(self, hazard_analysis: Dict, stress_analysis: Dict) -> str:
        """Determine dispatch priority level"""
        
        max_severity = hazard_analysis.get("max_severity", 0)
        voice_stress = stress_analysis.get("overall_stress_level", 0)
        
        if max_severity >= 9 or voice_stress >= 0.8:
            return "priority_1"  # Life-threatening emergency
        elif max_severity >= 7 or voice_stress >= 0.6:
            return "priority_2"  # Urgent emergency
        elif max_severity >= 5 or voice_stress >= 0.4:
            return "priority_3"  # Non-urgent emergency
        else:
            return "priority_4"  # Routine call
    
    def _generate_dispatch_recommendations(self, gemma_analysis: Dict, location_info: Dict) -> Dict[str, Any]:
        """Generate specific dispatch recommendations"""
        
        recommendations = {
            "units_needed": [],
            "estimated_response_time": "unknown",
            "special_equipment": [],
            "route_considerations": [],
            "scene_safety": []
        }
        
        # Determine units needed based on emergency type
        emergency_type = gemma_analysis.get("emergency_type", "unknown")
        
        if "fire" in emergency_type.lower():
            recommendations["units_needed"].extend(["fire_engine", "ladder_truck", "ems_unit"])
            recommendations["special_equipment"].extend(["breathing_apparatus", "thermal_camera"])
        elif "medical" in emergency_type.lower():
            recommendations["units_needed"].extend(["ambulance", "paramedic_unit"])
            recommendations["special_equipment"].extend(["advanced_life_support", "defibrillator"])
        elif "violence" in emergency_type.lower():
            recommendations["units_needed"].extend(["police_units", "supervisor"])
            recommendations["scene_safety"].extend(["secure_perimeter", "await_backup"])
        
        # Location-based considerations
        if location_info.get("addresses") or location_info.get("landmarks"):
            recommendations["route_considerations"].append("GPS_coordinates_available")
        else:
            recommendations["route_considerations"].append("location_unclear_triangulate")
        
        return recommendations
    
    # Helper methods for audio feature extraction
    def _calculate_speech_rate(self, y: np.ndarray, sr: int) -> float:
        """Calculate words per minute from audio"""
        try:
            # Simple syllable-based estimation
            duration = len(y) / sr
            # Rough estimation: average 1.5 syllables per word
            estimated_syllables = self._count_syllables(y, sr)
            estimated_words = estimated_syllables / 1.5
            words_per_minute = (estimated_words / duration) * 60
            return words_per_minute
        except:
            return 0.0
    
    def _count_syllables(self, y: np.ndarray, sr: int) -> int:
        """Estimate syllable count from audio energy"""
        try:
            # Simple peak detection in energy
            hop_length = 512
            energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            peaks, _ = scipy.signal.find_peaks(energy, height=np.mean(energy) * 0.5)
            return len(peaks)
        except:
            return 0
    
    def _analyze_volume_patterns(self, y: np.ndarray) -> Dict[str, float]:
        """Analyze volume patterns in audio"""
        try:
            rms = librosa.feature.rms(y=y)[0]
            db = librosa.amplitude_to_db(rms)
            
            return {
                "mean_db": float(np.mean(db)),
                "std_db": float(np.std(db)),
                "min_db": float(np.min(db)),
                "max_db": float(np.max(db)),
                "range": float(np.max(db) - np.min(db))
            }
        except:
            return {"mean_db": 0, "std_db": 0, "min_db": 0, "max_db": 0, "range": 0}
    
    def _analyze_frequency_content(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze frequency content"""
        try:
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            return {
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "frequency_range": "speech_range" if np.mean(spectral_centroids) < 4000 else "wide_range"
            }
        except:
            return {"spectral_centroid_mean": 0, "spectral_rolloff_mean": 0, "frequency_range": "unknown"}
    
    def _analyze_silence_patterns(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze silence and pause patterns"""
        try:
            # Simple energy-based silence detection
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.01 * sr)     # 10ms hop
            
            energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            threshold = np.mean(energy) * 0.1
            
            silent_frames = energy < threshold
            silence_ratio = np.sum(silent_frames) / len(silent_frames)
            
            return {
                "silence_ratio": float(silence_ratio),
                "speech_continuity": "continuous" if silence_ratio < 0.2 else "interrupted",
                "pause_frequency": "high" if silence_ratio > 0.4 else "normal"
            }
        except:
            return {"silence_ratio": 0, "speech_continuity": "unknown", "pause_frequency": "unknown"}
    
    def _assess_speech_clarity(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Assess speech clarity and intelligibility"""
        try:
            # Spectral clarity metrics
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            clarity_score = np.mean(spectral_contrast)
            
            return {
                "clarity_score": float(clarity_score),
                "intelligibility": "high" if clarity_score > 15 else "medium" if clarity_score > 10 else "low"
            }
        except:
            return {"clarity_score": 0, "intelligibility": "unknown"}
    
    def _extract_f0_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract fundamental frequency features"""
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            f0_values = []
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    f0_values.append(pitch)
            
            if f0_values:
                return {
                    "mean": float(np.mean(f0_values)),
                    "std": float(np.std(f0_values)),
                    "min": float(np.min(f0_values)),
                    "max": float(np.max(f0_values))
                }
            else:
                return {"mean": 0, "std": 0, "min": 0, "max": 0}
        except:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral features for voice analysis"""
        try:
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            return {
                "spectral_centroid": float(np.mean(spectral_centroids)),
                "spectral_bandwidth": float(np.mean(spectral_bandwidth)),
                "spectral_rolloff": float(np.mean(spectral_rolloff)),
                "zero_crossing_rate": float(np.mean(zero_crossing_rate))
            }
        except:
            return {"spectral_centroid": 0, "spectral_bandwidth": 0, "spectral_rolloff": 0, "zero_crossing_rate": 0}
    
    def _extract_prosodic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract prosodic features (rhythm, stress patterns)"""
        try:
            # Tempo and rhythm analysis
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            return {
                "tempo": float(tempo),
                "rhythm_regularity": "regular" if len(beats) > 10 else "irregular",
                "beat_strength": float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))
            }
        except:
            return {"tempo": 0, "rhythm_regularity": "unknown", "beat_strength": 0}
    
    def _analyze_background_noise(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze background noise and environmental sounds"""
        try:
            # Simple noise floor analysis
            rms_energy = librosa.feature.rms(y=y)[0]
            noise_floor = np.percentile(rms_energy, 10)  # Bottom 10% as noise floor
            signal_to_noise = np.mean(rms_energy) / noise_floor if noise_floor > 0 else 0
            
            return {
                "noise_floor": float(noise_floor),
                "signal_to_noise_ratio": float(signal_to_noise),
                "noise_level": "high" if signal_to_noise < 3 else "medium" if signal_to_noise < 10 else "low"
            }
        except:
            return {"noise_floor": 0, "signal_to_noise_ratio": 0, "noise_level": "unknown"}
    
    def _detect_environmental_sounds(self, y: np.ndarray, sr: int) -> List[str]:
        """Detect environmental sounds in background"""
        # This would typically use a trained model for sound event detection
        # For now, return basic frequency-based detection
        
        detected_sounds = []
        
        try:
            # Simple frequency analysis for common emergency sounds
            freqs = np.fft.fftfreq(len(y), 1/sr)
            fft = np.abs(np.fft.fft(y))
            
            # Siren detection (around 1000-2000 Hz modulation)
            siren_range = np.where((freqs >= 800) & (freqs <= 2500))[0]
            if len(siren_range) > 0 and np.max(fft[siren_range]) > np.mean(fft) * 2:
                detected_sounds.append("possible_siren")
            
            # Vehicle sounds (low frequency rumble)
            vehicle_range = np.where((freqs >= 50) & (freqs <= 300))[0]
            if len(vehicle_range) > 0 and np.max(fft[vehicle_range]) > np.mean(fft) * 1.5:
                detected_sounds.append("vehicle_sounds")
            
        except:
            pass
        
        return detected_sounds
    
    def _analyze_text_emotions(self, transcript: str) -> Dict[str, float]:
        """Analyze emotions from transcript text"""
        
        emotions = {emotion: 0.0 for emotion in EMOTIONAL_INDICATORS.keys()}
        text_lower = transcript.lower()
        
        for emotion, indicators in EMOTIONAL_INDICATORS.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            emotions[emotion] = min(score / len(indicators), 1.0)
        
        return emotions
    
    def _analyze_audio_emotions(self, audio_path: str) -> List[Dict]:
        """Analyze emotions from audio using ML model"""
        try:
            if self.emotion_classifier:
                results = self.emotion_classifier(audio_path)
                return sorted(results, key=lambda x: x["score"], reverse=True)
            else:
                return []
        except Exception as e:
            logger.error(f"Audio emotion analysis failed: {e}")
            return []
    
    def _detect_emotional_indicators(self, transcript: str) -> List[str]:
        """Detect emotional indicators in transcript"""
        
        indicators = []
        text_lower = transcript.lower()
        
        # Intensity words
        intensity_words = ["very", "extremely", "really", "so", "totally", "completely"]
        for word in intensity_words:
            if word in text_lower:
                indicators.append(f"intensity_{word}")
        
        # Urgency phrases
        urgency_phrases = ["right now", "immediately", "as soon as possible", "hurry", "quick"]
        for phrase in urgency_phrases:
            if phrase in text_lower:
                indicators.append(f"urgency_{phrase.replace(' ', '_')}")
        
        # Emotional expressions
        emotional_expressions = ["oh my god", "help me", "i can't", "please", "scared", "worried"]
        for expr in emotional_expressions:
            if expr in text_lower:
                indicators.append(f"emotional_{expr.replace(' ', '_')}")
        
        return indicators
    
    def _calculate_stress_level(self, text_emotions: Dict, emotional_indicators: List) -> float:
        """Calculate overall stress level from text analysis"""
        
        # Weight different emotions
        stress_weights = {
            "panic": 1.0,
            "urgent": 0.8,
            "confusion": 0.6,
            "pain": 0.7,
            "calm": -0.5  # Negative weight for calm
        }
        
        stress_score = 0.0
        for emotion, level in text_emotions.items():
            weight = stress_weights.get(emotion, 0.0)
            stress_score += level * weight
        
        # Add indicator-based stress
        indicator_stress = len(emotional_indicators) * 0.1
        
        # Normalize to 0-1 range
        total_stress = min(max(stress_score + indicator_stress, 0.0), 1.0)
        
        return total_stress
    
    def _extract_keyword_context(self, transcript: str, keyword: str) -> str:
        """Extract context around detected keywords"""
        
        text_lower = transcript.lower()
        keyword_pos = text_lower.find(keyword)
        
        if keyword_pos == -1:
            return ""
        
        # Extract 50 characters before and after the keyword
        start = max(0, keyword_pos - 50)
        end = min(len(transcript), keyword_pos + len(keyword) + 50)
        
        context = transcript[start:end].strip()
        return context
    
    def _detect_severity_phrases(self, transcript: str) -> List[str]:
        """Detect phrases indicating severity level"""
        
        severity_phrases = []
        text_lower = transcript.lower()
        
        # Critical severity indicators
        critical_phrases = ["life threatening", "critical condition", "not breathing", "unconscious", "massive bleeding"]
        for phrase in critical_phrases:
            if phrase in text_lower:
                severity_phrases.append(f"critical_{phrase.replace(' ', '_')}")
        
        # High severity indicators
        high_phrases = ["serious injury", "severe pain", "can't move", "losing consciousness", "heavy bleeding"]
        for phrase in high_phrases:
            if phrase in text_lower:
                severity_phrases.append(f"high_{phrase.replace(' ', '_')}")
        
        return severity_phrases
    
    def _assess_audio_quality(self, audio_features: Dict) -> Dict[str, Any]:
        """Assess overall audio quality for processing"""
        
        quality_score = 0.0
        quality_factors = []
        
        # Signal to noise ratio
        snr = audio_features.get("background_noise", {}).get("signal_to_noise_ratio", 0)
        if snr > 10:
            quality_score += 0.3
            quality_factors.append("good_snr")
        elif snr > 5:
            quality_score += 0.2
        else:
            quality_factors.append("poor_snr")
        
        # Speech clarity
        clarity = audio_features.get("speech_clarity", {}).get("clarity_score", 0)
        if clarity > 15:
            quality_score += 0.3
            quality_factors.append("clear_speech")
        elif clarity > 10:
            quality_score += 0.2
        else:
            quality_factors.append("unclear_speech")
        
        # Duration adequacy
        duration = audio_features.get("duration", 0)
        if duration > 10:
            quality_score += 0.2
            quality_factors.append("adequate_duration")
        elif duration > 5:
            quality_score += 0.1
        else:
            quality_factors.append("short_duration")
        
        # Continuity
        silence_ratio = audio_features.get("silence_analysis", {}).get("silence_ratio", 0)
        if silence_ratio < 0.3:
            quality_score += 0.2
            quality_factors.append("continuous_speech")
        else:
            quality_factors.append("fragmented_speech")
        
        quality_level = "excellent" if quality_score > 0.8 else "good" if quality_score > 0.6 else "fair" if quality_score > 0.4 else "poor"
        
        return {
            "overall_quality": quality_level,
            "quality_score": quality_score,
            "quality_factors": quality_factors,
            "processing_confidence": quality_score
        }
    
    def _determine_caller_state(self, emotional_analysis: Dict, stress_analysis: Dict) -> Dict[str, Any]:
        """Determine overall caller state and recommendations"""
        
        stress_level = emotional_analysis.get("stress_level", 0)
        voice_stress = stress_analysis.get("overall_stress_level", 0)
        primary_emotion = emotional_analysis.get("primary_emotion", "unknown")
        
        # Determine caller state
        if stress_level > 0.8 or voice_stress > 0.8:
            caller_state = "high_distress"
            support_needed = "immediate_emotional_support"
        elif stress_level > 0.6 or voice_stress > 0.6:
            caller_state = "moderate_distress"
            support_needed = "reassurance_and_guidance"
        elif stress_level > 0.4 or voice_stress > 0.4:
            caller_state = "mild_distress"
            support_needed = "calm_guidance"
        else:
            caller_state = "stable"
            support_needed = "standard_protocol"
        
        return {
            "state": caller_state,
            "primary_emotion": primary_emotion,
            "stress_level": max(stress_level, voice_stress),
            "support_needed": support_needed,
            "communication_approach": self._recommend_communication_approach(caller_state, primary_emotion)
        }
    
    def _recommend_communication_approach(self, caller_state: str, primary_emotion: str) -> List[str]:
        """Recommend communication approach for dispatcher"""
        
        approaches = []
        
        if caller_state == "high_distress":
            approaches.extend([
                "speak_slowly_and_clearly",
                "use_calming_tone",
                "repeat_key_information",
                "provide_frequent_reassurance"
            ])
        elif caller_state == "moderate_distress":
            approaches.extend([
                "maintain_professional_calm",
                "acknowledge_emotions",
                "provide_clear_instructions"
            ])
        else:
            approaches.extend([
                "standard_professional_tone",
                "focus_on_information_gathering"
            ])
        
        # Emotion-specific approaches
        if "panic" in primary_emotion:
            approaches.append("breathing_guidance")
        elif "confusion" in primary_emotion:
            approaches.append("simple_yes_no_questions")
        
        return approaches
    
    def _extract_word_timestamps(self, whisper_result: Dict) -> List[Dict]:
        """Extract word-level timestamps from Whisper result"""
        
        word_timestamps = []
        
        if "segments" in whisper_result:
            for segment in whisper_result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        word_timestamps.append({
                            "word": word_info.get("word", ""),
                            "start": word_info.get("start", 0),
                            "end": word_info.get("end", 0),
                            "confidence": word_info.get("probability", 0)
                        })
        
        return word_timestamps

# Enhanced functions for compatibility with existing codebase
def transcribe_audio(audio_path: str) -> dict:
    """Enhanced audio transcription with comprehensive analysis"""
    
    processor = VoiceEmergencyProcessor()
    return processor.process_emergency_call(audio_path)

def get_audio_analysis(file_path: str) -> dict:
    """Get comprehensive audio analysis"""
    
    processor = VoiceEmergencyProcessor()
    result = processor.process_emergency_call(file_path)
    
    # Return in expected format for existing code
    return {
        "transcript": result.get("transcript", ""),
        "language": result.get("language", "unknown"),
        "escalation": result.get("overall_urgency", "unknown"),
        "tone": result.get("emotional_state", {}).get("primary_emotion", "unknown"),
        "confidence": result.get("confidence", 0.0),
        "enhanced_analysis": result
    }