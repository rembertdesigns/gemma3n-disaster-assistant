#!/usr/bin/env python3
"""
complete_fix.py - One script to fix all the undefined variable errors

Run this script to automatically fix the missing models and imports in your
Emergency Response Assistant application.
"""

import os
import sys
import shutil
from pathlib import Path

def create_fallback_ai():
    """Create the fallback AI module"""
    
    fallback_content = '''# app/fallback_ai.py - Fallback implementations for missing AI modules

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
        r'\\b(?:at|on|near|in)\\s+([A-Z][a-zA-Z\\s]+(?:Street|St|Avenue|Ave|Road|Rd))\\b',
        r'\\b([A-Z][a-zA-Z\\s]+(?:Hospital|School|Mall|Center|Building))\\b'
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
'''
    
    app_dir = Path("app")
    app_dir.mkdir(exist_ok=True)
    
    fallback_file = app_dir / "fallback_ai.py"
    fallback_file.write_text(fallback_content)
    
    print(f"✅ Created {fallback_file}")

def backup_and_update_models():
    """Backup existing models.py and update with complete version"""
    
    models_file = Path("app/models.py")
    
    # Create backup
    if models_file.exists():
        backup_file = Path("app/models.py.backup")
        shutil.copy2(models_file, backup_file)
        print(f"✅ Backed up existing models.py to {backup_file}")
    
    # Complete models.py content with all missing models
    complete_models = '''# app/models.py - Complete models with all missing classes

from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

# ==========================
# 👤 USER MODEL
# ==========================
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    
    # Role and permissions
    role = Column(String(20), default="user", index=True)  # user, admin, responder, viewer
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    permissions = Column(JSON, nullable=True)  # Additional permissions
    
    # System fields
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"

# ==========================
# 🚨 EMERGENCY REPORT MODEL
# ==========================
class EmergencyReport(Base):
    __tablename__ = "emergency_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(50), unique=True, index=True)  # Unique report identifier
    
    # Core report information
    type = Column(String(100), nullable=False)  # fire, medical, accident, etc.
    description = Column(Text, nullable=False)
    location = Column(String(255), nullable=False)
    
    # Geographic coordinates
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Priority and status
    priority = Column(String(20), default="medium", index=True)  # low, medium, high, critical
    status = Column(String(20), default="pending", index=True)  # pending, active, resolved, closed
    severity = Column(Integer, nullable=True)  # 1-10 scale
    
    # Input method and source
    method = Column(String(20), default="text")  # text, voice, image, multimodal
    reporter = Column(String(100), nullable=True)  # Who reported it
    
    # File attachments
    evidence_file = Column(String(255), nullable=True)  # Evidence file path
    
    # AI Analysis results
    ai_analysis = Column(JSON, nullable=True)  # AI processing results
    confidence_score = Column(Float, nullable=True)  # AI confidence 0.0-1.0
    
    # System fields
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<EmergencyReport(id={self.id}, type='{self.type}', priority='{self.priority}')>"

# ==========================
# 📣 CROWD REPORT MODEL
# ==========================
class CrowdReport(Base):
    __tablename__ = "crowd_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    message = Column(Text)
    tone = Column(String)
    escalation = Column(String)
    user = Column(String)
    location = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Enhanced fields
    severity = Column(Integer, nullable=True, index=True)  # 1-10 severity score
    confidence_score = Column(Float, nullable=True)  # AI confidence 0.0-1.0
    ai_analysis = Column(JSON, nullable=True)  # Full AI analysis results
    source = Column(String, default="manual", index=True)  # manual, voice_analysis_system, etc.
    verified = Column(Boolean, default=False)  # Human verification
    response_dispatched = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<CrowdReport(id={self.id}, escalation='{self.escalation}', severity={self.severity})>"

# ==========================
# 🏥 TRIAGE PATIENT MODEL
# ==========================
class TriagePatient(Base):
    __tablename__ = "triage_patients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    injury_type = Column(String, nullable=False)
    consciousness = Column(String, nullable=False)
    breathing = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    triage_color = Column(String, nullable=False, index=True)
    status = Column(String, default="active", index=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    timestamp = Column(DateTime, default=datetime.utcnow)  # Legacy compatibility
    
    def __repr__(self):
        return f"<TriagePatient(id={self.id}, name='{self.name}', triage_color='{self.triage_color}')>"

# ==========================
# 🎤 VOICE ANALYSIS MODEL
# ==========================
class VoiceAnalysis(Base):
    __tablename__ = "voice_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    audio_file_path = Column(String(255))
    transcript = Column(Text)
    confidence = Column(Float, default=0.0)
    urgency_level = Column(String(20), index=True)  # critical, high, medium, low
    emergency_type = Column(String(100))  # fire, medical, violence, etc.
    hazards_detected = Column(JSON)  # List of detected hazards
    emotional_state = Column(JSON)  # Emotional analysis results
    analyst_id = Column(String(100))  # Who processed the analysis
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<VoiceAnalysis(id={self.id}, urgency='{self.urgency_level}')>"

# ==========================
# 🔀 MULTIMODAL ASSESSMENT MODEL
# ==========================
class MultimodalAssessment(Base):
    __tablename__ = "multimodal_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_type = Column(String(50), index=True)  # comprehensive, damage, medical, etc.
    text_input = Column(Text)
    image_path = Column(String(255))
    audio_path = Column(String(255))
    severity_score = Column(Float, default=0.0, index=True)  # 0.0-10.0
    emergency_type = Column(String(100), index=True)  # Primary emergency classification
    risk_factors = Column(JSON)  # Identified risk factors
    resource_requirements = Column(JSON)  # Required resources
    ai_confidence = Column(Float, default=0.0)  # Overall AI confidence
    analyst_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<MultimodalAssessment(id={self.id}, type='{self.assessment_type}')>"

# ==========================
# 🧠 CONTEXT ANALYSIS MODEL
# ==========================
class ContextAnalysis(Base):
    __tablename__ = "context_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(String(50), index=True)  # comprehensive, batch_summary, etc.
    input_tokens = Column(Integer, default=0)  # Tokens processed
    output_summary = Column(Text)  # Analysis summary
    key_insights = Column(JSON)  # Key insights extracted
    confidence = Column(Float, default=0.0)  # Analysis confidence
    analyst_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ContextAnalysis(id={self.id}, type='{self.analysis_type}')>"

# ==========================
# ⚡ DEVICE PERFORMANCE MODEL
# ==========================
class DevicePerformance(Base):
    __tablename__ = "device_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(100), index=True)  # Device identifier
    cpu_usage = Column(Float)  # CPU usage percentage
    memory_usage = Column(Float)  # Memory usage percentage
    gpu_usage = Column(Float)  # GPU usage percentage
    battery_level = Column(Float)  # Battery level percentage
    temperature = Column(Float)  # CPU temperature
    inference_speed = Column(Float)  # Tokens per second
    optimization_level = Column(String(20))  # speed, balanced, quality
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<DevicePerformance(device='{self.device_id}', cpu={self.cpu_usage}%)>"
'''
    
    models_file.write_text(complete_models)
    print(f"✅ Updated {models_file} with complete models")

def create_migration_script():
    """Create a simple migration script"""
    
    migration_content = '''#!/usr/bin/env python3
# quick_migration.py - Run this to update your database

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def run_migration():
    """Quick migration to create missing tables"""
    
    print("🔧 Quick Database Migration")
    print("=" * 40)
    
    try:
        # Import after adding to path
        from app.database import engine
        from app.models import Base
        
        print("✅ Successfully imported database modules")
        
        # Create all tables
        print("📊 Creating/updating database tables...")
        Base.metadata.create_all(bind=engine)
        
        print("✅ Database tables created successfully!")
        
        # Test database connection
        from app.database import get_db
        from app.models import User, CrowdReport, TriagePatient
        
        db = next(get_db())
        
        # Test basic queries
        user_count = db.query(User).count()
        report_count = db.query(CrowdReport).count()
        patient_count = db.query(TriagePatient).count()
        
        print(f"📋 Database Status:")
        print(f"   Users: {user_count}")
        print(f"   Reports: {report_count}")
        print(f"   Patients: {patient_count}")
        
        # Create default admin user if no users exist
        if user_count == 0:
            print("👤 Creating default admin user...")
            
            import hashlib
            hashed_password = hashlib.sha256("admin".encode()).hexdigest()
            
            admin_user = User(
                username="admin",
                email="admin@emergency.local",
                hashed_password=hashed_password,
                role="admin",
                is_active=True
            )
            
            db.add(admin_user)
            db.commit()
            
            print("✅ Default admin user created:")
            print("   Username: admin")
            print("   Password: admin")
        
        db.close()
        
        print("\\n" + "=" * 40)
        print("🎉 Migration completed successfully!")
        print("\\nNext steps:")
        print("1. Run: python api.py")
        print("2. Visit: http://localhost:8000")
        print("3. Login with admin/admin")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
        print("\\nThis might be normal if some tables already exist.")
        print("Try running your application anyway: python api.py")
        return False

if __name__ == "__main__":
    run_migration()
'''
    
    migration_file = Path("quick_migration.py")
    migration_file.write_text(migration_content)
    print(f"✅ Created {migration_file}")

def main():
    """Main function to run all fixes"""
    
    print("🔧 Emergency Response Assistant - Complete Fix Script")
    print("=" * 60)
    print("This script will fix all the undefined variable errors in your application.")
    print()
    
    try:
        # Step 1: Create fallback AI module
        print("Step 1: Creating fallback AI module...")
        create_fallback_ai()
        
        # Step 2: Update models.py
        print("\\nStep 2: Updating models.py with complete models...")
        backup_and_update_models()
        
        # Step 3: Create migration script
        print("\\nStep 3: Creating migration script...")
        create_migration_script()
        
        print("\\n" + "=" * 60)
        print("🎉 Fix script completed successfully!")
        print("=" * 60)
        
        print("\\n📋 What was fixed:")
        print("   ✅ Created app/fallback_ai.py with mock implementations")
        print("   ✅ Updated app/models.py with all missing models (User, EmergencyReport, etc.)")
        print("   ✅ Created quick_migration.py to update your database")
        
        print("\\n🚀 Next steps:")
        print("   1. Run the migration: python quick_migration.py")
        print("   2. Start your app: python api.py")
        print("   3. Visit: http://localhost:8000")
        print("   4. Login with: admin/admin")
        
        print("\\n💡 Notes:")
        print("   • Your original models.py was backed up as models.py.backup")
        print("   • The fallback AI provides simulated responses until real AI is set up")
        print("   • All undefined variable errors should now be resolved")
        
        print("\\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Error during fix: {e}")
        print("\\nYou may need to manually apply some fixes.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🎯 Ready to run: python quick_migration.py")
    else:
        print("\\n⚠️  Some fixes may need to be applied manually.")