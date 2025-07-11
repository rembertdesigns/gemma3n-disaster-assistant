# app/models.py - Complete models with all missing classes

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
