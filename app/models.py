from sqlalchemy import Column, Integer, String, DateTime, Float, Text
from datetime import datetime
from app.database import Base

# ==========================
# 📣 Crowd Report Model
# ==========================
class CrowdReport(Base):
    __tablename__ = "crowd_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    message = Column(String)
    tone = Column(String)
    escalation = Column(String)
    user = Column(String)
    location = Column(String)
    timestamp = Column(String)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

# ==========================
# 🏥 Triage Patient Model
# ==========================
class TriagePatient(Base):
    __tablename__ = "triage_patients"
    
    # Primary Key
    id = Column(Integer, primary_key=True, index=True)
    
    # Patient Information
    name = Column(String, nullable=False, index=True)  # Required field
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)  # male, female, other, unknown
    medical_id = Column(String, nullable=True)  # Medical ID/Bracelet
    
    # Medical Assessment
    injury_type = Column(String, nullable=False)  # Required field - Primary injury/condition
    mechanism = Column(String, nullable=True)  # fall, mva, assault, medical, burn, other
    consciousness = Column(String, nullable=False)  # Required - alert, verbal, pain, unresponsive
    breathing = Column(String, nullable=False)  # Required - normal, labored, shallow, absent
    
    # Vital Signs (Individual fields for better querying and analysis)
    heart_rate = Column(Integer, nullable=True)  # BPM
    bp_systolic = Column(Integer, nullable=True)  # mmHg
    bp_diastolic = Column(Integer, nullable=True)  # mmHg
    respiratory_rate = Column(Integer, nullable=True)  # breaths per minute
    temperature = Column(Float, nullable=True)  # degrees Fahrenheit
    oxygen_sat = Column(Integer, nullable=True)  # percentage
    
    # Assessment Results
    severity = Column(String, nullable=False)  # Required - mild, moderate, severe, critical
    triage_color = Column(String, nullable=False, index=True)  # Required - red, yellow, green, black
    
    # Additional Medical Information
    allergies = Column(Text, nullable=True)  # Known allergies
    medications = Column(Text, nullable=True)  # Current medications
    medical_history = Column(Text, nullable=True)  # Medical conditions, surgeries, etc.
    notes = Column(Text, nullable=True)  # Assessment notes and observations
    
    # System Fields
    status = Column(String, default="active", index=True)  # active, in_treatment, treated, discharged
    assigned_staff = Column(String, nullable=True)  # Staff member assigned to patient
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Legacy field (keeping for backward compatibility)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<TriagePatient(id={self.id}, name='{self.name}', triage_color='{self.triage_color}', severity='{self.severity}')>"
    
    @property
    def priority_score(self):
        """Calculate priority score for sorting (1=highest priority)"""
        color_priority = {"red": 1, "yellow": 2, "green": 3, "black": 4}
        return color_priority.get(self.triage_color, 5)
    
    @property
    def vital_signs_dict(self):
        """Return vital signs as a dictionary"""
        return {
            "heart_rate": self.heart_rate,
            "bp_systolic": self.bp_systolic,
            "bp_diastolic": self.bp_diastolic,
            "respiratory_rate": self.respiratory_rate,
            "temperature": self.temperature,
            "oxygen_sat": self.oxygen_sat
        }
    
    @property
    def is_critical_vitals(self):
        """Check if any vital signs are in critical range"""
        if self.heart_rate and (self.heart_rate < 50 or self.heart_rate > 120):
            return True
        if self.oxygen_sat and self.oxygen_sat < 90:
            return True
        if self.bp_systolic and (self.bp_systolic < 80 or self.bp_systolic > 180):
            return True
        if self.temperature and (self.temperature < 95 or self.temperature > 104):
            return True
        if self.respiratory_rate and (self.respiratory_rate < 10 or self.respiratory_rate > 30):
            return True
        return False