from sqlalchemy import Column, Integer, String, DateTime, Float
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

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer)
    injury_type = Column(String)
    severity = Column(String)
    vitals = Column(String)  # Could be JSON in future upgrade
    notes = Column(String)
    triage_color = Column(String)  # Red, Yellow, Green, Black
    timestamp = Column(DateTime, default=datetime.utcnow)