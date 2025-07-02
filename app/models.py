from sqlalchemy import Column, Integer, String
from app.database import Base

class CrowdReport(Base):
    __tablename__ = "crowd_reports"

    id = Column(Integer, primary_key=True, index=True)
    message = Column(String)
    tone = Column(String)
    escalation = Column(String)
    user = Column(String)
    location = Column(String)
    timestamp = Column(String)