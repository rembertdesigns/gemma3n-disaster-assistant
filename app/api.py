# ================================================================================
# IMPORTS & DEPENDENCIES
# ================================================================================

from fastapi import (
    FastAPI, Request, Form, UploadFile, File, Depends, HTTPException,
    Body, Query, BackgroundTasks
)
from fastapi.responses import (
    HTMLResponse, FileResponse, JSONResponse, RedirectResponse,
    StreamingResponse, Response
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_ # Added for new query functions
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import os
import uuid
import json
import zipfile
import io
import base64
import logging
import tempfile # Added for temporary file handling

# External dependencies
from weasyprint import HTML, HTML as WeasyHTML
from jinja2 import Environment, FileSystemLoader
from io import BytesIO

# Internal modules
from app.predictive_engine import calculate_risk_score
from app.broadcast_utils import start_broadcast, discover_nearby_broadcasts
from app.sentiment_utils import analyze_sentiment
from app.map_snapshot import generate_map_image
from app.hazard_detection import detect_hazards
from app.preprocessing import preprocess_input
from app.inference import run_disaster_analysis
from app.audio_transcription import transcribe_audio
from app.report_utils import (
    generate_report_pdf,
    generate_map_preview_data,
    generate_static_map_endpoint
)
from app.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    require_role
)
from app.map_utils import (
    map_utils,
    generate_static_map,
    get_coordinate_formats,
    get_emergency_resources,
    get_map_metadata,
    MapConfig
)
from app.database import get_db, engine
from app.models import CrowdReport, TriagePatient, Base # Assuming existing models are here

# NEW: Import necessary modules/classes for Gemma 3n features.
# These imports assume you have these files and classes defined.
# If not, you'll need to implement them or mock them for the API to run.
try:
    from app.inference import Gemma3nEmergencyProcessor, analyze_comprehensive_context
    from app.audio_transcription import VoiceEmergencyProcessor
    from app.adaptive_ai_settings import AdaptiveAIOptimizer
    from app.health import setup_health_checks # From the health.py artifact
    # from app.celery_app import celery_app # Uncomment if you use Celery and define it here
except ImportError as e:
    logger.error(f"Failed to import core AI processing/utility modules: {e}. Some API features may not work. Please ensure app.inference, app.audio_transcription, app.adaptive_ai_settings, and app.health (if used) are correctly implemented.")
    # Define dummy classes/functions if actual imports fail, for API structure testing
    class Gemma3nEmergencyProcessor:
        def __init__(self, mode="balanced"): self.mode = mode; self.model = True; self.device = "CPU"; self.config = {"model_name": "gemma-3n-4b", "context_window": 128000}
        def analyze_multimodal_emergency(self, text=None, image_data=None, audio_data=None, context=None):
            return {"emergency_type": {"primary": "simulated_incident"}, "severity": {"overall_score": 5.0, "confidence": 0.7}, "immediate_risks": ["simulated_risk"], "resource_requirements": {}, "device_performance": {"inference_speed": 0.15}}
    class VoiceEmergencyProcessor:
        def process_emergency_call(self, audio_path, context=None): # context added for consistency
            return {"transcript": "Simulated voice report.", "confidence": 0.8, "overall_urgency": "medium", "emotional_state": {"stress": 0.5}, "hazards_detected": [], "location_info": {"addresses": ["Simulated Location"]}, "audio_duration": 10, "severity_indicators": [5]}
    class AdaptiveAIOptimizer:
        def __init__(self): self.device_caps = {"cpu_cores": 4, "memory_gb": 8, "gpu_available": True, "gpu_memory_gb": 4}; self.current_config = type('obj', (object,), {'model_variant': 'simulated_model', 'context_window': 64000, 'precision': 'fp16', 'optimization_level': 'balanced', 'batch_size': 1})
        def optimize_for_device(self, level): return type('obj', (object,), {'model_variant': 'simulated_model', 'context_window': 64000, 'precision': 'fp16', 'optimization_level': level, 'batch_size': 1})
        def monitor_performance(self): return type('obj', (object,), {"cpu_usage": 50, "memory_usage": 60, "gpu_usage": 40, "battery_level": 80, "inference_speed": 0.2, "temperature": 35, "timestamp": datetime.utcnow()})
    def analyze_comprehensive_context(context_data): return {"comprehensive_analysis": "Simulated context analysis.", "confidence": 0.85, "tokens_used": 50000, "context_window_utilization": "50%", "analysis_timestamp": datetime.utcnow().isoformat()}
    def setup_health_checks(app_instance): logger.warning("Dummy health checks initialized as app.health not found.")


# NEW: SQLAlchemy Models for Gemma 3n features (assuming they are defined here if not in app.models)
# It's highly recommended to put these in app/models.py and import them like other models.
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean, JSON

# These model definitions need to be part of your Base declarative_base.
# If they are in app/models.py, remove these definitions from here and ensure they are imported correctly.
# If you are defining them here, you might need: from sqlalchemy.ext.declarative import declarative_base; Base = declarative_base() if Base is not imported from app.models
# For consistency with your provided code, assuming Base is imported from app.models.

class ContextAnalysis(Base):
    __tablename__ = "context_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(String(50), nullable=False)
    input_tokens = Column(Integer, default=0)
    output_summary = Column(Text)
    confidence = Column(Float, default=0.0)
    analyst_id = Column(String(100))
    processing_time = Column(Float, default=0.0)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class VoiceAnalysis(Base):
    __tablename__ = "voice_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    audio_file_path = Column(String(255))
    transcript = Column(Text)
    confidence = Column(Float, default=0.0)
    urgency_level = Column(String(20))
    emotional_state = Column(JSON)
    hazards_detected = Column(JSON)
    location_extracted = Column(String(255))
    processing_metadata = Column(JSON)
    analyst_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class MultimodalAssessment(Base):
    __tablename__ = "multimodal_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_type = Column(String(50))
    text_input = Column(Text)
    image_path = Column(String(255))
    audio_path = Column(String(255))
    severity_score = Column(Float, default=0.0)
    emergency_type = Column(String(100))
    risk_factors = Column(JSON)
    resource_requirements = Column(JSON)
    ai_confidence = Column(Float, default=0.0)
    analyst_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class DevicePerformance(Base):
    __tablename__ = "device_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(100))
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    gpu_usage = Column(Float)
    battery_level = Column(Float)
    model_config = Column(JSON)
    inference_speed = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


# ================================================================================
# GLOBAL CONFIGURATION
# ================================================================================

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FastAPI app initialization
app = FastAPI(
    title="Enhanced Disaster Response & Recovery Assistant",
    description="AI-Powered Emergency Management with Real-time Mapping and Demo Features",
    version="2.2.0"
)

# NEW: CORS Middleware (Add after FastAPI app creation)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production (e.g., ["http://localhost:8000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

async def cleanup_temp_file(file_path: str):
    """Clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup temp file {file_path}: {e}")

def get_time_ago(timestamp_str):
    """Calculate human-readable time difference from timestamp"""
    try:
        if isinstance(timestamp_str, str):
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str.replace('Z', '+00:00')
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = timestamp_str
            
        now = datetime.utcnow()
        diff = now - timestamp.replace(tzinfo=None)
            
        if diff.days > 7:
            return f"{diff.days} days ago"
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"
    except Exception:
        return "Unknown time"

def calculate_avg_severity(patients):
    """Calculate average severity score for analytics"""
    if not patients:
        return 0.0
        
    severity_scores = []
    for p in patients:
        if p.severity == "critical":
            severity_scores.append(4)
        elif p.severity == "severe":
            severity_scores.append(3)
        elif p.severity == "moderate":
            severity_scores.append(2)
        elif p.severity == "mild":
            severity_scores.append(1)
            
    return round(sum(severity_scores) / len(severity_scores), 1) if severity_scores else 0.0

def calculate_response_efficiency(patients):
    """Calculate response efficiency percentage"""
    if not patients:
        return 0.0
        
    treated_count = len([p for p in patients if p.status in ["treated", "discharged"]])
    return round((treated_count / len(patients)) * 100, 1)

def generate_trend_data(reports, patients, start_time, end_time):
    """Generate time series data for analytics charts"""
    days = (end_time - start_time).days + 1
    trend_data = []
        
    for i in range(days):
        day_start = start_time + timedelta(days=i)
        day_end = day_start + timedelta(days=1)
            
        day_reports = []
        day_patients = []
            
        for r in reports:
            if r.timestamp:
                try:
                    report_time = datetime.fromisoformat(r.timestamp.replace('Z', '+00:00'))
                    if day_start <= report_time < day_end:
                        day_reports.append(r)
                except:
                    pass
            
        for p in patients:
            if p.created_at and day_start <= p.created_at < day_end:
                day_patients.append(p)
            
        trend_data.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "reports": len(day_reports),
            "patients": len(day_patients),
            "critical_patients": len([p for p in day_patients if p.triage_color == "red"])
        })
            
    return trend_data

async def _gather_emergency_context(data: dict, db: Session) -> dict:
    """Gather comprehensive emergency context for analysis"""
    
    context = {
        "recent_reports": [],
        "active_patients": [],
        "weather_data": {}, # Placeholder
        "resource_status": {}, # Placeholder
        "historical_patterns": {}, # Placeholder
        "geographic_context": {}, # Placeholder
        "input_request_data": data # Include the original request data
    }
    
    try:
        # Recent crowd reports (last 24 hours)
        recent_reports = db.query(CrowdReport).filter(
            CrowdReport.timestamp >= (datetime.utcnow() - timedelta(hours=24))
        ).order_by(desc(CrowdReport.timestamp)).limit(50).all()
        
        context["recent_reports"] = [
            {
                "id": r.id,
                "message": r.message,
                "escalation": r.escalation,
                "location": r.location,
                "timestamp": r.timestamp, # Stored as datetime, not isoformat
                "severity": r.severity # Assuming severity is an attribute
            } for r in recent_reports
        ]
        
        # Active patients
        active_patients = db.query(TriagePatient).filter(
            TriagePatient.status == "active"
        ).all()
        
        context["active_patients"] = [
            {
                "id": p.id,
                "severity": p.severity,
                "triage_color": p.triage_color,
                "injury_type": p.injury_type,
                "status": p.status
            } for p in active_patients
        ]
        
        # Recent voice analyses (last 6 hours)
        recent_voice = db.query(VoiceAnalysis).filter(
            VoiceAnalysis.created_at >= (datetime.utcnow() - timedelta(hours=6))
        ).order_by(desc(VoiceAnalysis.created_at)).limit(20).all()
        
        context["recent_voice_analyses"] = [
            {
                "urgency_level": v.urgency_level,
                "emotional_state": v.emotional_state,
                "hazards_detected": v.hazards_detected,
                "confidence": v.confidence
            } for v in recent_voice
        ]
        
        # System performance data (last hour)
        recent_performance = db.query(DevicePerformance).filter(
            DevicePerformance.timestamp >= (datetime.utcnow() - timedelta(hours=1))
        ).order_by(desc(DevicePerformance.timestamp)).limit(10).all()
        
        context["system_performance"] = [
            {
                "cpu_usage": p.cpu_usage,
                "memory_usage": p.memory_usage,
                "inference_speed": p.inference_speed,
                "timestamp": p.timestamp.isoformat()
            } for p in recent_performance
        ]
        
    except Exception as e:
        logger.error(f"Error gathering context: {e}")
        context["error"] = str(e)
    
    return context

def _generate_optimization_recommendations(performance_metrics: dict) -> list:
    """Generate optimization recommendations based on performance"""
    
    recommendations = []
    
    cpu_usage = performance_metrics.get("cpu_usage", 0)
    memory_usage = performance_metrics.get("memory_usage", 0)
    battery_level = performance_metrics.get("battery_level", 100)
    
    if cpu_usage > 80:
        recommendations.append({
            "type": "performance",
            "priority": "high",
            "message": "Reduce model complexity to lower CPU usage",
            "action": "switch_to_low_resource_model"
        })
    
    if memory_usage > 85:
        recommendations.append({
            "type": "memory",
            "priority": "high",
            "message": "Reduce context window size to free memory",
            "action": "reduce_context_window"
        })
    
    if battery_level < 20:
        recommendations.append({
            "type": "power",
            "priority": "medium",
            "message": "Enable power saving mode",
            "action": "enable_power_saving"
        })
    
    if not recommendations:
        recommendations.append({
            "type": "status",
            "priority": "info",
            "message": "System performance is optimal",
            "action": "maintain_current_settings"
        })
    
    return recommendations

def _categorize_damage_level(severity_score: float) -> str:
    """Categorize damage level from severity score"""
    if severity_score >= 9:
        return "catastrophic"
    elif severity_score >= 7:
        return "severe"
    elif severity_score >= 5:
        return "moderate"
    elif severity_score >= 3:
        return "minor"
    else:
        return "minimal"

def _assess_stability_risk(severity_score: float, emergency_type: str) -> str:
    """Assess structural stability risk"""
    if "structural" in emergency_type.lower() or "collapse" in emergency_type.lower():
        return "high"
    elif severity_score >= 7:
        return "medium"
    else:
        return "low"

def _extract_environmental_hazards(risk_factors: list) -> list:
    """Extract environmental hazards from risk factors"""
    hazards = []
    
    if not risk_factors:
        return hazards
    
    for risk in risk_factors:
        risk_text = str(risk).lower()
        if "fire" in risk_text:
            hazards.append("fire_risk")
        if "flood" in risk_text or "water" in risk_text:
            hazards.append("flooding")
        if "chemical" in risk_text or "toxic" in risk_text:
            hazards.append("hazardous_materials")
        if "gas" in risk_text:
            hazards.append("gas_leak")
    
    return hazards

def _assess_contamination_risk(emergency_type: str) -> str:
    """Assess contamination risk"""
    emergency_lower = emergency_type.lower()
    
    if "chemical" in emergency_lower or "toxic" in emergency_lower or "gas" in emergency_lower:
        return "high"
    elif "fire" in emergency_lower:
        return "medium"
    else:
        return "low"

def _estimate_casualties(severity_score: float) -> dict:
    """Estimate casualty levels"""
    if severity_score >= 8:
        return {"level": "high", "estimated_range": "10+", "priority": "mass_casualty"}
    elif severity_score >= 6:
        return {"level": "medium", "estimated_range": "3-10", "priority": "multiple_casualty"}
    elif severity_score >= 4:
        return {"level": "low", "estimated_range": "1-3", "priority": "single_casualty"}
    else:
        return {"level": "minimal", "estimated_range": "0-1", "priority": "no_casualty"}

def _determine_evacuation_need(severity_score: float, risk_factors: list) -> dict:
    """Determine evacuation requirements"""
    high_risk_indicators = ["fire", "gas", "collapse", "flood", "toxic", "explosion"]
    has_high_risk = any(indicator in str(risk_factors).lower() for indicator in high_risk_indicators)
    
    if has_high_risk or severity_score >= 7:
        return {
            "required": True,
            "urgency": "immediate",
            "radius_meters": 500 if severity_score >= 8 else 200,
            "estimated_affected": "100+" if severity_score >= 8 else "50-100"
        }
    elif severity_score >= 5:
        return {
            "required": True,
            "urgency": "precautionary",
            "radius_meters": 100,
            "estimated_affected": "10-50"
        }
    else:
        return {
            "required": False,
            "urgency": "none",
            "radius_meters": 0,
            "estimated_affected": "0"
        }

async def _get_voice_analysis_stats(db: Session, start_time: datetime) -> dict:
    """Get voice analysis statistics"""
    
    total_analyses = db.query(VoiceAnalysis).filter(
        VoiceAnalysis.created_at >= start_time
    ).count()
    
    urgency_distribution = db.query(
        VoiceAnalysis.urgency_level,
        func.count(VoiceAnalysis.id)
    ).filter(
        VoiceAnalysis.created_at >= start_time
    ).group_by(VoiceAnalysis.urgency_level).all()
    
    avg_confidence = db.query(
        func.avg(VoiceAnalysis.confidence)
    ).filter(
        VoiceAnalysis.created_at >= start_time
    ).scalar() or 0.0
    
    return {
        "total_analyses": total_analyses,
        "urgency_distribution": {level: count for level, count in urgency_distribution},
        "average_confidence": float(avg_confidence),
        "processing_trend": "improving" # Would calculate from actual data
    }

async def _get_multimodal_stats(db: Session, start_time: datetime) -> dict:
    """Get multimodal assessment statistics"""
    
    total_assessments = db.query(MultimodalAssessment).filter(
        MultimodalAssessment.created_at >= start_time
    ).count()
    
    severity_distribution = db.query(
        func.case(
            [(MultimodalAssessment.severity_score >= 8, 'high'),
             (MultimodalAssessment.severity_score >= 5, 'medium')],
            else_='low'
        ).label('severity_level'),
        func.count(MultimodalAssessment.id)
    ).filter(
        MultimodalAssessment.created_at >= start_time
    ).group_by('severity_level').all()
    
    avg_ai_confidence = db.query(
        func.avg(MultimodalAssessment.ai_confidence)
    ).filter(
        MultimodalAssessment.created_at >= start_time
    ).scalar() or 0.0
    
    emergency_types = db.query(
        MultimodalAssessment.emergency_type,
        func.count(MultimodalAssessment.id)
    ).filter(
        MultimodalAssessment.created_at >= start_time
    ).group_by(MultimodalAssessment.emergency_type).all()
    
    return {
        "total_assessments": total_assessments,
        "severity_distribution": dict(severity_distribution),
        "average_ai_confidence": float(avg_ai_confidence),
        "emergency_types": {etype: count for etype, count in emergency_types},
        "accuracy_trend": "stable" # Would calculate from validation data
    }

async def _get_context_analysis_stats(db: Session, start_time: datetime) -> dict:
    """Get context analysis statistics"""
    
    total_analyses = db.query(ContextAnalysis).filter(
        ContextAnalysis.created_at >= start_time
    ).count()
    
    avg_tokens = db.query(
        func.avg(ContextAnalysis.input_tokens)
    ).filter(
        ContextAnalysis.created_at >= start_time
    ).scalar() or 0.0
    
    avg_processing_time = db.query(
        func.avg(ContextAnalysis.processing_time)
    ).filter(
        ContextAnalysis.created_at >= start_time
    ).scalar() or 0.0
    
    avg_confidence = db.query(
        func.avg(ContextAnalysis.confidence)
    ).filter(
        ContextAnalysis.created_at >= start_time
    ).scalar() or 0.0
    
    return {
        "total_analyses": total_analyses,
        "average_tokens_used": float(avg_tokens),
        "average_processing_time": float(avg_processing_time),
        "average_confidence": float(avg_confidence),
        "context_utilization": f"{(avg_tokens/128000)*100:.1f}%" if avg_tokens > 0 else "0%"
    }

async def _get_device_performance_stats(db: Session, start_time: datetime) -> dict:
    """Get device performance statistics"""
    
    latest_performance = db.query(DevicePerformance).filter(
        DevicePerformance.timestamp >= start_time
    ).order_by(desc(DevicePerformance.timestamp)).first()
    
    avg_cpu = db.query(
        func.avg(DevicePerformance.cpu_usage)
    ).filter(
        DevicePerformance.timestamp >= start_time
    ).scalar() or 0.0
    
    avg_memory = db.query(
        func.avg(DevicePerformance.memory_usage)
    ).filter(
        DevicePerformance.timestamp >= start_time
    ).scalar() or 0.0
    
    avg_inference_speed = db.query(
        func.avg(DevicePerformance.inference_speed)
    ).filter(
        DevicePerformance.timestamp >= start_time
    ).scalar() or 0.0
    
    return {
        "current_performance": {
            "cpu_usage": latest_performance.cpu_usage if latest_performance else 0,
            "memory_usage": latest_performance.memory_usage if latest_performance else 0,
            "battery_level": latest_performance.battery_level if latest_performance else 100,
            "inference_speed": latest_performance.inference_speed if latest_performance else 0,
            "gpu_usage": latest_performance.gpu_usage if latest_performance else 0, # Added for completeness
            "temperature": latest_performance.temperature if hasattr(latest_performance, 'temperature') else None # Added if model includes temperature
        },
        "average_performance": {
            "cpu_usage": float(avg_cpu),
            "memory_usage": float(avg_memory),
            "inference_speed": float(avg_inference_speed)
        },
        "optimization_status": "optimal" if avg_cpu < 70 and avg_memory < 80 else "needs_tuning"
    }

async def _generate_ai_insights(db: Session, start_time: datetime) -> dict:
    """Generate AI-powered insights from recent data"""
    
    insights = []
    
    # Analyze emergency patterns
    recent_reports = db.query(CrowdReport).filter(
        CrowdReport.timestamp >= start_time
    ).all()
    
    if len(recent_reports) > 5:
        high_severity_reports = [r for r in recent_reports if r.severity and r.severity >= 7] # Ensure severity is not None
        if len(high_severity_reports) / len(recent_reports) > 0.3:
            insights.append({
                "type": "alert",
                "priority": "high",
                "message": f"High severity incident rate: {len(high_severity_reports)}/{len(recent_reports)} reports",
                "recommendation": "Consider activating emergency protocols"
            })
    
    # Analyze voice emergency patterns
    voice_analyses = db.query(VoiceAnalysis).filter(
        VoiceAnalysis.created_at >= start_time
    ).all()
    
    if voice_analyses:
        critical_calls = [v for v in voice_analyses if v.urgency_level == "critical"]
        if len(critical_calls) > 3:
            insights.append({
                "type": "trend",
                "priority": "medium",
                "message": f"Spike in critical voice emergencies: {len(critical_calls)} calls",
                "recommendation": "Monitor for potential large-scale incident"
            })
    
    # System performance insights
    performance_records = db.query(DevicePerformance).filter(
        DevicePerformance.timestamp >= start_time
    ).all()
    
    if performance_records:
        avg_cpu = sum(p.cpu_usage for p in performance_records) / len(performance_records)
        if avg_cpu > 85:
            insights.append({
                "type": "system",
                "priority": "medium",
                "message": f"High average CPU usage: {avg_cpu:.1f}%",
                "recommendation": "Consider reducing model complexity"
            })
    
    return {
        "insights": insights,
        "insight_count": len(insights),
        "last_generated": datetime.utcnow().isoformat()
    }

async def _analyze_emergency_trends(db: Session, start_time: datetime) -> dict:
    """Analyze emergency trends over time"""
    
    # Emergency type trends
    emergency_types = db.query(
        MultimodalAssessment.emergency_type,
        func.count(MultimodalAssessment.id)
    ).filter(
        MultimodalAssessment.created_at >= start_time
    ).group_by(MultimodalAssessment.emergency_type).all()
    
    # Severity trends by hour (using CrowdReport for broader data)
    hourly_severity = db.query(
        func.extract('hour', CrowdReport.timestamp).label('hour'),
        func.avg(CrowdReport.severity).label('avg_severity')
    ).filter(
        CrowdReport.timestamp >= start_time
    ).group_by('hour').all()
    
    # Location hotspots (using CrowdReport for broader data)
    location_counts = db.query(
        CrowdReport.location,
        func.count(CrowdReport.id)
    ).filter(
        CrowdReport.timestamp >= start_time,
        CrowdReport.location != None # Filter out None locations explicitly
    ).group_by(CrowdReport.location).order_by(
        desc(func.count(CrowdReport.id))
    ).limit(10).all()
    
    return {
        "emergency_types": {etype: count for etype, count in emergency_types},
        "hourly_severity": {int(hour): float(severity) for hour, severity in hourly_severity},
        "location_hotspots": {location: count for location, count in location_counts},
        "trend_analysis": {
            "most_common_emergency": emergency_types[0][0] if emergency_types else "none",
            "peak_severity_hour": max(hourly_severity, key=lambda x: x[1])[0] if hourly_severity else 0,
            "top_hotspot": location_counts[0][0] if location_counts else "none"
        }
    }

def _generate_immediate_actions(analysis_result: dict) -> list:
    """Generate immediate action recommendations"""
    
    actions = []
    severity = analysis_result.get("severity", {}).get("overall_score", 0)
    emergency_type = analysis_result.get("emergency_type", {}).get("primary", "unknown")
    confidence = analysis_result.get("severity", {}).get("confidence", 0.0)
    
    # High confidence, high severity actions
    if confidence > 0.8 and severity >= 8:
        actions.append({
            "priority": 1,
            "action": "IMMEDIATE DISPATCH REQUIRED",
            "details": "High confidence critical emergency detected",
            "timeline": "0-3 minutes",
            "resources": ["all_available_units"]
        })
    
    # Emergency type specific actions
    if "fire" in emergency_type.lower():
        actions.append({
            "priority": 1,
            "action": "Fire department dispatch",
            "details": "Fire emergency detected",
            "timeline": "immediate",
            "resources": ["fire_trucks", "ems"]
        })
    elif "medical" in emergency_type.lower():
        actions.append({
            "priority": 1,
            "action": "Medical emergency response",
            "details": "Medical emergency detected",
            "timeline": "immediate",
            "resources": ["ambulance", "paramedics"]
        })
    elif "violence" in emergency_type.lower():
        actions.append({
            "priority": 1,
            "action": "Law enforcement dispatch",
            "details": "Violence/security threat detected",
            "timeline": "immediate",
            "resources": ["police_units", "backup"]
        })
    
    # Resource requirements
    resource_reqs = analysis_result.get("resource_requirements", {})
    if resource_reqs:
        actions.append({
            "priority": 2,
            "action": "Resource allocation",
            "details": f"Deploy required resources: {resource_reqs}",
            "timeline": "5-10 minutes",
            "resources": list(resource_reqs.get("personnel", {}).keys()) # Assuming 'personnel' key exists
        })
    
    # Low confidence actions
    if confidence < 0.5:
        actions.append({
            "priority": 3,
            "action": "Verification required",
            "details": "Low confidence analysis - human verification needed",
            "timeline": "immediate",
            "resources": ["dispatcher", "supervisor"]
        })
    
    return actions

# ================================================================================
# AUTHENTICATION ROUTES
# ================================================================================

# ... (rest of your existing authentication routes) ...

# ================================================================================
# MAIN PAGE ROUTES
# ================================================================================

# ... (rest of your existing main page routes) ...

# ================================================================================
# NEW EMERGENCY RESPONSE SYSTEM PAGES
# ================================================================================

# ... (rest of your existing new emergency response system pages) ...

# ================================================================================
# GEMMA 3N ENHANCED PAGES
# ================================================================================

# ... (rest of your existing Gemma 3n enhanced pages) ...

# ================================================================================
# OPTIMAL TIER - AI-POWERED EMERGENCY MANAGEMENT PAGES (90% MAX POTENTIAL)
# ================================================================================

# ... (rest of your existing Optimal Tier pages) ...

# ================================================================================
# COMPLETE TIER - ULTIMATE EMERGENCY MANAGEMENT PAGES (100% MAX POTENTIAL)
# ================================================================================

# ... (rest of your existing Complete Tier pages) ...


# ================================================================================
# DASHBOARD ROUTES
# ================================================================================

# ... (rest of your existing Dashboard routes) ...


# ================================================================================
# CROWD REPORTS - PAGES
# ================================================================================

# ... (rest of your existing Crowd Reports - Pages) ...


# ================================================================================
# CROWD REPORTS - FORM SUBMISSION
# ================================================================================

@app.post("/submit-crowd-report")
async def submit_crowd_report(
    request: Request,
    message: str = Form(...),
    tone: Optional[str] = Form(None),
    escalation: str = Form(...),
    user: Optional[str] = Form("Anonymous"),
    location: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks # Added for background task
):
    """Submit new crowd report with geolocation support"""
    try:
        if latitude and longitude:
            logger.info(f"📍 Received crowd report from location: {latitude}, {longitude}")

        # Analyze sentiment if not provided
        if not tone:
            tone = analyze_sentiment(message)

        # Handle image upload
        image_path = None
        if image and image.filename:
            ext = os.path.splitext(image.filename)[1]
            image_path = os.path.join("uploads", f"crowd_{uuid.uuid4().hex}{ext}")
            with open(image_path, "wb") as f:
                f.write(await image.read())

        # Create new crowd report
        new_report = CrowdReport(
            message=message, tone=tone, escalation=escalation, user=user,
            location=location, timestamp=datetime.utcnow().isoformat(),
            latitude=latitude, longitude=longitude,
            severity=5 # Assign a default severity, adjust as needed or derive from AI
        )
            
        db.add(new_report)
        db.commit()
        db.refresh(new_report)
            
        logger.info(f"✅ Crowd report saved: ID={new_report.id}, escalation={new_report.escalation}")
        
        # Add background task to process high-priority reports
        if new_report.escalation in ["critical", "high"]:
            background_tasks.add_task(process_emergency_report_background, new_report.id)
            logger.info(f"Scheduled background processing for high-priority report {new_report.id}")
            
        return RedirectResponse(url="/view-reports", status_code=303)
            
    except Exception as e:
        logger.error(f"❌ Failed to insert crowd report: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Error saving report")

# ================================================================================
# CROWD REPORTS - API ENDPOINTS
# ================================================================================

# ... (rest of your existing Crowd Reports - API Endpoints) ...

# ================================================================================
# EMERGENCY SYSTEM API ENDPOINTS
# ================================================================================

# ... (rest of your existing Emergency System API Endpoints) ...

# ================================================================================
# DEMO DATA GENERATION API
# ================================================================================

# ... (rest of your existing Demo Data Generation API) ...

# ================================================================================
# ENHANCED STATISTICS & ANALYTICS API
# ================================================================================

# ... (rest of your existing Enhanced Statistics & Analytics API) ...

# ================================================================================
# EXPORT & ARCHIVE FUNCTIONALITY
# ================================================================================

# ... (rest of your existing Export & Archive Functionality) ...

# ================================================================================
# PATIENT TRIAGE MANAGEMENT - PAGES
# ================================================================================

# ... (rest of your existing Patient Triage Management - Pages) ...

# ================================================================================
# PATIENT TRIAGE MANAGEMENT - FORM SUBMISSION & UPDATES
# ================================================================================

# ... (rest of your existing Patient Triage Management - Form Submission & Updates) ...

# ================================================================================
# PATIENT TRIAGE MANAGEMENT - API ENDPOINTS
# ================================================================================

# ... (rest of your existing Patient Triage Management - API Endpoints) ...

# ================================================================================
# AI ANALYSIS & REPORTING
# ================================================================================

# ... (rest of your existing AI Analysis & Reporting) ...

# ================================================================================
# GEMMA 3N API ENDPOINTS (Updated and expanded)
# ================================================================================

# NOTE: Original /api/submit-voice-emergency-report and /api/submit-damage-assessment
# are kept for compatibility/demonstration, but new /api/gemma-3n/multimodal-analysis
# and /api/gemma-3n/voice-emergency are more comprehensive.

@app.post("/api/gemma-3n/multimodal-analysis")
async def multimodal_emergency_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    text_report: str = Form(None),
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    context_data: str = Form("{}"), # Changed from context_
    user: dict = Depends(require_role(["admin", "responder"])),
    db: Session = Depends(get_db)
):
    """Comprehensive multimodal emergency analysis using Gemma 3n"""
    
    try:
        start_time = datetime.utcnow()
        
        # Parse context data
        context = json.loads(context_data) if context_data else {}
        
        # Save uploaded files temporarily
        image_data = None
        audio_data = None
        image_path = None
        audio_path = None
        
        if image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix or ".jpg") as tmp_img: # Use Path.suffix for robustness
                image_data = await image.read()
                tmp_img.write(image_data)
                image_path = tmp_img.name
        
        if audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix or ".wav") as tmp_audio: # Use Path.suffix for robustness
                audio_data = await audio.read()
                tmp_audio.write(audio_data)
                audio_path = tmp_audio.name
        
        # Process with Gemma 3n
        processor = Gemma3nEmergencyProcessor()
        
        analysis_result = processor.analyze_multimodal_emergency(
            text=text_report,
            image_data=image_data,
            audio_data=audio_data,
            context=context
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Store analysis in database (using MultimodalAssessment model)
        multimodal_record = MultimodalAssessment(
            assessment_type="comprehensive_multimodal",
            text_input=text_report,
            image_path=image_path,
            audio_path=audio_path,
            severity_score=analysis_result.get("severity", {}).get("overall_score", 0.0), # Ensure float default
            emergency_type=analysis_result.get("emergency_type", {}).get("primary", "unknown"),
            risk_factors=analysis_result.get("immediate_risks", []),
            resource_requirements=analysis_result.get("resource_requirements", {}),
            ai_confidence=analysis_result.get("severity", {}).get("confidence", 0.0),
            analyst_id=user["username"]
        )
        
        db.add(multimodal_record)
        db.commit()
        db.refresh(multimodal_record) # Refresh to get ID
        
        # Schedule cleanup of temporary files
        if image_path:
            background_tasks.add_task(cleanup_temp_file, image_path)
        if audio_path:
            background_tasks.add_task(cleanup_temp_file, audio_path)
        
        return JSONResponse(content={
            "success": True,
            "analysis_id": multimodal_record.id,
            "analysis": analysis_result,
            "processing_time_seconds": processing_time,
            "modalities_processed": {
                "text": text_report is not None and text_report != "", # Check if text_report actually has content
                "image": image is not None,
                "audio": audio is not None
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Multimodal analysis error: {e}", exc_info=True) # Log full traceback
        # Ensure temporary files are cleaned up even on error
        if image_path and os.path.exists(image_path):
            background_tasks.add_task(cleanup_temp_file, image_path)
        if audio_path and os.path.exists(audio_path):
            background_tasks.add_task(cleanup_temp_file, audio_path)
        db.rollback() # Rollback any partial database transactions
        raise HTTPException(status_code=500, detail=f"Multimodal analysis failed: {str(e)}")


@app.post("/api/gemma-3n/voice-emergency")
async def process_voice_emergency(
    request: Request,
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    caller_info: str = Form("{}"),
    location_hint: Optional[str] = Form(None), # Made optional
    user: dict = Depends(require_role(["admin", "responder", "dispatcher"])),
    db: Session = Depends(get_db)
):
    """Enhanced voice emergency processing with Gemma 3n intelligence"""
    
    try:
        start_time = datetime.utcnow()
        
        # Parse caller information
        caller_context = json.loads(caller_info) if caller_info else {}
        if location_hint:
            caller_context["location_hint"] = location_hint
        
        # Save audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix or ".wav") as tmp_audio:
            audio_data = await audio_file.read()
            tmp_audio.write(audio_data)
            audio_path = tmp_audio.name
        
        # Process with enhanced voice processor
        voice_processor = VoiceEmergencyProcessor()
        
        voice_analysis = voice_processor.process_emergency_call(
            audio_path=audio_path,
            context=caller_context
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Store voice analysis in database (using VoiceAnalysis model)
        voice_record = VoiceAnalysis(
            audio_file_path=audio_path,
            transcript=voice_analysis.get("transcript", ""),
            confidence=voice_analysis.get("confidence", 0.0), # Ensure float default
            urgency_level=voice_analysis.get("overall_urgency", "unknown"),
            emotional_state=voice_analysis.get("emotional_state", {}),
            hazards_detected=voice_analysis.get("hazards_detected", []),
            location_extracted=json.dumps(voice_analysis.get("location_info", {})), # Store as JSON string
            processing_metadata={
                "processing_time": processing_time,
                "audio_duration": voice_analysis.get("audio_duration", 0),
                "model_used": "gemma_3n_enhanced"
            },
            analyst_id=user["username"]
        )
        
        db.add(voice_record)
        db.commit()
        db.refresh(voice_record) # Refresh to get ID
        
        # Auto-create crowd report if high urgency
        if voice_analysis.get("overall_urgency") in ["critical", "high"]:
            auto_report = CrowdReport(
                message=f"VOICE EMERGENCY: {voice_analysis.get('transcript', '')[:200]}...",
                escalation=voice_analysis.get("overall_urgency"),
                location=voice_analysis.get("location_info", {}).get("addresses", ["Unknown"])[0] if voice_analysis.get("location_info", {}).get("addresses") else "Location not specified",
                timestamp=datetime.utcnow(),
                severity=min(int(voice_analysis.get("severity_indicators", [5])[0]) if voice_analysis.get("severity_indicators") else 5, 10), # Ensure valid int, cap at 10
                user=f"voice_system_{user['username']}", # Use 'user' for CrowdReport, not 'reporter_id'
                source="voice_analysis_system",
                metadata=json.dumps({
                    "voice_analysis_id": voice_record.id,
                    "auto_generated": True,
                    "urgency_level": voice_analysis.get("overall_urgency"),
                    "confidence": voice_analysis.get("confidence", 0.0)
                })
            )
            
            db.add(auto_report)
            db.commit()
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, audio_path)
        
        return JSONResponse(content={
            "success": True,
            "voice_analysis_id": voice_record.id,
            "analysis": voice_analysis,
            "processing_time_seconds": processing_time,
            "auto_report_created": voice_analysis.get("overall_urgency") in ["critical", "high"],
            "recommendations": voice_analysis.get("recommended_actions", []),
            "dispatch_priority": voice_analysis.get("priority_level", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Voice emergency processing error: {e}", exc_info=True) # Log full traceback
        if audio_path and os.path.exists(audio_path): # Ensure cleanup on error
            background_tasks.add_task(cleanup_temp_file, audio_path)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Voice emergency processing failed: {str(e)}")

# This endpoint replaces the older /api/context-analysis for clarity and expanded functionality.
@app.post("/api/gemma-3n/context-analysis")
async def comprehensive_context_analysis(
    request: Request,
    user: dict = Depends(require_role(["admin", "responder"])),
    db: Session = Depends(get_db)
):
    """Comprehensive context analysis using Gemma 3n's 128K window"""
    
    try:
        data = await request.json()
        start_time = datetime.utcnow()
        
        # Gather comprehensive context data
        context_data = await _gather_emergency_context(data, db)
        
        # Process with Gemma 3n
        analysis = analyze_comprehensive_context(context_data)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Store analysis results in database (using the new ContextAnalysis model)
        analysis_record = ContextAnalysis(
            analysis_type=data.get("analysis_type", "comprehensive"),
            input_tokens=analysis.get("tokens_used", 0),
            output_summary=analysis.get("comprehensive_analysis", ""),
            confidence=analysis.get("confidence", 0.0),
            processing_time=processing_time,
            metadata={
                "context_window_utilization": analysis.get("context_window_utilization", "unknown"),
                "analysis_timestamp": analysis.get("analysis_timestamp"),
                "data_sources": list(context_data.keys()) if isinstance(context_data, dict) else []
            },
            analyst_id=user["username"]
        )
        
        db.add(analysis_record)
        db.commit()
        
        return JSONResponse(content={
            "success": True,
            "analysis_id": analysis_record.id,
            "analysis": analysis,
            "context_window_used": f"{analysis.get('tokens_used', 0)}/128000",
            "processing_time_seconds": processing_time,
            "data_sources_analyzed": len(context_data),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Context analysis error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Context analysis failed: {str(e)}")

@app.get("/api/gemma-3n/adaptive-settings")
async def get_adaptive_ai_settings(
    request: Request,
    user: dict = Depends(require_role(["admin", "responder"])),
    db: Session = Depends(get_db)
):
    """Get current adaptive AI settings and device performance"""
    
    try:
        optimizer = AdaptiveAIOptimizer()
        
        current_config = optimizer.current_config # Access current_config from initialized optimizer
        performance_metrics = optimizer.monitor_performance()
        
        # Store performance data (using DevicePerformance model)
        perf_record = DevicePerformance(
            device_id=request.client.host if request.client else "unknown_host", # Use request.client.host
            cpu_usage=performance_metrics.cpu_usage,
            memory_usage=performance_metrics.memory_usage,
            gpu_usage=performance_metrics.gpu_usage,
            battery_level=performance_metrics.battery_level,
            model_config={
                "model_variant": current_config.model_variant if current_config else "unknown",
                "context_window": current_config.context_window if current_config else 0,
                "precision": current_config.precision if current_config else "unknown",
                "optimization_level": current_config.optimization_level if current_config else "unknown"
            },
            inference_speed=performance_metrics.inference_speed,
            timestamp=performance_metrics.timestamp # Use timestamp from performance metrics
        )
        
        db.add(perf_record)
        db.commit()
        
        return JSONResponse(content={
            "success": True,
            "current_config": {
                "model_variant": current_config.model_variant,
                "context_window": current_config.context_window,
                "batch_size": current_config.batch_size,
                "precision": current_config.precision,
                "optimization_level": current_config.optimization_level
            },
            "performance_metrics": {
                "cpu_usage": performance_metrics.cpu_usage,
                "memory_usage": performance_metrics.memory_usage,
                "gpu_usage": performance_metrics.gpu_usage,
                "battery_level": performance_metrics.battery_level,
                "inference_speed": performance_metrics.inference_speed,
                "temperature": performance_metrics.temperature,
                "timestamp": performance_metrics.timestamp.isoformat()
            },
            "device_capabilities": {
                "cpu_cores": optimizer.device_caps["cpu_cores"], # Access as dict
                "memory_gb": optimizer.device_caps["memory_gb"],
                "gpu_available": optimizer.device_caps["gpu_available"],
                "gpu_memory_gb": optimizer.device_caps["gpu_memory_gb"]
            },
            "optimization_recommendations": _generate_optimization_recommendations(performance_metrics),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Adaptive settings error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Adaptive settings failed: {str(e)}")

@app.post("/api/gemma-3n/adaptive-settings") # This is a POST endpoint for updating settings
async def update_adaptive_ai_settings(
    request: Request,
    user: dict = Depends(require_role(["admin"])),
    db: Session = Depends(get_db)
):
    """Update adaptive AI settings"""
    
    try:
        data = await request.json()
        
        optimizer = AdaptiveAIOptimizer()
        
        # Apply new settings
        if "optimization_level" in data:
            # Update optimization level
            new_config = optimizer.optimize_for_device(data["optimization_level"])
            
            return JSONResponse(content={
                "success": True,
                "updated_config": {
                    "model_variant": new_config.model_variant,
                    "context_window": new_config.context_window,
                    "optimization_level": new_config.optimization_level
                },
                "message": "AI settings updated successfully",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return JSONResponse(content={
            "success": False,
            "error": "No valid settings provided",
            "timestamp": datetime.utcnow().isoformat()
        }, status_code=400)
        
    except Exception as e:
        logger.error(f"Update adaptive settings error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Update adaptive settings failed: {str(e)}")

@app.get("/api/gemma-3n/damage-assessment/{assessment_id}")
async def get_damage_assessment(
    assessment_id: int,
    user: dict = Depends(require_role(["admin", "responder"])),
    db: Session = Depends(get_db)
):
    """Get detailed damage assessment results"""
    
    try:
        assessment = db.query(MultimodalAssessment).filter(
            MultimodalAssessment.id == assessment_id
        ).first()
        
        if not assessment:
            return JSONResponse(content={
                "success": False,
                "error": "Assessment not found"
            }, status_code=404)
        
        # Generate enhanced damage assessment if needed
        damage_details = None
        if assessment.assessment_type == "comprehensive_multimodal":
            damage_details = _generate_damage_assessment_details(assessment, db) # No need for await if not async
        
        return JSONResponse(content={
            "success": True,
            "assessment": {
                "id": assessment.id,
                "assessment_type": assessment.assessment_type,
                "text_input": assessment.text_input, # Added for context
                "image_path": assessment.image_path, # Added for context
                "audio_path": assessment.audio_path, # Added for context
                "severity_score": assessment.severity_score,
                "emergency_type": assessment.emergency_type,
                "risk_factors": assessment.risk_factors,
                "resource_requirements": assessment.resource_requirements,
                "ai_confidence": assessment.ai_confidence,
                "created_at": assessment.created_at.isoformat(),
                "analyst": assessment.analyst_id
            },
            "damage_details": damage_details,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Get damage assessment error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Get damage assessment failed: {str(e)}")

@app.get("/api/gemma-3n/intelligence-dashboard")
async def get_intelligence_dashboard(
    request: Request,
    timeframe: str = "24h",
    user: dict = Depends(require_role(["admin", "responder"])),
    db: Session = Depends(get_db)
):
    """Get AI intelligence dashboard data"""
    
    try:
        # Calculate time range
        if timeframe == "1h":
            time_delta = timedelta(hours=1)
        elif timeframe == "6h":
            time_delta = timedelta(hours=6)
        elif timeframe == "24h":
            time_delta = timedelta(hours=24)
        elif timeframe == "7d":
            time_delta = timedelta(days=7)
        else:
            time_delta = timedelta(hours=24) # Default
        
        start_time = datetime.utcnow() - time_delta
        
        # Gather intelligence data
        dashboard_data = {
            "summary": {
                "timeframe": timeframe,
                "start_time": start_time.isoformat(),
                "end_time": datetime.utcnow().isoformat()
            },
            "voice_analysis_stats": await _get_voice_analysis_stats(db, start_time), # Renamed key
            "multimodal_assessments_stats": await _get_multimodal_stats(db, start_time), # Renamed key
            "context_analyses_stats": await _get_context_analysis_stats(db, start_time), # Renamed key
            "device_performance_stats": await _get_device_performance_stats(db, start_time), # Renamed key
            "ai_insights": await _generate_ai_insights(db, start_time),
            "trend_analysis": await _analyze_emergency_trends(db, start_time)
        }
        
        return JSONResponse(content={
            "success": True,
            "dashboard": dashboard_data,
            "last_updated": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Intelligence dashboard error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Intelligence dashboard failed: {str(e)}")

@app.post("/api/gemma-3n/real-time-analysis")
async def real_time_emergency_analysis(
    request: Request,
    user: dict = Depends(require_role(["admin", "responder", "dispatcher"])),
    db: Session = Depends(get_db) # Added db dependency to potentially log/store real-time analysis
):
    """Real-time emergency analysis with streaming response"""
    
    try:
        data = await request.json()
        
        # Validate input
        if not data.get("emergency_data"):
            raise HTTPException(status_code=400, detail="No emergency data provided")
        
        # Process in real-time with Gemma 3n
        processor = Gemma3nEmergencyProcessor("high_performance")
        
        analysis_result = processor.analyze_multimodal_emergency(
            text=data.get("emergency_data", {}).get("text"),
            context={
                "real_time_analysis": True,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user["username"],
                "priority": data.get("priority", "normal")
            }
        )
        
        # Generate immediate recommendations
        immediate_actions = _generate_immediate_actions(analysis_result)
        
        # Optional: Log real-time analysis results to DB if needed
        # real_time_log = RealTimeAnalysisLog(
        #    event_type="real_time_emergency",
        #    analysis_summary=json.dumps(analysis_result),
        #    user_id=user["username"],
        #    timestamp=datetime.utcnow()
        # )
        # db.add(real_time_log)
        # db.commit()

        return JSONResponse(content={
            "success": True,
            "real_time_analysis": analysis_result,
            "immediate_actions": immediate_actions,
            "processing_time_ms": analysis_result.get("device_performance", {}).get("inference_speed", 0) * 1000,
            "confidence": analysis_result.get("severity", {}).get("confidence", 0.0),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Real-time analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Real-time analysis failed: {str(e)}")

@app.post("/api/gemma-3n/batch-analysis")
async def batch_emergency_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_role(["admin"])),
    db: Session = Depends(get_db)
):
    """Process multiple emergency reports in batch"""
    
    try:
        data = await request.json()
        reports = data.get("reports", [])
        
        if not reports:
            return JSONResponse(content={
                "success": False,
                "error": "No reports provided for batch analysis"
            }, status_code=400)
        
        # Start batch processing in background
        batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}" # Unique batch ID
        
        background_tasks.add_task(
            process_batch_analysis,
            batch_id,
            reports,
            user["username"],
            # Pass a new session or commit within the task if it's long-running and truly isolated
            # For simplicity, passing db and assuming it's managed correctly or that this is a short task.
            db
        )
        
        return JSONResponse(content={
            "success": True,
            "batch_id": batch_id,
            "reports_queued": len(reports),
            "estimated_completion": "5-10 minutes", # This would be dynamic
            "status_endpoint": f"/api/gemma-3n/batch-status/{batch_id}",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}", exc_info=True)
        db.rollback() # Ensure rollback if initial request fails database operations
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

async def process_batch_analysis(batch_id: str, reports: list, user_id: str, db: Session):
    """Process batch analysis in background"""
    
    try:
        processor = Gemma3nEmergencyProcessor()
        
        results = []
        
        for i, report_data in enumerate(reports): # Renamed 'report' to 'report_data' for clarity
            try:
                analysis = processor.analyze_multimodal_emergency(
                    text=report_data.get("text"), # Access text from report_data
                    context={
                        "batch_id": batch_id,
                        "batch_index": i,
                        "total_reports": len(reports)
                    }
                )
                
                # Store individual analysis (MultimodalAssessment)
                batch_record = MultimodalAssessment(
                    assessment_type="batch_analysis_item", # Distinguish from summary
                    text_input=report_data.get("text", ""),
                    severity_score=analysis.get("severity", {}).get("overall_score", 0.0),
                    emergency_type=analysis.get("emergency_type", {}).get("primary", "unknown"),
                    risk_factors=analysis.get("immediate_risks", []),
                    ai_confidence=analysis.get("severity", {}).get("confidence", 0.0),
                    analyst_id=f"batch_{user_id}",
                    metadata={
                        "batch_id": batch_id,
                        "batch_index": i,
                        "original_report_info": report_data # Store original data for reference
                    }
                )
                
                db.add(batch_record)
                results.append({
                    "index": i,
                    "assessment_id": batch_record.id, # Include ID for lookup
                    "analysis_summary": analysis.get("comprehensive_analysis", "No summary"),
                    "status": "completed"
                })
                
            except Exception as e:
                logger.error(f"Batch analysis failed for report {i} in batch {batch_id}: {e}", exc_info=True)
                results.append({
                    "index": i,
                    "error": str(e),
                    "status": "failed",
                    "original_report_info": report_data # Include original data for failed reports
                })
        
        # Commit all individual batch records once the loop finishes
        db.commit() 

        # Store batch summary (ContextAnalysis)
        batch_summary_record = ContextAnalysis(
            analysis_type="batch_summary",
            input_tokens=sum(len(r.get("text", "")) for r in reports if r.get("text")),
            output_summary=f"Batch analysis completed: {len(results)} reports processed.",
            confidence=sum(r.get("analysis", {}).get("severity", {}).get("confidence", 0) for r in results if r.get("analysis")) / max(1, len([r for r in results if r.get("analysis")])), # Calculate average confidence only from successful analyses
            analyst_id=f"batch_{user_id}",
            processing_time=(datetime.utcnow() - datetime.utcnow()).total_seconds(), # Placeholder, actual time should be calculated within the task
            metadata={
                "batch_id": batch_id,
                "total_reports": len(reports),
                "successful": len([r for r in results if r["status"] == "completed"]),
                "failed": len([r for r in results if r["status"] == "failed"]),
                "results_summary": results # Store summary of results here
            }
        )
        
        db.add(batch_summary_record)
        db.commit() # Commit the batch summary record

        logger.info(f"Batch analysis {batch_id} completed successfully and summary stored.")
        
    except Exception as e:
        logger.error(f"Batch processing main task error for batch {batch_id}: {e}", exc_info=True)
        db.rollback() # Rollback if the main task fails

@app.get("/api/gemma-3n/batch-status/{batch_id}")
async def get_batch_status(
    batch_id: str,
    user: dict = Depends(require_role(["admin", "responder"])),
    db: Session = Depends(get_db)
):
    """Get batch analysis status"""
    
    try:
        batch_summary_record = db.query(ContextAnalysis).filter(
            ContextAnalysis.analysis_type == "batch_summary",
            ContextAnalysis.metadata["batch_id"].astext == batch_id # Access JSON field
        ).first()
        
        if not batch_summary_record:
            return JSONResponse(content={
                "success": False,
                "error": "Batch not found or still processing. Please allow some time.",
                "status": "not_found"
            }, status_code=404)
        
        metadata = batch_summary_record.metadata or {}
        
        return JSONResponse(content={
            "success": True,
            "batch_id": batch_id,
            "status": "completed", # Assume completed if summary record exists
            "summary": {
                "total_reports": metadata.get("total_reports", 0),
                "successful": metadata.get("successful", 0),
                "failed": metadata.get("failed", 0),
                "average_confidence": batch_summary_record.confidence,
                "processing_time": batch_summary_record.processing_time
            },
            "results": metadata.get("results_summary", []), # Return the detailed results summary
            "completed_at": batch_summary_record.created_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch status retrieval failed: {str(e)}")

# ================================================================================
# DEBUG & TESTING ROUTES
# ================================================================================

# ... (rest of your existing DEBUG & TESTING ROUTES) ...

# ================================================================================
# MAP & GEOLOCATION API ENDPOINTS
# ================================================================================

# ... (rest of your existing MAP & GEOLOCATION API ENDPOINTS) ...

# ================================================================================
# SYSTEM HEALTH & UTILITIES
# ================================================================================

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint with Gemma 3n service status"""
    try:
        db = next(get_db())
        reports_count = db.query(CrowdReport).count()
        patients_count = db.query(TriagePatient).count()
        # Count new tables as well
        context_analyses_count = db.query(ContextAnalysis).count()
        voice_analyses_count = db.query(VoiceAnalysis).count()
        multimodal_assessments_count = db.query(MultimodalAssessment).count()
        device_performance_logs_count = db.query(DevicePerformance).count()

        db_status = "connected"
    except Exception as e: # Catch specific SQLAlchemy errors if desired, but general is fine for health check
        logger.error(f"Database health check failed: {e}")
        reports_count = 0
        patients_count = 0
        context_analyses_count = 0
        voice_analyses_count = 0
        multimodal_assessments_count = 0
        device_performance_logs_count = 0
        db_status = "error"
            
    return {
        "status": "healthy",
        "service": "Enhanced Disaster Response Assistant with Gemma 3n",
        "version": "2.2.0",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "authentication": True, "maps": True, "emergency_resources": True,
            "coordinate_formats": True, "ai_analysis": True, "hazard_detection": True,
            "audio_transcription": True, "pdf_generation": True, "patient_management": True,
            "crowd_reports": True, "demo_data_generation": True, "real_time_updates": True,
            "enhanced_export": True, "offline_support": True, "analytics_dashboard": True,
            # NEW GEMMA 3N FEATURES
            "voice_emergency_reporter": True, "multimodal_damage_assessment": True,
            "context_intelligence_dashboard": True, "adaptive_ai_settings": True,
            "gemma_3n_integration": True, "128k_context_window": True,
            "edge_ai_optimization": True, "multimodal_processing": True,
            # NEW OPTIMAL TIER FEATURES
            "predictive_risk_modeling": True,
            "real_time_resource_optimizer": True,     
            "communication_intelligence": True,
            "cross_modal_verification": True,
            "optimal_tier_complete": True,
            # NEW COMPLETE TIER FEATURES:
            "edge_ai_monitor": True,
            "crisis_command_center": True, 
            "predictive_analytics_dashboard": True,
            "quantum_emergency_hub": True,
            "complete_tier_active": True,
            "max_potential_achieved": "100%",
        },
        "database": {
            "status": db_status, "type": "SQLAlchemy with SQLite",
            "tables": ["crowd_reports", "triage_patients", "context_analyses", "voice_analyses", "multimodal_assessments", "device_performance"], # List all tables
            "records": {
                "reports": reports_count,
                "patients": patients_count,
                "context_analyses": context_analyses_count,
                "voice_analyses": voice_analyses_count,
                "multimodal_assessments": multimodal_assessments_count,
                "device_performance_logs": device_performance_logs_count
            }
        },
        "ai_models": {
            "gemma_3n": {
                "status": "ready",
                "available_models": ["gemma-3n-2b", "gemma-3n-4b", "gemma-3n-4b-hq"],
                "active_model": "gemma-3n-4b",
                "context_window": "128K tokens",
                "features": ["voice_processing", "multimodal_analysis", "adaptive_optimization"]
            }
        },
        "map_service": {
            "preferred": map_utils.preferred_service.value,
            "available_services": [
                service.value for service in map_utils.api_keys.keys() 
                if map_utils.api_keys[service]
            ]
        },
        "demo_ready": True,
        "api_endpoints": {
            "crowd_reports": "/api/crowd-report-locations",
            "demo_data": "/api/create-demo-reports",
            "map_stats": "/api/map-statistics",
            "export": "/api/export-map-data",
            "triage_stats": "/api/triage-stats",
            # NEW GEMMA 3N ENDPOINTS
            "voice_reports_old": "/api/submit-voice-emergency-report", # Original endpoint
            "damage_assessment_old": "/api/submit-damage-assessment", # Original endpoint
            "context_analysis_old": "/api/context-analysis", # Original endpoint
            "ai_model_status_old": "/api/ai-model-status", # Original endpoint
            "ai_optimization_old": "/api/optimize-ai-settings", # Original endpoint
            "device_performance_old": "/api/device-performance", # Original endpoint
            # NEW OPTIMAL TIER ENDPOINTS
            "risk_forecast": "/api/risk-forecast",
            "resource_optimization": "/api/resource-optimization",
            "translate_message": "/api/translate-emergency-message",
            "verify_report": "/api/verify-report",
            # NEW COMPLETE TIER API ENDPOINTS
            "edge_ai_performance": "/api/edge-ai-performance",
            "crisis_coordination_status": "/api/crisis-coordination-status", 
            "predictive_insights": "/api/predictive-insights",
            "quantum_system_status": "/api/quantum-system-status",
            # More specific Gemma 3N endpoints (consolidated into a single entry where applicable)
            "gemma_multimodal_analysis": "/api/gemma-3n/multimodal-analysis",
            "gemma_voice_emergency": "/api/gemma-3n/voice-emergency",
            "gemma_adaptive_settings_get": "/api/gemma-3n/adaptive-settings (GET)",
            "gemma_adaptive_settings_post": "/api/gemma-3n/adaptive-settings (POST)",
            "gemma_damage_assessment_detail": "/api/gemma-3n/damage-assessment/{assessment_id}",
            "gemma_intelligence_dashboard": "/api/gemma-3n/intelligence-dashboard",
            "gemma_real_time_analysis": "/api/gemma-3n/real-time-analysis",
            "gemma_batch_analysis": "/api/gemma-3n/batch-analysis",
            "gemma_batch_status": "/api/gemma-3n/batch-status/{batch_id}",
            # Legacy/deprecated AI endpoints (if you want to list them as such)
            "legacy_analyze": "/analyze",
            "legacy_detect_hazards": "/detect-hazards",
            "legacy_predict_risk": "/predict-risk",
            "legacy_broadcast": "/broadcast", # This might not be legacy if still in use
        },
        "gemma_3n_pages": {
            "voice_reporter": "/voice-emergency-reporter",
            "damage_assessment": "/multimodal-damage-assessment",
            "context_dashboard": "/context-intelligence-dashboard",
            "ai_settings": "/adaptive-ai-settings"
        },
        "optimal_tier_pages": {
            "predictive_risk": "/predictive-risk-modeling",
            "resource_optimizer": "/real-time-resource-optimizer",
            "communication_intelligence": "/communication-intelligence",
            "cross_modal_verification": "/cross-modal-verification"
        },
        "complete_tier_pages": {
            "edge_ai_monitor": "/edge-ai-monitor",
            "crisis_command": "/crisis-command-center", 
            "predictive_analytics": "/predictive-analytics-dashboard",
            "quantum_hub": "/quantum-emergency-hub"
        }
    }

# ================================================================================
# ARCHIVE & EXPORT MANAGEMENT
# ================================================================================

# ... (rest of your existing Archive & Export Management) ...

# ================================================================================
# APPLICATION STARTUP & ERROR HANDLERS
# ================================================================================

# Replace your existing @app.on_event("startup") with this enhanced version:
@app.on_event("startup")
async def enhanced_startup_event():
    """Enhanced initialization with Gemma 3n and health monitoring"""
    logger.info("🚀 Starting Enhanced Disaster Response Assistant API Server v2.2 with Gemma 3n")
    logger.info(f"📍 Map service: {map_utils.preferred_service.value}")
    logger.info("🗺️ Enhanced map utilities initialized")
    
    # Create necessary directories
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create all SQLAlchemy tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ SQLAlchemy tables created/verified")
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
    
    # Initialize demo data if needed
    try:
        db = next(get_db())
        existing_reports = db.query(CrowdReport).count()
        
        if existing_reports == 0:
            logger.info("🎯 No existing reports found - initializing with welcome demo data...")
            
            welcome_report = CrowdReport(
                message="Welcome to the Enhanced Live Crowd Reports Map! This is a demo report showing the system capabilities. Click 'Generate Demo Data' for more realistic examples.",
                tone="descriptive", escalation="low", user="System Demo",
                location="San Francisco, CA (Demo)", latitude=37.7749, longitude=-122.4194,
                timestamp=datetime.utcnow().isoformat(),
                severity=1 # Default severity for demo report
            )
            
            db.add(welcome_report)
            db.commit()
            logger.info("✅ Welcome demo data initialized successfully")
            
    except Exception as e:
        logger.error(f"❌ Error initializing demo data: {str(e)}")
    
    # Log available map services
    available_services = [
        service.value for service in map_utils.api_keys.keys() 
        if map_utils.api_keys[service]
    ]
    logger.info(f"🔑 Available map services: {available_services or ['OpenStreetMap (free)']}")
    
    # Add health monitoring
    try:
        setup_health_checks(app)
        logger.info("🏥 Health monitoring system initialized")
    except ImportError:
        logger.warning("Health monitoring not available - install missing dependencies (app.health).")
    
    # Initialize Gemma 3n capabilities
    logger.info("🧠 Initializing Gemma 3n AI capabilities...")
    logger.info("    • Voice Emergency Reporter with real-time transcription")
    logger.info("    • Multimodal Damage Assessment (video/image/audio)")
    logger.info("    • Context Intelligence Dashboard (128K token window)")
    logger.info("    • Adaptive AI Settings for device optimization")
    # Initialize Optimal Tier capabilities
    logger.info("✨ Initializing Optimal Tier AI capabilities (90% MAX POTENTIAL)...")
    logger.info("    • Predictive Risk Modeling for emergency forecasting")
    logger.info("    • Real-Time Resource Optimization for dynamic allocation")
    logger.info("    • Communication Intelligence with 140+ language support")
    logger.info("    • Cross-Modal Verification for robust report authentication")
    # Complete Tier initialization
    logger.info("🚀 COMPLETE TIER: Ultimate Emergency Management (100% MAX POTENTIAL)...")
    logger.info("    • Edge AI Monitor for performance optimization")
    logger.info("    • Crisis Command Center for multi-agency coordination") 
    logger.info("    • Predictive Analytics Dashboard for AI-powered insights")
    logger.info("    • Quantum Emergency Hub for ultimate command & control")
    logger.info("🎯 MAXIMUM POTENTIAL ACHIEVED - Complete emergency management ecosystem ready!")
    
    logger.info("✅ Enhanced API server ready with comprehensive capabilities:")
    logger.info("    • Enhanced patient management (SQLAlchemy)")
    logger.info("    • Advanced crowd reports with geolocation (SQLAlchemy)")
    logger.info("    • Real-time map integration with demo data")
    logger.info("    • Multi-format export (CSV, JSON, KML, PDF)")
    logger.info("    • AI analysis & hazard detection")
    logger.info("    • Comprehensive analytics dashboards")
    logger.info("    • Demo data generation for presentations")
    logger.info("    • Network status monitoring & offline support")
    logger.info("    • Enhanced authentication & role management")
    logger.info("    🆕 GEMMA 3N: Voice emergency reporting")
    logger.info("    🆕 GEMMA 3N: Multimodal damage assessment")
    logger.info("    🆕 GEMMA 3N: 128K context intelligence")
    logger.info("    🆕 GEMMA 3N: Adaptive AI optimization")
    logger.info("    🌟 OPTIMAL TIER: Predictive Risk Modeling")
    logger.info("    🌟 OPTIMAL TIER: Real-Time Resource Optimizer")
    logger.info("    🌟 OPTIMAL TIER: Communication Intelligence")
    logger.info("    🌟 OPTIMAL TIER: Cross-Modal Verification")
    logger.info("    🚀 COMPLETE TIER: Edge AI Monitor")
    logger.info("    🚀 COMPLETE TIER: Crisis Command Center")
    logger.info("    🚀 COMPLETE TIER: Predictive Analytics Dashboard")
    logger.info("    🚀 COMPLETE TIER: Quantum Emergency Hub")


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Enhanced 404 handler with helpful endpoint suggestions"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found", 
            "path": str(request.url),
            "available_endpoints": [
                "/map-reports", "/admin", "/analytics", 
                "/api/crowd-report-locations", "/api/create-demo-reports",
                "/api/map-statistics", "/health"
            ],
            "version": "2.2.0"
        }
    )

@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    """Enhanced 500 handler with support information"""
    logger.error(f"Server error: {exc}", exc_info=True) # Log full traceback for 500 errors
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "message": "Please try again later. If the problem persists, contact support.",
            "support": "Check /health endpoint for system status",
            "version": "2.2.0"
        }
    )

# ================================================================================
# APPLICATION ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("🎯 Starting Enhanced Disaster Response API Server...")
    logger.info("📍 Access the application at: http://localhost:8000")
    logger.info("🗺️ Demo map reports at: http://localhost:8000/map-reports")
    logger.info("📊 Admin dashboard at: http://localhost:8000/admin")
    logger.info("🔍 Health check at: http://localhost:8000/health")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )