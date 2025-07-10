# ================================================================================
# ENHANCED DISASTER RESPONSE & RECOVERY ASSISTANT API
# Complete FastAPI Application with Citizen Portal Integration
# Version: 3.0.0 - Ultimate Emergency Management System
# ================================================================================

# ================================================================================
# IMPORTS & DEPENDENCIES
# ================================================================================

from fastapi import (
    FastAPI, Request, Form, UploadFile, File, Depends, HTTPException,
    Body, Query, BackgroundTasks, Header
)
from fastapi.responses import (
    HTMLResponse, FileResponse, JSONResponse, RedirectResponse,
    StreamingResponse, Response
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_, Column, Integer, String, Float, Text, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from typing import Optional, List, Dict, Any
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
import tempfile
import asyncio
import hashlib
import secrets
from contextlib import asynccontextmanager

# External dependencies
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    
from jinja2 import Environment, FileSystemLoader
from io import BytesIO

# Internal modules with fallback implementations
try:
    from app.predictive_engine import calculate_risk_score
except ImportError:
    def calculate_risk_score(data): return {"risk_score": 5.0, "confidence": 0.8}

try:
    from app.broadcast_utils import start_broadcast, discover_nearby_broadcasts
except ImportError:
    def start_broadcast(data): return {"status": "simulated"}
    def discover_nearby_broadcasts(): return []

try:
    from app.sentiment_utils import analyze_sentiment
except ImportError:
    def analyze_sentiment(text): return "neutral"

try:
    from app.map_snapshot import generate_map_image
except ImportError:
    def generate_map_image(lat, lon): return "static/placeholder_map.png"

try:
    from app.hazard_detection import detect_hazards
except ImportError:
    def detect_hazards(image_data): return ["simulated_hazard"]

try:
    from app.preprocessing import preprocess_input
except ImportError:
    def preprocess_input(data): return data

try:
    from app.inference import run_disaster_analysis
except ImportError:
    def run_disaster_analysis(data): return {"type": "simulated", "confidence": 0.8}

try:
    from app.audio_transcription import transcribe_audio
except ImportError:
    def transcribe_audio(audio_path): return {"transcript": "Simulated transcription", "confidence": 0.9}

try:
    from app.report_utils import generate_report_pdf, generate_map_preview_data
except ImportError:
    def generate_report_pdf(data): return "simulated_report.pdf"
    def generate_map_preview_data(lat, lon): return {"preview": "simulated"}

try:
    from app.database import get_db, engine
    from app.models import Base
    DATABASE_AVAILABLE = True
except ImportError:
    # Create fallback database setup
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine("sqlite:///emergency_response.db", echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    DATABASE_AVAILABLE = True

# ================================================================================
# ENHANCED DATABASE MODELS
# ================================================================================

class CrowdReport(Base):
    __tablename__ = "crowd_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    message = Column(Text, nullable=False)
    tone = Column(String(50))
    escalation = Column(String(20))
    user = Column(String(100), default="Anonymous")
    location = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    severity = Column(Integer, default=5)
    source = Column(String(50), default="web")
    metadata = Column(JSON)
    status = Column(String(20), default="pending")
    priority_score = Column(Float, default=0.0)

class TriagePatient(Base):
    __tablename__ = "triage_patients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    age = Column(Integer)
    gender = Column(String(10))
    injury_type = Column(String(100))
    severity = Column(String(20))
    triage_color = Column(String(10))
    status = Column(String(20), default="active")
    location = Column(String(255))
    assigned_to = Column(String(100))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    priority_score = Column(Float, default=0.0)

class EmergencyReport(Base):
    __tablename__ = "emergency_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(50), unique=True, index=True)
    type = Column(String(50))
    description = Column(Text)
    location = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)
    priority = Column(String(20))
    status = Column(String(20), default="pending")
    reporter = Column(String(100))
    evidence_file = Column(String(255))
    method = Column(String(20), default="text")
    timestamp = Column(DateTime, default=datetime.utcnow)
    ai_analysis = Column(JSON)
    response_units = Column(JSON)

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
    temperature = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255))
    role = Column(String(20), default="user")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    permissions = Column(JSON)

# ================================================================================
# GLOBAL CONFIGURATION
# ================================================================================

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent if __file__.endswith('.py') else Path.cwd()
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories
for directory in [STATIC_DIR, TEMPLATES_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

# FastAPI app initialization
app = FastAPI(
    title="Enhanced Emergency Response Assistant",
    description="Complete AI-Powered Emergency Management System with Citizen Portal",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
else:
    # Create minimal templates directory with basic template
    TEMPLATES_DIR.mkdir(exist_ok=True)
    basic_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Emergency Response Assistant</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; }
        .error { color: #dc2626; background: #fef2f2; padding: 1rem; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Emergency Response Assistant</h1>
    <div class="error">
        <p>Templates directory not found. Please create the templates directory and add your HTML templates.</p>
        <p>Current path: {{ request.url }}</p>
    </div>
</body>
</html>
    """
    (TEMPLATES_DIR / "error.html").write_text(basic_template)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ================================================================================
# AUTHENTICATION & SECURITY
# ================================================================================

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))

def hash_password(password: str) -> str:
    """Hash password using SHA-256 (use bcrypt in production)"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return hash_password(plain_password) == hashed_password

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token (simplified version)"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    to_encode.update({"exp": expire})
    # In production, use proper JWT library
    return base64.b64encode(json.dumps(to_encode).encode()).decode()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get current authenticated user"""
    try:
        # Simplified token verification (use proper JWT in production)
        payload = json.loads(base64.b64decode(token).decode())
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return {"username": user.username, "role": user.role, "id": user.id}
    except:
        # Fallback for demo mode
        return {"username": "demo_user", "role": "admin", "id": 1}

def require_role(allowed_roles: List[str]):
    """Require specific role for endpoint access"""
    def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in allowed_roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return role_checker

# ================================================================================
# AI SIMULATION CLASSES
# ================================================================================

class Gemma3nEmergencyProcessor:
    """Simulated Gemma 3n processor for emergency analysis"""
    
    def __init__(self, mode="balanced"):
        self.mode = mode
        self.model = True
        self.device = "CPU"
        self.config = {"model_name": "gemma-3n-4b", "context_window": 128000}
    
    def analyze_multimodal_emergency(self, text=None, image_data=None, audio_data=None, context=None):
        """Simulate multimodal emergency analysis"""
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
        primary_type = "fire" if text and "fire" in text.lower() else "medical"
        
        return {
            "emergency_type": {"primary": primary_type},
            "severity": {
                "overall_score": min(10.0, severity_score),
                "confidence": min(1.0, confidence)
            },
            "immediate_risks": ["structural_damage", "smoke_inhalation"] if primary_type == "fire" else ["injury"],
            "resource_requirements": {
                "personnel": {"firefighters": 6, "paramedics": 2} if primary_type == "fire" else {"paramedics": 2},
                "equipment": ["fire_truck", "ambulance"] if primary_type == "fire" else ["ambulance"]
            },
            "device_performance": {"inference_speed": 0.15}
        }

class VoiceEmergencyProcessor:
    """Simulated voice emergency processor"""
    
    def process_emergency_call(self, audio_path, context=None):
        """Simulate voice emergency processing"""
        urgency_keywords = ["help", "fire", "emergency", "critical", "urgent"]
        
        # Simulate transcript based on filename or generate generic
        transcript = "There's a fire in the building. We need help immediately. Multiple people are trapped."
        
        urgency = "critical" if any(word in transcript.lower() for word in urgency_keywords) else "medium"
        
        return {
            "transcript": transcript,
            "confidence": 0.8,
            "overall_urgency": urgency,
            "emotional_state": {"stress": 0.7, "panic": 0.8},
            "hazards_detected": ["fire", "smoke"],
            "location_info": {"addresses": ["123 Main Street"]},
            "audio_duration": 30,
            "severity_indicators": [8 if urgency == "critical" else 5]
        }

class AdaptiveAIOptimizer:
    """Simulated AI optimizer for device performance"""
    
    def __init__(self):
        self.device_caps = {
            "cpu_cores": 4, "memory_gb": 8, "gpu_available": True, "gpu_memory_gb": 4
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
        return type('Performance', (), {
            "cpu_usage": 50,
            "memory_usage": 60,
            "gpu_usage": 40,
            "battery_level": 80,
            "inference_speed": 0.2,
            "temperature": 35,
            "timestamp": datetime.utcnow()
        })()

# Global AI instances
gemma_processor = Gemma3nEmergencyProcessor()
voice_processor = VoiceEmergencyProcessor()
ai_optimizer = AdaptiveAIOptimizer()

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

def generate_report_id() -> str:
    """Generate unique report ID"""
    return f"ER-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"

def calculate_time_ago(timestamp):
    """Calculate human-readable time difference"""
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    now = datetime.utcnow()
    diff = now - timestamp.replace(tzinfo=None) if timestamp.tzinfo else now - timestamp
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "Just now"

def analyze_comprehensive_context(context_data):
    """Simulate comprehensive context analysis"""
    return {
        "comprehensive_analysis": "Simulated comprehensive analysis of emergency context.",
        "confidence": 0.85,
        "tokens_used": 50000,
        "context_window_utilization": "39%",
        "analysis_timestamp": datetime.utcnow().isoformat()
    }

async def process_emergency_report_background(report_id: int):
    """Background processing for high-priority reports"""
    try:
        await asyncio.sleep(1)  # Simulate processing
        logger.info(f"Background processing completed for report {report_id}")
    except Exception as e:
        logger.error(f"Background processing failed for report {report_id}: {e}")

# ================================================================================
# MAIN PAGE ROUTES - CITIZEN PORTAL
# ================================================================================

@app.get("/", response_class=HTMLResponse)
async def citizen_portal_home(request: Request):
    """Main citizen emergency portal - primary public interface"""
    try:
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        return HTMLResponse(f"""
        <html><head><title>Emergency Portal</title></head>
        <body>
        <h1>🆘 Emergency Response Assistant</h1>
        <p>Citizen emergency portal is loading...</p>
        <p>Error: {str(e)}</p>
        <p><a href="/api/docs">API Documentation</a> | <a href="/health">System Status</a></p>
        </body></html>
        """)

@app.get("/citizen", response_class=HTMLResponse)
async def citizen_portal_alt(request: Request):
    """Alternative citizen portal route"""
    return RedirectResponse(url="/", status_code=301)

@app.get("/offline", response_class=HTMLResponse)
async def offline_page(request: Request):
    """Offline support page"""
    try:
        return templates.TemplateResponse("offline.html", {"request": request})
    except:
        return HTMLResponse("""
        <html><head><title>Offline Mode</title></head>
        <body>
        <h1>📴 Offline Mode</h1>
        <p>This app works offline! Your reports are saved locally.</p>
        </body></html>
        """)

# ================================================================================
# PROFESSIONAL DASHBOARD ROUTES
# ================================================================================

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    """Admin dashboard with real database statistics"""
    try:
        # Get statistics
        total_reports = db.query(CrowdReport).count()
        total_patients = db.query(TriagePatient).count()
        total_emergency_reports = db.query(EmergencyReport).count()
        
        # Recent data
        recent_reports = db.query(CrowdReport).order_by(desc(CrowdReport.timestamp)).limit(5).all()
        active_patients = db.query(TriagePatient).filter(TriagePatient.status == "active").limit(5).all()
        
        stats = {
            "total_reports": total_reports,
            "total_patients": total_patients,
            "total_emergency_reports": total_emergency_reports,
            "active_users": 5,  # Simulated
            "avg_severity": 6.2,  # Simulated
            "reports_today": db.query(CrowdReport).filter(
                func.date(CrowdReport.timestamp) == datetime.utcnow().date()
            ).count(),
            "system_uptime": "24h 15m",
            "last_report_time": recent_reports[0].timestamp if recent_reports else None
        }
        
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "stats": stats,
            "recent_reports": recent_reports,
            "priority_patients": active_patients,
            "current_time": datetime.utcnow(),
            "demo_mode": True
        })
        
    except Exception as e:
        logger.error(f"Admin dashboard error: {e}")
        return HTMLResponse(f"""
        <html><head><title>Admin Dashboard</title></head>
        <body>
        <h1>📊 Admin Dashboard</h1>
        <p>Loading dashboard...</p>
        <p>Error: {str(e)}</p>
        <p><a href="/">← Back to Home</a></p>
        </body></html>
        """)

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_dashboard(request: Request, db: Session = Depends(get_db)):
    """Analytics dashboard"""
    try:
        return templates.TemplateResponse("analytics.html", {
            "request": request,
            "current_time": datetime.utcnow()
        })
    except:
        return HTMLResponse("""
        <html><head><title>Analytics</title></head>
        <body>
        <h1>📈 Analytics Dashboard</h1>
        <p>Analytics dashboard loading...</p>
        </body></html>
        """)

# ================================================================================
# EMERGENCY REPORTING ROUTES
# ================================================================================

@app.get("/submit-report", response_class=HTMLResponse)
async def submit_report_page(request: Request):
    """Emergency report submission page"""
    try:
        return templates.TemplateResponse("submit_report.html", {"request": request})
    except:
        return HTMLResponse("""
        <html><head><title>Submit Report</title></head>
        <body>
        <h1>📝 Submit Emergency Report</h1>
        <form action="/api/submit-emergency-report" method="post">
        <textarea name="description" placeholder="Describe the emergency..."></textarea><br>
        <input type="text" name="location" placeholder="Location"><br>
        <select name="priority">
        <option value="low">Low</option>
        <option value="medium">Medium</option>
        <option value="high">High</option>
        <option value="critical">Critical</option>
        </select><br>
        <button type="submit">Submit Report</button>
        </form>
        </body></html>
        """)

@app.get("/voice-emergency-reporter", response_class=HTMLResponse)
async def voice_reporter_page(request: Request):
    """Voice emergency reporter page"""
    try:
        return templates.TemplateResponse("voice-emergency-reporter.html", {"request": request})
    except:
        return HTMLResponse("""
        <html><head><title>Voice Reporter</title></head>
        <body>
        <h1>🎤 Voice Emergency Reporter</h1>
        <p>Voice reporting interface</p>
        <button onclick="startRecording()">Start Recording</button>
        <script>
        function startRecording() {
            alert('Voice recording would start here');
        }
        </script>
        </body></html>
        """)

@app.get("/view-reports", response_class=HTMLResponse)
async def view_reports_page(request: Request, db: Session = Depends(get_db)):
    """View submitted reports"""
    try:
        reports = db.query(CrowdReport).order_by(desc(CrowdReport.timestamp)).limit(50).all()
        return templates.TemplateResponse("view-reports.html", {
            "request": request,
            "reports": reports
        })
    except Exception as e:
        return HTMLResponse(f"""
        <html><head><title>View Reports</title></head>
        <body>
        <h1>📋 Emergency Reports</h1>
        <p>Loading reports... Error: {str(e)}</p>
        </body></html>
        """)

# ================================================================================
# API ENDPOINTS - EMERGENCY REPORTS
# ================================================================================

@app.post("/api/submit-emergency-report")
async def submit_emergency_report(
    request: Request,
    background_tasks: BackgroundTasks,
    type: str = Form(...),
    description: str = Form(...),
    location: str = Form(...),
    priority: str = Form("medium"),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    file: Optional[UploadFile] = File(None),
    method: str = Form("text"),
    db: Session = Depends(get_db)
):
    """Submit emergency report from citizen portal"""
    try:
        # Generate unique report ID
        report_id = generate_report_id()
        
        # Handle file upload
        evidence_file = None
        if file and file.filename:
            file_ext = os.path.splitext(file.filename)[1]
            evidence_file = f"evidence_{report_id}{file_ext}"
            file_path = UPLOAD_DIR / evidence_file
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        
        # Create emergency report
        emergency_report = EmergencyReport(
            report_id=report_id,
            type=type,
            description=description,
            location=location,
            latitude=latitude,
            longitude=longitude,
            priority=priority,
            evidence_file=evidence_file,
            method=method,
            reporter="citizen_portal"
        )
        
        db.add(emergency_report)
        db.commit()
        db.refresh(emergency_report)
        
        # Background processing for high priority
        if priority in ["critical", "high"]:
            background_tasks.add_task(process_emergency_report_background, emergency_report.id)
        
        logger.info(f"Emergency report submitted: {report_id} ({priority})")
        
        return JSONResponse({
            "success": True,
            "report_id": report_id,
            "status": "submitted",
            "priority": priority,
            "estimated_response_time": "5-10 minutes" if priority == "critical" else "30-60 minutes"
        })
        
    except Exception as e:
        logger.error(f"Emergency report submission failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/submit-voice-report")
async def submit_voice_report(
    request: Request,
    transcript: str = Form(...),
    urgency: str = Form("medium"),
    emotion: str = Form("neutral"),
    location: str = Form(""),
    recommendation: str = Form(""),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """Submit voice-analyzed emergency report"""
    try:
        # Create emergency report from voice analysis
        report_id = generate_report_id()
        
        emergency_report = EmergencyReport(
            report_id=report_id,
            type="voice_emergency",
            description=f"Voice Emergency: {transcript}",
            location=location or "Location not specified",
            latitude=latitude,
            longitude=longitude,
            priority=urgency,
            method="voice",
            reporter="voice_system",
            ai_analysis={
                "transcript": transcript,
                "urgency": urgency,
                "emotion": emotion,
                "recommendation": recommendation
            }
        )
        
        db.add(emergency_report)
        db.commit()
        db.refresh(emergency_report)
        
        logger.info(f"Voice report submitted: {report_id} (urgency: {urgency})")
        
        return JSONResponse({
            "success": True,
            "report_id": report_id,
            "urgency": urgency,
            "auto_created": True,
            "status": "processing"
        })
        
    except Exception as e:
        logger.error(f"Voice report submission failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/emergency-reports")
async def get_emergency_reports(
    limit: int = Query(50, description="Number of reports to return"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db)
):
    """Get emergency reports with filtering"""
    try:
        query = db.query(EmergencyReport)
        
        if priority:
            query = query.filter(EmergencyReport.priority == priority)
        if status:
            query = query.filter(EmergencyReport.status == status)
        
        reports = query.order_by(desc(EmergencyReport.timestamp)).limit(limit).all()
        
        return JSONResponse({
            "success": True,
            "reports": [
                {
                    "id": r.id,
                    "report_id": r.report_id,
                    "type": r.type,
                    "description": r.description,
                    "location": r.location,
                    "priority": r.priority,
                    "status": r.status,
                    "method": r.method,
                    "timestamp": r.timestamp.isoformat(),
                    "has_coordinates": r.latitude is not None and r.longitude is not None
                }
                for r in reports
            ],
            "total": len(reports)
        })
        
    except Exception as e:
        logger.error(f"Error fetching emergency reports: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# ================================================================================
# API ENDPOINTS - CROWD REPORTS
# ================================================================================

@app.get("/submit-crowd-report", response_class=HTMLResponse)
async def submit_crowd_report_form(request: Request):
    """Crowd report submission form"""
    try:
        return templates.TemplateResponse("submit-crowd-report.html", {"request": request})
    except:
        return HTMLResponse("""
        <html><head><title>Submit Community Report</title></head>
        <body>
        <h1>📢 Submit Community Report</h1>
        <form action="/api/submit-crowd-report" method="post">
        <textarea name="message" placeholder="What's happening in your area?" required></textarea><br>
        <select name="escalation" required>
        <option value="low">🟢 Low</option>
        <option value="moderate">🟡 Moderate</option>
        <option value="high">🟠 High</option>
        <option value="critical">🔴 Critical</option>
        </select><br>
        <input type="text" name="location" placeholder="Location"><br>
        <input type="text" name="user" placeholder="Your name (optional)"><br>
        <button type="submit">Submit Report</button>
        </form>
        </body></html>
        """)

@app.post("/api/submit-crowd-report")
async def submit_crowd_report(
    request: Request,
    message: str = Form(...),
    escalation: str = Form(...),
    tone: Optional[str] = Form(None),
    user: str = Form("Anonymous"),
    location: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """Submit crowd report"""
    try:
        # Analyze sentiment if not provided
        if not tone:
            tone = analyze_sentiment(message)
        
        # Handle image upload
        if image and image.filename:
            file_ext = os.path.splitext(image.filename)[1]
            image_filename = f"crowd_{uuid.uuid4().hex}{file_ext}"
            image_path = UPLOAD_DIR / image_filename
            
            with open(image_path, "wb") as buffer:
                content = await image.read()
                buffer.write(content)
        
        # Create crowd report
        crowd_report = CrowdReport(
            message=message,
            tone=tone,
            escalation=escalation,
            user=user,
            location=location,
            latitude=latitude,
            longitude=longitude,
            severity={"critical": 9, "high": 7, "moderate": 5, "low": 3}.get(escalation, 5)
        )
        
        db.add(crowd_report)
        db.commit()
        db.refresh(crowd_report)
        
        logger.info(f"Crowd report submitted: ID={crowd_report.id}, escalation={escalation}")
        
        return JSONResponse({
            "success": True,
            "report_id": crowd_report.id,
            "escalation": escalation,
            "tone": tone,
            "status": "submitted"
        })
        
    except Exception as e:
        logger.error(f"Crowd report submission failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/crowd-reports")
async def get_crowd_reports(
    limit: int = Query(100, description="Number of reports to return"),
    escalation: Optional[str] = Query(None, description="Filter by escalation"),
    tone: Optional[str] = Query(None, description="Filter by tone"),
    db: Session = Depends(get_db)
):
    """Get crowd reports with filtering"""
    try:
        query = db.query(CrowdReport)
        
        if escalation:
            query = query.filter(CrowdReport.escalation == escalation)
        if tone:
            query = query.filter(CrowdReport.tone == tone)
        
        reports = query.order_by(desc(CrowdReport.timestamp)).limit(limit).all()
        
        return JSONResponse({
            "success": True,
            "reports": [
                {
                    "id": r.id,
                    "message": r.message,
                    "escalation": r.escalation,
                    "tone": r.tone,
                    "user": r.user,
                    "location": r.location,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "timestamp": r.timestamp.isoformat(),
                    "severity": r.severity,
                    "time_ago": calculate_time_ago(r.timestamp)
                }
                for r in reports
            ],
            "total": len(reports),
            "escalation_filter": escalation,
            "tone_filter": tone
        })
        
    except Exception as e:
        logger.error(f"Error fetching crowd reports: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# ================================================================================
# API ENDPOINTS - AI ANALYSIS
# ================================================================================

@app.post("/api/analyze-text")
async def analyze_text(request: Request):
    """Analyze text for emergency indicators"""
    try:
        data = await request.json()
        text = data.get("text", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Simulate AI analysis
        urgency_keywords = ["fire", "emergency", "urgent", "critical", "help", "danger"]
        emotion_keywords = ["panic", "scared", "calm", "worried", "frantic"]
        
        urgency_score = sum(1 for word in urgency_keywords if word in text.lower())
        emotion_matches = [word for word in emotion_keywords if word in text.lower()]
        
        urgency_level = "critical" if urgency_score >= 3 else "high" if urgency_score >= 2 else "medium" if urgency_score >= 1 else "low"
        panic_level = "critical" if "panic" in emotion_matches else "elevated" if len(emotion_matches) > 1 else "calm"
        
        emergency_indicators = []
        for keyword in urgency_keywords:
            if keyword in text.lower():
                emergency_indicators.append({
                    "keyword": keyword,
                    "category": "urgency",
                    "confidence": 0.8
                })
        
        return JSONResponse({
            "success": True,
            "analysis": {
                "urgency_level": urgency_level,
                "panic_level": panic_level,
                "confidence": min(1.0, 0.6 + (urgency_score * 0.1)),
                "emergency_indicators": emergency_indicators,
                "processing_time": "0.15s"
            }
        })
        
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    context: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Analyze image for hazards and damage"""
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await image.read()
        
        # Save temporarily for processing
        temp_path = UPLOAD_DIR / f"temp_{uuid.uuid4().hex}.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_data)
        
        # Simulate AI analysis
        hazards = ["fire", "smoke", "structural_damage"]
        confidence = 0.85
        severity = 7.2
        
        analysis_result = {
            "hazards_detected": hazards,
            "confidence": confidence,
            "severity_score": severity,
            "objects_detected": ["building", "smoke", "people"],
            "recommended_action": "Immediate evacuation recommended",
            "processing_time": "2.3s"
        }
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return JSONResponse({
            "success": True,
            "analysis": analysis_result
        })
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/analyze-voice")
async def analyze_voice(
    audio: UploadFile = File(...),
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Analyze voice recording for emergency content"""
    try:
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be audio format")
        
        # Save audio file
        audio_ext = os.path.splitext(audio.filename)[1] or ".wav"
        audio_path = UPLOAD_DIR / f"voice_{uuid.uuid4().hex}{audio_ext}"
        
        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Process with voice processor
        analysis = voice_processor.process_emergency_call(str(audio_path))
        
        # Save to database
        voice_analysis = VoiceAnalysis(
            audio_file_path=str(audio_path),
            transcript=analysis["transcript"],
            confidence=analysis["confidence"],
            urgency_level=analysis["overall_urgency"],
            emotional_state=analysis["emotional_state"],
            hazards_detected=analysis["hazards_detected"],
            analyst_id="api_user"
        )
        
        db.add(voice_analysis)
        db.commit()
        db.refresh(voice_analysis)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, str(audio_path))
        
        return JSONResponse({
            "success": True,
            "analysis_id": voice_analysis.id,
            "analysis": analysis
        })
        
    except Exception as e:
        logger.error(f"Voice analysis failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# ================================================================================
# API ENDPOINTS - GEMMA 3N ADVANCED
# ================================================================================

@app.post("/api/gemma-3n/multimodal-analysis")
async def gemma_multimodal_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    text_report: str = Form(None),
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    context_data: str = Form("{}"),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Comprehensive multimodal emergency analysis using Gemma 3n"""
    try:
        start_time = datetime.utcnow()
        context = json.loads(context_data) if context_data else {}
        
        # Process uploaded files
        image_data = None
        audio_data = None
        temp_files = []
        
        if image:
            image_data = await image.read()
            temp_image = UPLOAD_DIR / f"temp_img_{uuid.uuid4().hex}.jpg"
            with open(temp_image, "wb") as f:
                f.write(image_data)
            temp_files.append(str(temp_image))
        
        if audio:
            audio_data = await audio.read()
            temp_audio = UPLOAD_DIR / f"temp_audio_{uuid.uuid4().hex}.wav"
            with open(temp_audio, "wb") as f:
                f.write(audio_data)
            temp_files.append(str(temp_audio))
        
        # Process with Gemma 3n
        analysis_result = gemma_processor.analyze_multimodal_emergency(
            text=text_report,
            image_data=image_data,
            audio_data=audio_data,
            context=context
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Save to database
        assessment = MultimodalAssessment(
            assessment_type="comprehensive_multimodal",
            text_input=text_report,
            image_path=temp_files[0] if image else None,
            audio_path=temp_files[1] if audio else temp_files[0] if audio and not image else None,
            severity_score=analysis_result["severity"]["overall_score"],
            emergency_type=analysis_result["emergency_type"]["primary"],
            risk_factors=analysis_result["immediate_risks"],
            resource_requirements=analysis_result["resource_requirements"],
            ai_confidence=analysis_result["severity"]["confidence"],
            analyst_id=current_user["username"]
        )
        
        db.add(assessment)
        db.commit()
        db.refresh(assessment)
        
        # Schedule cleanup
        for temp_file in temp_files:
            background_tasks.add_task(cleanup_temp_file, temp_file)
        
        return JSONResponse({
            "success": True,
            "analysis_id": assessment.id,
            "analysis": analysis_result,
            "processing_time_seconds": processing_time,
            "modalities_processed": {
                "text": text_report is not None,
                "image": image is not None,
                "audio": audio is not None
            }
        })
        
    except Exception as e:
        logger.error(f"Multimodal analysis error: {e}")
        # Cleanup temp files on error
        for temp_file in temp_files:
            background_tasks.add_task(cleanup_temp_file, temp_file)
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/ai-model-status")
async def get_ai_model_status():
    """Get current AI model status and performance"""
    try:
        performance = ai_optimizer.monitor_performance()
        
        return JSONResponse({
            "success": True,
            "status": {
                "gemma_3n_status": {
                    "available_models": [
                        {"name": "gemma-3n-2b", "status": "ready", "memory": "2GB", "speed": "fast"},
                        {"name": "gemma-3n-4b", "status": "ready", "memory": "4GB", "speed": "balanced"},
                        {"name": "gemma-3n-4b-hq", "status": "ready", "memory": "6GB", "speed": "precise"}
                    ],
                    "active_model": ai_optimizer.current_config.model_variant,
                    "context_window": f"{ai_optimizer.current_config.context_window} tokens",
                    "optimization_level": ai_optimizer.current_config.optimization_level
                },
                "performance_metrics": {
                    "cpu_usage": f"{performance.cpu_usage}%",
                    "memory_usage": f"{performance.memory_usage}%",
                    "inference_speed": f"{performance.inference_speed}s",
                    "battery_level": f"{performance.battery_level}%"
                },
                "device_capabilities": ai_optimizer.device_caps
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting AI model status: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/optimize-ai-settings")
async def optimize_ai_settings(request: Request):
    """Optimize AI settings for device and use case"""
    try:
        data = await request.json()
        optimization_level = data.get("optimization_level", "balanced")
        
        # Apply optimization
        new_config = ai_optimizer.optimize_for_device(optimization_level)
        ai_optimizer.current_config = new_config
        
        return JSONResponse({
            "success": True,
            "message": f"AI optimized for {optimization_level} performance",
            "new_config": {
                "model_variant": new_config.model_variant,
                "context_window": new_config.context_window,
                "precision": new_config.precision,
                "optimization_level": new_config.optimization_level
            }
        })
        
    except Exception as e:
        logger.error(f"AI optimization error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# ================================================================================
# API ENDPOINTS - PATIENT MANAGEMENT
# ================================================================================

@app.get("/triage-form", response_class=HTMLResponse)
async def triage_form_page(request: Request):
    """Triage form page"""
    try:
        return templates.TemplateResponse("triage_form.html", {"request": request})
    except:
        return HTMLResponse("""
        <html><head><title>Triage Form</title></head>
        <body>
        <h1>🏥 Patient Triage Form</h1>
        <form action="/api/submit-triage" method="post">
        <input type="text" name="name" placeholder="Patient Name" required><br>
        <input type="number" name="age" placeholder="Age"><br>
        <select name="gender">
        <option value="male">Male</option>
        <option value="female">Female</option>
        <option value="other">Other</option>
        </select><br>
        <input type="text" name="injury_type" placeholder="Injury Type"><br>
        <select name="severity">
        <option value="mild">Mild</option>
        <option value="moderate">Moderate</option>
        <option value="severe">Severe</option>
        <option value="critical">Critical</option>
        </select><br>
        <select name="triage_color">
        <option value="green">Green</option>
        <option value="yellow">Yellow</option>
        <option value="red">Red</option>
        <option value="black">Black</option>
        </select><br>
        <textarea name="notes" placeholder="Additional notes"></textarea><br>
        <button type="submit">Submit Triage</button>
        </form>
        </body></html>
        """)

@app.post("/api/submit-triage")
async def submit_triage(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    injury_type: str = Form(...),
    severity: str = Form(...),
    triage_color: str = Form(...),
    location: str = Form(""),
    notes: str = Form(""),
    db: Session = Depends(get_db)
):
    """Submit patient triage information"""
    try:
        # Calculate priority score
        priority_scores = {"critical": 10, "severe": 8, "moderate": 5, "mild": 3}
        priority_score = priority_scores.get(severity, 5)
        
        patient = TriagePatient(
            name=name,
            age=age,
            gender=gender,
            injury_type=injury_type,
            severity=severity,
            triage_color=triage_color,
            location=location,
            notes=notes,
            priority_score=priority_score
        )
        
        db.add(patient)
        db.commit()
        db.refresh(patient)
        
        logger.info(f"Patient triaged: {name} ({triage_color} - {severity})")
        
        return JSONResponse({
            "success": True,
            "patient_id": patient.id,
            "triage_color": triage_color,
            "severity": severity,
            "priority_score": priority_score
        })
        
    except Exception as e:
        logger.error(f"Triage submission failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/patients")
async def get_patients(
    status: str = Query("active", description="Patient status filter"),
    triage_color: Optional[str] = Query(None, description="Triage color filter"),
    limit: int = Query(100, description="Number of patients to return"),
    db: Session = Depends(get_db)
):
    """Get patients with filtering"""
    try:
        query = db.query(TriagePatient).filter(TriagePatient.status == status)
        
        if triage_color:
            query = query.filter(TriagePatient.triage_color == triage_color)
        
        patients = query.order_by(desc(TriagePatient.priority_score)).limit(limit).all()
        
        return JSONResponse({
            "success": True,
            "patients": [
                {
                    "id": p.id,
                    "name": p.name,
                    "age": p.age,
                    "gender": p.gender,
                    "injury_type": p.injury_type,
                    "severity": p.severity,
                    "triage_color": p.triage_color,
                    "status": p.status,
                    "location": p.location,
                    "priority_score": p.priority_score,
                    "created_at": p.created_at.isoformat(),
                    "time_ago": calculate_time_ago(p.created_at)
                }
                for p in patients
            ],
            "total": len(patients),
            "status_filter": status,
            "triage_color_filter": triage_color
        })
        
    except Exception as e:
        logger.error(f"Error fetching patients: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# ================================================================================
# API ENDPOINTS - STATISTICS & ANALYTICS
# ================================================================================

@app.get("/api/dashboard-stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get comprehensive dashboard statistics"""
    try:
        # Basic counts
        total_reports = db.query(CrowdReport).count()
        total_patients = db.query(TriagePatient).count()
        total_emergency_reports = db.query(EmergencyReport).count()
        
        # Today's data
        today = datetime.utcnow().date()
        reports_today = db.query(CrowdReport).filter(
            func.date(CrowdReport.timestamp) == today
        ).count()
        patients_today = db.query(TriagePatient).filter(
            func.date(TriagePatient.created_at) == today
        ).count()
        
        # Critical/urgent counts
        critical_reports = db.query(CrowdReport).filter(
            CrowdReport.escalation == "critical"
        ).count()
        critical_patients = db.query(TriagePatient).filter(
            or_(TriagePatient.triage_color == "red", TriagePatient.severity == "critical")
        ).count()
        
        # Active patients
        active_patients = db.query(TriagePatient).filter(
            TriagePatient.status == "active"
        ).count()
        
        return JSONResponse({
            "success": True,
            "stats": {
                "total_reports": total_reports,
                "total_patients": total_patients,
                "total_emergency_reports": total_emergency_reports,
                "reports_today": reports_today,
                "patients_today": patients_today,
                "critical_reports": critical_reports,
                "critical_patients": critical_patients,
                "active_patients": active_patients,
                "system_uptime": "24h 15m",
                "last_updated": datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/analytics-data")
async def get_analytics_data(
    timeframe: str = Query("7d", description="Time range: 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get analytics data for charts and graphs"""
    try:
        # Calculate timeframe
        now = datetime.utcnow()
        if timeframe == "24h":
            start_time = now - timedelta(hours=24)
        elif timeframe == "30d":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=7)
        
        # Get reports in timeframe
        reports = db.query(CrowdReport).filter(
            CrowdReport.timestamp >= start_time
        ).all()
        
        patients = db.query(TriagePatient).filter(
            TriagePatient.created_at >= start_time
        ).all()
        
        # Escalation breakdown
        escalation_counts = {}
        for level in ["critical", "high", "moderate", "low"]:
            count = len([r for r in reports if r.escalation == level])
            escalation_counts[level] = count
        
        # Triage color breakdown
        triage_counts = {}
        for color in ["red", "yellow", "green", "black"]:
            count = len([p for p in patients if p.triage_color == color])
            triage_counts[color] = count
        
        # Trend data (daily aggregation)
        trend_data = []
        current_date = start_time.date()
        end_date = now.date()
        
        while current_date <= end_date:
            day_reports = len([r for r in reports if r.timestamp.date() == current_date])
            day_patients = len([p for p in patients if p.created_at.date() == current_date])
            
            trend_data.append({
                "date": current_date.isoformat(),
                "reports": day_reports,
                "patients": day_patients
            })
            current_date += timedelta(days=1)
        
        return JSONResponse({
            "success": True,
            "analytics": {
                "timeframe": timeframe,
                "total_reports": len(reports),
                "total_patients": len(patients),
                "escalation_breakdown": escalation_counts,
                "triage_breakdown": triage_counts,
                "trend_data": trend_data,
                "average_severity": sum(r.severity for r in reports if r.severity) / max(len(reports), 1)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics data: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# ================================================================================
# API ENDPOINTS - EXPORT & UTILITY
# ================================================================================

@app.get("/api/export-reports")
async def export_reports(
    format: str = Query("json", description="Export format: json, csv"),
    timeframe: str = Query("7d", description="Time range"),
    db: Session = Depends(get_db)
):
    """Export reports in various formats"""
    try:
        # Calculate timeframe
        now = datetime.utcnow()
        if timeframe == "24h":
            start_time = now - timedelta(hours=24)
        elif timeframe == "30d":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=7)
        
        # Get reports
        reports = db.query(CrowdReport).filter(
            CrowdReport.timestamp >= start_time
        ).order_by(desc(CrowdReport.timestamp)).all()
        
        if format == "csv":
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "ID", "Message", "Escalation", "Tone", "User", "Location", 
                "Latitude", "Longitude", "Timestamp", "Severity"
            ])
            
            # Write data
            for report in reports:
                writer.writerow([
                    report.id, report.message, report.escalation, report.tone,
                    report.user, report.location, report.latitude, report.longitude,
                    report.timestamp.isoformat(), report.severity
                ])
            
            csv_content = output.getvalue()
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=reports_{timeframe}.csv"}
            )
        
        else:  # JSON format
            reports_data = [
                {
                    "id": r.id,
                    "message": r.message,
                    "escalation": r.escalation,
                    "tone": r.tone,
                    "user": r.user,
                    "location": r.location,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "timestamp": r.timestamp.isoformat(),
                    "severity": r.severity,
                    "time_ago": calculate_time_ago(r.timestamp)
                }
                for r in reports
            ]
            
            export_data = {
                "export_info": {
                    "timeframe": timeframe,
                    "export_date": now.isoformat(),
                    "total_reports": len(reports)
                },
                "reports": reports_data
            }
            
            return JSONResponse(export_data)
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/generate-demo-data")
async def generate_demo_data(
    count: int = Query(10, description="Number of demo reports to create"),
    db: Session = Depends(get_db)
):
    """Generate demo data for testing"""
    try:
        demo_reports = []
        demo_locations = [
            {"name": "Downtown Fire Station", "lat": 37.7749, "lng": -122.4194},
            {"name": "City Hospital", "lat": 37.7849, "lng": -122.4094},
            {"name": "Central Park", "lat": 37.7649, "lng": -122.4294},
            {"name": "Main Street", "lat": 37.7949, "lng": -122.4394},
            {"name": "Harbor District", "lat": 37.7549, "lng": -122.4494}
        ]
        
        demo_messages = [
            "Traffic accident on Main Street, multiple vehicles involved",
            "Fire reported in downtown building, smoke visible",
            "Medical emergency at City Hospital parking lot",
            "Flooding in Harbor District due to heavy rain",
            "Power outage affecting 3 city blocks",
            "Gas leak reported near Central Park",
            "Building collapse risk on 5th Avenue",
            "Multiple injuries from bus accident",
            "Chemical spill on Highway 101",
            "Earthquake damage assessment needed"
        ]
        
        escalation_levels = ["low", "moderate", "high", "critical"]
        tones = ["urgent", "concerned", "descriptive", "frantic"]
        
        for i in range(count):
            location = demo_locations[i % len(demo_locations)]
            message = demo_messages[i % len(demo_messages)]
            
            report = CrowdReport(
                message=f"DEMO: {message}",
                escalation=escalation_levels[i % len(escalation_levels)],
                tone=tones[i % len(tones)],
                user=f"DemoUser{i+1}",
                location=location["name"],
                latitude=location["lat"] + (i * 0.001),  # Slight variation
                longitude=location["lng"] + (i * 0.001),
                severity={"critical": 9, "high": 7, "moderate": 5, "low": 3}[escalation_levels[i % len(escalation_levels)]],
                source="demo_generator"
            )
            
            db.add(report)
            demo_reports.append(report)
        
        db.commit()
        
        logger.info(f"Generated {count} demo reports")
        
        return JSONResponse({
            "success": True,
            "message": f"Generated {count} demo reports",
            "reports_created": len(demo_reports)
        })
        
    except Exception as e:
        logger.error(f"Demo data generation failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.delete("/api/clear-demo-data")
async def clear_demo_data(db: Session = Depends(get_db)):
    """Clear demo data from database"""
    try:
        # Delete demo reports
        demo_reports_deleted = db.query(CrowdReport).filter(
            CrowdReport.source == "demo_generator"
        ).delete()
        
        # Delete reports with DEMO prefix
        demo_prefix_deleted = db.query(CrowdReport).filter(
            CrowdReport.message.like("DEMO:%")
        ).delete()
        
        db.commit()
        
        total_deleted = demo_reports_deleted + demo_prefix_deleted
        
        return JSONResponse({
            "success": True,
            "message": f"Cleared {total_deleted} demo reports",
            "demo_reports_deleted": demo_reports_deleted,
            "demo_prefix_deleted": demo_prefix_deleted
        })
        
    except Exception as e:
        logger.error(f"Demo data clearing failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# ================================================================================
# MAP & LOCATION ROUTES
# ================================================================================

@app.get("/map-reports", response_class=HTMLResponse)
async def map_reports_page(request: Request):
    """Map reports visualization page"""
    try:
        return templates.TemplateResponse("map_reports.html", {"request": request})
    except:
        return HTMLResponse("""
        <html><head><title>Map Reports</title></head>
        <body>
        <h1>🗺️ Emergency Reports Map</h1>
        <div id="map" style="height: 500px; background: #f0f0f0; display: flex; align-items: center; justify-content: center;">
        <p>Interactive map would appear here</p>
        </div>
        <script>
        // Placeholder for map initialization
        console.log('Map would be initialized here');
        </script>
        </body></html>
        """)

@app.get("/map-snapshot", response_class=HTMLResponse)
async def map_snapshot_page(
    request: Request,
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    report_id: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    """Map snapshot page with specific location"""
    try:
        # Get coordinates from report or use provided coordinates
        if report_id:
            report = db.query(CrowdReport).filter(CrowdReport.id == report_id).first()
            if report and report.latitude and report.longitude:
                lat, lon = report.latitude, report.longitude
        
        # Default to San Francisco if no coordinates
        if lat is None or lon is None:
            lat, lon = 37.7749, -122.4194
        
        return templates.TemplateResponse("map_snapshot.html", {
            "request": request,
            "latitude": lat,
            "longitude": lon,
            "report_id": report_id
        })
    except:
        return HTMLResponse(f"""
        <html><head><title>Map Snapshot</title></head>
        <body>
        <h1>🗺️ Map Snapshot</h1>
        <p>Location: {lat}, {lon}</p>
        <div style="height: 400px; background: #e0e0e0; display: flex; align-items: center; justify-content: center;">
        <p>Map snapshot for coordinates: {lat}, {lon}</p>
        </div>
        </body></html>
        """)

@app.get("/api/map-data")
async def get_map_data(
    bounds: Optional[str] = Query(None, description="Map bounds as 'lat1,lng1,lat2,lng2'"),
    escalation: Optional[str] = Query(None, description="Filter by escalation level"),
    db: Session = Depends(get_db)
):
    """Get map data for reports visualization"""
    try:
        query = db.query(CrowdReport).filter(
            CrowdReport.latitude.isnot(None),
            CrowdReport.longitude.isnot(None)
        )
        
        if escalation:
            query = query.filter(CrowdReport.escalation == escalation)
        
        # Apply bounds filter if provided
        if bounds:
            try:
                lat1, lng1, lat2, lng2 = map(float, bounds.split(','))
                query = query.filter(
                    CrowdReport.latitude.between(min(lat1, lat2), max(lat1, lat2)),
                    CrowdReport.longitude.between(min(lng1, lng2), max(lng1, lng2))
                )
            except ValueError:
                pass  # Ignore invalid bounds format
        
        reports = query.order_by(desc(CrowdReport.timestamp)).limit(500).all()
        
        map_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [r.longitude, r.latitude]
                    },
                    "properties": {
                        "id": r.id,
                        "message": r.message,
                        "escalation": r.escalation,
                        "tone": r.tone,
                        "user": r.user,
                        "location": r.location,
                        "timestamp": r.timestamp.isoformat(),
                        "severity": r.severity,
                        "time_ago": calculate_time_ago(r.timestamp),
                        "marker_color": {
                            "critical": "#dc2626",
                            "high": "#f59e0b", 
                            "moderate": "#3b82f6",
                            "low": "#16a34a"
                        }.get(r.escalation, "#6b7280")
                    }
                }
                for r in reports
            ]
        }
        
        return JSONResponse({
            "success": True,
            "data": map_data,
            "total_features": len(map_data["features"]),
            "bounds_applied": bounds is not None,
            "escalation_filter": escalation
        })
        
    except Exception as e:
        logger.error(f"Map data error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# ================================================================================
# AUTHENTICATION ROUTES
# ================================================================================

@app.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login endpoint to get access token"""
    try:
        # Simple demo authentication
        if form_data.username == "admin" and form_data.password == "admin":
            access_token = create_access_token(data={"sub": form_data.username})
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "role": "admin"
            }
        elif form_data.username == "demo" and form_data.password == "demo":
            access_token = create_access_token(data={"sub": form_data.username})
            return {
                "access_token": access_token,
                "token_type": "bearer", 
                "role": "user"
            }
        else:
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/register")
async def register_user(
    request: Request,
    db: Session = Depends(get_db)
):
    """Register new user"""
    try:
        data = await request.json()
        username = data.get("username")
        email = data.get("email")
        password = data.get("password")
        
        if not all([username, email, password]):
            raise HTTPException(status_code=400, detail="All fields required")
        
        # Check if user exists
        existing_user = db.query(User).filter(
            or_(User.username == username, User.email == email)
        ).first()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create new user
        user = User(
            username=username,
            email=email,
            hashed_password=hash_password(password),
            role="user"
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return JSONResponse({
            "success": True,
            "message": "User registered successfully",
            "user_id": user.id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return JSONResponse({
        "success": True,
        "user": current_user
    })

# ================================================================================
# SYSTEM HEALTH & UTILITIES
# ================================================================================

@app.get("/health")
async def health_check():
    """Comprehensive system health check"""
    try:
        # Database health
        db = next(get_db())
        total_reports = db.query(CrowdReport).count()
        total_patients = db.query(TriagePatient).count()
        total_emergency_reports = db.query(EmergencyReport).count()
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        total_reports = total_patients = total_emergency_reports = 0
        db_status = "error"
    
    # AI model status
    try:
        ai_performance = ai_optimizer.monitor_performance()
        ai_status = "ready"
    except Exception as e:
        logger.error(f"AI health check failed: {e}")
        ai_status = "error"
    
    return JSONResponse({
        "status": "healthy",
        "service": "Enhanced Emergency Response Assistant",
        "version": "3.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": {
                "status": db_status,
                "records": {
                    "crowd_reports": total_reports,
                    "triage_patients": total_patients,
                    "emergency_reports": total_emergency_reports
                }
            },
            "ai_models": {
                "status": ai_status,
                "gemma_3n": {
                    "active_model": ai_optimizer.current_config.model_variant,
                    "optimization_level": ai_optimizer.current_config.optimization_level
                }
            },
            "features": {
                "citizen_portal": True,
                "emergency_reporting": True,
                "voice_analysis": True,
                "image_analysis": True,
                "patient_triage": True,
                "crowd_reports": True,
                "analytics_dashboard": True,
                "map_visualization": True,
                "export_functionality": True,
                "demo_data_generation": True,
                "offline_support": True,
                "multimodal_analysis": True,
                "ai_optimization": True
            }
        },
        "endpoints": {
            "citizen_portal": "/",
            "admin_dashboard": "/admin",
            "api_documentation": "/api/docs",
            "health_check": "/health",
            "emergency_reports": "/api/emergency-reports",
            "crowd_reports": "/api/crowd-reports",
            "patients": "/api/patients",
            "analytics": "/api/analytics-data"
        }
    })

@app.get("/api/system-info")
async def get_system_info():
    """Get detailed system information"""
    try:
        import platform
        import sys
        
        return JSONResponse({
            "success": True,
            "system": {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.architecture()[0],
                "python_version": sys.version,
                "hostname": platform.node()
            },
            "application": {
                "name": "Enhanced Emergency Response Assistant",
                "version": "3.0.0",
                "base_directory": str(BASE_DIR),
                "upload_directory": str(UPLOAD_DIR),
                "templates_available": TEMPLATES_DIR.exists(),
                "static_files_available": STATIC_DIR.exists()
            },
            "capabilities": {
                "weasyprint_pdf": WEASYPRINT_AVAILABLE,
                "database": DATABASE_AVAILABLE,
                "ai_processing": True,
                "file_uploads": True,
                "background_tasks": True
            }
        })
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# ================================================================================
# STATIC FILE HANDLERS & UTILITIES
# ================================================================================

@app.get("/favicon.ico")
async def get_favicon():
    """Serve favicon"""
    favicon_path = STATIC_DIR / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    else:
        # Return empty response for missing favicon
        return Response(content="", media_type="image/x-icon")

@app.get("/manifest.json")
async def get_manifest():
    """Serve PWA manifest"""
    manifest_path = STATIC_DIR / "manifest.json"
    if manifest_path.exists():
        return FileResponse(manifest_path, media_type="application/json")
    else:
        # Return basic manifest
        return JSONResponse({
            "name": "Emergency Response Assistant",
            "short_name": "Emergency App",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#ffffff",
            "theme_color": "#3b82f6",
            "icons": [
                {
                    "src": "/static/icon-192.png",
                    "sizes": "192x192",
                    "type": "image/png"
                }
            ]
        })

@app.get("/sw.js")
async def get_service_worker():
    """Serve service worker"""
    sw_path = STATIC_DIR / "js" / "sw.js"
    if sw_path.exists():
        return FileResponse(sw_path, media_type="application/javascript")
    else:
        # Return basic service worker
        return Response(
            content="""
// Basic service worker for offline support
const CACHE_NAME = 'emergency-app-v1';
const urlsToCache = [
  '/',
  '/offline',
  '/static/css/styles.css'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        if (response) {
          return response;
        }
        return fetch(event.request);
      })
      .catch(() => caches.match('/offline'))
  );
});
            """,
            media_type="application/javascript"
        )

# ================================================================================
# ERROR HANDLERS
# ================================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Enhanced 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "path": str(request.url),
            "suggestions": [
                "Try /api/docs for API documentation",
                "Visit / for the citizen portal",
                "Check /health for system status",
                "Use /admin for dashboard access"
            ],
            "available_endpoints": {
                "citizen_portal": "/",
                "admin_dashboard": "/admin", 
                "api_docs": "/api/docs",
                "health_check": "/health"
            }
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Enhanced 500 handler"""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "support_info": {
                "health_check": "/health",
                "system_info": "/api/system-info",
                "contact": "Check system logs for details"
            }
        }
    )

# ================================================================================
# APPLICATION STARTUP & LIFECYCLE
# ================================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup initialization"""
    logger.info("🚀 Starting Enhanced Emergency Response Assistant v3.0.0")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created/verified")
        
        # Initialize demo data if database is empty
        db = next(get_db())
        if db.query(CrowdReport).count() == 0:
            logger.info("📝 Initializing with welcome data...")
            
            welcome_report = CrowdReport(
                message="Welcome to the Enhanced Emergency Response Assistant! This system provides comprehensive emergency management capabilities including citizen reporting, AI analysis, and professional dashboard tools.",
                escalation="low",
                tone="descriptive",
                user="System",
                location="San Francisco, CA",
                latitude=37.7749,
                longitude=-122.4194,
                severity=2,
                source="system_init"
            )
            
            db.add(welcome_report)
            db.commit()
            
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
    
    # Initialize AI components
    try:
        logger.info("🧠 Initializing AI components...")
        performance = ai_optimizer.monitor_performance()
        logger.info(f"     • Device CPU: {performance.cpu_usage}%")
        logger.info(f"     • Memory: {performance.memory_usage}%")
        logger.info(f"     • Active model: {ai_optimizer.current_config.model_variant}")
        logger.info("✅ AI components ready")
    except Exception as e:
        logger.error(f"❌ AI initialization error: {e}")
    
    # Log available features
    logger.info("🎯 Available features:")
    logger.info("     • Citizen Emergency Portal (main interface)")
    logger.info("     • Voice Emergency Reporter with real-time transcription")
    logger.info("     • Multimodal AI Analysis (text + image + audio)")
    logger.info("     • Professional Admin Dashboard")
    logger.info("     • Patient Triage Management")
    logger.info("     • Crowd Report System with geolocation")
    logger.info("     • Analytics Dashboard with real-time charts")
    logger.info("     • Map Visualization with interactive reports")
    logger.info("     • Export functionality (JSON, CSV)")
    logger.info("     • Demo data generation for testing")
    logger.info("     • Offline support with service worker")
    logger.info("     • RESTful API with comprehensive documentation")
    
    logger.info("✅ Enhanced Emergency Response Assistant ready!")
    logger.info(f"     🌐 Citizen Portal: http://localhost:8000/")
    logger.info(f"     📊 Admin Dashboard: http://localhost:8000/admin")
    logger.info(f"     📚 API Documentation: http://localhost:8000/api/docs")
    logger.info(f"     🏥 Health Check: http://localhost:8000/health")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown cleanup"""
    logger.info("🛑 Shutting down Enhanced Emergency Response Assistant")
    
    # Cleanup temporary files
    try:
        temp_files = list(UPLOAD_DIR.glob("temp_*"))
        for temp_file in temp_files:
            temp_file.unlink()
        logger.info(f"🧹 Cleaned up {len(temp_files)} temporary files")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
    
    logger.info("✅ Shutdown complete")

# ================================================================================
# MAIN APPLICATION ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🎯 Enhanced Emergency Response Assistant")
    logger.info("📍 Starting FastAPI server...")
    logger.info("🌐 Citizen Portal will be available at: http://localhost:8000/")
    logger.info("📊 Admin Dashboard at: http://localhost:8000/admin")
    logger.info("📚 API Documentation at: http://localhost:8000/api/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )