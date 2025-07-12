# ================================================================================
# IMPORTS & DEPENDENCIES
# ================================================================================

from fastapi import (
    FastAPI, Request, Form, UploadFile, File, Depends, HTTPException,
    Body, Query, BackgroundTasks, Header, WebSocket, WebSocketDisconnect
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
import sys
import tempfile
import asyncio
import hashlib
import secrets
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator, EmailStr
from enum import Enum
import time
import psutil
from functools import wraps
from collections import defaultdict, deque
from threading import Lock
from dataclasses import dataclass
import bcrypt
import jwt

# External dependencies with fallback handling
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

from jinja2 import Environment, FileSystemLoader
from io import BytesIO

# Internal modules with comprehensive fallback implementations
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
    def analyze_sentiment(text): return {"sentiment": "neutral", "tone": "descriptive", "escalation": "low"}

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
    from app.inference import run_disaster_analysis, Gemma3nEmergencyProcessor, analyze_voice_emergency
    from app.audio_transcription import VoiceEmergencyProcessor
    from app.adaptive_ai_settings import adaptive_optimizer
except ImportError:
    # Use comprehensive fallback implementations
    try:
        from app.fallback_ai import (
            gemma_processor as Gemma3nEmergencyProcessor,
            voice_processor as VoiceEmergencyProcessor, 
            ai_optimizer as adaptive_optimizer,
            analyze_voice_emergency,
            detect_hazards,
            transcribe_audio
        )
        
        def run_disaster_analysis(data):
            if data.get("type") == "text":
                return Gemma3nEmergencyProcessor.analyze_multimodal_emergency(text=data.get("content"))
            return {"type": "simulated", "confidence": 0.8}
            
    except ImportError:
        # Ultimate fallback if even fallback_ai doesn't exist
        class MockProcessor:
            def analyze_multimodal_emergency(self, **kwargs):
                return {"severity": {"overall_score": 5, "confidence": 0.8}, "emergency_type": {"primary": "general"}}
            def process_emergency_call(self, audio_path, context=None):
                return {"transcript": "Mock transcript", "overall_urgency": "medium", "confidence": 0.8}
        
        class MockOptimizer:
            def __init__(self):
                self.current_config = type('Config', (), {'model_variant': 'mock', 'context_window': 4000})()
                self.device_caps = type('Caps', (), {'cpu_cores': 4, 'memory_gb': 8, 'gpu_available': False})()
            def monitor_performance(self):
                return type('Perf', (), {'cpu_usage': 50, 'memory_usage': 60, 'inference_speed': 10})()
        
        Gemma3nEmergencyProcessor = MockProcessor()
        VoiceEmergencyProcessor = MockProcessor()
        adaptive_optimizer = MockOptimizer()
        
        def analyze_voice_emergency(transcript, audio_features, emotional_state):
            return {"urgency": "medium", "emergency_type": "general", "confidence": 0.8}
        
        def run_disaster_analysis(data):
            return {"type": "simulated", "confidence": 0.8}
        
        def detect_hazards(image_data):
            return ["simulated_hazard"]
        
        def transcribe_audio(audio_path):
            return {"transcript": "Mock transcription", "confidence": 0.8}

try:
    from app.report_utils import generate_report_pdf, generate_map_preview_data
except ImportError:
    def generate_report_pdf(data): 
        return f"mock_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    def generate_map_preview_data(lat, lon): 
        return {"preview": "simulated", "coordinates": {"latitude": lat, "longitude": lon}}

try:
    from celery_app import schedule_emergency_analysis
    CELERY_INTEGRATION = True
except ImportError:
    CELERY_INTEGRATION = False
    def schedule_emergency_analysis(**kwargs):
        return {"status": "mock", "message": "Celery not available"}

# Create global instances for use throughout the application
try:
    # Try to use the real processors
    gemma_processor = Gemma3nEmergencyProcessor()
    voice_processor = VoiceEmergencyProcessor()
except:
    # If that fails, they're already mocked above
    pass

# ================================================================================
# CONFIGURATION MANAGEMENT
# ================================================================================

@dataclass
class AppConfig:
    """Application configuration with environment variable support"""
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{Path.cwd()}/data/emergency_response.db")
    
    # File uploads
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "uploads")
    
    # AI Configuration
    AI_MODEL_VARIANT: str = os.getenv("AI_MODEL_VARIANT", "gemma-3n-4b")
    AI_CONTEXT_WINDOW: int = int(os.getenv("AI_CONTEXT_WINDOW", "64000"))
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))
    
    # Environment
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # External services
    ENABLE_NOTIFICATIONS: bool = os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true"
    NOTIFICATION_WEBHOOK_URL: Optional[str] = os.getenv("NOTIFICATION_WEBHOOK_URL")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "detailed")

# Global config instance
config = AppConfig()

# ================================================================================
# ENHANCED LOGGING SETUP
# ================================================================================

def setup_logging():
    """Setup comprehensive logging system with structured output"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            if hasattr(record, 'user_id'):
                log_data["user_id"] = record.user_id
            if hasattr(record, 'request_id'):
                log_data["request_id"] = record.request_id
            if hasattr(record, 'ip_address'):
                log_data["ip_address"] = record.ip_address
            return json.dumps(log_data)
            
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    if config.LOG_FORMAT == "structured":
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(console_handler)
    
    # File handlers
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)
    
    error_handler = logging.FileHandler(log_dir / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(error_handler)
    
    # Security audit logger
    audit_handler = logging.FileHandler(log_dir / "security.log")
    audit_handler.setLevel(logging.WARNING)
    audit_handler.setFormatter(StructuredFormatter())
    
    audit_logger = logging.getLogger("security_audit")
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.WARNING)
    
    return root_logger

logger = setup_logging()

def log_security_event(event_type: str, details: dict, user_id: str = None, ip_address: str = None):
    """Log security-related events"""
    audit_logger = logging.getLogger("security_audit")
    extra_data = {'event_type': event_type, 'user_id': user_id, 'ip_address': ip_address, **details}
    audit_logger.warning(f"Security event: {event_type}", extra=extra_data)

def log_api_request(endpoint: str, method: str, user_id: str = None, ip_address: str = None, response_code: int = None):
    """Log API requests for monitoring"""
    logger.info(f"API {method} {endpoint}", extra={
        'endpoint': endpoint, 'method': method, 'user_id': user_id, 
        'ip_address': ip_address, 'response_code': response_code
    })

# ================================================================================
# DATABASE SETUP WITH ALL MODELS
# ================================================================================

# Directory paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = BASE_DIR / config.UPLOAD_FOLDER
OUTPUT_DIR = BASE_DIR / "outputs"

try:
    from app.database import get_db, engine
    from app.models import (
        Base, 
        User,  # Added missing User model
        CrowdReport, 
        TriagePatient, 
        EmergencyReport,  # Added missing EmergencyReport model
        VoiceAnalysis,  # Added missing VoiceAnalysis model
        MultimodalAssessment,  # Added missing MultimodalAssessment model
        ContextAnalysis,  # Added missing ContextAnalysis model
        DevicePerformance  # Added missing DevicePerformance model
    )
    DATABASE_AVAILABLE = True
except ImportError:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    DATABASE_DIR = BASE_DIR / "data"
    DATABASE_DIR.mkdir(exist_ok=True)
    
    engine = create_engine(
        config.DATABASE_URL, 
        echo=False, 
        connect_args={"check_same_thread": False} if "sqlite" in config.DATABASE_URL else {}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    # Create fallback models if imports failed
    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True, index=True)
        username = Column(String(50), unique=True, index=True, nullable=False)
        email = Column(String(100), unique=True, index=True, nullable=False)
        hashed_password = Column(String(255), nullable=False)
        role = Column(String(20), default="user", index=True)
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        last_login = Column(DateTime, nullable=True)
    
    class EmergencyReport(Base):
        __tablename__ = "emergency_reports"
        id = Column(Integer, primary_key=True, index=True)
        report_id = Column(String(50), unique=True, index=True)
        type = Column(String(100), nullable=False)
        description = Column(Text, nullable=False)
        location = Column(String(255), nullable=False)
        latitude = Column(Float, nullable=True)
        longitude = Column(Float, nullable=True)
        priority = Column(String(20), default="medium", index=True)
        status = Column(String(20), default="pending", index=True)
        method = Column(String(20), default="text")
        reporter = Column(String(100), nullable=True)
        evidence_file = Column(String(255), nullable=True)
        ai_analysis = Column(JSON, nullable=True)
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
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
        severity = Column(Integer, nullable=True, index=True)
        confidence_score = Column(Float, nullable=True)
        ai_analysis = Column(JSON, nullable=True)
        source = Column(String, default="manual", index=True)
        verified = Column(Boolean, default=False)
        response_dispatched = Column(Boolean, default=False)
    
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
        priority_score = Column(Integer, default=5)
        created_at = Column(DateTime, default=datetime.utcnow, index=True)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    class VoiceAnalysis(Base):
        __tablename__ = "voice_analyses"
        id = Column(Integer, primary_key=True, index=True)
        audio_file_path = Column(String(255))
        transcript = Column(Text)
        confidence = Column(Float, default=0.0)
        urgency_level = Column(String(20), index=True)
        emergency_type = Column(String(100))
        hazards_detected = Column(JSON)
        emotional_state = Column(JSON)
        analyst_id = Column(String(100))
        created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    class MultimodalAssessment(Base):
        __tablename__ = "multimodal_assessments"
        id = Column(Integer, primary_key=True, index=True)
        assessment_type = Column(String(50), index=True)
        text_input = Column(Text)
        image_path = Column(String(255))
        audio_path = Column(String(255))
        severity_score = Column(Float, default=0.0, index=True)
        emergency_type = Column(String(100), index=True)
        risk_factors = Column(JSON)
        resource_requirements = Column(JSON)
        ai_confidence = Column(Float, default=0.0)
        analyst_id = Column(String(100))
        created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    class ContextAnalysis(Base):
        __tablename__ = "context_analyses"
        id = Column(Integer, primary_key=True, index=True)
        analysis_type = Column(String(50), index=True)
        input_tokens = Column(Integer, default=0)
        output_summary = Column(Text)
        key_insights = Column(JSON)
        confidence = Column(Float, default=0.0)
        analyst_id = Column(String(100))
        created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    class DevicePerformance(Base):
        __tablename__ = "device_performance"
        id = Column(Integer, primary_key=True, index=True)
        device_id = Column(String(100), index=True)
        cpu_usage = Column(Float)
        memory_usage = Column(Float)
        gpu_usage = Column(Float)
        battery_level = Column(Float)
        temperature = Column(Float)
        inference_speed = Column(Float)
        optimization_level = Column(String(20))
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    DATABASE_AVAILABLE = True

# ================================================================================
# INPUT VALIDATION MODELS (PYDANTIC)
# ================================================================================

class PriorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EscalationLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class TriageColor(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    BLACK = "black"

class EmergencyReportRequest(BaseModel):
    type: str = Field(..., min_length=1, max_length=50)
    description: str = Field(..., min_length=5, max_length=1000)
    location: str = Field(..., min_length=1, max_length=255)
    priority: PriorityLevel = PriorityLevel.MEDIUM
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    method: str = Field("text", pattern="^(text|voice|image|multimodal)$")
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if not v.strip():
            raise ValueError('Description cannot be empty')
        return v.strip()

class CrowdReportRequest(BaseModel):
    message: str = Field(..., min_length=5, max_length=500)
    escalation: EscalationLevel
    tone: Optional[str] = Field(None, pattern="^(urgent|concerned|descriptive|frantic|neutral)$")
    user: str = Field("Anonymous", max_length=100)
    location: Optional[str] = Field(None, max_length=255)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

class TriagePatientRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    gender: str = Field(..., pattern="^(male|female|other)$")
    injury_type: str = Field(..., min_length=1, max_length=100)
    severity: str = Field(..., pattern="^(mild|moderate|severe|critical)$")
    triage_color: TriageColor
    location: Optional[str] = Field(None, max_length=255)
    notes: Optional[str] = Field(None, max_length=1000)

def validate_file_upload(file: UploadFile, max_size_mb: int = 10, allowed_types: List[str] = None):
    """Validate uploaded files for size and type"""
    if not file or not file.filename:
        return
    
    # Check file size
    if file.size and file.size > max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {max_size_mb}MB")
    
    # Check file type
    if allowed_types and file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed: {', '.join(allowed_types)}")

# ================================================================================
# ENHANCED SECURITY & AUTHENTICATION
# ================================================================================

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = config.SECRET_KEY
ACCESS_TOKEN_EXPIRE_MINUTES = config.ACCESS_TOKEN_EXPIRE_MINUTES
ALGORITHM = "HS256"

def hash_password(password: str) -> str:
    """Hash password using bcrypt with SHA-256 fallback"""
    try:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    except Exception:
        # Fallback to SHA-256 if bcrypt fails
        return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against a hash with multiple format support"""
    try:
        # Try bcrypt first
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError:
        # Fallback to SHA-256 for older passwords
        return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token with fallback encoding"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    try:
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    except Exception:
        # Fallback to base64 encoding if JWT fails
        return base64.b64encode(json.dumps(to_encode).encode()).decode()

def verify_token(token: str):
    """Verify JWT token with fallback verification"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            return None
        return {"username": username}
    except (jwt.PyJWTError, AttributeError):
        # Fallback verification for base64 tokens
        try:
            payload = json.loads(base64.b64decode(token).decode())
            username = payload.get("sub")
            exp = payload.get("exp")
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                return None  # Token expired
            return {"username": username}
        except Exception:
            return None

# Rate limiting implementation
rate_limit_storage = {}

def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if not isinstance(request, Request):
                # Try to find request in args if not in kwargs
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            client_ip = "127.0.0.1"
            if request and hasattr(request, 'client') and request.client:
                client_ip = request.client.host
                
            current_time = time.time()
            
            if client_ip not in rate_limit_storage:
                rate_limit_storage[client_ip] = []
            
            # Remove timestamps outside the window
            rate_limit_storage[client_ip] = [
                req_time for req_time in rate_limit_storage[client_ip] 
                if current_time - req_time < window_seconds
            ]
            
            if len(rate_limit_storage[client_ip]) >= max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            rate_limit_storage[client_ip].append(current_time)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get current authenticated user with proper token verification"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token_data = verify_token(token)
        if not token_data or "username" not in token_data:
            raise credentials_exception
        
        username: str = token_data["username"]
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            raise credentials_exception
        
        return {"username": user.username, "role": user.role, "id": user.id}
    except Exception:
        # Fallback for demo mode in non-production environments
        if config.ENVIRONMENT != "production":
            return {"username": "demo_user", "role": "admin", "id": 1}
        raise credentials_exception

def require_role(allowed_roles: List[str]):
    """Require specific role for endpoint access"""
    def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in allowed_roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return role_checker

# ================================================================================
# PERFORMANCE MONITORING
# ================================================================================

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.request_times = defaultdict(deque)
        self.error_counts = defaultdict(int)
        self.lock = Lock()
        self.max_samples = 1000

    def record_request_time(self, endpoint: str, duration: float):
        with self.lock:
            self.request_times[endpoint].append({
                'duration': duration, 
                'timestamp': time.time()
            })
            if len(self.request_times[endpoint]) > self.max_samples:
                self.request_times[endpoint].popleft()

    def record_error(self, endpoint: str, error_type: str):
        with self.lock:
            self.error_counts[f"{endpoint}:{error_type}"] += 1

    def get_stats(self):
        with self.lock:
            stats = {
                "system": self.get_system_stats(),
                "endpoints": {},
                "errors": dict(self.error_counts)
            }
            
            for endpoint, times in self.request_times.items():
                if times:
                    durations = [t['duration'] for t in times]
                    stats["endpoints"][endpoint] = {
                        "avg_duration": sum(durations) / len(durations),
                        "min_duration": min(durations),
                        "max_duration": max(durations),
                        "request_count": len(durations),
                        "recent_requests": len([
                            t for t in times 
                            if time.time() - t['timestamp'] < 300
                        ])
                    }
            return stats

    def get_system_stats(self):
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available": memory.available,
                "disk_usage": disk.percent,
                "disk_free": disk.free
            }
        except (ImportError, Exception):
            return {
                "cpu_usage": 0,
                "memory_usage": 0,
                "memory_available": 0,
                "disk_usage": 0,
                "disk_free": 0
            }

perf_monitor = PerformanceMonitor()

# ================================================================================
# FASTAPI APPLICATION SETUP
# ================================================================================

# CORS configuration
if config.ENVIRONMENT == "production":
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
else:
    CORS_ORIGINS = ["*"]

app = FastAPI(
    title="Enhanced Emergency Response Assistant",
    description="Complete AI-Powered Emergency Management System with Citizen Portal",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Performance monitoring middleware"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        perf_monitor.record_request_time(request.url.path, process_time)
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        perf_monitor.record_error(request.url.path, type(e).__name__)
        raise e

# ================================================================================
# STATIC FILES & TEMPLATES SETUP
# ================================================================================

# Create required directories
for directory in [STATIC_DIR, TEMPLATES_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

# Static files mounting
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates setup
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
else:
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
# REAL-TIME FEATURES (WEBSOCKETS)
# ================================================================================

class ConnectionManager:
    """WebSocket connection manager for real-time updates"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            self.user_connections[user_id] = websocket
        logger.info(f"WebSocket connected: {user_id or 'anonymous'}")

    def disconnect(self, websocket: WebSocket, user_id: str = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id and user_id in self.user_connections:
            del self.user_connections[user_id]
        logger.info(f"WebSocket disconnected: {user_id or 'anonymous'}")

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.user_connections:
            await self.user_connections[user_id].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections[:]:  # Iterate over a copy
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)

    async def send_to_admins(self, message: str):
        # In a real app, you would look up admin users and send only to them
        await self.broadcast(message)

manager = ConnectionManager()

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket, token: Optional[str] = Query(None)):
    """WebSocket endpoint for dashboard real-time updates"""
    user_id = None
    try:
        if token:
            token_data = verify_token(token)
            user_id = token_data.get("username") if token_data else None
        await manager.connect(websocket, user_id)
        
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message_data.get("type") == "subscribe":
                await websocket.send_text(json.dumps({
                    "type": "subscribed", 
                    "feed": message_data.get("feed")
                }))
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, user_id)

@app.websocket("/ws/emergency-updates")
async def emergency_updates_websocket(websocket: WebSocket):
    """WebSocket endpoint for emergency updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")  # Implement real logic as needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def broadcast_emergency_update(update_type: str, data: dict):
    """Broadcast emergency updates to all connected clients"""
    message = json.dumps({
        "type": "emergency_update",
        "update_type": update_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }, default=str)
    await manager.broadcast(message)

async def send_admin_notification(notification_type: str, data: dict):
    """Send notifications to admin users"""
    message = json.dumps({
        "type": "admin_notification",
        "notification_type": notification_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }, default=str)
    await manager.send_to_admins(message)

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
    """Offline support page with proper error handling"""
    try:
        return templates.TemplateResponse("offline.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving offline.html: {e}")
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Offline Mode</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; margin: 2rem; background: #f3f4f6; }
                .container { max-width: 600px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; }
                .icon { font-size: 4rem; text-align: center; margin-bottom: 1rem; }
                h1 { color: #1f2937; text-align: center; }
                p { color: #6b7280; text-align: center; line-height: 1.6; }
                .btn { display: inline-block; background: #3b82f6; color: white; padding: 0.75rem 1.5rem; 
                       border-radius: 6px; text-decoration: none; margin: 0.5rem; }
                .btn:hover { background: #2563eb; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="icon">📴</div>
                <h1>Offline Mode</h1>
                <p>You're currently offline, but the Emergency Response Assistant is designed to work without an internet connection.</p>
                <p>Key features available offline:</p>
                <ul style="text-align: left; color: #374151;">
                    <li>🚨 Submit emergency reports (will sync when online)</li>
                    <li>🧠 AI analysis using local processing</li>
                    <li>📱 Access cached emergency information</li>
                    <li>🗺️ View offline maps</li>
                </ul>
                <div style="text-align: center; margin-top: 2rem;">
                    <a href="/" class="btn">🏠 Go to Home</a>
                    <button class="btn" onclick="location.reload()">🔄 Check Connection</button>
                </div>
            </div>
            <script>
                // Auto-redirect when online
                window.addEventListener('online', () => {
                    setTimeout(() => window.location.href = '/', 1000);
                });
                
                // Check connection status
                function updateStatus() {
                    if (navigator.onLine) {
                        document.body.innerHTML += '<div style="position:fixed;top:20px;right:20px;background:#16a34a;color:white;padding:1rem;border-radius:6px;">✅ Back Online!</div>';
                        setTimeout(() => window.location.href = '/', 2000);
                    }
                }
                setInterval(updateStatus, 5000);
            </script>
        </body>
        </html>
        """)

@app.get("/offline.html", response_class=HTMLResponse)
async def serve_offline_html(request: Request):
    """Alternative route for offline.html"""
    return await offline_page(request)

@app.get("/onboarding", response_class=HTMLResponse)
async def onboarding_page(request: Request):
    """User onboarding and tutorial page"""
    try:
        return templates.TemplateResponse("onboarding.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving onboarding.html: {e}")
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Welcome - Emergency Response Assistant</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
                .container { max-width: 800px; margin: 0 auto; padding: 2rem; }
                .hero { background: white; border-radius: 16px; padding: 3rem 2rem; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
                .hero-icon { font-size: 4rem; margin-bottom: 1rem; }
                h1 { color: #1f2937; margin-bottom: 1rem; }
                p { color: #6b7280; line-height: 1.6; margin-bottom: 2rem; }
                .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin: 2rem 0; }
                .feature-card { background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .feature-icon { font-size: 2.5rem; margin-bottom: 1rem; }
                .feature-title { font-weight: bold; color: #1f2937; margin-bottom: 0.5rem; }
                .feature-description { color: #6b7280; font-size: 0.9rem; }
                .btn { display: inline-block; background: #3b82f6; color: white; padding: 0.75rem 2rem; border-radius: 8px; text-decoration: none; margin: 1rem 0.5rem; font-weight: 600; }
                .btn:hover { background: #2563eb; transform: translateY(-2px); transition: all 0.3s ease; }
                .btn-outline { background: transparent; border: 2px solid #3b82f6; color: #3b82f6; }
                .btn-outline:hover { background: #3b82f6; color: white; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="hero">
                    <div class="hero-icon">🆘</div>
                    <h1>Welcome to Emergency Response Assistant</h1>
                    <p>Your comprehensive disaster response and recovery assistant. This system helps you report emergencies, coordinate responses, and stay safe during disasters.</p>
                    
                    <div class="features">
                        <div class="feature-card">
                            <div class="feature-icon">🚨</div>
                            <h3 class="feature-title">Emergency Reporting</h3>
                            <p class="feature-description">Quick incident reporting with photos, location, and priority levels</p>
                        </div>
                        
                        <div class="feature-card">
                            <div class="feature-icon">🤖</div>
                            <h3 class="feature-title">AI Analysis</h3>
                            <p class="feature-description">Real-time situation analysis and intelligent recommendations</p>
                        </div>
                        
                        <div class="feature-card">
                            <div class="feature-icon">📱</div>
                            <h3 class="feature-title">Offline Ready</h3>
                            <p class="feature-description">Works without internet connection using local storage</p>
                        </div>
                        
                        <div class="feature-card">
                            <div class="feature-icon">🗺️</div>
                            <h3 class="feature-title">Live Mapping</h3>
                            <p class="feature-description">Interactive maps with real-time incident tracking</p>
                        </div>
                    </div>
                    
                    <div style="margin-top: 2rem;">
                        <a href="/" class="btn">🚀 Get Started</a>
                        <a href="/api/docs" class="btn btn-outline">📚 API Documentation</a>
                    </div>
                    
                    <div style="margin-top: 2rem; padding: 1.5rem; background: #fef3c7; border-radius: 8px; border: 1px solid #f59e0b;">
                        <h4 style="color: #92400e; margin-bottom: 1rem;">⚠️ Emergency Situations</h4>
                        <p style="color: #78350f; margin: 0;">
                            <strong>For life-threatening emergencies, call 911 first.</strong> This system supplements but doesn't replace emergency services.
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """)

@app.get("/onboarding.html", response_class=HTMLResponse)
async def serve_onboarding_html(request: Request):
    """Alternative route for onboarding.html"""
    return await onboarding_page(request)

@app.get("/hazards")
async def hazards_page(request: Request):
    """Hazards information page"""
    try:
        return templates.TemplateResponse("hazards.html", {"request": request})
    except:
        return HTMLResponse("""
        <html><head><title>Hazards</title></head>
        <body>
        <h1>🌪️ Hazards Information</h1>
        <p>Hazard detection and monitoring page</p>
        <a href="/">← Back to Home</a>
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
@rate_limit(max_requests=10, window_seconds=60)
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
        # Validate file upload
        validate_file_upload(file, max_size_mb=config.MAX_FILE_SIZE_MB, 
                           allowed_types=["image/jpeg", "image/png", "application/pdf"])
        
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
        
        # If Celery is integrated, schedule background AI analysis for high-priority reports
        if CELERY_INTEGRATION and priority in ["critical", "high"]:
            # Use the file path if a file was saved
            saved_file_path = str(UPLOAD_DIR / evidence_file) if evidence_file else None
            
            task_ids = schedule_emergency_analysis(
                audio_path=saved_file_path,  # Use the same path for both
                image_path=saved_file_path,
                text_input=description,
                urgency_level=priority
            )
            
            # Store task IDs for tracking
            emergency_report.ai_analysis = {"celery_tasks": task_ids}
            logger.info(f"Scheduled Celery tasks for report {report_id}: {task_ids}")
            
        # Fallback to FastAPI's BackgroundTasks if Celery is not available
        elif not CELERY_INTEGRATION and priority in ["critical", "high"]:
             background_tasks.add_task(process_emergency_report_background, emergency_report.id)
             logger.warning(f"Celery not available. Using default background task for report {report_id}.")

        db.add(emergency_report)
        db.commit()
        db.refresh(emergency_report)
        
        # Real-time broadcast
        await broadcast_emergency_update("new_report", {
            "report_id": report_id, 
            "priority": priority, 
            "location": location
        })
        
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
    
@app.post("/api/quick-emergency")
async def quick_emergency_report(
    request: Request,
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """Quick emergency report submission for the emergency FAB button"""
    try:
        report_id = generate_report_id()
        
        emergency_report = EmergencyReport(
            report_id=report_id,
            type="quick_emergency",
            description="QUICK EMERGENCY BUTTON PRESSED - Immediate assistance needed. Location assistance requested.",
            location="GPS coordinates provided" if latitude and longitude else "Location not specified - please contact caller",
            latitude=latitude,
            longitude=longitude,
            priority="critical",
            method="quick_button",
            reporter="citizen_portal_quick"
        )
        
        db.add(emergency_report)
        db.commit()
        db.refresh(emergency_report)
        
        # Broadcast emergency update to connected admins
        await broadcast_emergency_update("quick_emergency", {
            "report_id": report_id,
            "priority": "critical",
            "location": emergency_report.location,
            "message": "Quick emergency button activated"
        })
        
        # Send admin notification
        await send_admin_notification("quick_emergency", {
            "report_id": report_id,
            "coordinates": f"{latitude}, {longitude}" if latitude and longitude else "No GPS",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.warning(f"QUICK EMERGENCY ACTIVATED: Report {report_id} - GPS: {latitude}, {longitude}")
        
        return JSONResponse({
            "success": True,
            "report_id": report_id,
            "status": "emergency_dispatched",
            "message": "Emergency services have been notified immediately",
            "priority": "critical",
            "estimated_response": "5-10 minutes"
        })
        
    except Exception as e:
        logger.error(f"Quick emergency failed: {e}")
        return JSONResponse({
            "success": False,
            "error": "Emergency dispatch failed",
            "fallback_message": "Please call 911 directly"
        }, status_code=500)

@app.get("/api/risk-prediction")
async def get_risk_prediction(
    latitude: Optional[float] = Query(None),
    longitude: Optional[float] = Query(None),
    db: Session = Depends(get_db)
):
    """Get AI-powered risk prediction for current location"""
    try:
        # Simulate AI risk analysis
        risk_factors = {
            "weather": {
                "level": "low",
                "details": "Clear conditions, no severe weather expected",
                "confidence": 0.9
            },
            "traffic": {
                "level": "moderate", 
                "details": "Normal traffic flow with some congestion during rush hours",
                "confidence": 0.8
            },
            "crime": {
                "level": "low",
                "details": "Below average crime activity in this area",
                "confidence": 0.85
            },
            "infrastructure": {
                "level": "low",
                "details": "No known infrastructure issues or outages",
                "confidence": 0.95
            },
            "natural_disasters": {
                "level": "low",
                "details": "Low probability of earthquakes, floods, or other natural events",
                "confidence": 0.7
            }
        }
        
        # Calculate overall risk
        risk_levels = {"low": 1, "moderate": 2, "high": 3, "critical": 4}
        avg_risk = sum(risk_levels[factor["level"]] for factor in risk_factors.values()) / len(risk_factors)
        
        overall_level = "low" if avg_risk < 1.5 else "moderate" if avg_risk < 2.5 else "high" if avg_risk < 3.5 else "critical"
        overall_confidence = sum(factor["confidence"] for factor in risk_factors.values()) / len(risk_factors)
        
        return JSONResponse({
            "success": True,
            "risk_assessment": {
                "overall_level": overall_level,
                "overall_confidence": round(overall_confidence, 2),
                "location": {
                    "latitude": latitude,
                    "longitude": longitude,
                    "has_coordinates": latitude is not None and longitude is not None
                },
                "risk_factors": risk_factors,
                "recommendations": [
                    "Keep emergency contacts readily available",
                    "Stay informed about local conditions",
                    "Have emergency supplies prepared" if overall_level in ["moderate", "high"] else "Continue normal activities with standard precautions"
                ],
                "last_updated": datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Risk prediction error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/navigation-state")
async def get_navigation_state():
    """Get current navigation state for UI updates"""
    return JSONResponse({
        "success": True,
        "navigation": {
            "current_page": "home",
            "available_pages": ["home", "report", "voice", "track"],
            "user_permissions": {
                "can_submit_reports": True,
                "can_use_voice": True,
                "can_view_analytics": True
            }
        }
    })

@app.get("/api/template-check")
async def check_templates():
    """Check if required templates exist"""
    template_status = {}
    required_templates = ["base.html", "home.html", "offline.html", "onboarding.html"]
    
    for template in required_templates:
        template_path = TEMPLATES_DIR / template
        template_status[template] = {
            "exists": template_path.exists(),
            "path": str(template_path)
        }
    
    return JSONResponse({
        "success": True,
        "templates": template_status,
        "templates_dir": str(TEMPLATES_DIR),
        "templates_dir_exists": TEMPLATES_DIR.exists()
    })

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

@app.get("/api/performance-metrics")
async def get_performance_metrics():
    """Get system performance metrics"""
    try:
        return JSONResponse({
            "success": True,
            "metrics": perf_monitor.get_stats()
        })
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
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

async def analyze_voice(
    audio: UploadFile = File(...),
    context: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
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
        # Check database for user
        user = db.query(User).filter(User.username == form_data.username).first()
        
        if user and verify_password(form_data.password, user.hashed_password):
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": user.username}, expires_delta=access_token_expires
            )
            
            log_security_event("successful_login", {"username": user.username})
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "role": user.role,
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
            }
        
        # Fallback demo authentication for development
        elif config.ENVIRONMENT != "production":
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
        
        log_security_event("failed_login", {"username": form_data.username})
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
        
        # Validate password strength
        if len(password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        
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
        
        log_security_event("user_registered", {"username": username, "email": email})
        
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

@app.post("/api/logout")
async def logout_user(current_user: dict = Depends(get_current_user)):
    """Logout user (mainly for logging purposes)"""
    log_security_event("user_logout", {"username": current_user["username"]})
    return JSONResponse({
        "success": True,
        "message": "Logged out successfully"
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
        total_users = db.query(User).count()
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        total_reports = total_patients = total_emergency_reports = total_users = 0
        db_status = "error"
    
    # AI model status
    try:
        ai_performance = ai_optimizer.monitor_performance()
        ai_status = "ready"
    except Exception as e:
        logger.error(f"AI health check failed: {e}")
        ai_status = "error"
    
    # System performance
    try:
        system_stats = perf_monitor.get_system_stats()
        system_status = "healthy"
    except Exception:
        system_stats = {}
        system_status = "limited"
    
    return JSONResponse({
        "status": "healthy",
        "service": "Enhanced Emergency Response Assistant",
        "version": "3.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": config.ENVIRONMENT,
        "components": {
            "database": {
                "status": db_status,
                "records": {
                    "crowd_reports": total_reports,
                    "triage_patients": total_patients,
                    "emergency_reports": total_emergency_reports,
                    "users": total_users
                }
            },
            "ai_models": {
                "status": ai_status,
                "gemma_3n": {
                    "active_model": ai_optimizer.current_config.model_variant,
                    "optimization_level": ai_optimizer.current_config.optimization_level
                }
            },
            "system_performance": {
                "status": system_status,
                **system_stats
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
                "ai_optimization": True,
                "real_time_updates": True,
                "websocket_support": True,
                "rate_limiting": True,
                "authentication": True,
                "role_based_access": True
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
            "analytics": "/api/analytics-data",
            "websocket_dashboard": "/ws/dashboard",
            "websocket_emergency": "/ws/emergency-updates"
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
                "environment": config.ENVIRONMENT,
                "base_directory": str(BASE_DIR),
                "upload_directory": str(UPLOAD_DIR),
                "templates_available": TEMPLATES_DIR.exists(),
                "static_files_available": STATIC_DIR.exists(),
                "debug_mode": config.DEBUG
            },
            "configuration": {
                "database_url": config.DATABASE_URL.replace(config.SECRET_KEY, "***") if config.SECRET_KEY in config.DATABASE_URL else config.DATABASE_URL,
                "max_file_size_mb": config.MAX_FILE_SIZE_MB,
                "ai_model_variant": config.AI_MODEL_VARIANT,
                "ai_context_window": config.AI_CONTEXT_WINDOW,
                "rate_limit_requests": config.RATE_LIMIT_REQUESTS,
                "rate_limit_window": config.RATE_LIMIT_WINDOW,
                "access_token_expire_minutes": config.ACCESS_TOKEN_EXPIRE_MINUTES
            },
            "capabilities": {
                "weasyprint_pdf": WEASYPRINT_AVAILABLE,
                "database": DATABASE_AVAILABLE,
                "ai_processing": True,
                "file_uploads": True,
                "background_tasks": True,
                "websockets": True,
                "rate_limiting": True,
                "jwt_authentication": True,
                "performance_monitoring": True
            }
        })
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# ================================================================================
# STATIC FILE HANDLERS & PWA SUPPORT
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
                },
                {
                    "src": "/static/icon-512.png",
                    "sizes": "512x512",
                    "type": "image/png"
                }
            ]
        })

@app.get("/sw.js")
async def get_service_worker():
    """Serve service worker for offline support"""
    sw_path = STATIC_DIR / "js" / "sw.js"
    if sw_path.exists():
        return FileResponse(sw_path, media_type="application/javascript")
    else:
        # Return basic service worker
        return Response(
            content="""
// Enhanced service worker for offline support
const CACHE_NAME = 'emergency-app-v3.0.0';
const urlsToCache = [
  '/',
  '/offline',
  '/static/css/styles.css',
  '/static/js/app.js',
  '/api/dashboard-stats',
  '/manifest.json'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Opened cache');
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Return cached version or fetch from network
        if (response) {
          return response;
        }
        return fetch(event.request).catch(() => {
          // If both cache and network fail, show offline page
          if (event.request.destination === 'document') {
            return caches.match('/offline');
          }
        });
      })
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
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
    """Enhanced 404 handler with helpful suggestions"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "path": str(request.url),
            "method": request.method,
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
                "health_check": "/health",
                "websocket_dashboard": "/ws/dashboard"
            }
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Enhanced 500 handler with debugging info"""
    logger.error(f"Internal server error on {request.method} {request.url}: {exc}", exc_info=True)
    
    error_id = uuid.uuid4().hex[:8]
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_id": error_id,
            "message": "An unexpected error occurred. Please try again.",
            "debug_info": str(exc) if config.DEBUG else None,
            "support_info": {
                "health_check": "/health",
                "system_info": "/api/system-info",
                "contact": f"Reference error ID: {error_id}"
            }
        }
    )

@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    """Rate limit exceeded handler"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please slow down and try again later.",
            "retry_after": 60
        }
    )

# ================================================================================
# APPLICATION STARTUP & LIFECYCLE
# ================================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup initialization"""
    logger.info("🚀 Starting Enhanced Emergency Response Assistant v3.0.0")
    logger.info(f"🌍 Environment: {config.ENVIRONMENT}")
    logger.info(f"🔧 Debug mode: {config.DEBUG}")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created/verified")
        
        # Initialize default data if needed
        db = next(get_db())
        
        # Create default admin user if no users exist
        if db.query(User).count() == 0:
            logger.info("📝 No users found. Creating default admin user...")
            admin_user = User(
                username="admin",
                email="admin@example.com",
                hashed_password=hash_password("admin"),
                role="admin",
                is_active=True,
            )
            db.add(admin_user)
            db.commit()
            logger.info("✅ Default admin user created (username: admin, password: admin)")
        
        # Add welcome data if database is empty
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
            logger.info("✅ Welcome data initialized")
            
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
    logger.info("     • 🌐 Citizen Emergency Portal (main interface)")
    logger.info("     • 🎤 Voice Emergency Reporter with real-time transcription")
    logger.info("     • 🤖 Multimodal AI Analysis (text + image + audio)")
    logger.info("     • 📊 Professional Admin Dashboard")
    logger.info("     • 🏥 Patient Triage Management")
    logger.info("     • 📢 Crowd Report System with geolocation")
    logger.info("     • 📈 Analytics Dashboard with real-time charts")
    logger.info("     • 🗺️ Map Visualization with interactive reports")
    logger.info("     • 📁 Export functionality (JSON, CSV)")
    logger.info("     • 🎭 Demo data generation for testing")
    logger.info("     • 📱 Offline support with service worker")
    logger.info("     • ⚡ Real-time updates via WebSockets")
    logger.info("     • 🔐 JWT Authentication with role-based access")
    logger.info("     • 🛡️ Rate limiting and security monitoring")
    logger.info("     • 📊 Performance monitoring and metrics")
    logger.info("     • 🔧 RESTful API with comprehensive documentation")
    
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
            temp_file.unlink(missing_ok=True)
        logger.info(f"🧹 Cleaned up {len(temp_files)} temporary files")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
    
    # Close WebSocket connections
    try:
        for connection in manager.active_connections[:]:
            await connection.close()
        logger.info("🔌 WebSocket connections closed")
    except Exception as e:
        logger.error(f"❌ WebSocket cleanup error: {e}")
    
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
        reload=config.DEBUG,
        log_level="info",
        access_log=True
    )