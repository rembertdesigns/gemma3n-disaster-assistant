# ================================================================================
# IMPORTS & DEPENDENCIES
# ================================================================================

from fastapi import (
    FastAPI, Request, Form, UploadFile, File, Depends, HTTPException,
    Body, Query
)
from fastapi.responses import (
    HTMLResponse, FileResponse, JSONResponse, RedirectResponse,
    StreamingResponse, Response
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
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
from app.models import CrowdReport, TriagePatient, Base

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

# Static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

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

# ================================================================================
# AUTHENTICATION ROUTES
# ================================================================================

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """User authentication endpoint"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me")
async def read_current_user(user: dict = Depends(get_current_user)):
    """Get current user information"""
    return {"username": user["username"], "role": user["role"]}

# ================================================================================
# MAIN PAGE ROUTES
# ================================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """Home page with AI analysis"""
    return templates.TemplateResponse("home.html", {"request": request, "result": None})

@app.get("/hazards", response_class=HTMLResponse)
async def serve_hazard_page(request: Request):
    """Hazard detection page"""
    return templates.TemplateResponse("hazards.html", {"request": request, "result": None})

@app.get("/generate", response_class=HTMLResponse)
async def serve_generate_page(request: Request):
    """Report generation page"""
    return templates.TemplateResponse("generate.html", {"request": request})

@app.get("/live-generate", response_class=HTMLResponse)
async def serve_live_generate_page(request: Request):
    """Live report builder"""
    return templates.TemplateResponse("live_generate.html", {"request": request})

@app.get("/map-reports", response_class=HTMLResponse)
async def map_reports_page(request: Request):
    """Enhanced map reports page with demo features"""
    return templates.TemplateResponse("map_reports.html", {"request": request})

@app.get("/map-snapshot", response_class=HTMLResponse)
async def map_snapshot_view(
    request: Request,
    report_id: Optional[int] = Query(None, description="Specific report ID to show"),
    lat: Optional[float] = Query(None, description="Manual latitude"),
    lon: Optional[float] = Query(None, description="Manual longitude"),
    db: Session = Depends(get_db)
):
    """Enhanced map snapshot with real report data or manual coordinates"""
    try:
        if report_id:
            # Get specific report from database
            report = db.query(CrowdReport).filter(CrowdReport.id == report_id).first()
            if report and report.latitude and report.longitude:
                return templates.TemplateResponse("map_snapshot.html", {
                    "request": request,
                    "report_id": report.id,
                    "latitude": report.latitude,
                    "longitude": report.longitude,
                    "report_message": report.message,
                    "report_user": report.user or "Anonymous",
                    "report_escalation": report.escalation,
                    "report_timestamp": report.timestamp,
                    "report_location": report.location,
                    "has_real_data": True
                })
            else:
                logger.warning(f"Report {report_id} not found or has no coordinates")
                
        # Use manual coordinates if provided
        if lat is not None and lon is not None:
            return templates.TemplateResponse("map_snapshot.html", {
                "request": request,
                "report_id": f"Manual-{int(lat*1000)}{int(lon*1000)}",
                "latitude": lat,
                "longitude": lon,
                "report_message": "Manual coordinate plotting",
                "report_user": "System",
                "report_escalation": "low",
                "report_timestamp": datetime.utcnow().isoformat(),
                "report_location": f"Coordinates: {lat}, {lon}",
                "has_real_data": False
            })
            
        # Get latest report with coordinates as default
        latest_report = db.query(CrowdReport).filter(
            CrowdReport.latitude.isnot(None),
            CrowdReport.longitude.isnot(None)
        ).order_by(CrowdReport.timestamp.desc()).first()
            
        if latest_report:
            return templates.TemplateResponse("map_snapshot.html", {
                "request": request,
                "report_id": latest_report.id,
                "latitude": latest_report.latitude,
                "longitude": latest_report.longitude,
                "report_message": latest_report.message,
                "report_user": latest_report.user or "Anonymous",
                "report_escalation": latest_report.escalation,
                "report_timestamp": latest_report.timestamp,
                "report_location": latest_report.location,
                "has_real_data": True
            })
            
        # Fallback to San Francisco demo coordinates
        return templates.TemplateResponse("map_snapshot.html", {
            "request": request,
            "report_id": "DEMO",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "report_message": "Demo location - San Francisco City Hall",
            "report_user": "Demo System",
            "report_escalation": "low",
            "report_timestamp": datetime.utcnow().isoformat(),
            "report_location": "San Francisco, CA (Demo)",
            "has_real_data": False
        })
            
    except Exception as e:
        logger.error(f"❌ Error loading map snapshot: {str(e)}")
        # Error fallback
        return templates.TemplateResponse("map_snapshot.html", {
            "request": request,
            "report_id": "ERROR",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "report_message": f"Error loading data: {str(e)}",
            "report_user": "System",
            "report_escalation": "low",
            "report_timestamp": datetime.utcnow().isoformat(),
            "report_location": "Error - Using default coordinates",
            "has_real_data": False,
            "error": str(e)
        })

@app.get("/map-snapshot/{report_id}", response_class=HTMLResponse)
async def map_snapshot_by_id(report_id: int, request: Request, db: Session = Depends(get_db)):
    """Direct map snapshot for a specific report ID"""
    return await map_snapshot_view(request, report_id=report_id, db=db)

@app.get("/offline.html", response_class=HTMLResponse)
async def offline_page(request: Request):
    """Offline support page"""
    return templates.TemplateResponse("offline.html", {"request": request})

@app.get("/submit-report", response_class=HTMLResponse)
async def submit_report_page(request: Request):
    """Report submission page"""
    return templates.TemplateResponse("submit_report.html", {"request": request})

@app.get("/manifest.json")
async def manifest():
    """Serve PWA manifest"""
    return FileResponse("static/manifest.json", media_type="application/json")

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Risk prediction page"""
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/sw.js")
async def service_worker():
    """Serve service worker from root path"""
    return FileResponse("static/js/sw.js", media_type="application/javascript")

@app.get("/triage-form", response_class=HTMLResponse)
async def triage_form_page(request: Request):
    """Triage assessment form"""
    return templates.TemplateResponse("triage_form.html", {"request": request})

@app.get("/test-offline", response_class=HTMLResponse)
async def test_offline_page(request: Request):
    """Offline testing suite for service worker validation"""
    return templates.TemplateResponse("test-offline.html", {"request": request})

# ================================================================================
# NEW EMERGENCY RESPONSE SYSTEM PAGES
# ================================================================================

@app.get("/sync-status", response_class=HTMLResponse)
async def sync_status_page(request: Request):
    """Sync status and offline data management"""
    return templates.TemplateResponse("sync_status.html", {"request": request})

@app.get("/device-status", response_class=HTMLResponse)
async def device_status_page(request: Request):
    """Device monitoring and sensor status"""
    return templates.TemplateResponse("device_status.html", {"request": request})

@app.get("/report-archive", response_class=HTMLResponse)
async def report_archive_page(request: Request):
    """Historical report archive and management"""
    return templates.TemplateResponse("report_archive.html", {"request": request})

@app.get("/onboarding", response_class=HTMLResponse)
async def onboarding_page(request: Request):
    """User onboarding and system tutorial"""
    return templates.TemplateResponse("onboarding.html", {"request": request})

@app.get("/admin-dashboard", response_class=HTMLResponse)
async def admin_dashboard_page(request: Request, user: dict = Depends(require_role(["admin"]))):
    """Comprehensive admin dashboard (protected route)"""
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "user": user,
        "current_time": datetime.utcnow()
    })

@app.get("/feedback", response_class=HTMLResponse)
async def feedback_page(request: Request):
    """User feedback and bug reporting system"""
    return templates.TemplateResponse("feedback.html", {"request": request})

@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    """Comprehensive help documentation"""
    return templates.TemplateResponse("help.html", {"request": request})

# ================================================================================
# GEMMA 3N ENHANCED PAGES - ADD TO YOUR EXISTING api.py
# Add these routes after the existing page routes section
# ================================================================================

# Add after line ~400 in the "MAIN PAGE ROUTES" section:

@app.get("/voice-emergency-reporter", response_class=HTMLResponse)
async def voice_emergency_reporter_page(request: Request):
    """Voice Emergency Reporter - Hands-free reporting with Gemma 3n AI"""
    return templates.TemplateResponse("voice-emergency-reporter.html", {"request": request})

@app.get("/multimodal-damage-assessment", response_class=HTMLResponse)
async def multimodal_damage_assessment_page(request: Request):
    """Multimodal Damage Assessment - AI-powered analysis using video, images, and audio"""
    return templates.TemplateResponse("multimodal-damage-assessment.html", {"request": request})

@app.get("/context-intelligence-dashboard", response_class=HTMLResponse)
async def context_intelligence_dashboard_page(
    request: Request, 
    user: dict = Depends(require_role(["admin", "responder"]))
):
    """Context Intelligence Dashboard - Deep situation analysis with 128K context (Protected route)"""
    return templates.TemplateResponse("context-intelligence-dashboard.html", {
        "request": request,
        "user": user,
        "current_time": datetime.utcnow()
    })

@app.get("/adaptive-ai-settings", response_class=HTMLResponse)
async def adaptive_ai_settings_page(request: Request):
    """Adaptive AI Settings - Optimize Gemma 3n performance for device and use case"""
    return templates.TemplateResponse("adaptive-ai-settings.html", {"request": request})

# ================================================================================
# DASHBOARD ROUTES
# ================================================================================

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    """Admin dashboard with real database statistics (Demo Mode)"""
    try:
        logger.info("🔧 Loading admin dashboard in DEMO MODE")
            
        # Get real data from database
        all_reports = db.query(CrowdReport).all()
        all_patients = db.query(TriagePatient).all()
            
        # Filter today's data
        today = datetime.utcnow().date()
        today_reports = [r for r in all_reports if r.timestamp and 
                            datetime.fromisoformat(r.timestamp.replace('Z', '+00:00')).date() == today]
        today_patients = [p for p in all_patients if p.created_at and p.created_at.date() == today]
            
        # Calculate statistics
        active_patients = [p for p in all_patients if p.status == "active"]
        critical_reports = [r for r in all_reports if r.escalation == "critical"]
        critical_patients = [p for p in active_patients if p.triage_color == "red" or p.severity == "critical"]
        avg_severity = calculate_avg_severity(all_patients)
        active_users = 3 + len(set([r.user for r in all_reports if r.user and r.user != "Anonymous"]))
            
        # Build comprehensive stats
        stats = {
            "total_reports": len(all_reports),
            "active_users": active_users,
            "avg_severity": avg_severity,
            "reports_today": len(today_reports),
            "total_patients": len(all_patients),
            "active_patients": len(active_patients),
            "patients_today": len(today_patients),
            "critical_reports": len(critical_reports),
            "critical_patients": len(critical_patients),
            "system_uptime": "12h 34m",
            "last_report_time": all_reports[-1].timestamp if all_reports else None
        }
            
        # Get recent reports and priority patients
        recent_reports = db.query(CrowdReport).order_by(CrowdReport.timestamp.desc()).limit(5).all()
        priority_patients = sorted(active_patients, key=lambda p: (p.priority_score if hasattr(p, 'priority_score') else 0, -p.id))[:5]
            
        logger.info(f"✅ Admin dashboard loaded: {stats['total_reports']} reports, {stats['total_patients']} patients")
            
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "username": "Demo Administrator",
            "role": "ADMIN",
            "stats": stats,
            "recent_reports": recent_reports,
            "priority_patients": priority_patients,
            "current_time": datetime.utcnow(),
            "demo_mode": True,
            "user_info": {
                "full_role": "admin",
                "login_time": datetime.utcnow().strftime("%H:%M"),
                "access_level": "Demo Administrative Access"
            }
        })
            
    except Exception as e:
        logger.error(f"❌ Error loading admin dashboard: {str(e)}")
            
        # Fallback with safe defaults
        empty_stats = {
            "total_reports": 0, "active_users": 0, "avg_severity": 0.0,
            "reports_today": 0, "total_patients": 0, "active_patients": 0,
            "patients_today": 0, "critical_reports": 0, "critical_patients": 0,
            "system_uptime": "Unknown", "last_report_time": None
        }
            
        return templates.TemplateResponse("admin.html", {
            "request": request, "username": "Demo User", "role": "ADMIN",
            "stats": empty_stats, "recent_reports": [], "priority_patients": [],
            "current_time": datetime.utcnow(), "error": f"Dashboard error: {str(e)}",
            "demo_mode": True
        })

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_dashboard(
    request: Request,
    timeframe: str = Query("7d", description="Time range: 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Enhanced analytics dashboard with real-time data visualizations"""
    try:
        logger.info(f"📊 Loading analytics dashboard (timeframe: {timeframe})")
            
        # Get all data
        all_reports = db.query(CrowdReport).all()
        all_patients = db.query(TriagePatient).all()
            
        # Calculate timeframe filter
        now = datetime.utcnow()
        if timeframe == "24h":
            cutoff = now - timedelta(hours=24)
            timeframe_label = "Last 24 Hours"
        elif timeframe == "30d":
            cutoff = now - timedelta(days=30)
            timeframe_label = "Last 30 Days"
        else:  # default 7d
            cutoff = now - timedelta(days=7)
            timeframe_label = "Last 7 Days"
            
        # Filter data by timeframe
        recent_reports = [r for r in all_reports if r.timestamp and 
                            datetime.fromisoformat(r.timestamp.replace('Z', '+00:00')) >= cutoff]
        recent_patients = [p for p in all_patients if p.created_at and p.created_at >= cutoff]
            
        # Generate analytics data
        report_trends = {
            "total_reports": len(recent_reports),
            "daily_average": round(len(recent_reports) / max(1, (now - cutoff).days), 1),
            "escalation_breakdown": {
                "critical": len([r for r in recent_reports if r.escalation == "critical"]),
                "high": len([r for r in recent_reports if r.escalation == "high"]),
                "moderate": len([r for r in recent_reports if r.escalation == "moderate"]),
                "low": len([r for r in recent_reports if r.escalation == "low"])
            },
            "tone_analysis": {
                "urgent": len([r for r in recent_reports if r.tone == "urgent"]),
                "frantic": len([r for r in recent_reports if r.tone == "frantic"]),
                "helpless": len([r for r in recent_reports if r.tone == "helpless"]),
                "descriptive": len([r for r in recent_reports if r.tone == "descriptive"])
            }
        }
            
        triage_insights = {
            "total_patients": len(recent_patients),
            "color_distribution": {
                "red": len([p for p in recent_patients if p.triage_color == "red"]),
                "yellow": len([p for p in recent_patients if p.triage_color == "yellow"]),
                "green": len([p for p in recent_patients if p.triage_color == "green"]),
                "black": len([p for p in recent_patients if p.triage_color == "black"])
            },
            "severity_trend": {
                "critical": len([p for p in recent_patients if p.severity == "critical"]),
                "severe": len([p for p in recent_patients if p.severity == "severe"]),
                "moderate": len([p for p in recent_patients if p.severity == "moderate"]),
                "mild": len([p for p in recent_patients if p.severity == "mild"])
            },
            "average_severity": calculate_avg_severity(recent_patients)
        }
            
        ai_metrics = {
            "sentiment_accuracy": 87.3,
            "triage_confidence": 92.1,
            "auto_classifications": len([r for r in recent_reports if r.tone]),
            "manual_overrides": 3,
            "processing_speed": "0.24s",
        }
            
        trend_data = generate_trend_data(recent_reports, recent_patients, cutoff, now)
            
        kpis = {
            "response_efficiency": calculate_response_efficiency(recent_patients),
            "critical_ratio": round((triage_insights["color_distribution"]["red"] / max(1, len(recent_patients))) * 100, 1),
            "system_utilization": min(100, len(recent_reports) + len(recent_patients)),
            "geographic_coverage": len(set([r.location for r in recent_reports if r.location])),
        }
            
        logger.info(f"✅ Analytics loaded: {report_trends['total_reports']} reports, {triage_insights['total_patients']} patients")
            
        return templates.TemplateResponse("analytics_dashboard.html", {
            "request": request, "timeframe": timeframe, "timeframe_label": timeframe_label,
            "report_trends": report_trends, "triage_insights": triage_insights,
            "ai_metrics": ai_metrics, "trend_data": trend_data, "kpis": kpis,
            "current_time": now, "total_all_time": {"reports": len(all_reports), "patients": len(all_patients)}
        })
            
    except Exception as e:
        logger.error(f"❌ Error loading analytics dashboard: {str(e)}")
            
        # Return error page with safe defaults
        return templates.TemplateResponse("analytics_dashboard.html", {
            "request": request, "timeframe": timeframe, "timeframe_label": "Error",
            "error": str(e), "current_time": datetime.utcnow(),
            "report_trends": {"total_reports": 0, "daily_average": 0, "escalation_breakdown": {}, "tone_analysis": {}},
            "triage_insights": {"total_patients": 0, "color_distribution": {}, "severity_trend": {}, "average_severity": 0},
            "ai_metrics": {"sentiment_accuracy": 0, "triage_confidence": 0, "auto_classifications": 0, "manual_overrides": 0},
            "kpis": {"response_efficiency": 0, "critical_ratio": 0, "system_utilization": 0, "geographic_coverage": 0}
        })

# ================================================================================
# CROWD REPORTS - PAGES
# ================================================================================

@app.get("/view-reports", response_class=HTMLResponse)
async def view_reports(
    request: Request,
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """View crowd reports with filtering"""
    try:
        query = db.query(CrowdReport)
            
        if tone:
            query = query.filter(CrowdReport.tone == tone)
        if escalation:
            query = query.filter(CrowdReport.escalation == escalation)
        if keyword:
            query = query.filter(CrowdReport.message.ilike(f"%{keyword}%"))
            
        reports = query.order_by(CrowdReport.timestamp.desc()).all()
            
        logger.info(f"📋 Loaded {len(reports)} crowd reports")
            
        return templates.TemplateResponse("view-reports.html", {
            "request": request, "reports": reports,
            "tone": tone, "escalation": escalation, "keyword": keyword
        })
            
    except Exception as e:
        logger.error(f"❌ Error loading reports: {str(e)}")
        return templates.TemplateResponse("view-reports.html", {
            "request": request, "reports": [], "error": str(e)
        })

@app.get("/submit-crowd-report", response_class=HTMLResponse)
async def submit_crowd_report_form(request: Request):
    """Crowd report submission form"""
    return templates.TemplateResponse("submit-crowd-report.html", {"request": request})

@app.get("/crowd-reports", response_class=HTMLResponse)
async def view_crowd_reports(
    request: Request,
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Enhanced crowd reports view"""
    try:
        query = db.query(CrowdReport)

        if tone:
            query = query.filter(CrowdReport.tone == tone)
        if escalation:
            query = query.filter(CrowdReport.escalation == escalation)
        if keyword:
            query = query.filter(CrowdReport.message.ilike(f"%{keyword}%"))

        reports = query.order_by(CrowdReport.timestamp.desc()).all()
            
        logger.info(f"📋 Crowd reports loaded: {len(reports)} reports found")

        return templates.TemplateResponse("crowd_reports.html", {
            "request": request, "reports": reports, "tone": tone,
            "escalation": escalation, "keyword": keyword, "current_time": datetime.utcnow()
        })
            
    except Exception as e:
        logger.error(f"❌ Error loading crowd reports: {str(e)}")
        return templates.TemplateResponse("crowd_reports.html", {
            "request": request, "reports": [], "tone": tone, "escalation": escalation,
            "keyword": keyword, "current_time": datetime.utcnow(), "error": str(e)
        })

@app.get("/reports", response_class=HTMLResponse)
async def reports_redirect(request: Request):
    """Redirect /reports to /view-reports for compatibility"""
    return RedirectResponse(url="/view-reports", status_code=301)

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
    db: Session = Depends(get_db)
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
            latitude=latitude, longitude=longitude
        )
            
        db.add(new_report)
        db.commit()
        db.refresh(new_report)
            
        logger.info(f"✅ Crowd report saved: ID={new_report.id}, escalation={new_report.escalation}")
            
        return RedirectResponse(url="/view-reports", status_code=303)
            
    except Exception as e:
        logger.error(f"❌ Failed to insert crowd report: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Error saving report")

# ================================================================================
# CROWD REPORTS - API ENDPOINTS
# ================================================================================

@app.get("/api/crowd-report-locations", response_class=JSONResponse)
async def crowd_report_locations_enhanced(
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None),
    timeRange: Optional[str] = Query(None),
    include_media: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Enhanced API for crowd reports with geolocation, filtering, and time range support"""
    try:
        query = db.query(CrowdReport).filter(
            CrowdReport.latitude.isnot(None),
            CrowdReport.longitude.isnot(None)
        )
            
        # Apply filters
        if tone:
            query = query.filter(CrowdReport.tone == tone)
        if escalation:
            query = query.filter(CrowdReport.escalation == escalation)
            
        # Apply time range filter
        if timeRange:
            cutoff_time = datetime.utcnow()
            if timeRange == "1h":
                cutoff_time -= timedelta(hours=1)
            elif timeRange == "24h":
                cutoff_time -= timedelta(hours=24)
            elif timeRange == "7d":
                cutoff_time -= timedelta(days=7)
                
            cutoff_iso = cutoff_time.isoformat()
            query = query.filter(CrowdReport.timestamp >= cutoff_iso)
            
        reports = query.order_by(CrowdReport.timestamp.desc()).all()
            
        reports_data = []
        for report in reports:
            report_dict = {
                "id": report.id, "message": report.message, "user": report.user,
                "timestamp": report.timestamp, "tone": report.tone, "escalation": report.escalation,
                "location": report.location, "latitude": float(report.latitude), "longitude": float(report.longitude),
                "time_ago": get_time_ago(report.timestamp) if report.timestamp else "Unknown"
            }
                
            if include_media:
                report_dict["has_image"] = False
                report_dict["has_audio"] = False
                report_dict["media_count"] = 0
                
            reports_data.append(report_dict)
            
        logger.info(f"📍 Enhanced locations API: returned {len(reports_data)} reports")
            
        return JSONResponse(content={
            "success": True, "reports": reports_data, "count": len(reports_data),
            "filters_applied": {"tone": tone, "escalation": escalation, "timeRange": timeRange, "include_media": include_media},
            "generated_at": datetime.utcnow().isoformat()
        })
            
    except Exception as e:
        logger.error(f"❌ Error getting enhanced crowd report locations: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/api/community-stats", response_class=JSONResponse)
async def get_community_stats(db: Session = Depends(get_db)):
    """Get community statistics for dashboard widgets"""
    try:
        all_reports = db.query(CrowdReport).all()
        today_reports = [r for r in all_reports if r.timestamp and 
                            datetime.fromisoformat(r.timestamp.replace('Z', '+00:00')).date() == datetime.utcnow().date()]
            
        stats = {
            "total_reports": len(all_reports), "reports_today": len(today_reports),
            "critical_reports": len([r for r in all_reports if r.escalation == "critical"]),
            "active_locations": len(set([r.location for r in all_reports if r.location])),
            "escalation_breakdown": {
                "critical": len([r for r in all_reports if r.escalation == "critical"]),
                "high": len([r for r in all_reports if r.escalation == "high"]),
                "moderate": len([r for r in all_reports if r.escalation == "moderate"]),
                "low": len([r for r in all_reports if r.escalation == "low"])
            },
            "tone_breakdown": {
                "urgent": len([r for r in all_reports if r.tone == "urgent"]),
                "frantic": len([r for r in all_reports if r.tone == "frantic"]),
                "helpless": len([r for r in all_reports if r.tone == "helpless"]),
                "descriptive": len([r for r in all_reports if r.tone == "descriptive"])
            }
        }
            
        logger.info(f"📊 Community stats requested: {stats['total_reports']} total reports")
            
        return JSONResponse(content={
            "success": True, "stats": stats, "generated_at": datetime.utcnow().isoformat()
        })
            
    except Exception as e:
        logger.error(f"❌ Error getting community stats: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/api/recent-reports", response_class=JSONResponse)
async def get_recent_reports(
    limit: int = Query(5, description="Number of recent reports to return"),
    db: Session = Depends(get_db)
):
    """Get recent crowd reports for widgets and notifications"""
    try:
        if limit < 1 or limit > 50:
            limit = 5
                
        recent_reports = db.query(CrowdReport).order_by(CrowdReport.timestamp.desc()).limit(limit).all()
            
        reports_data = [{
            "id": report.id, "message": report.message, "tone": report.tone,
            "escalation": report.escalation, "user": report.user or "Anonymous",
            "location": report.location, "timestamp": report.timestamp,
            "latitude": report.latitude, "longitude": report.longitude,
            "time_ago": get_time_ago(report.timestamp) if report.timestamp else "Unknown"
        } for report in recent_reports]
            
        logger.info(f"📋 Recent reports API called: returned {len(reports_data)} reports")
            
        return JSONResponse(content={
            "success": True, "reports": reports_data, "count": len(reports_data),
            "generated_at": datetime.utcnow().isoformat()
        })
            
    except Exception as e:
        logger.error(f"❌ Error getting recent reports: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)
        
# ================================================================================
# EMERGENCY SYSTEM API ENDPOINTS
# ================================================================================

@app.get("/api/system-status", response_class=JSONResponse)
async def get_system_status():
    """Get comprehensive system status for device monitoring"""
    try:
        return JSONResponse(content={
            "success": True,
            "status": {
                "server": "online",
                "database": "connected",
                "services": {
                    "sync": "active",
                    "mapping": "operational",
                    "ai_analysis": "ready",
                    "offline_support": "enabled"
                },
                "uptime": "12h 34m",
                "last_sync": datetime.utcnow().isoformat(),
                "performance": {
                    "response_time": "45ms",
                    "memory_usage": "67%",
                    "cpu_usage": "23%"
                }
            },
            "generated_at": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/api/sync-queue", response_class=JSONResponse)
async def get_sync_queue():
    """Get offline sync queue status"""
    try:
        # This would normally read from a sync queue database/storage
        # For now, return demo data
        return JSONResponse(content={
            "success": True,
            "queue": {
                "pending_reports": 0,
                "failed_syncs": 0,
                "last_sync": datetime.utcnow().isoformat(),
                "sync_status": "all_synced"
            },
            "generated_at": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.post("/api/feedback", response_class=JSONResponse)
async def submit_feedback(request: Request):
    """Submit user feedback"""
    try:
        data = await request.json()
            
        # Log feedback (in production, save to database)
        logger.info(f"📝 Feedback received: {data.get('type', 'general')} - {data.get('title', 'No title')}")
            
        return JSONResponse(content={
            "success": True,
            "message": "Feedback submitted successfully",
            "ticket_id": f"FB-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}",
            "submitted_at": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Error submitting feedback: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

# ================================================================================
# DEMO DATA GENERATION API
# ================================================================================

@app.post("/api/create-demo-reports")
async def create_demo_reports_api(db: Session = Depends(get_db)):
    """Create comprehensive demo crowd reports for testing and presentations"""
    try:
        # Check if demo reports already exist
        existing_count = db.query(CrowdReport).filter(
            CrowdReport.user.like("%Chief%") |
            CrowdReport.user.like("%EMT%") |
            CrowdReport.user.like("%Officer%") |
            CrowdReport.user.like("%Captain%") |
            CrowdReport.user.like("%Paramedic%") |
            CrowdReport.user.like("%Search & Rescue%")
        ).count()
            
        if existing_count > 5:
            return JSONResponse(content={
                "success": True,
                "message": f"Demo reports already exist ({existing_count} found)",
                "reports_created": 0,
                "existing_reports": existing_count
            })
            
        # Define realistic demo reports
        demo_reports = [
            {
                "user": "Fire Chief Martinez",
                "message": "Major structure fire at downtown warehouse. Multiple units responding. Heavy smoke visible from several blocks away. Requesting additional water supply and air support.",
                "tone": "urgent", "escalation": "critical",
                "latitude": 37.7749, "longitude": -122.4194,
                "location": "Downtown San Francisco, CA"
            },
            {
                "user": "EMT Johnson",
                "message": "Multi-vehicle accident on Highway 101. Three vehicles involved, possible injuries. Traffic severely backed up. Air ambulance requested for critical patient.",
                "tone": "concerned", "escalation": "high",
                "latitude": 37.7849, "longitude": -122.4094,
                "location": "Highway 101, San Francisco, CA"
            },
            {
                "user": "Citizen Reporter",
                "message": "Power lines down on Elm Street after strong winds. Area residents evacuated as precaution. PG&E crews en route to repair damage.",
                "tone": "descriptive", "escalation": "moderate",
                "latitude": 37.7649, "longitude": -122.4294,
                "location": "Elm Street, San Francisco, CA"
            },
            {
                "user": "Police Officer Chen",
                "message": "Minor fender bender resolved. Traffic flow restored. No injuries reported. Tow truck clearing vehicles from roadway.",
                "tone": "neutral", "escalation": "low",
                "latitude": 37.7549, "longitude": -122.4394,
                "location": "Market Street, San Francisco, CA"
            },
            {
                "user": "Anonymous",
                "message": "Flooding reported in underground parking garage. Water level rising rapidly. Residents in building notified. Emergency pumps being deployed.",
                "tone": "frantic", "escalation": "high",
                "latitude": 37.7949, "longitude": -122.3994,
                "location": "Financial District, San Francisco, CA"
            },
            {
                "user": "Search & Rescue Team Alpha",
                "message": "Missing hiker found safe and uninjured. Team returning to base. False alarm on emergency beacon activation.",
                "tone": "descriptive", "escalation": "low",
                "latitude": 37.7449, "longitude": -122.4494,
                "location": "Golden Gate Park, San Francisco, CA"
            },
            {
                "user": "Captain Rodriguez",
                "message": "Gas leak detected at residential complex. Area evacuated, gas company on scene. Hazmat team standing by for assessment.",
                "tone": "urgent", "escalation": "high",
                "latitude": 37.7349, "longitude": -122.4594,
                "location": "Mission District, San Francisco, CA"
            },
            {
                "user": "Paramedic Williams",
                "message": "Medical emergency at school resolved. Patient stable and transported to hospital. Normal school operations resumed.",
                "tone": "concerned", "escalation": "moderate",
                "latitude": 37.7149, "longitude": -122.4694,
                "location": "Richmond District, San Francisco, CA"
            }
        ]
            
        created_reports = []
        for report_data in demo_reports:
            new_report = CrowdReport(
                message=report_data["message"], tone=report_data["tone"],
                escalation=report_data["escalation"], user=report_data["user"],
                location=report_data["location"], latitude=report_data["latitude"],
                longitude=report_data["longitude"], timestamp=datetime.utcnow().isoformat()
            )
            
            db.add(new_report)
            db.flush()
            created_reports.append({
                "id": new_report.id, "user": new_report.user,
                "escalation": new_report.escalation, "location": new_report.location
            })
            
        db.commit()
            
        logger.info(f"✅ Created {len(created_reports)} demo crowd reports")
            
        return JSONResponse(content={
            "success": True,
            "message": f"Successfully created {len(created_reports)} demo reports",
            "reports_created": len(created_reports),
            "reports": created_reports,
            "timestamp": datetime.utcnow().isoformat()
        })
            
    except Exception as e:
        logger.error(f"❌ Error creating demo reports: {str(e)}")
        db.rollback()
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/api/demo-status")
async def get_demo_status(db: Session = Depends(get_db)):
    """Check if demo data exists and provide summary"""
    try:
        all_reports = db.query(CrowdReport).all()
        demo_reports = db.query(CrowdReport).filter(
            CrowdReport.user.like("%Chief%") |
            CrowdReport.user.like("%EMT%") |
            CrowdReport.user.like("%Officer%") |
            CrowdReport.user.like("%Search & Rescue%") |
            CrowdReport.user.like("%Captain%") |
            CrowdReport.user.like("%Paramedic%")
        ).all()
            
        reports_with_coords = db.query(CrowdReport).filter(
            CrowdReport.latitude.isnot(None),
            CrowdReport.longitude.isnot(None)
        ).all()
            
        return JSONResponse(content={
            "total_reports": len(all_reports),
            "demo_reports": len(demo_reports),
            "reports_with_coordinates": len(reports_with_coords),
            "has_demo_data": len(demo_reports) > 0,
            "demo_ready": len(reports_with_coords) >= 3,
            "escalation_breakdown": {
                "critical": len([r for r in all_reports if r.escalation == "critical"]),
                "high": len([r for r in all_reports if r.escalation == "high"]),
                "moderate": len([r for r in all_reports if r.escalation == "moderate"]),
                "low": len([r for r in all_reports if r.escalation == "low"])
            }
        })
            
    except Exception as e:
        logger.error(f"❌ Error getting demo status: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/clear-demo-data")
async def clear_demo_data(db: Session = Depends(get_db)):
    """Clear all demo data for fresh testing"""
    try:
        demo_patterns = ["%Chief%", "%EMT%", "%Officer%", "%Search & Rescue%", "%Captain%", "%Paramedic%"]
            
        deleted_count = 0
        for pattern in demo_patterns:
            demo_reports = db.query(CrowdReport).filter(CrowdReport.user.like(pattern)).all()
            for report in demo_reports:
                db.delete(report)
                deleted_count += 1
            
        db.commit()
            
        logger.info(f"🗑️ Cleared {deleted_count} demo reports")
            
        return JSONResponse(content={
            "success": True,
            "message": f"Cleared {deleted_count} demo reports",
            "reports_deleted": deleted_count,
            "timestamp": datetime.utcnow().isoformat()
        })
            
    except Exception as e:
        logger.error(f"❌ Error clearing demo data: {str(e)}")
        db.rollback()
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

# ================================================================================
# ENHANCED STATISTICS & ANALYTICS API
# ================================================================================

@app.get("/api/dashboard-stats", response_class=JSONResponse)
async def dashboard_stats_api(db: Session = Depends(get_db)):
    """API endpoint for real-time dashboard stats refresh"""
    try:
        all_reports = db.query(CrowdReport).all()
        all_patients = db.query(TriagePatient).all()
        active_patients = [p for p in all_patients if p.status == "active"]
            
        avg_severity = calculate_avg_severity(all_patients)
        active_users = 3 + len(set([r.user for r in all_reports if r.user and r.user != "Anonymous"]))
            
        stats = {
            "total_reports": len(all_reports), "active_users": active_users,
            "avg_severity": avg_severity, "total_patients": len(all_patients),
            "active_patients": len(active_patients),
            "critical_alerts": len([p for p in active_patients if p.triage_color == "red" or p.severity == "critical"]),
            "system_status": "online", "last_updated": datetime.utcnow().isoformat()
        }
            
        return JSONResponse(content={
            "success": True, "stats": stats, "generated_by": "Demo User",
            "timestamp": datetime.utcnow().isoformat()
        })
            
    except Exception as e:
        logger.error(f"❌ Dashboard stats API error: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/api/map-statistics", response_class=JSONResponse)
async def get_map_statistics(db: Session = Depends(get_db)):
    """Get comprehensive statistics for the map dashboard"""
    try:
        all_reports = db.query(CrowdReport).all()
        reports_with_coords = [r for r in all_reports if r.latitude and r.longitude]
            
        # Time-based statistics
        now = datetime.utcnow()
        last_24h = [r for r in all_reports if r.timestamp and 
                            datetime.fromisoformat(r.timestamp.replace('Z', '+00:00')) >= now - timedelta(hours=24)]
        last_week = [r for r in all_reports if r.timestamp and 
                            datetime.fromisoformat(r.timestamp.replace('Z', '+00:00')) >= now - timedelta(days=7)]
            
        # Geographic and activity statistics
        unique_locations = len(set([r.location for r in all_reports if r.location]))
            
        escalation_stats = {
            "critical": len([r for r in all_reports if r.escalation == "critical"]),
            "high": len([r for r in all_reports if r.escalation == "high"]),
            "moderate": len([r for r in all_reports if r.escalation == "moderate"]),
            "low": len([r for r in all_reports if r.escalation == "low"])
        }
            
        tone_stats = {
            "urgent": len([r for r in all_reports if r.tone == "urgent"]),
            "frantic": len([r for r in all_reports if r.tone == "frantic"]),
            "concerned": len([r for r in all_reports if r.tone == "concerned"]),
            "descriptive": len([r for r in all_reports if r.tone == "descriptive"]),
            "neutral": len([r for r in all_reports if r.tone == "neutral"]),
            "helpless": len([r for r in all_reports if r.tone == "helpless"])
        }
            
        most_active_users = {}
        for report in all_reports:
            user = report.user or "Anonymous"
            most_active_users[user] = most_active_users.get(user, 0) + 1
            
        top_users = sorted(most_active_users.items(), key=lambda x: x[1], reverse=True)[:5]
            
        stats = {
            "overview": {
                "total_reports": len(all_reports),
                "reports_with_coordinates": len(reports_with_coords),
                "unique_locations": unique_locations,
                "reports_last_24h": len(last_24h),
                "reports_last_week": len(last_week)
            },
            "escalation_breakdown": escalation_stats,
            "tone_breakdown": tone_stats,
            "activity": {
                "top_users": top_users,
                "average_reports_per_user": round(len(all_reports) / max(1, len(most_active_users)), 2),
                "total_active_users": len(most_active_users)
            },
            "geographic": {
                "coverage_area": "San Francisco Bay Area",
                "coordinate_bounds": {
                    "north": max([r.latitude for r in reports_with_coords]) if reports_with_coords else None,
                    "south": min([r.latitude for r in reports_with_coords]) if reports_with_coords else None,
                    "east": max([r.longitude for r in reports_with_coords]) if reports_with_coords else None,
                    "west": min([r.longitude for r in reports_with_coords]) if reports_with_coords else None
                }
            },
            "timeline": {
                "oldest_report": min([r.timestamp for r in all_reports if r.timestamp]) if all_reports else None,
                "newest_report": max([r.timestamp for r in all_reports if r.timestamp]) if all_reports else None,
                "peak_activity_24h": len(last_24h),
                "trend": "increasing" if len(last_24h) > len(last_week) / 7 else "stable"
            }
        }
            
        return JSONResponse(content={
            "success": True, "statistics": stats, "generated_at": datetime.utcnow().isoformat(),
            "cache_duration": 300
        })
            
    except Exception as e:
        logger.error(f"❌ Error getting map statistics: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

# ================================================================================
# EXPORT & ARCHIVE FUNCTIONALITY
# ================================================================================

@app.get("/export-reports.csv")
async def export_reports_csv(
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Export filtered reports as CSV"""
    try:
        query = db.query(CrowdReport)

        if tone:
            query = query.filter(CrowdReport.tone == tone)
        if escalation:
            query = query.filter(CrowdReport.escalation == escalation)
        if keyword:
            query = query.filter(CrowdReport.message.ilike(f"%{keyword}%"))

        reports = query.order_by(CrowdReport.timestamp.desc()).all()

        csv_data = "id,message,tone,escalation,timestamp,user,location,latitude,longitude\n"
        for r in reports:
            csv_data += f'"{r.id}","{r.message}","{r.tone}","{r.escalation}","{r.timestamp}","{r.user or ""}","{r.location or ""}","{r.latitude or ""}","{r.longitude or ""}"\n'

        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=crowd_reports.csv"}
        )
            
    except Exception as e:
        logger.error(f"❌ Error exporting reports CSV: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export CSV")

@app.get("/export-reports.json", response_class=JSONResponse)
async def export_reports_json(
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Export filtered reports as JSON"""
    try:
        query = db.query(CrowdReport)

        if tone:
            query = query.filter(CrowdReport.tone == tone)
        if escalation:
            query = query.filter(CrowdReport.escalation == escalation)
        if keyword:
            query = query.filter(CrowdReport.message.ilike(f"%{keyword}%"))

        reports = query.order_by(CrowdReport.timestamp.desc()).all()

        report_list = [{
            "id": r.id, "message": r.message, "tone": r.tone, "escalation": r.escalation,
            "timestamp": r.timestamp, "user": r.user, "location": r.location,
            "latitude": r.latitude, "longitude": r.longitude
        } for r in reports]

        return {"reports": report_list}
            
    except Exception as e:
        logger.error(f"❌ Error exporting reports JSON: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export JSON")

@app.get("/api/export-map-data")
async def export_map_data(
    format: str = Query("json", description="Export format: json, csv, kml"),
    include_filters: bool = Query(True),
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Export map data in various formats with filtering"""
    try:
        query = db.query(CrowdReport).filter(
            CrowdReport.latitude.isnot(None),
            CrowdReport.longitude.isnot(None)
        )
            
        if tone:
            query = query.filter(CrowdReport.tone == tone)
        if escalation:
            query = query.filter(CrowdReport.escalation == escalation)
            
        reports = query.all()
            
        if format.lower() == "csv":
            csv_data = "id,timestamp,user,message,latitude,longitude,tone,escalation,location\n"
            for r in reports:
                csv_data += f'"{r.id}","{r.timestamp}","{r.user or ""}","{(r.message or "").replace(chr(34), chr(34)+chr(34))}","{r.latitude}","{r.longitude}","{r.tone}","{r.escalation}","{r.location or ""}"\n'
            
            return Response(
                content=csv_data,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=map_reports.csv"}
            )
            
        elif format.lower() == "kml":
            kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
        <name>Crowd Reports Map</name>
        <description>Emergency crowd reports exported from Disaster Response System</description>
        
        <Style id="critical">
            <IconStyle><color>ff0000ff</color><scale>1.2</scale>
                <Icon><href>http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png</href></Icon>
            </IconStyle>
        </Style>
        <Style id="high">
            <IconStyle><color>ff0066ff</color><scale>1.0</scale>
                <Icon><href>http://maps.google.com/mapfiles/kml/pushpin/orange-pushpin.png</href></Icon>
            </IconStyle>
        </Style>
        <Style id="moderate">
            <IconStyle><color>ff00ffff</color><scale>0.8</scale>
                <Icon><href>http://maps.google.com/mapfiles/kml/pushpin/yellow-pushpin.png</href></Icon>
            </IconStyle>
        </Style>
        <Style id="low">
            <IconStyle><color>ff00ff00</color><scale>0.6</scale>
                <Icon><href>http://maps.google.com/mapfiles/kml/pushpin/green-pushpin.png</href></Icon>
            </IconStyle>
        </Style>
'''
            
            for report in reports:
                kml_content += f'''
        <Placemark>
            <name>{report.user or "Anonymous"} - {report.escalation.title()}</name>
            <description><![CDATA[
                <b>Message:</b> {report.message or "No message"}<br/>
                <b>Tone:</b> {report.tone or "Unknown"}<br/>
                <b>Escalation:</b> {report.escalation or "Unknown"}<br/>
                <b>Time:</b> {report.timestamp or "Unknown"}<br/>
                <b>Location:</b> {report.location or "Unknown"}
            ]]></description>
            <styleUrl>#{report.escalation or "low"}</styleUrl>
            <Point>
                <coordinates>{report.longitude},{report.latitude},0</coordinates>
            </Point>
        </Placemark>'''
            
            kml_content += '''
    </Document>
</kml>'''
            
            return Response(
                content=kml_content,
                media_type="application/vnd.google-earth.kml+xml",
                headers={"Content-Disposition": "attachment; filename=map_reports.kml"}
            )
            
        else:
            # JSON export (default) - GeoJSON format
            export_data = {
                "export_info": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_reports": len(reports),
                    "format": "geojson",
                    "filters_applied": {"tone": tone, "escalation": escalation} if include_filters else None
                },
                "type": "FeatureCollection",
                "features": []
            }
            
            for report in reports:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(report.longitude), float(report.latitude)]
                    },
                    "properties": {
                        "id": report.id, "user": report.user, "message": report.message,
                        "tone": report.tone, "escalation": report.escalation,
                        "timestamp": report.timestamp, "location": report.location,
                        "marker-color": {
                            "critical": "#991b1b", "high": "#dc2626",  
                            "moderate": "#f59e0b", "low": "#16a34a"
                        }.get(report.escalation, "#6b7280"),
                        "marker-symbol": {
                            "critical": "emergency", "high": "fire-station",
                            "moderate": "warning", "low": "information"
                        }.get(report.escalation, "marker")
                    }
                }
                export_data["features"].append(feature)
            
            return JSONResponse(content=export_data)
            
    except Exception as e:
        logger.error(f"❌ Error exporting map data: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

# ================================================================================
# PATIENT TRIAGE MANAGEMENT - PAGES
# ================================================================================

@app.get("/triage", response_class=HTMLResponse)
async def triage_form_page(request: Request):
    """Triage assessment form"""
    return templates.TemplateResponse("triage_form.html", {"request": request})

@app.get("/patients", response_class=HTMLResponse)
async def get_patient_tracker(
    request: Request, 
    severity: Optional[str] = None, 
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Patient tracker with filtering"""
    try:
        query = db.query(TriagePatient)
            
        if severity:
            query = query.filter(TriagePatient.severity == severity)
        if status:
            query = query.filter(TriagePatient.status == status)
            
        patients = query.order_by(TriagePatient.created_at.desc()).all()
            
        return templates.TemplateResponse("patient_tracker.html", {
            "request": request, "patients": patients, "now": datetime.utcnow(),
            "severity_filter": severity, "status_filter": status
        })
            
    except Exception as e:
        logger.error(f"❌ Error loading patient tracker: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load patients: {str(e)}")

@app.get("/patient-list", response_class=HTMLResponse)
async def patient_list_page(
    request: Request,
    triage_color: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Enhanced patient list dashboard"""
    try:
        query = db.query(TriagePatient)
            
        if triage_color:
            query = query.filter(TriagePatient.triage_color == triage_color)
        if status:
            query = query.filter(TriagePatient.status == status)
        if severity:
            query = query.filter(TriagePatient.severity == severity)
            
        patients = query.order_by(TriagePatient.created_at.desc()).all()
        patients = sorted(patients, key=lambda p: (p.priority_score, -p.id))
            
        # Calculate statistics
        total_patients = len(patients)
        active_patients = len([p for p in patients if p.status == "active"])
        critical_patients = len([p for p in patients if p.triage_color == "red"])
            
        color_counts = {
            "red": len([p for p in patients if p.triage_color == "red"]),
            "yellow": len([p for p in patients if p.triage_color == "yellow"]),
            "green": len([p for p in patients if p.triage_color == "green"]),
            "black": len([p for p in patients if p.triage_color == "black"])
        }
            
        status_counts = {
            "active": len([p for p in patients if p.status == "active"]),
            "in_treatment": len([p for p in patients if p.status == "in_treatment"]),
            "treated": len([p for p in patients if p.status == "treated"]),
            "discharged": len([p for p in patients if p.status == "discharged"])
        }
            
        logger.info(f"📋 Patient list accessed: {total_patients} total, {active_patients} active")
            
        return templates.TemplateResponse("patient_list.html", {
            "request": request, "patients": patients, "total_patients": total_patients,
            "active_patients": active_patients, "critical_patients": critical_patients,
            "color_counts": color_counts, "status_counts": status_counts,
            "filters": {"triage_color": triage_color, "status": status, "severity": severity}
        })
            
    except Exception as e:
        logger.error(f"❌ Error loading patient list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load patient list: {str(e)}")

@app.get("/triage-dashboard", response_class=HTMLResponse)
async def triage_dashboard_page(request: Request, db: Session = Depends(get_db)):
    """Comprehensive triage dashboard with real-time statistics"""
    try:
        all_patients = db.query(TriagePatient).all()
        active_patients = db.query(TriagePatient).filter(TriagePatient.status == "active").all()
            
        stats = {
            "total_patients": len(all_patients),
            "active_patients": len(active_patients),
            "patients_today": len([p for p in all_patients if p.created_at.date() == datetime.utcnow().date()]),
            "critical_alerts": len([p for p in active_patients if p.triage_color == "red" or p.severity == "critical"])
        }
            
        triage_breakdown = {
            "red": {"count": len([p for p in active_patients if p.triage_color == "red"]), "percentage": 0},
            "yellow": {"count": len([p for p in active_patients if p.triage_color == "yellow"]), "percentage": 0},
            "green": {"count": len([p for p in active_patients if p.triage_color == "green"]), "percentage": 0},
            "black": {"count": len([p for p in active_patients if p.triage_color == "black"]), "percentage": 0}
        }
            
        if stats["active_patients"] > 0:
            for color in triage_breakdown:
                triage_breakdown[color]["percentage"] = round(
                    (triage_breakdown[color]["count"] / stats["active_patients"]) * 100, 1
                )
            
        severity_breakdown = {
            "critical": len([p for p in active_patients if p.severity == "critical"]),
            "severe": len([p for p in active_patients if p.severity == "severe"]),
            "moderate": len([p for p in active_patients if p.severity == "moderate"]),
            "mild": len([p for p in active_patients if p.severity == "mild"])
        }
            
        critical_vitals_patients = [p for p in active_patients if p.is_critical_vitals]
        recent_patients = [p for p in all_patients if (datetime.utcnow() - p.created_at).total_seconds() < 86400]
        recent_patients = sorted(recent_patients, key=lambda p: p.created_at, reverse=True)[:10]
        priority_queue = sorted(active_patients, key=lambda p: (p.priority_score, -p.id))[:15]
            
        logger.info(f"📊 Triage dashboard accessed: {stats['total_patients']} total, {stats['active_patients']} active")
            
        return templates.TemplateResponse("triage_dashboard.html", {
            "request": request, "stats": stats, "triage_breakdown": triage_breakdown,
            "severity_breakdown": severity_breakdown, "critical_vitals_patients": critical_vitals_patients,
            "recent_patients": recent_patients, "priority_queue": priority_queue,
            "current_time": datetime.utcnow()
        })
            
    except Exception as e:
        logger.error(f"❌ Error loading triage dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load dashboard: {str(e)}")

# ================================================================================
# PATIENT TRIAGE MANAGEMENT - FORM SUBMISSION & UPDATES
# ================================================================================

@app.post("/submit-triage")
async def submit_triage(
    request: Request,
    db: Session = Depends(get_db),
    # Patient Information
    name: str = Form(...),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    medical_id: Optional[str] = Form(None),
    # Medical Assessment
    injury_type: str = Form(...),
    mechanism: Optional[str] = Form(None),
    consciousness: str = Form(...),
    breathing: str = Form(...),
    # Vital Signs
    heart_rate: Optional[int] = Form(None),
    bp_systolic: Optional[int] = Form(None),
    bp_diastolic: Optional[int] = Form(None),
    respiratory_rate: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    oxygen_sat: Optional[int] = Form(None),
    # Assessment
    severity: str = Form(...),
    triage_color: str = Form(...),
    # Additional Information
    allergies: Optional[str] = Form(None),
    medications: Optional[str] = Form(None),
    medical_history: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    assessment_timestamp: Optional[str] = Form(None)
):
    """Submit new triage assessment"""
    try:
        logger.info(f"🚑 Receiving triage submission for patient: {name}")
            
        # Validation
        if not name or not name.strip():
            raise HTTPException(status_code=400, detail="Patient name is required")
        if not injury_type or not injury_type.strip():
            raise HTTPException(status_code=400, detail="Injury type is required")
        if consciousness not in ["alert", "verbal", "pain", "unresponsive"]:
            raise HTTPException(status_code=400, detail="Invalid consciousness level")
        if breathing not in ["normal", "labored", "shallow", "absent"]:
            raise HTTPException(status_code=400, detail="Invalid breathing status")
        if severity not in ["mild", "moderate", "severe", "critical"]:
            raise HTTPException(status_code=400, detail="Invalid severity level")
        if triage_color not in ["red", "yellow", "green", "black"]:
            raise HTTPException(status_code=400, detail="Invalid triage color")
            
        # Validate vital signs ranges
        if heart_rate is not None and (heart_rate < 0 or heart_rate > 300):
            raise HTTPException(status_code=400, detail="Invalid heart rate")
        if bp_systolic is not None and (bp_systolic < 0 or bp_systolic > 300):
            raise HTTPException(status_code=400, detail="Invalid systolic blood pressure")
        if bp_diastolic is not None and (bp_diastolic < 0 or bp_diastolic > 200):
            raise HTTPException(status_code=400, detail="Invalid diastolic blood pressure")
        if respiratory_rate is not None and (respiratory_rate < 0 or respiratory_rate > 100):
            raise HTTPException(status_code=400, detail="Invalid respiratory rate")
        if temperature is not None and (temperature < 80 or temperature > 115):
            raise HTTPException(status_code=400, detail="Invalid temperature")
        if oxygen_sat is not None and (oxygen_sat < 0 or oxygen_sat > 100):
            raise HTTPException(status_code=400, detail="Invalid oxygen saturation")
        if age is not None and (age < 0 or age > 120):
            raise HTTPException(status_code=400, detail="Invalid age")
            
        # Create new triage patient record
        new_patient = TriagePatient(
            name=name.strip(), age=age, gender=gender, medical_id=medical_id,
            injury_type=injury_type.strip(), mechanism=mechanism, consciousness=consciousness,
            breathing=breathing, heart_rate=heart_rate, bp_systolic=bp_systolic,
            bp_diastolic=bp_diastolic, respiratory_rate=respiratory_rate,
            temperature=temperature, oxygen_sat=oxygen_sat, severity=severity,
            triage_color=triage_color, allergies=allergies, medications=medications,
            medical_history=medical_history, notes=notes, status="active",
            created_at=datetime.utcnow(), updated_at=datetime.utcnow()
        )
            
        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)
            
        logger.info(f"✅ Triage patient saved: ID={new_patient.id}, Name={new_patient.name}, Color={new_patient.triage_color}")
            
        if triage_color == "red" or severity == "critical":
            logger.warning(f"🚨 CRITICAL PATIENT ALERT: {name} - {triage_color.upper()} triage, {severity} severity")
            
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Triage assessment submitted successfully for {name}",
                "patient_id": new_patient.id,
                "triage_color": new_patient.triage_color,
                "severity": new_patient.severity,
                "priority_score": new_patient.priority_score,
                "critical_vitals": new_patient.is_critical_vitals,
                "timestamp": new_patient.created_at.isoformat()
            }
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error saving triage patient: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save triage assessment: {str(e)}")

@app.get("/patients/{patient_id}/edit", response_class=HTMLResponse)
async def edit_patient_form(patient_id: int, request: Request, db: Session = Depends(get_db)):
    """Render patient edit form"""
    try:
        patient = db.query(TriagePatient).filter(TriagePatient.id == patient_id).first()
            
        if not patient:
            logger.warning(f"❌ Patient not found for edit: ID={patient_id}")
            raise HTTPException(status_code=404, detail="Patient not found")
            
        logger.info(f"📝 Rendering edit form for patient: ID={patient_id}, Name={patient.name}")
            
        return templates.TemplateResponse("edit_patient.html", {
            "request": request, "patient": patient
        })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error loading patient edit form: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load edit form: {str(e)}")

@app.post("/patients/{patient_id}/update")
async def update_patient(
    patient_id: int, 
    request: Request, 
    db: Session = Depends(get_db),
    name: str = Form(...),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    medical_id: Optional[str] = Form(None),
    injury_type: str = Form(...),
    mechanism: Optional[str] = Form(None),
    consciousness: str = Form(...),
    breathing: str = Form(...),
    heart_rate: Optional[int] = Form(None),
    bp_systolic: Optional[int] = Form(None),
    bp_diastolic: Optional[int] = Form(None),
    respiratory_rate: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    oxygen_sat: Optional[int] = Form(None),
    severity: str = Form(...),
    triage_color: str = Form(...),
    allergies: Optional[str] = Form(None),
    medications: Optional[str] = Form(None),
    medical_history: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    status: str = Form(...)
):
    """Update patient information"""
    try:
        patient = db.query(TriagePatient).filter(TriagePatient.id == patient_id).first()
            
        if not patient:
            logger.warning(f"❌ Patient not found for update: ID={patient_id}")
            raise HTTPException(status_code=404, detail="Patient not found")
            
        # Validation (same as submit_triage)
        if not name or not name.strip():
            raise HTTPException(status_code=400, detail="Patient name is required")
        if not injury_type or not injury_type.strip():
            raise HTTPException(status_code=400, detail="Injury type is required")
        if consciousness not in ["alert", "verbal", "pain", "unresponsive"]:
            raise HTTPException(status_code=400, detail="Invalid consciousness level")
        if breathing not in ["normal", "labored", "shallow", "absent"]:
            raise HTTPException(status_code=400, detail="Invalid breathing status")
        if severity not in ["mild", "moderate", "severe", "critical"]:
            raise HTTPException(status_code=400, detail="Invalid severity level")
        if triage_color not in ["red", "yellow", "green", "black"]:
            raise HTTPException(status_code=400, detail="Invalid triage color")
        if status not in ["active", "in_treatment", "treated", "discharged"]:
            raise HTTPException(status_code=400, detail="Invalid status")
                
        # Store original values for change tracking
        original_values = {
            "severity": patient.severity,
            "triage_color": patient.triage_color,
            "status": patient.status
        }
            
        # Update patient fields
        patient.name = name.strip()
        patient.age = age
        patient.gender = gender
        patient.medical_id = medical_id
        patient.injury_type = injury_type.strip()
        patient.mechanism = mechanism
        patient.consciousness = consciousness
        patient.breathing = breathing
        patient.heart_rate = heart_rate
        patient.bp_systolic = bp_systolic
        patient.bp_diastolic = bp_diastolic
        patient.respiratory_rate = respiratory_rate
        patient.temperature = temperature
        patient.oxygen_sat = oxygen_sat
        patient.severity = severity
        patient.triage_color = triage_color
        patient.allergies = allergies
        patient.medications = medications
        patient.medical_history = medical_history
        patient.notes = notes
        patient.status = status
        patient.updated_at = datetime.utcnow()
            
        db.commit()
        db.refresh(patient)
            
        # Track changes
        changes = []
        if original_values["severity"] != severity:
            changes.append(f"severity: {original_values['severity']} → {severity}")
        if original_values["triage_color"] != triage_color:
            changes.append(f"triage: {original_values['triage_color']} → {triage_color}")
        if original_values["status"] != status:
            changes.append(f"status: {original_values['status']} → {status}")
            
        change_summary = ", ".join(changes) if changes else "general updates"
        logger.info(f"✅ Patient updated: ID={patient_id}, Name={patient.name}, Changes=({change_summary})")
            
        if triage_color == "red" or severity == "critical":
            logger.warning(f"🚨 CRITICAL PATIENT UPDATED: {patient.name} - {triage_color.upper()} triage, {severity} severity")
            
        return RedirectResponse(
            url=f"/patient-list?success={f'Patient {patient.name} updated successfully!'.replace(' ', '+')}", 
            status_code=303
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating patient: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update patient: {str(e)}")

@app.get("/patients/{patient_id}/view", response_class=HTMLResponse)
async def view_patient_details(patient_id: int, request: Request, db: Session = Depends(get_db)):
    """View detailed patient information"""
    try:
        patient = db.query(TriagePatient).filter(TriagePatient.id == patient_id).first()
            
        if not patient:
            logger.warning(f"❌ Patient not found for view: ID={patient_id}")
            raise HTTPException(status_code=404, detail="Patient not found")
            
        logger.info(f"👀 Viewing patient details: ID={patient_id}, Name={patient.name}")
            
        return templates.TemplateResponse("patient_details.html", {
            "request": request, "patient": patient, "current_time": datetime.utcnow()
        })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error loading patient details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load patient details: {str(e)}")

@app.post("/patients/{patient_id}/discharge")
async def discharge_patient(patient_id: int, db: Session = Depends(get_db)):
    """Discharge patient"""
    try:
        patient = db.query(TriagePatient).filter(TriagePatient.id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
            
        patient.status = "discharged"
        patient.updated_at = datetime.utcnow()
        db.commit()
            
        logger.info(f"✅ Patient discharged: ID={patient_id}, Name={patient.name}")
        return RedirectResponse(url="/patients", status_code=303)
            
    except Exception as e:
        logger.error(f"❌ Error discharging patient: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to discharge patient")

# ================================================================================
# PATIENT TRIAGE MANAGEMENT - API ENDPOINTS
# ================================================================================

@app.get("/api/triage-stats", response_class=JSONResponse)
async def get_triage_stats(db: Session = Depends(get_db)):
    """Get triage statistics for dashboard widgets"""
    try:
        all_patients = db.query(TriagePatient).all()
        active_patients = [p for p in all_patients if p.status == "active"]
        today_patients = [p for p in all_patients if p.created_at.date() == datetime.utcnow().date()]
            
        stats = {
            "total_patients": len(all_patients),
            "active_patients": len(active_patients),
            "patients_today": len(today_patients),
            "critical_alerts": len([p for p in active_patients if p.triage_color == "red" or p.severity == "critical" or p.is_critical_vitals]),
            "triage_breakdown": {
                "red": len([p for p in active_patients if p.triage_color == "red"]),
                "yellow": len([p for p in active_patients if p.triage_color == "yellow"]),
                "green": len([p for p in active_patients if p.triage_color == "green"]),
                "black": len([p for p in active_patients if p.triage_color == "black"])
            },
            "severity_breakdown": {
                "critical": len([p for p in active_patients if p.severity == "critical"]),
                "severe": len([p for p in active_patients if p.severity == "severe"]),
                "moderate": len([p for p in active_patients if p.severity == "moderate"]),
                "mild": len([p for p in active_patients if p.severity == "mild"])
            },
            "status_breakdown": {
                "active": len([p for p in all_patients if p.status == "active"]),
                "in_treatment": len([p for p in all_patients if p.status == "in_treatment"]),
                "treated": len([p for p in all_patients if p.status == "treated"]),
                "discharged": len([p for p in all_patients if p.status == "discharged"])
            }
        }
            
        logger.info(f"🏥 Triage stats requested: {stats['total_patients']} total patients")
            
        return JSONResponse(content={
            "success": True, "stats": stats, "generated_at": datetime.utcnow().isoformat()
        })
            
    except Exception as e:
        logger.error(f"❌ Error getting triage stats: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/api/priority-queue", response_class=JSONResponse)
async def get_priority_queue(
    limit: int = Query(10, description="Number of priority patients to return"),
    db: Session = Depends(get_db)
):
    """Get priority patient queue for real-time updates"""
    try:
        active_patients = db.query(TriagePatient).filter(TriagePatient.status == "active").all()
        priority_patients = sorted(active_patients, key=lambda p: (p.priority_score, -p.id))[:limit]
            
        queue_data = [{
            "id": patient.id, "name": patient.name, "injury_type": patient.injury_type,
            "severity": patient.severity, "triage_color": patient.triage_color,
            "priority_score": patient.priority_score, "critical_vitals": patient.is_critical_vitals,
            "created_at": patient.created_at.isoformat(), "time_ago": get_time_ago(patient.created_at.isoformat())
        } for patient in priority_patients]
            
        logger.info(f"🚨 Priority queue requested: {len(queue_data)} patients")
            
        return JSONResponse(content={
            "success": True, "queue": queue_data, "count": len(queue_data),
            "generated_at": datetime.utcnow().isoformat()
        })
            
    except Exception as e:
        logger.error(f"❌ Error getting priority queue: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/export-patients-pdf")
async def export_patients_pdf(db: Session = Depends(get_db)):
    """Export patients as PDF"""
    try:
        patients = db.query(TriagePatient).order_by(TriagePatient.created_at.desc()).all()
            
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("patient_tracker.html")
        html_out = template.render(
            patients=patients, now=datetime.utcnow(), 
            severity_filter=None, status_filter=None
        )
            
        pdf_path = os.path.join("outputs", f"patients_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
        WeasyHTML(string=html_out).write_pdf(pdf_path)
            
        return FileResponse(pdf_path, filename="patient_tracker.pdf", media_type="application/pdf")
            
    except Exception as e:
        logger.error(f"❌ Error exporting patients PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export patients PDF")

# ================================================================================
# AI ANALYSIS & REPORTING
# ================================================================================

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_input(
    request: Request,
    report_text: str = Form(""),
    file: UploadFile = File(None),
    audio: UploadFile = File(None),
    user: dict = Depends(require_role(["admin", "responder"]))
):
    """AI analysis of reports, images, and audio"""
    input_payload = {}
    hazards = []

    if file and file.filename != "":
        extension = os.path.splitext(file.filename)[1]
        unique_filename = f"upload_{uuid.uuid4().hex}{extension}"
        saved_path = os.path.join("static", unique_filename)
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        input_payload = {"type": "image", "content": saved_path}

    elif audio and audio.filename != "":
        extension = os.path.splitext(audio.filename)[1]
        audio_path = os.path.join(UPLOAD_DIR, f"audio_{uuid.uuid4().hex}{extension}")
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        transcription = transcribe_audio(audio_path)
        if "error" in transcription:
            return templates.TemplateResponse("home.html", {
                "request": request, "result": transcription, "input_text": None
            })

        input_payload = {"type": "text", "content": transcription["text"]}
        hazards = transcription.get("hazards", [])

    else:
        input_payload = {"type": "text", "content": report_text.strip()}

    processed = preprocess_input(input_payload)
    result = run_disaster_analysis(processed)

    return templates.TemplateResponse("home.html", {
        "request": request, "result": result, "original_input": input_payload["content"], "hazards": hazards
    })

@app.post("/export-pdf")
async def export_pdf(request: Request, report_text: str = Form(...)):
    """Export analysis as PDF"""
    html_content = templates.get_template("pdf_template.html").render({"report_text": report_text})
    pdf_path = os.path.join(OUTPUT_DIR, f"report_{uuid.uuid4().hex}.pdf")
    WeasyHTML(string=html_content).write_pdf(pdf_path)

    return templates.TemplateResponse("pdf_success.html", {
        "request": request, "pdf_url": f"/{pdf_path}"
    })

@app.post("/generate-report")
async def generate_report(
    request: Request,
    file: UploadFile = File(None),
    user: dict = Depends(require_role(["admin", "responder"]))
):
    """Generate enhanced PDF report with MAP integration"""
    content_type = request.headers.get("Content-Type", "")

    if "application/json" in content_type:
        payload = await request.json()
    elif "multipart/form-data" in content_type:
        form = await request.form()
        payload_raw = form.get("json")
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            return JSONResponse(content={"error": "Invalid JSON format"}, status_code=400)

        if file and file.filename:
            ext = os.path.splitext(file.filename)[1]
            filename = f"upload_{uuid.uuid4().hex}{ext}"
            filepath = os.path.join(UPLOAD_DIR, filename)
            with open(filepath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            payload["image_url"] = f"uploaded://{filename}"
    else:
        return JSONResponse(content={"error": "Unsupported content type"}, status_code=415)

    logger.info(f"Generating enhanced PDF report with maps for user {user['username']}")
        
    try:
        pdf_path = generate_report_pdf(payload)
        logger.info(f"PDF report generated successfully: {pdf_path}")
            
        return FileResponse(
            pdf_path, media_type="application/pdf", 
            filename="emergency_incident_report.pdf"
        )
            
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.post("/detect-hazards")
async def detect_hazards_api(
    file: UploadFile = File(...),
    user: dict = Depends(require_role(["admin", "responder"]))
):
    """AI-powered hazard detection from images"""
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)
    try:
        image_bytes = await file.read()
        result = detect_hazards(image_bytes)
            
        logger.info(f"Hazard detection performed by user {user['username']}")
            
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Hazard detection failed: {e}")
        return JSONResponse(content={"error": f"Hazard detection failed: {str(e)}"}, status_code=500)

@app.post("/predict-risk")
async def predict_risk_api(payload: dict = Body(...)):
    """Predict risk scores based on location, weather, and hazard type"""
    location = payload.get("location", {})
    weather = payload.get("weather", {})
    hazard = payload.get("hazard_type", "unknown")

    result = calculate_risk_score(location, weather, hazard)
    return JSONResponse(content=result)

@app.post("/broadcast")
async def trigger_broadcast(request: Request):
    """Trigger emergency broadcast"""
    payload = await request.json()
    message = payload.get("message", "Emergency Broadcast")
    location = payload.get("location", {})
    severity = payload.get("severity", "High")
    result = start_broadcast(message, location, severity)
    return JSONResponse(content=result)

@app.get("/broadcasts")
async def get_active_broadcasts():
    """Get active emergency broadcasts"""
    result = discover_nearby_broadcasts(location={})
    return JSONResponse(content=result)

# ================================================================================
# GEMMA 3N API ENDPOINTS - ADD TO YOUR EXISTING API SECTION
# Add these after the existing API endpoints (around line ~1200)
# ================================================================================

@app.post("/api/submit-voice-emergency-report", response_class=JSONResponse)
async def submit_voice_emergency_report(request: Request, db: Session = Depends(get_db)):
    """Submit voice emergency report with AI analysis"""
    try:
        data = await request.json()
        
        # Extract voice report data
        transcript = data.get("transcript", "")
        analysis = data.get("analysis", {})
        language = data.get("language", "en-US")
        model_used = data.get("model_used", "gemma-3n-4b")
        
        # Validate required fields
        if not transcript or not transcript.strip():
            raise HTTPException(status_code=400, detail="Transcript is required")
        
        # Create emergency report from voice data
        new_report = CrowdReport(
            message=transcript,
            tone=analysis.get("emotionalState", "unknown"),
            escalation=analysis.get("urgencyLevel", "medium").lower(),
            user=f"Voice Reporter ({language})",
            location=analysis.get("location", "Not specified"),
            timestamp=datetime.utcnow().isoformat(),
            # Store additional voice analysis data in notes or separate fields
        )
        
        db.add(new_report)
        db.commit()
        db.refresh(new_report)
        
        logger.info(f"🎤 Voice emergency report submitted: ID={new_report.id}, urgency={analysis.get('urgencyLevel')}")
        
        return JSONResponse(content={
            "success": True,
            "message": "Voice emergency report submitted successfully",
            "report_id": new_report.id,
            "analysis_summary": {
                "urgency_level": analysis.get("urgencyLevel"),
                "emergency_type": analysis.get("emergencyType"),
                "confidence": analysis.get("confidence")
            },
            "submitted_at": new_report.timestamp
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error submitting voice emergency report: {str(e)}")
        db.rollback()
        return JSONResponse(
            content={"success": False, "error": str(e)}, 
            status_code=500
        )

@app.post("/api/submit-damage-assessment", response_class=JSONResponse)
async def submit_damage_assessment(request: Request, db: Session = Depends(get_db)):
    """Submit multimodal damage assessment"""
    try:
        data = await request.json()
        
        analysis = data.get("analysis", {})
        media_metadata = data.get("mediaMetadata", {})
        model_used = data.get("modelUsed", "gemma-3n-4b")
        resolution = data.get("resolution", 512)
        
        # Extract key assessment data
        structural_damage = analysis.get("structural", {})
        environmental_hazards = analysis.get("environmental", {})
        recommendations = analysis.get("recommendations", [])
        
        # Determine overall escalation level
        escalation_level = "low"
        if structural_damage.get("level") == "severe":
            escalation_level = "critical"
        elif structural_damage.get("level") == "moderate":
            escalation_level = "high"
        elif len(environmental_hazards.get("hazards", [])) > 2:
            escalation_level = "high"
        
        # Create assessment report
        assessment_summary = f"""MULTIMODAL DAMAGE ASSESSMENT
Structural: {structural_damage.get('level', 'unknown')} ({structural_damage.get('confidence', 0)*100:.1f}% confidence)
Hazards: {len(environmental_hazards.get('hazards', []))} detected
Media: {media_metadata.get('videoCount', 0)} videos, {media_metadata.get('imageCount', 0)} images, {media_metadata.get('audioCount', 0)} audio
Analysis Model: {model_used} @ {resolution}px
Top Recommendations: {'; '.join([r.get('action', '') for r in recommendations[:3]])}
"""
        
        new_report = CrowdReport(
            message=assessment_summary,
            tone="analytical",
            escalation=escalation_level,
            user=f"AI Assessment ({model_used})",
            location="Assessment Area",
            timestamp=datetime.utcnow().isoformat()
        )
        
        db.add(new_report)
        db.commit()
        db.refresh(new_report)
        
        logger.info(f"📹 Damage assessment submitted: ID={new_report.id}, level={structural_damage.get('level')}")
        
        return JSONResponse(content={
            "success": True,
            "message": "Damage assessment submitted successfully",
            "assessment_id": new_report.id,
            "overall_level": escalation_level,
            "structural_damage": structural_damage.get("level"),
            "hazards_detected": len(environmental_hazards.get("hazards", [])),
            "confidence": structural_damage.get("confidence", 0),
            "submitted_at": new_report.timestamp
        })
        
    except Exception as e:
        logger.error(f"❌ Error submitting damage assessment: {str(e)}")
        db.rollback()
        return JSONResponse(
            content={"success": False, "error": str(e)}, 
            status_code=500
        )

@app.post("/api/context-analysis", response_class=JSONResponse)
async def submit_context_analysis(
    request: Request, 
    user: dict = Depends(require_role(["admin", "responder"])),
    db: Session = Depends(get_db)
):
    """Submit comprehensive context analysis using 128K token window"""
    try:
        data = await request.json()
        
        analysis_type = data.get("analysis_type", "comprehensive")
        data_sources = data.get("data_sources", [])
        insights = data.get("insights", [])
        synthesis = data.get("synthesis", {})
        
        # Store analysis results (in production, you'd have a dedicated table)
        analysis_summary = f"""CONTEXT INTELLIGENCE ANALYSIS ({analysis_type.upper()})
Performed by: {user['username']} ({user['role']})
Data Sources: {', '.join(data_sources)}
Key Insights: {len(insights)} patterns identified
Context Window: 128K tokens processed
Model: Gemma 3n with enhanced context processing

Synthesis: {synthesis.get('summary', 'Analysis completed successfully')}
"""
        
        # Create analysis report
        analysis_report = CrowdReport(
            message=analysis_summary,
            tone="analytical",
            escalation="low",  # Context analysis is typically informational
            user=f"Context AI ({user['username']})",
            location="System Analysis",
            timestamp=datetime.utcnow().isoformat()
        )
        
        db.add(analysis_report)
        db.commit()
        db.refresh(analysis_report)
        
        logger.info(f"🧠 Context analysis submitted by {user['username']}: {len(insights)} insights")
        
        return JSONResponse(content={
            "success": True,
            "message": "Context analysis completed and stored",
            "analysis_id": analysis_report.id,
            "insights_count": len(insights),
            "data_sources_processed": len(data_sources),
            "analyst": user['username'],
            "completed_at": analysis_report.timestamp
        })
        
    except Exception as e:
        logger.error(f"❌ Error submitting context analysis: {str(e)}")
        db.rollback()
        return JSONResponse(
            content={"success": False, "error": str(e)}, 
            status_code=500
        )

@app.get("/api/ai-model-status", response_class=JSONResponse)
async def get_ai_model_status():
    """Get current AI model status and performance metrics"""
    try:
        # Simulate AI model status (in production, this would query actual model status)
        status = {
            "gemma_3n_status": {
                "available_models": [
                    {"name": "gemma-3n-2b", "status": "ready", "memory": "2GB", "speed": "fast"},
                    {"name": "gemma-3n-4b", "status": "ready", "memory": "4GB", "speed": "balanced"},
                    {"name": "gemma-3n-4b-hq", "status": "ready", "memory": "6GB", "speed": "slow"}
                ],
                "active_model": "gemma-3n-4b",
                "context_window": "128K tokens",
                "multimodal_support": True,
                "audio_processing": True,
                "vision_encoder": "MobileNet-V5-300M"
            },
            "performance_metrics": {
                "average_response_time": "0.8s",
                "accuracy": "94.2%",
                "uptime": "99.7%",
                "requests_processed_today": 1247
            },
            "device_optimization": {
                "memory_usage": "65%",
                "cpu_usage": "34%",
                "battery_optimized": True,
                "adaptive_quality": True
            },
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return JSONResponse(content={
            "success": True,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting AI model status: {str(e)}")
        return JSONResponse(
            content={"success": False, "error": str(e)}, 
            status_code=500
        )

@app.post("/api/optimize-ai-settings", response_class=JSONResponse)
async def optimize_ai_settings(request: Request):
    """Apply AI optimization settings"""
    try:
        data = await request.json()
        
        selected_model = data.get("selected_model", "gemma-3n-4b")
        performance_settings = data.get("performance_settings", {})
        auto_optimization = data.get("auto_optimization", False)
        
        # Validate model selection
        valid_models = ["gemma-3n-2b", "gemma-3n-4b", "gemma-3n-4b-hq"]
        if selected_model not in valid_models:
            raise HTTPException(status_code=400, detail=f"Invalid model. Must be one of: {valid_models}")
        
        # In production, this would actually configure the AI models
        optimization_result = {
            "model_switched": selected_model,
            "settings_applied": performance_settings,
            "auto_optimization": auto_optimization,
            "estimated_performance_change": {
                "speed": "+15%" if selected_model == "gemma-3n-2b" else "baseline",
                "quality": "+10%" if selected_model == "gemma-3n-4b-hq" else "baseline",
                "memory_usage": "-30%" if selected_model == "gemma-3n-2b" else "baseline"
            },
            "applied_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"⚙️ AI settings optimized: model={selected_model}, auto={auto_optimization}")
        
        return JSONResponse(content={
            "success": True,
            "message": "AI settings optimized successfully",
            "result": optimization_result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error optimizing AI settings: {str(e)}")
        return JSONResponse(
            content={"success": False, "error": str(e)}, 
            status_code=500
        )

@app.get("/api/device-performance", response_class=JSONResponse)
async def get_device_performance():
    """Get real-time device performance metrics"""
    try:
        import psutil
        import platform
        
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        battery = psutil.sensors_battery()
        
        performance_data = {
            "system": {
                "platform": platform.system(),
                "processor": platform.processor(),
                "architecture": platform.architecture()[0]
            },
            "cpu": {
                "usage_percent": cpu_percent,
                "core_count": psutil.cpu_count(),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": memory.percent,
                "free_gb": round(memory.free / (1024**3), 2)
            },
            "battery": {
                "percent": battery.percent if battery else None,
                "charging": battery.power_plugged if battery else None,
                "time_left": str(battery.secsleft // 60) + " minutes" if battery and battery.secsleft > 0 else "N/A"
            } if battery else None,
            "ai_optimization_recommendations": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Generate AI optimization recommendations
        if cpu_percent > 80:
            performance_data["ai_optimization_recommendations"].append({
                "type": "cpu_throttling",
                "message": "High CPU usage detected. Consider switching to Gemma 3n 2B model for better performance.",
                "severity": "medium"
            })
        
        if memory.percent > 85:
            performance_data["ai_optimization_recommendations"].append({
                "type": "memory_optimization",
                "message": "High memory usage. Enable aggressive memory cleanup and reduce context window size.",
                "severity": "high"
            })
        
        if battery and battery.percent < 20:
            performance_data["ai_optimization_recommendations"].append({
                "type": "battery_saving",
                "message": "Low battery. Switch to power-saving mode and reduce AI processing intensity.",
                "severity": "high"
            })
        
        return JSONResponse(content={
            "success": True,
            "performance": performance_data
        })
        
    except ImportError:
        # Fallback for systems without psutil
        return JSONResponse(content={
            "success": True,
            "performance": {
                "system": {"platform": "Unknown", "note": "Install psutil for detailed metrics"},
                "cpu": {"usage_percent": 45, "note": "Simulated data"},
                "memory": {"total_gb": 8, "used_percent": 67, "note": "Simulated data"},
                "battery": {"percent": 78, "charging": False, "note": "Simulated data"},
                "ai_optimization_recommendations": [],
                "timestamp": datetime.utcnow().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"❌ Error getting device performance: {str(e)}")
        return JSONResponse(
            content={"success": False, "error": str(e)}, 
            status_code=500
        )

# ================================================================================
# DEBUG & TESTING ROUTES
# ================================================================================

@app.get("/debug/patients")
async def debug_patients(db: Session = Depends(get_db)):
    """Debug endpoint to see all patients in database"""
    try:
        patients = db.query(TriagePatient).all()
            
        patient_data = [{
            "id": p.id, "name": p.name, "injury_type": p.injury_type,
            "severity": p.severity, "triage_color": p.triage_color,
            "status": p.status, "created_at": p.created_at.isoformat() if p.created_at else None
        } for p in patients]
            
        return JSONResponse(content={
            "total_patients": len(patients), "patients": patient_data,
            "message": f"Found {len(patients)} patients in database"
        })
            
    except Exception as e:
        logger.error(f"❌ Debug patients error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/create-test-patients-get")
async def create_test_patients_browser_friendly(db: Session = Depends(get_db)):
    """Browser-friendly GET version of create test patients"""
    try:
        existing_count = db.query(TriagePatient).count()
        if existing_count > 0:
            return HTMLResponse(content=f"""
            <html>
            <head><title>Test Patients</title></head>
            <body style="font-family: Arial; padding: 20px;">
                <h2>✅ Patients Already Exist</h2>
                <p>Found {existing_count} patients in database.</p>
                <p><a href="/debug/patients">View existing patients</a></p>
                <p><a href="/patient-list">Go to patient list</a></p>
                <p><a href="/patients/1/edit">Edit first patient</a></p>
            </body>
            </html>
            """)
            
        test_patients = [
            {
                "name": "John Smith", "age": 35, "gender": "Male", "injury_type": "Chest trauma",
                "consciousness": "alert", "breathing": "labored", "heart_rate": 120,
                "bp_systolic": 90, "bp_diastolic": 60, "severity": "critical",
                "triage_color": "red", "status": "active",
                "notes": "Motor vehicle accident victim, possible internal bleeding"
            },
            {
                "name": "Sarah Johnson", "age": 28, "gender": "Female", "injury_type": "Broken arm",
                "consciousness": "alert", "breathing": "normal", "heart_rate": 85,
                "bp_systolic": 120, "bp_diastolic": 80, "severity": "moderate",
                "triage_color": "yellow", "status": "active",
                "notes": "Fall from ladder, stable vital signs"
            },
            {
                "name": "Mike Wilson", "age": 45, "gender": "Male", "injury_type": "Minor cuts",
                "consciousness": "alert", "breathing": "normal", "heart_rate": 75,
                "bp_systolic": 125, "bp_diastolic": 82, "severity": "mild",
                "triage_color": "green", "status": "treated",
                "notes": "Glass cuts from broken window, cleaned and bandaged"
            }
        ]
            
        created_patients = []
        for patient_data in test_patients:
            new_patient = TriagePatient(
                **patient_data,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(new_patient)
            db.flush()
            created_patients.append({
                "id": new_patient.id, "name": new_patient.name,
                "triage_color": new_patient.triage_color
            })
            
        db.commit()
            
        logger.info(f"✅ Created {len(created_patients)} test patients")
            
        patient_links = ""
        for patient in created_patients:
            patient_links += f'<li><a href="/patients/{patient["id"]}/edit">Edit {patient["name"]} ({patient["triage_color"]} priority)</a></li>'
            
        return HTMLResponse(content=f"""
        <html>
        <head><title>Test Patients Created</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h2>✅ Test Patients Created Successfully!</h2>
            <p>Created {len(created_patients)} test patients.</p>
            
            <h3>Quick Links:</h3>
            <ul>
                {patient_links}
            </ul>
            
            <h3>Other Actions:</h3>
            <ul>
                <li><a href="/debug/patients">View all patients (JSON)</a></li>
                <li><a href="/patient-list">Patient List Dashboard</a></li>
                <li><a href="/triage-dashboard">Triage Dashboard</a></li>
                <li><a href="/admin">Admin Dashboard</a></li>
            </ul>
        </body>
        </html>
        """)
            
    except Exception as e:
        logger.error(f"❌ Error creating test patients: {str(e)}")
        db.rollback()
        return HTMLResponse(content=f"""
        <html>
        <head><title>Error</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h2>❌ Error Creating Test Patients</h2>
            <p>Error: {str(e)}</p>
            <p><a href="/debug/patients">Check existing patients</a></p>
        </body>
        </html>
        """, status_code=500)

# ================================================================================
# MAP & GEOLOCATION API ENDPOINTS
# ================================================================================

@app.get("/api/static-map")
async def api_static_map(
    latitude: float,
    longitude: float,
    width: int = 600,
    height: int = 400,
    zoom: int = 15,
    format: str = "png",
    user: dict = Depends(require_role(["admin", "responder"]))
):
    """Generate static map image for given coordinates"""
    try:
        if not (-90 <= latitude <= 90):
            raise HTTPException(status_code=400, detail="Invalid latitude (-90 to 90)")
        if not (-180 <= longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid longitude (-180 to 180)")
        if not (1 <= zoom <= 20):
            raise HTTPException(status_code=400, detail="Invalid zoom level (1 to 20)")
        if not (100 <= width <= 2000) or not (100 <= height <= 2000):
            raise HTTPException(status_code=400, detail="Invalid dimensions (100-2000px)")
            
        logger.info(f"Generating static map for {latitude}, {longitude} by user {user['username']}")
            
        result = generate_static_map_endpoint(latitude, longitude, width, height, zoom)
            
        if not result['success']:
            logger.warning(f"Map generation failed: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error', 'Map generation failed'))
            
        if format.lower() == 'json':
            return JSONResponse(content=result)
        elif format.lower() == 'base64':
            return {"image_data": result['image_data']}
        else:
            try:
                image_bytes = base64.b64decode(result['image_data'])
                return Response(
                    content=image_bytes,
                    media_type="image/png",
                    headers={
                        "Content-Disposition": f"inline; filename=emergency_map_{latitude}_{longitude}.png",
                        "Cache-Control": "public, max-age=3600"
                    }
                )
            except Exception as e:
                logger.error(f"Failed to decode map image: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to decode image: {e}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Static map API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/map-preview")
async def api_map_preview(
    latitude: float, 
    longitude: float,
    user: dict = Depends(require_role(["admin", "responder"]))
):
    """Get comprehensive map preview data for web interface"""
    try:
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinates")
            
        logger.info(f"Generating map preview for {latitude}, {longitude} by user {user['username']}")
            
        preview_data = generate_map_preview_data(latitude, longitude)
            
        if not preview_data['success']:
            logger.warning(f"Map preview failed: {preview_data.get('error')}")
            raise HTTPException(status_code=500, detail=preview_data.get('error', 'Preview generation failed'))
            
        return JSONResponse(content=preview_data)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Map preview API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/emergency-resources")
async def api_emergency_resources(
    latitude: float, 
    longitude: float, 
    radius: int = 25,
    type_filter: str = None,
    user: dict = Depends(require_role(["admin", "responder"]))
):
    """Find emergency resources near coordinates"""
    try:
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinates")
            
        if not (1 <= radius <= 100):
            raise HTTPException(status_code=400, detail="Radius must be between 1 and 100 km")
            
        valid_types = ['hospital', 'fire_station', 'police', 'evacuation_route']
        if type_filter and type_filter not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid type filter. Must be one of: {valid_types}")
            
        logger.info(f"Finding emergency resources near {latitude}, {longitude} within {radius}km by user {user['username']}")
            
        resources = get_emergency_resources(latitude, longitude, radius)
            
        if type_filter:
            resources = [r for r in resources if r.type == type_filter]
            
        resources_data = [{
            'name': resource.name, 'type': resource.type,
            'latitude': resource.latitude, 'longitude': resource.longitude,
            'distance_km': round(resource.distance_km, 2), 'estimated_time': resource.estimated_time,
            'capacity': resource.capacity, 'contact': resource.contact
        } for resource in resources]
            
        return JSONResponse(content={
            'success': True, 'search_center': {'latitude': latitude, 'longitude': longitude},
            'search_radius_km': radius, 'type_filter': type_filter,
            'total_found': len(resources_data), 'resources': resources_data,
            'generated_by': user['username'], 'timestamp': datetime.now().isoformat()
        })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Emergency resources API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================================
# SYSTEM HEALTH & UTILITIES
# ================================================================================

# Replace or update your existing health_check function with this enhanced version:
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint with Gemma 3n service status"""
    try:
        db = next(get_db())
        reports_count = db.query(CrowdReport).count()
        patients_count = db.query(TriagePatient).count()
        db_status = "connected"
    except:
        reports_count = 0
        patients_count = 0
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
            "edge_ai_optimization": True, "multimodal_processing": True
        },
        "database": {
            "status": db_status, "type": "SQLAlchemy with SQLite",
            "tables": ["crowd_reports", "triage_patients"],
            "records": {"reports": reports_count, "patients": patients_count}
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
            "voice_reports": "/api/submit-voice-emergency-report",
            "damage_assessment": "/api/submit-damage-assessment", 
            "context_analysis": "/api/context-analysis",
            "ai_model_status": "/api/ai-model-status",
            "ai_optimization": "/api/optimize-ai-settings",
            "device_performance": "/api/device-performance"
        },
        "gemma_3n_pages": {
            "voice_reporter": "/voice-emergency-reporter",
            "damage_assessment": "/multimodal-damage-assessment",
            "context_dashboard": "/context-intelligence-dashboard",
            "ai_settings": "/adaptive-ai-settings"
        }
    }

# ================================================================================
# ARCHIVE & EXPORT MANAGEMENT
# ================================================================================

@app.get("/reports/export", response_class=HTMLResponse)
async def export_archive_page(
    request: Request, 
    user: dict = Depends(require_role(["admin", "responder"]))
):
    """Export archive page with multiple export options"""
    try:
        db = next(get_db())
        all_reports = db.query(CrowdReport).all()
        all_patients = db.query(TriagePatient).all()
            
        export_stats = {
            "total_reports": len(all_reports), "total_patients": len(all_patients),
            "export_formats": ["PDF", "CSV", "JSON", "ZIP Archive"],
            "last_export": "Never", "export_size_estimate": f"{len(all_reports) + len(all_patients)} records"
        }
            
        return templates.TemplateResponse("export_archive.html", {
            "request": request, "stats": export_stats, "user": user
        })
            
    except Exception as e:
        logger.error(f"❌ Error loading export page: {str(e)}")
        # Simple fallback export page
        simple_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Export Archive</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
                .export-option {{ background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 8px; }}
                .export-option a {{ text-decoration: none; color: #1e40af; font-weight: bold; }}
                .export-option:hover {{ background: #e0e0e0; }}
            </style>
        </head>
        <body>
            <h1>📦 Export Archive</h1>
            <p>Export your disaster response data in multiple formats:</p>
            
            <div class="export-option">
                <a href="/export-reports.csv">📄 Download Reports as CSV</a>
                <p>All crowd reports in spreadsheet format</p>
            </div>
            
            <div class="export-option">
                <a href="/export-reports.json">🔧 Download Reports as JSON</a>
                <p>All crowd reports in JSON format for API integration</p>
            </div>
            
            <div class="export-option">
                <a href="/export-patients-pdf">🏥 Download Patient Records as PDF</a>
                <p>All triage patient records in PDF format</p>
            </div>
            
            <div class="export-option">
                <a href="/export-full-archive">📦 Download Complete Archive (ZIP)</a>
                <p>All data, reports, and system logs in a single ZIP file</p>
            </div>
            
            <p><a href="/admin">← Back to Admin Dashboard</a></p>
        </body>
        </html>
        """
        return HTMLResponse(content=simple_html)

@app.get("/export-full-archive")
async def export_full_archive(user: dict = Depends(require_role(["admin"]))):
    """Export complete system archive as ZIP file"""
    try:
        db = next(get_db())
            
        # Create ZIP file in memory
        zip_buffer = BytesIO()
            
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Export reports as CSV
            reports = db.query(CrowdReport).all()
            reports_csv = "id,message,tone,escalation,timestamp,user,location,latitude,longitude\n"
            for r in reports:
                reports_csv += f'"{r.id}","{r.message}","{r.tone}","{r.escalation}","{r.timestamp}","{r.user or ""}","{r.location or ""}","{r.latitude or ""}","{r.longitude or ""}"\n'
            zip_file.writestr("crowd_reports.csv", reports_csv)
            
            # Export patients as CSV
            patients = db.query(TriagePatient).all()
            patients_csv = "id,name,age,injury_type,severity,triage_color,status,created_at\n"
            for p in patients:
                patients_csv += f'"{p.id}","{p.name}","{p.age or ""}","{p.injury_type}","{p.severity}","{p.triage_color}","{p.status}","{p.created_at}"\n'
            zip_file.writestr("triage_patients.csv", patients_csv)
            
            # Add system info
            system_info = f"""Enhanced Disaster Response System Export
Generated: {datetime.utcnow().isoformat()}
Exported by: {user['username']} ({user['role']})
Total Reports: {len(reports)}
Total Patients: {len(patients)}
Export Type: Complete Archive
System Version: 2.2.0
Features: Enhanced mapping, demo data, real-time updates
"""
            zip_file.writestr("export_info.txt", system_info)
            
        zip_buffer.seek(0)
            
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_disaster_response_archive_{timestamp}.zip"
            
        logger.info(f"📦 Full archive exported by {user['username']}: {len(reports)} reports, {len(patients)} patients")
            
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
            
    except Exception as e:
        logger.error(f"❌ Archive export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Archive export failed: {str(e)}")

# ================================================================================
# APPLICATION STARTUP & ERROR HANDLERS
# ================================================================================

@app.on_event("startup")
async def startup_event():
    """Enhanced initialization with Gemma 3n capabilities"""
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
                timestamp=datetime.utcnow().isoformat()
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
    
    # NEW: Initialize Gemma 3n capabilities
    logger.info("🧠 Initializing Gemma 3n AI capabilities...")
    logger.info("   • Voice Emergency Reporter with real-time transcription")
    logger.info("   • Multimodal Damage Assessment (video/image/audio)")
    logger.info("   • Context Intelligence Dashboard (128K token window)")
    logger.info("   • Adaptive AI Settings for device optimization")
    
    logger.info("✅ Enhanced API server ready with comprehensive capabilities:")
    logger.info("   • Enhanced patient management (SQLAlchemy)")
    logger.info("   • Advanced crowd reports with geolocation (SQLAlchemy)")
    logger.info("   • Real-time map integration with demo data")
    logger.info("   • Multi-format export (CSV, JSON, KML, PDF)")
    logger.info("   • AI analysis & hazard detection")
    logger.info("   • Comprehensive analytics dashboards")
    logger.info("   • Demo data generation for presentations")
    logger.info("   • Network status monitoring & offline support")
    logger.info("   • Enhanced authentication & role management")
    logger.info("   🆕 GEMMA 3N: Voice emergency reporting")
    logger.info("   🆕 GEMMA 3N: Multimodal damage assessment")
    logger.info("   🆕 GEMMA 3N: 128K context intelligence")
    logger.info("   🆕 GEMMA 3N: Adaptive AI optimization")

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
    logger.error(f"Server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "message": "Please try again later",
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