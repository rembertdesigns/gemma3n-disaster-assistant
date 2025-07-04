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

from weasyprint import HTML, HTML as WeasyHTML
from jinja2 import Environment, FileSystemLoader
from io import BytesIO

# Core app utilities
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

# Map-related tools
from app.map_utils import (
    map_utils,
    generate_static_map,
    get_coordinate_formats,
    get_emergency_resources,
    get_map_metadata,
    MapConfig
)

# 🆕 Database session and models - SQLAlchemy ONLY
from app.database import get_db, engine
from app.models import CrowdReport, TriagePatient, Base

# ================================
# CONFIGURATION & SETUP
# ================================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Resolve static directory using absolute path
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

# ✅ Initialize FastAPI app
app = FastAPI(
    title="Disaster Response & Recovery Assistant",
    description="AI-Powered Emergency Analysis & Support with Interactive Maps",
    version="2.1.0"
)

# ✅ Mount static files safely
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ✅ Setup Jinja templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Global directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# UTILITY FUNCTIONS
# ================================

def get_time_ago(timestamp_str):
    """Calculate human-readable time ago from timestamp string"""
    try:
        if isinstance(timestamp_str, str):
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str.replace('Z', '+00:00')
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = timestamp_str
            
        now = datetime.utcnow()
        diff = now - timestamp.replace(tzinfo=None)
        
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
    except Exception:
        return "Unknown"

# ================================
# AUTHENTICATION ROUTES
# ================================

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me")
async def read_current_user(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "role": user["role"]}

# ================================
# MAIN PAGE ROUTES
# ================================

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "result": None})

@app.get("/hazards", response_class=HTMLResponse)
async def serve_hazard_page(request: Request):
    return templates.TemplateResponse("hazards.html", {"request": request, "result": None})

@app.get("/generate", response_class=HTMLResponse)
async def serve_generate_page(request: Request):
    return templates.TemplateResponse("generate.html", {"request": request})

@app.get("/live-generate", response_class=HTMLResponse)
async def serve_live_generate_page(request: Request):
    return templates.TemplateResponse("live_generate.html", {"request": request})

@app.get("/offline.html", response_class=HTMLResponse)
async def offline_page(request: Request):
    return templates.TemplateResponse("offline.html", {"request": request})

@app.get("/submit-report", response_class=HTMLResponse)
async def submit_report_page(request: Request):
    return templates.TemplateResponse("submit-report.html", {"request": request})

# ================================
# CROWD REPORTS ROUTES - SQLAlchemy Only
# ================================

@app.get("/view-reports", response_class=HTMLResponse)
async def view_reports(
    request: Request,
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """View crowd reports using SQLAlchemy"""
    try:
        query = db.query(CrowdReport)
        
        # Apply filters
        if tone:
            query = query.filter(CrowdReport.tone == tone)
        if escalation:
            query = query.filter(CrowdReport.escalation == escalation)
        if keyword:
            query = query.filter(CrowdReport.message.ilike(f"%{keyword}%"))
        
        reports = query.order_by(CrowdReport.timestamp.desc()).all()
        
        logger.info(f"📋 Loaded {len(reports)} crowd reports")
        
        return templates.TemplateResponse("view-reports.html", {
            "request": request,
            "reports": reports,
            "tone": tone,
            "escalation": escalation,
            "keyword": keyword
        })
        
    except Exception as e:
        logger.error(f"❌ Error loading reports: {str(e)}")
        return templates.TemplateResponse("view-reports.html", {
            "request": request,
            "reports": [],
            "error": str(e)
        })

@app.get("/submit-crowd-report", response_class=HTMLResponse)
async def submit_crowd_report_form(request: Request):
    return templates.TemplateResponse("submit-crowd-report.html", {"request": request})

@app.get("/map-reports", response_class=HTMLResponse)
async def map_reports_page(request: Request):
    return templates.TemplateResponse("map_reports.html", {"request": request})

@app.get("/crowd-reports", response_class=HTMLResponse)
async def view_crowd_reports(
    request: Request,
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Alternative crowd reports view"""
    query = db.query(CrowdReport)

    if tone:
        query = query.filter(CrowdReport.tone == tone)
    if escalation:
        query = query.filter(CrowdReport.escalation == escalation)
    if keyword:
        query = query.filter(CrowdReport.message.ilike(f"%{keyword}%"))

    reports = query.order_by(CrowdReport.timestamp.desc()).all()

    return templates.TemplateResponse("crowd_reports.html", {
        "request": request,
        "reports": reports,
        "tone": tone,
        "escalation": escalation,
        "keyword": keyword
    })

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
    """Submit crowd report using SQLAlchemy"""
    try:
        # Log coordinates if provided
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

        # Create new crowd report using SQLAlchemy
        new_report = CrowdReport(
            message=message,
            tone=tone,
            escalation=escalation,
            user=user,
            location=location,
            timestamp=datetime.utcnow().isoformat(),
            latitude=latitude,
            longitude=longitude
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

# ================================
# TRIAGE & PATIENT MANAGEMENT ROUTES - SQLAlchemy Only
# ================================

@app.get("/triage", response_class=HTMLResponse)
async def triage_form_page(request: Request):
    return templates.TemplateResponse("triage_form.html", {"request": request})

@app.get("/patients", response_class=HTMLResponse)
async def get_patient_tracker(
    request: Request, 
    severity: Optional[str] = None, 
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Patient tracker using SQLAlchemy"""
    try:
        query = db.query(TriagePatient)
        
        if severity:
            query = query.filter(TriagePatient.severity == severity)
        if status:
            query = query.filter(TriagePatient.status == status)
        
        patients = query.order_by(TriagePatient.created_at.desc()).all()
        
        return templates.TemplateResponse("patient_tracker.html", {
            "request": request,
            "patients": patients,
            "now": datetime.utcnow(),
            "severity_filter": severity,
            "status_filter": status
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
    """Enhanced patient list dashboard using SQLAlchemy"""
    try:
        query = db.query(TriagePatient)
        
        # Apply filters
        if triage_color:
            query = query.filter(TriagePatient.triage_color == triage_color)
        if status:
            query = query.filter(TriagePatient.status == status)
        if severity:
            query = query.filter(TriagePatient.severity == severity)
        
        patients = query.order_by(TriagePatient.created_at.desc()).all()
        
        # Custom sort by priority score
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
            "request": request,
            "patients": patients,
            "total_patients": total_patients,
            "active_patients": active_patients,
            "critical_patients": critical_patients,
            "color_counts": color_counts,
            "status_counts": status_counts,
            "filters": {
                "triage_color": triage_color,
                "status": status,
                "severity": severity
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error loading patient list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load patient list: {str(e)}")

@app.get("/triage-dashboard", response_class=HTMLResponse)
async def triage_dashboard_page(
    request: Request,
    db: Session = Depends(get_db)
):
    """Comprehensive triage dashboard with real-time statistics"""
    try:
        all_patients = db.query(TriagePatient).all()
        active_patients = db.query(TriagePatient).filter(TriagePatient.status == "active").all()
        
        stats = {
            "total_patients": len(all_patients),
            "active_patients": len(active_patients),
            "patients_today": len([
                p for p in all_patients 
                if p.created_at.date() == datetime.utcnow().date()
            ]),
            "critical_alerts": len([
                p for p in active_patients 
                if p.triage_color == "red" or p.severity == "critical"
            ])
        }
        
        triage_breakdown = {
            "red": {"count": len([p for p in active_patients if p.triage_color == "red"]), "percentage": 0},
            "yellow": {"count": len([p for p in active_patients if p.triage_color == "yellow"]), "percentage": 0},
            "green": {"count": len([p for p in active_patients if p.triage_color == "green"]), "percentage": 0},
            "black": {"count": len([p for p in active_patients if p.triage_color == "black"]), "percentage": 0}
        }
        
        # Calculate percentages
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
        
        recent_patients = [
            p for p in all_patients 
            if (datetime.utcnow() - p.created_at).total_seconds() < 86400
        ]
        recent_patients = sorted(recent_patients, key=lambda p: p.created_at, reverse=True)[:10]
        
        priority_queue = sorted(active_patients, key=lambda p: (p.priority_score, -p.id))[:15]
        
        logger.info(f"📊 Triage dashboard accessed: {stats['total_patients']} total, {stats['active_patients']} active")
        
        return templates.TemplateResponse("triage_dashboard.html", {
            "request": request,
            "stats": stats,
            "triage_breakdown": triage_breakdown,
            "severity_breakdown": severity_breakdown,
            "critical_vitals_patients": critical_vitals_patients,
            "recent_patients": recent_patients,
            "priority_queue": priority_queue,
            "current_time": datetime.utcnow()
        })
        
    except Exception as e:
        logger.error(f"❌ Error loading triage dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load dashboard: {str(e)}")

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
    """Submit a new triage assessment and save to database"""
    try:
        logger.info(f"🚑 Receiving triage submission for patient: {name}")
        
        # Validate required fields
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
            name=name.strip(),
            age=age,
            gender=gender,
            medical_id=medical_id,
            injury_type=injury_type.strip(),
            mechanism=mechanism,
            consciousness=consciousness,
            breathing=breathing,
            heart_rate=heart_rate,
            bp_systolic=bp_systolic,
            bp_diastolic=bp_diastolic,
            respiratory_rate=respiratory_rate,
            temperature=temperature,
            oxygen_sat=oxygen_sat,
            severity=severity,
            triage_color=triage_color,
            allergies=allergies,
            medications=medications,
            medical_history=medical_history,
            notes=notes,
            status="active",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)
        
        logger.info(f"✅ Triage patient saved: ID={new_patient.id}, Name={new_patient.name}, Color={new_patient.triage_color}")
        
        # Log critical cases
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

@app.post("/patients/{patient_id}/discharge")
async def discharge_patient(patient_id: int, db: Session = Depends(get_db)):
    """Discharge patient using SQLAlchemy"""
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

# ================================
# CROWD REPORTS API ROUTES - All SQLAlchemy
# ================================

@app.get("/api/community-stats", response_class=JSONResponse)
async def get_community_stats(db: Session = Depends(get_db)):
    """Get community statistics for dashboard widgets"""
    try:
        all_reports = db.query(CrowdReport).all()
        today_reports = [
            r for r in all_reports 
            if r.timestamp and datetime.fromisoformat(r.timestamp.replace('Z', '+00:00')).date() == datetime.utcnow().date()
        ]
        
        stats = {
            "total_reports": len(all_reports),
            "reports_today": len(today_reports),
            "critical_reports": len([r for r in all_reports if r.escalation == "Critical"]),
            "active_locations": len(set([r.location for r in all_reports if r.location])),
            "escalation_breakdown": {
                "Critical": len([r for r in all_reports if r.escalation == "Critical"]),
                "High": len([r for r in all_reports if r.escalation == "High"]),
                "Moderate": len([r for r in all_reports if r.escalation == "Moderate"]),
                "Low": len([r for r in all_reports if r.escalation == "Low"])
            },
            "tone_breakdown": {
                "Urgent": len([r for r in all_reports if r.tone == "Urgent"]),
                "Frantic": len([r for r in all_reports if r.tone == "Frantic"]),
                "Helpless": len([r for r in all_reports if r.tone == "Helpless"]),
                "Descriptive": len([r for r in all_reports if r.tone == "Descriptive"])
            }
        }
        
        logger.info(f"📊 Community stats requested: {stats['total_reports']} total reports")
        
        return JSONResponse(content={
            "success": True,
            "stats": stats,
            "generated_at": datetime.utcnow().isoformat()
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
            
        recent_reports = db.query(CrowdReport).order_by(
            CrowdReport.timestamp.desc()
        ).limit(limit).all()
        
        reports_data = []
        for report in recent_reports:
            reports_data.append({
                "id": report.id,
                "message": report.message,
                "tone": report.tone,
                "escalation": report.escalation,
                "user": report.user or "Anonymous",
                "location": report.location,
                "timestamp": report.timestamp,
                "latitude": report.latitude,
                "longitude": report.longitude,
                "time_ago": get_time_ago(report.timestamp) if report.timestamp else "Unknown"
            })
        
        logger.info(f"📋 Recent reports API called: returned {len(reports_data)} reports")
        
        return JSONResponse(content={
            "success": True,
            "reports": reports_data,
            "count": len(reports_data),
            "generated_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting recent reports: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/api/triage-stats", response_class=JSONResponse)
async def get_triage_stats(db: Session = Depends(get_db)):
    """Get triage statistics for dashboard widgets"""
    try:
        all_patients = db.query(TriagePatient).all()
        active_patients = [p for p in all_patients if p.status == "active"]
        today_patients = [
            p for p in all_patients 
            if p.created_at.date() == datetime.utcnow().date()
        ]
        
        stats = {
            "total_patients": len(all_patients),
            "active_patients": len(active_patients),
            "patients_today": len(today_patients),
            "critical_alerts": len([
                p for p in active_patients 
                if p.triage_color == "red" or p.severity == "critical" or p.is_critical_vitals
            ]),
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
            "success": True,
            "stats": stats,
            "generated_at": datetime.utcnow().isoformat()
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
        active_patients = db.query(TriagePatient).filter(
            TriagePatient.status == "active"
        ).all()
        
        priority_patients = sorted(active_patients, key=lambda p: (p.priority_score, -p.id))[:limit]
        
        queue_data = []
        for patient in priority_patients:
            queue_data.append({
                "id": patient.id,
                "name": patient.name,
                "injury_type": patient.injury_type,
                "severity": patient.severity,
                "triage_color": patient.triage_color,
                "priority_score": patient.priority_score,
                "critical_vitals": patient.is_critical_vitals,
                "created_at": patient.created_at.isoformat(),
                "time_ago": get_time_ago(patient.created_at.isoformat())
            })
        
        logger.info(f"🚨 Priority queue requested: {len(queue_data)} patients")
        
        return JSONResponse(content={
            "success": True,
            "queue": queue_data,
            "count": len(queue_data),
            "generated_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting priority queue: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/api/crowd-report-locations", response_class=JSONResponse)
async def crowd_report_locations(
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get crowd reports with GPS coordinates for mapping"""
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
        
        reports_data = [{
            "id": report.id,
            "message": report.message,
            "user": report.user,
            "timestamp": report.timestamp,
            "tone": report.tone,
            "escalation": report.escalation,
            "location": report.location,
            "latitude": float(report.latitude),
            "longitude": float(report.longitude),
        } for report in reports]
        
        return JSONResponse(content={"reports": reports_data})
        
    except Exception as e:
        logger.error(f"❌ Error getting crowd report locations: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/export-reports/pdf")
async def export_reports_pdf(
    request: Request,
    tone: str = Form(None),
    escalation: str = Form(None),
    keyword: str = Form(None),
    db: Session = Depends(get_db)
):
    """Export filtered reports as PDF"""
    try:
        query = db.query(CrowdReport)

        if tone:
            query = query.filter(CrowdReport.tone == tone)
        if escalation:
            query = query.filter(CrowdReport.escalation == escalation)
        if keyword:
            query = query.filter(CrowdReport.message.ilike(f"%{keyword}%"))

        reports = query.order_by(CrowdReport.timestamp.desc()).all()

        # Inject base64 map images
        for report in reports:
            if hasattr(report, 'latitude') and hasattr(report, 'longitude'):
                if report.latitude and report.longitude:
                    try:
                        map_bytes = generate_map_image(report.latitude, report.longitude, str(report.id))
                        report.map_image_base64 = base64.b64encode(map_bytes).decode("utf-8")
                    except Exception as e:
                        report.map_image_base64 = None
                else:
                    report.map_image_base64 = None
            else:
                report.map_image_base64 = None

        html_content = templates.get_template("export_pdf.html").render({
            "reports": reports
        })

        pdf_io = BytesIO()
        HTML(string=html_content).write_pdf(pdf_io)
        pdf_io.seek(0)

        return StreamingResponse(
            pdf_io,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline; filename=crowd_reports.pdf"}
        )
        
    except Exception as e:
        logger.error(f"❌ Error exporting reports PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export PDF")

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
            "id": r.id,
            "message": r.message,
            "tone": r.tone,
            "escalation": r.escalation,
            "timestamp": r.timestamp,
            "user": r.user,
            "location": r.location,
            "latitude": r.latitude,
            "longitude": r.longitude
        } for r in reports]

        return {"reports": report_list}
        
    except Exception as e:
        logger.error(f"❌ Error exporting reports JSON: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export JSON")

# ================================
# PATIENT EXPORT ROUTES - SQLAlchemy
# ================================

@app.get("/export-patients-pdf")
async def export_patients_pdf(db: Session = Depends(get_db)):
    """Export patients as PDF using SQLAlchemy"""
    try:
        patients = db.query(TriagePatient).order_by(TriagePatient.created_at.desc()).all()
        
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("patient_tracker.html")
        html_out = template.render(
            patients=patients, 
            now=datetime.utcnow(), 
            severity_filter=None, 
            status_filter=None
        )
        
        pdf_path = os.path.join("outputs", f"patients_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
        WeasyHTML(string=html_out).write_pdf(pdf_path)
        
        return FileResponse(pdf_path, filename="patient_tracker.pdf", media_type="application/pdf")
        
    except Exception as e:
        logger.error(f"❌ Error exporting patients PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export patients PDF")

# ================================
# MAP API ROUTES
# ================================

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
        
        resources_data = [
            {
                'name': resource.name,
                'type': resource.type,
                'latitude': resource.latitude,
                'longitude': resource.longitude,
                'distance_km': round(resource.distance_km, 2),
                'estimated_time': resource.estimated_time,
                'capacity': resource.capacity,
                'contact': resource.contact
            }
            for resource in resources
        ]
        
        return JSONResponse(content={
            'success': True,
            'search_center': {'latitude': latitude, 'longitude': longitude},
            'search_radius_km': radius,
            'type_filter': type_filter,
            'total_found': len(resources_data),
            'resources': resources_data,
            'generated_by': user['username'],
            'timestamp': datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Emergency resources API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/coordinate-formats")
async def api_coordinate_formats(
    latitude: float, 
    longitude: float,
    user: dict = Depends(require_role(["admin", "responder"]))
):
    """Get coordinates in multiple formats"""
    try:
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinates")
        
        logger.info(f"Generating coordinate formats for {latitude}, {longitude} by user {user['username']}")
        
        formats = get_coordinate_formats(latitude, longitude)
        
        return JSONResponse(content={
            'success': True,
            'input_coordinates': {'latitude': latitude, 'longitude': longitude},
            'formats': formats,
            'generated_by': user['username'],
            'timestamp': datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Coordinate formats API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/map-metadata")
async def api_map_metadata(
    latitude: float,
    longitude: float, 
    user: dict = Depends(require_role(["admin", "responder"]))
):
    """Get comprehensive location metadata for emergency planning"""
    try:
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinates")
        
        logger.info(f"Generating map metadata for {latitude}, {longitude} by user {user['username']}")
        
        metadata = get_map_metadata(latitude, longitude)
        
        metadata['request_info'] = {
            'generated_by': user['username'],
            'user_role': user['role'],
            'timestamp': datetime.now().isoformat(),
            'map_service': map_utils.preferred_service.value
        }
        
        return JSONResponse(content=metadata)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Map metadata API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/map-config")
async def get_map_config(user: dict = Depends(require_role(["admin", "responder"]))):
    """Get current map configuration"""
    return {
        "preferred_service": map_utils.preferred_service.value,
        "available_services": [
            service.value for service in map_utils.api_keys.keys() 
            if map_utils.api_keys[service]
        ],
        "default_config": {
            "width": map_utils.config.width,
            "height": map_utils.config.height,
            "zoom": map_utils.config.zoom,
            "map_type": map_utils.config.map_type,
            "marker_color": map_utils.config.marker_color
        },
        "user": user['username']
    }

# ================================
# ANALYSIS & REPORTING ROUTES
# ================================

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
                "request": request,
                "result": transcription,
                "input_text": None
            })

        input_payload = {"type": "text", "content": transcription["text"]}
        hazards = transcription.get("hazards", [])

    else:
        input_payload = {"type": "text", "content": report_text.strip()}

    processed = preprocess_input(input_payload)
    result = run_disaster_analysis(processed)

    return templates.TemplateResponse("home.html", {
        "request": request,
        "result": result,
        "original_input": input_payload["content"],
        "hazards": hazards
    })

@app.post("/export-pdf")
async def export_pdf(request: Request, report_text: str = Form(...)):
    """Export analysis as PDF"""
    html_content = templates.get_template("pdf_template.html").render({
        "report_text": report_text
    })
    pdf_path = os.path.join(OUTPUT_DIR, f"report_{uuid.uuid4().hex}.pdf")
    WeasyHTML(string=html_content).write_pdf(pdf_path)

    return templates.TemplateResponse("pdf_success.html", {
        "request": request,
        "pdf_url": f"/{pdf_path}"
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
            pdf_path, 
            media_type="application/pdf", 
            filename="emergency_incident_report.pdf"
        )
        
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

# ================================
# HAZARD DETECTION ROUTES
# ================================

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

# ================================
# RISK PREDICTION ROUTES
# ================================

@app.post("/predict-risk")
async def predict_risk_api(payload: dict = Body(...)):
    """Predict risk scores based on location, weather, and hazard type"""
    location = payload.get("location", {})
    weather = payload.get("weather", {})
    hazard = payload.get("hazard_type", "unknown")

    result = calculate_risk_score(location, weather, hazard)
    return JSONResponse(content=result)

# ================================
# EMERGENCY BROADCAST ROUTES
# ================================

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

# ================================
# UTILITY & HEALTH ROUTES
# ================================

@app.get("/health")
async def health_check():
    """Health check endpoint with service status"""
    return {
        "status": "healthy",
        "service": "Disaster Response Assistant",
        "version": "2.1.0",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "authentication": True,
            "maps": True,
            "emergency_resources": True,
            "coordinate_formats": True,
            "ai_analysis": True,
            "hazard_detection": True,
            "audio_transcription": True,
            "pdf_generation": True,
            "patient_management": True,
            "crowd_reports": True
        },
        "database": {
            "status": "connected",
            "type": "SQLAlchemy with SQLite"
        },
        "map_service": {
            "preferred": map_utils.preferred_service.value,
            "available_services": [
                service.value for service in map_utils.api_keys.keys() 
                if map_utils.api_keys[service]
            ]
        }
    }

# ================================
# STARTUP EVENT & ERROR HANDLERS
# ================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting Disaster Response Assistant API Server v2.1")
    logger.info(f"📍 Map service: {map_utils.preferred_service.value}")
    logger.info("🗺️ Map utilities initialized")
    
    # Create necessary directories
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create all SQLAlchemy tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ SQLAlchemy tables created/verified")
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
    
    # Log available map services
    available_services = [
        service.value for service in map_utils.api_keys.keys() 
        if map_utils.api_keys[service]
    ]
    logger.info(f"🔑 Available map services: {available_services or ['OpenStreetMap (free)']}")
    
    logger.info("✅ API server ready with enhanced capabilities:")
    logger.info("   • Patient management (SQLAlchemy)")
    logger.info("   • Crowd reports (SQLAlchemy)")
    logger.info("   • Map integration")
    logger.info("   • AI analysis & hazard detection")
    logger.info("   • Real-time dashboards")

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    logger.error(f"Server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "Please try again later"}
    )