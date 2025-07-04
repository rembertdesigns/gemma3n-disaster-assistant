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
from app.db import (
    get_db_connection,
    save_report_metadata,
    get_all_reports,
    get_report_by_id,
    get_dashboard_stats, 
    run_migrations
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

# 🆕 Database session and models for crowd report queue
from app.database import get_db  # Your SQLAlchemy session
from app.models import CrowdReport, TriagePatient

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

# Global storage and directories
crowd_reports = []  # In-memory store; consider replacing with DB later
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

@app.get("/view-reports", response_class=HTMLResponse)
async def view_reports(request: Request):
    conn = get_db_connection()
    reports = get_all_reports(conn)
    return templates.TemplateResponse("view-reports.html", {
        "request": request,
        "reports": reports
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

# ================================
# TRIAGE & PATIENT MANAGEMENT ROUTES
# ================================

@app.get("/triage", response_class=HTMLResponse)
async def triage_form_page(request: Request):
    return templates.TemplateResponse("triage_form.html", {"request": request})

@app.get("/patients", response_class=HTMLResponse)
async def get_patient_tracker(request: Request, severity: Optional[str] = None, status: Optional[str] = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM triage_patients WHERE 1=1"
    params = []

    if severity:
        query += " AND severity = ?"
        params.append(severity)

    if status:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY timestamp DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    now = datetime.utcnow()
    return templates.TemplateResponse("patient_tracker.html", {
        "request": request,
        "patients": rows,
        "now": now,
        "severity_filter": severity,
        "status_filter": status
    })

@app.post("/submit-triage")
async def submit_triage(
    request: Request,
    db: Session = Depends(get_db),
    # Patient Information
    name: str = Form(...),  # Required
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    medical_id: Optional[str] = Form(None),
    # Medical Assessment
    injury_type: str = Form(...),  # Required
    mechanism: Optional[str] = Form(None),
    consciousness: str = Form(...),  # Required
    breathing: str = Form(...),  # Required
    # Vital Signs
    heart_rate: Optional[int] = Form(None),
    bp_systolic: Optional[int] = Form(None),
    bp_diastolic: Optional[int] = Form(None),
    respiratory_rate: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    oxygen_sat: Optional[int] = Form(None),
    # Assessment
    severity: str = Form(...),  # Required
    triage_color: str = Form(...),  # Required
    # Additional Information
    allergies: Optional[str] = Form(None),
    medications: Optional[str] = Form(None),
    medical_history: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    # Timestamp from form
    assessment_timestamp: Optional[str] = Form(None)
):
    """
    Submit a new triage assessment and save to database
    """
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
        
        # Validate vital signs ranges (if provided)
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
            # Patient Information
            name=name.strip(),
            age=age,
            gender=gender,
            medical_id=medical_id,
            # Medical Assessment
            injury_type=injury_type.strip(),
            mechanism=mechanism,
            consciousness=consciousness,
            breathing=breathing,
            # Vital Signs
            heart_rate=heart_rate,
            bp_systolic=bp_systolic,
            bp_diastolic=bp_diastolic,
            respiratory_rate=respiratory_rate,
            temperature=temperature,
            oxygen_sat=oxygen_sat,
            # Assessment
            severity=severity,
            triage_color=triage_color,
            # Additional Information
            allergies=allergies,
            medications=medications,
            medical_history=medical_history,
            notes=notes,
            # System fields
            status="active",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Save to database
        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)
        
        logger.info(f"✅ Triage patient saved: ID={new_patient.id}, Name={new_patient.name}, Color={new_patient.triage_color}")
        
        # Log critical cases for immediate attention
        if triage_color == "red" or severity == "critical":
            logger.warning(f"🚨 CRITICAL PATIENT ALERT: {name} - {triage_color.upper()} triage, {severity} severity")
        
        # Return success response
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
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        logger.error(f"❌ Error saving triage patient: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to save triage assessment: {str(e)}"
        )

@app.post("/patients/{patient_id}/discharge")
async def discharge_patient(patient_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE triage_patients SET status = 'Discharged' WHERE id = ?", (patient_id,))
    conn.commit()
    conn.close()
    return RedirectResponse(url="/patients", status_code=303)

@app.get("/patient-list", response_class=HTMLResponse)
async def patient_list_page(
    request: Request,
    triage_color: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Enhanced patient list dashboard using SQLAlchemy
    """
    try:
        # Query triage patients with filters
        query = db.query(TriagePatient)
        
        # Apply filters
        if triage_color:
            query = query.filter(TriagePatient.triage_color == triage_color)
        if status:
            query = query.filter(TriagePatient.status == status)
        if severity:
            query = query.filter(TriagePatient.severity == severity)
        
        # Order by priority (red first, then by creation time)
        patients = query.order_by(
            TriagePatient.triage_color.desc(),  # This will need custom ordering
            TriagePatient.created_at.desc()
        ).all()
        
        # Custom sort by priority score (red=1, yellow=2, green=3, black=4)
        patients = sorted(patients, key=lambda p: (p.priority_score, -p.id))
        
        # Calculate statistics
        total_patients = len(patients)
        active_patients = len([p for p in patients if p.status == "active"])
        critical_patients = len([p for p in patients if p.triage_color == "red"])
        
        # Count by triage color
        color_counts = {
            "red": len([p for p in patients if p.triage_color == "red"]),
            "yellow": len([p for p in patients if p.triage_color == "yellow"]),
            "green": len([p for p in patients if p.triage_color == "green"]),
            "black": len([p for p in patients if p.triage_color == "black"])
        }
        
        # Count by status
        status_counts = {
            "active": len([p for p in patients if p.status == "active"]),
            "in_treatment": len([p for p in patients if p.status == "in_treatment"]),
            "treated": len([p for p in patients if p.status == "treated"]),
            "discharged": len([p for p in patients if p.status == "discharged"])
        }
        
        logger.info(f"📋 Patient list accessed: {total_patients} total, {active_patients} active, {critical_patients} critical")
        
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
    """
    Comprehensive triage dashboard with real-time statistics
    """
    try:
        # Get all active patients
        all_patients = db.query(TriagePatient).all()
        active_patients = db.query(TriagePatient).filter(TriagePatient.status == "active").all()
        
        # Calculate comprehensive statistics
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
        
        # Triage color breakdown
        triage_breakdown = {
            "red": {
                "count": len([p for p in active_patients if p.triage_color == "red"]),
                "percentage": 0
            },
            "yellow": {
                "count": len([p for p in active_patients if p.triage_color == "yellow"]),
                "percentage": 0
            },
            "green": {
                "count": len([p for p in active_patients if p.triage_color == "green"]),
                "percentage": 0
            },
            "black": {
                "count": len([p for p in active_patients if p.triage_color == "black"]),
                "percentage": 0
            }
        }
        
        # Calculate percentages
        if stats["active_patients"] > 0:
            for color in triage_breakdown:
                triage_breakdown[color]["percentage"] = round(
                    (triage_breakdown[color]["count"] / stats["active_patients"]) * 100, 1
                )
        
        # Severity breakdown
        severity_breakdown = {
            "critical": len([p for p in active_patients if p.severity == "critical"]),
            "severe": len([p for p in active_patients if p.severity == "severe"]),
            "moderate": len([p for p in active_patients if p.severity == "moderate"]),
            "mild": len([p for p in active_patients if p.severity == "mild"])
        }
        
        # Critical vitals alerts
        critical_vitals_patients = [p for p in active_patients if p.is_critical_vitals]
        
        # Recent patients (last 24 hours)
        recent_patients = [
            p for p in all_patients 
            if (datetime.utcnow() - p.created_at).total_seconds() < 86400  # 24 hours
        ]
        recent_patients = sorted(recent_patients, key=lambda p: p.created_at, reverse=True)[:10]
        
        # Priority queue (active patients sorted by priority)
        priority_queue = sorted(
            active_patients, 
            key=lambda p: (p.priority_score, -p.id)
        )[:15]  # Top 15 priority patients
        
        # Hourly activity (last 24 hours)
        hourly_activity = {}
        now = datetime.utcnow()
        for i in range(24):
            hour_start = now - timedelta(hours=i+1)
            hour_end = now - timedelta(hours=i)
            hour_patients = len([
                p for p in all_patients 
                if hour_start <= p.created_at < hour_end
            ])
            hourly_activity[hour_start.strftime("%H:00")] = hour_patients
        
        logger.info(f"📊 Triage dashboard accessed: {stats['total_patients']} total, {stats['active_patients']} active")
        
        return templates.TemplateResponse("triage_dashboard.html", {
            "request": request,
            "stats": stats,
            "triage_breakdown": triage_breakdown,
            "severity_breakdown": severity_breakdown,
            "critical_vitals_patients": critical_vitals_patients,
            "recent_patients": recent_patients,
            "priority_queue": priority_queue,
            "hourly_activity": hourly_activity,
            "current_time": datetime.utcnow()
        })
        
    except Exception as e:
        logger.error(f"❌ Error loading triage dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load dashboard: {str(e)}")

@app.get("/export-patients-pdf")
async def export_patients_pdf():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM triage_patients ORDER BY timestamp DESC")
    patients = cursor.fetchall()
    conn.close()
    
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

# ================================
# CROWD REPORTS API ROUTES
# ================================

@app.get("/api/reports", response_class=JSONResponse)
async def filtered_reports(
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None)
):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM reports WHERE 1=1"
    params = []

    if tone:
        query += " AND tone = ?"
        params.append(tone)

    if escalation:
        query += " AND escalation = ?"
        params.append(escalation)

    if keyword:
        query += " AND message LIKE ?"
        params.append(f"%{keyword}%")

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    reports = [{
        "id": row["id"],
        "message": row["message"],
        "tone": row["tone"],
        "escalation": row["escalation"],
        "timestamp": row["timestamp"],
        "user": row["user"]
    } for row in rows]

    return JSONResponse(content={"reports": reports})

@app.post("/api/submit-crowd-report")
async def submit_crowd_report(
    request: Request,
    message: str = Form(...),
    tone: Optional[str] = Form(None),
    escalation: str = Form(...),
    user: Optional[str] = Form("Anonymous"),
    location: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    latitude: Optional[str] = Form(None),
    longitude: Optional[str] = Form(None)
):
    # Log lat/lon
    if latitude and longitude:
        logger.info(f"📍 Received crowd report from location: {latitude}, {longitude}")

    if not tone:
        tone = analyze_sentiment(message)

    timestamp = datetime.utcnow().isoformat()
    image_path = None

    if image and image.filename:
        ext = os.path.splitext(image.filename)[1]
        image_path = os.path.join("uploads", f"crowd_{uuid.uuid4().hex}{ext}")
        with open(image_path, "wb") as f:
            f.write(await image.read())

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (message, tone, escalation, timestamp, user, location, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (message, tone, escalation, timestamp, user, location, image_path))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to insert crowd report: {e}")
        raise HTTPException(status_code=500, detail="Error saving report")

    return RedirectResponse(url="/view-reports", status_code=303)

@app.get("/api/crowd-report-locations", response_class=JSONResponse)
async def crowd_report_locations(
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None)
):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT id, message, user, timestamp, tone, escalation, location, latitude, longitude
        FROM crowd_reports
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    """
    params = []

    if tone:
        query += " AND tone = ?"
        params.append(tone)
    if escalation:
        query += " AND escalation = ?"
        params.append(escalation)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    reports = [{
        "id": row["id"],
        "message": row["message"],
        "user": row["user"],
        "timestamp": row["timestamp"],
        "tone": row["tone"],
        "escalation": row["escalation"],
        "location": row["location"],
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
    } for row in rows]

    return JSONResponse(content={"reports": reports})

@app.post("/api/submit-report")
async def submit_crowd_report_api(payload: dict = Body(...)):
    message = payload.get("message", "")
    location = payload.get("location", {})
    user = payload.get("user", "anonymous")

    sentiment_result = analyze_sentiment(message)

    report = {
        "id": str(uuid.uuid4()),
        "message": message,
        "location": location,
        "user": user,
        "timestamp": datetime.now().isoformat(),
        "sentiment": sentiment_result.get("sentiment"),
        "tone": sentiment_result.get("tone"),
        "escalation": sentiment_result.get("escalation"),
    }

    crowd_reports.append(report)
    return JSONResponse(content={"success": True, "report": report})

@app.post("/export-reports/pdf")
async def export_reports_pdf(
    request: Request,
    tone: str = Form(None),
    escalation: str = Form(None),
    keyword: str = Form(None),
    db: Session = Depends(get_db)
):
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

@app.get("/export-reports.csv")
async def export_reports_csv(
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(CrowdReport)

    if tone:
        query = query.filter(CrowdReport.tone == tone)
    if escalation:
        query = query.filter(CrowdReport.escalation == escalation)
    if keyword:
        query = query.filter(CrowdReport.message.ilike(f"%{keyword}%"))

    reports = query.order_by(CrowdReport.timestamp.desc()).all()

    # Build CSV
    csv_data = "id,message,tone,escalation,timestamp,user,location,image_url\n"
    for r in reports:
        csv_data += f'"{r.id}","{r.message}","{r.tone}","{r.escalation}","{r.timestamp}","{r.user or ""}","{r.location or ""}","{r.image_url or ""}"\n'

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=crowd_reports.csv"}
    )

@app.get("/export-reports.json", response_class=JSONResponse)
async def export_reports_json(
    tone: Optional[str] = Query(None),
    escalation: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
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
        "image_url": r.image_url
    } for r in reports]

    return {"reports": report_list}

# ================================
# ADMIN DASHBOARD ROUTES
# ================================

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    conn = get_db_connection()
    stats = get_dashboard_stats(conn)
    conn.close()
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "username": "Test Admin",
        "role": "admin",
        "stats": stats
    })

@app.get("/reports", response_class=HTMLResponse)
async def list_reports(request: Request, user: dict = Depends(require_role(["admin"]))):
    conn = get_db_connection()
    rows = get_all_reports(conn)
    conn.close()
    return templates.TemplateResponse("reports.html", {"request": request, "reports": rows})

@app.get("/reports/{report_id}")
async def download_report(report_id: str, user: dict = Depends(require_role(["admin"]))):
    conn = get_db_connection()
    report = get_report_by_id(conn, report_id)
    conn.close()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    file_path = os.path.join(OUTPUT_DIR, report["filename"])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="application/pdf", filename=report["filename"])

@app.post("/reports/{report_id}/status")
async def update_report_status(
    report_id: str,
    new_status: str = Form(...),
    user: dict = Depends(require_role(["admin"]))
):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE reports SET status = ? WHERE id = ?", (new_status, report_id))
    conn.commit()
    conn.close()
    return RedirectResponse("/reports", status_code=303)

@app.get("/reports/export")
async def export_reports_zip(user: dict = Depends(require_role(["admin"]))):
    conn = get_db_connection()
    reports = get_all_reports(conn)
    conn.close()

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for report in reports:
            pdf_path = os.path.join(OUTPUT_DIR, report["filename"])
            if os.path.exists(pdf_path):
                zipf.write(pdf_path, arcname=f"{report['id']}.pdf")

        metadata = [{
            "id": r["id"],
            "location": r["location"],
            "severity": r["severity"],
            "timestamp": r["timestamp"],
            "user": r["user"],
            "status": r["status"]
        } for r in reports]

        zipf.writestr("metadata.json", json.dumps(metadata, indent=2))

    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
        "Content-Disposition": "attachment; filename=report_archive.zip"
    })

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
    user: dict = Depends(require_role(["admin", "responder"]))  # Require auth for maps
):
    """
    Generate static map image for given coordinates
    
    Query params:
    - latitude: Incident latitude (-90 to 90)
    - longitude: Incident longitude (-180 to 180)
    - width: Map width in pixels (default: 600)
    - height: Map height in pixels (default: 400)
    - zoom: Map zoom level 1-20 (default: 15)
    - format: Response format ('png', 'json', 'base64')
    """
    try:
        # Validate coordinates
        if not (-90 <= latitude <= 90):
            raise HTTPException(status_code=400, detail="Invalid latitude (-90 to 90)")
        if not (-180 <= longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid longitude (-180 to 180)")
        if not (1 <= zoom <= 20):
            raise HTTPException(status_code=400, detail="Invalid zoom level (1 to 20)")
        if not (100 <= width <= 2000) or not (100 <= height <= 2000):
            raise HTTPException(status_code=400, detail="Invalid dimensions (100-2000px)")
        
        logger.info(f"Generating static map for {latitude}, {longitude} by user {user['username']}")
        
        # Generate map using our utilities
        result = generate_static_map_endpoint(latitude, longitude, width, height, zoom)
        
        if not result['success']:
            logger.warning(f"Map generation failed: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error', 'Map generation failed'))
        
        if format.lower() == 'json':
            # Return JSON with base64 image data
            return JSONResponse(content=result)
        
        elif format.lower() == 'base64':
            # Return just base64 string
            return {"image_data": result['image_data']}
        
        else:
            # Return raw image bytes
            try:
                image_bytes = base64.b64decode(result['image_data'])
                return Response(
                    content=image_bytes,
                    media_type="image/png",
                    headers={
                        "Content-Disposition": f"inline; filename=emergency_map_{latitude}_{longitude}.png",
                        "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
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
    """
    Get comprehensive map preview data for web interface
    
    Returns:
    - Coordinate formats (decimal, DMS, UTM, MGRS, Plus Codes, etc.)
    - Emergency resources within 25km
    - Location analysis metadata
    """
    try:
        # Validate coordinates
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinates")
        
        logger.info(f"Generating map preview for {latitude}, {longitude} by user {user['username']}")
        
        # Generate preview data using our utilities
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
    """
    Find emergency resources near coordinates
    
    Query params:
    - latitude, longitude: Search center
    - radius: Search radius in kilometers (1-100, default: 25)
    - type_filter: Filter by resource type (hospital, fire_station, police, evacuation_route)
    """
    try:
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinates")
        
        if not (1 <= radius <= 100):
            raise HTTPException(status_code=400, detail="Radius must be between 1 and 100 km")
        
        valid_types = ['hospital', 'fire_station', 'police', 'evacuation_route']
        if type_filter and type_filter not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid type filter. Must be one of: {valid_types}")
        
        logger.info(f"Finding emergency resources near {latitude}, {longitude} within {radius}km by user {user['username']}")
        
        # Get emergency resources
        resources = get_emergency_resources(latitude, longitude, radius)
        
        # Filter by type if specified
        if type_filter:
            resources = [r for r in resources if r.type == type_filter]
        
        # Convert to JSON-serializable format
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
    """
    Get coordinates in multiple formats
    
    Returns all supported coordinate system formats:
    - Decimal Degrees, DMS, UTM, MGRS, Plus Codes, Emergency Grid
    """
    try:
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinates")
        
        logger.info(f"Generating coordinate formats for {latitude}, {longitude} by user {user['username']}")
        
        # Get coordinate formats using our utilities
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
    """
    Get comprehensive location metadata for emergency planning
    
    Returns:
    - Coordinate formats
    - Emergency resources
    - Location analysis (terrain, accessibility, risk factors)
    - Map service information
    """
    try:
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinates")
        
        logger.info(f"Generating map metadata for {latitude}, {longitude} by user {user['username']}")
        
        # Get comprehensive metadata
        metadata = get_map_metadata(latitude, longitude)
        
        # Add user context
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
            return templates.TemplateResponse("index.html", {
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

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "original_input": input_payload["content"],
        "hazards": hazards
    })

@app.post("/export-pdf")
async def export_pdf(request: Request, report_text: str = Form(...)):
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
    """
    ENHANCED: Generate PDF report with MAP integration
    """
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

    # NEW: Enhanced PDF generation with maps
    logger.info(f"Generating enhanced PDF report with maps for user {user['username']}")
    
    try:
        pdf_path = generate_report_pdf(payload)
        
        # Enhanced metadata with map information
        metadata = {
            "id": str(uuid.uuid4()),
            "timestamp": payload.get("timestamp", datetime.now().isoformat()),
            "location": payload.get("location", "Unknown"),
            "severity": payload.get("severity", "N/A"),
            "filename": os.path.basename(pdf_path),
            "user": user["username"],
            "status": "submitted",
            "image_url": payload.get("image_url"),
            "checklist": payload.get("checklist", []),
            # NEW: Map metadata
            "has_coordinates": bool(payload.get("coordinates")),
            "gps_accuracy": payload.get("gps_accuracy"),
            "map_included": bool(payload.get("coordinates"))
        }
        
        save_report_metadata(metadata)
        
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
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)
    try:
        image_bytes = await file.read()
        result = detect_hazards(image_bytes)
        
        # NEW: Log hazard detection with user info
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
    payload = await request.json()
    message = payload.get("message", "Emergency Broadcast")
    location = payload.get("location", {})
    severity = payload.get("severity", "High")
    result = start_broadcast(message, location, severity)
    return JSONResponse(content=result)

@app.get("/broadcasts")
async def get_active_broadcasts():
    result = discover_nearby_broadcasts(location={})  # Replace with real location filtering if needed
    return JSONResponse(content=result)

# ================================
# UTILITY & HEALTH ROUTES
# ================================

@app.get("/health")
async def health_check():
    """Health check endpoint with map service status"""
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
            "patient_management": True
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
    
    # Log available map services
    available_services = [
        service.value for service in map_utils.api_keys.keys() 
        if map_utils.api_keys[service]
    ]
    logger.info(f"🔑 Available map services: {available_services or ['OpenStreetMap (free)']}")
    
    # ✅ Run DB migrations
    run_migrations()

    logger.info("✅ API server ready with enhanced map capabilities and patient management")

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