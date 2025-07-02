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
from datetime import datetime
from pathlib import Path
import shutil
import os
import uuid
import json
import zipfile
import io
import base64
import logging

from weasyprint import HTML as WeasyHTML

# Core app utilities
from app.predictive_engine import calculate_risk_score
from app.broadcast_utils import start_broadcast, discover_nearby_broadcasts
from app.sentiment_utils import analyze_sentiment
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
    get_dashboard_stats
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
from app.models import CrowdReport  # SQLAlchemy model for crowd reports

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

crowd_reports = []  # In-memory store; consider replacing with DB later

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# EXISTING AUTH ROUTES (unchanged)
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
# EXISTING PAGE ROUTES (unchanged)
# ================================

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

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

@app.get("/submit-crowd-report", response_class=HTMLResponse)
async def submit_crowd_report_form(request: Request):
    return templates.TemplateResponse("submit-crowd-report.html", {"request": request})

@app.post("/api/submit-crowd-report")
async def submit_crowd_report(
    request: Request,
    message: str = Form(...),
    tone: Optional[str] = Form(None),
    escalation: str = Form(...),
    user: Optional[str] = Form("Anonymous"),
    location: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
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

@app.get("/crowd-reports", response_class=HTMLResponse)
async def view_crowd_reports(request: Request, db: Session = Depends(get_db)):
    reports = db.query(CrowdReport).order_by(CrowdReport.timestamp.desc()).all()
    return templates.TemplateResponse("crowd_reports.html", {"request": request, "reports": reports})

# ================================
# EXISTING ADMIN ROUTES (unchanged)
# ================================

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, user: dict = Depends(require_role(["admin"]))):
    conn = get_db_connection()
    stats = get_dashboard_stats(conn)
    conn.close()
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "username": user["username"],
        "role": user["role"],
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
# NEW: MAP API ROUTES
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

# ================================
# EXISTING ANALYSIS & REPORTING ROUTES (unchanged)
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
# EXISTING HAZARD DETECTION (unchanged)
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
# NEW: UTILITY ROUTES
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
            "pdf_generation": True
        },
        "map_service": {
            "preferred": map_utils.preferred_service.value,
            "available_services": [
                service.value for service in map_utils.api_keys.keys() 
                if map_utils.api_keys[service]
            ]
        }
    }

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

@app.post("/predict-risk")
async def predict_risk_api(payload: dict = Body(...)):
    location = payload.get("location", {})
    weather = payload.get("weather", {})
    hazard = payload.get("hazard_type", "unknown")

    result = calculate_risk_score(location, weather, hazard)
    return JSONResponse(content=result)

@app.post("/api/submit-report")
async def submit_crowd_report(payload: dict = Body(...)):
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

# ================================
# PHASE 1: EMERGENCY BROADCAST ROUTES
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
# STARTUP EVENT
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
    logger.info("✅ API server ready with enhanced map capabilities")

# ================================
# ERROR HANDLERS
# ================================

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