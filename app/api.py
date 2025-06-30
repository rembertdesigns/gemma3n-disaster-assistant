from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm

from app.hazard_detection import detect_hazards
from app.preprocessing import preprocess_input
from app.inference import run_disaster_analysis
from app.audio_transcription import transcribe_audio
from app.report_utils import generate_report_pdf
from app.auth import authenticate_user, create_access_token, get_current_user, require_role
from app.db import (
    get_db_connection,
    save_report_metadata,
    get_all_reports,
    get_report_by_id,
    get_dashboard_stats
)

from weasyprint import HTML as WeasyHTML
from datetime import datetime
import shutil
import os
import uuid
import json
import zipfile
import io

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- AUTH ----------------

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

# ---------------- ROUTES ----------------

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

# ---------------- ADMIN ----------------

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

# ---------------- ANALYSIS & REPORTING ----------------

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

    pdf_path = generate_report_pdf(payload)

    save_report_metadata({
        "id": str(uuid.uuid4()),
        "timestamp": payload.get("timestamp", datetime.now().isoformat()),
        "location": payload.get("location", "Unknown"),
        "severity": payload.get("severity", "N/A"),
        "filename": os.path.basename(pdf_path),
        "user": user["username"],
        "status": "submitted",
        "image_url": payload.get("image_url"),
        "checklist": payload.get("checklist", [])
    })

    return FileResponse(pdf_path, media_type="application/pdf", filename="incident_report.pdf")

# ---------------- HAZARD DETECTION ----------------

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
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": f"Hazard detection failed: {str(e)}"}, status_code=500)

