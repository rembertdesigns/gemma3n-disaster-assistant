from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.hazard_detection import detect_hazards
from app.preprocessing import preprocess_input
from app.inference import run_disaster_analysis
from app.audio_transcription import transcribe_audio
from app.report_utils import generate_report_pdf

from weasyprint import HTML as WeasyHTML
import shutil
import os
import uuid

# Initialize app
app = FastAPI()

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create required directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# ROUTES
# ----------------------------

# Home page
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Hazards page
@app.get("/hazards", response_class=HTMLResponse)
async def serve_hazard_page(request: Request):
    return templates.TemplateResponse("hazards.html", {"request": request, "result": None})

# Generate page (JSON-to-PDF UI)
@app.get("/generate", response_class=HTMLResponse)
async def serve_generate_page(request: Request):
    return templates.TemplateResponse("generate.html", {"request": request})

# Offline fallback page
@app.get("/offline.html", response_class=HTMLResponse)
async def offline_page(request: Request):
    return templates.TemplateResponse("offline.html", {"request": request})


# ----------------------------
# ANALYSIS & REPORTING
# ----------------------------

# Analyze input (text, image, audio)
@app.post("/analyze", response_class=HTMLResponse)
async def analyze_input(
    request: Request,
    report_text: str = Form(""),
    file: UploadFile = File(None),
    audio: UploadFile = File(None)
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


# Export PDF from textarea input (classic)
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


# Export PDF from JSON input (structured)
@app.post("/generate-report")
async def generate_report(request: Request):
    payload = await request.json()
    pdf_path = generate_report_pdf(payload)
    return FileResponse(pdf_path, media_type="application/pdf", filename="incident_report.pdf")


# ----------------------------
# HAZARD DETECTION API
# ----------------------------

@app.post("/detect-hazards")
async def detect_hazards_api(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)

    try:
        image_bytes = await file.read()
        result = detect_hazards(image_bytes)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": f"Hazard detection failed: {str(e)}"}, status_code=500)
    
@app.get("/live-generate", response_class=HTMLResponse)
async def serve_live_generate_page(request: Request):
    return templates.TemplateResponse("live_generate.html", {"request": request})
