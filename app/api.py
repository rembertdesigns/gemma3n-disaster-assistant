from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import uuid

from app.preprocessing import preprocess_input
from app.inference import run_disaster_analysis

app = FastAPI()

# Static files (e.g., CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML Templates
templates = Jinja2Templates(directory="templates")

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_input(
    request: Request,
    report_text: str = Form(...),
    file: UploadFile = File(None)
):
    user_input = report_text.strip()
    saved_path = None

    # Save uploaded file if present
    if file and file.filename != "":
        extension = os.path.splitext(file.filename)[1]
        unique_filename = f"upload_{uuid.uuid4().hex}{extension}"
        saved_path = os.path.join(UPLOAD_DIR, unique_filename)
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        user_input = saved_path

    # Preprocess and analyze
    processed = preprocess_input(user_input)
    result = run_disaster_analysis(processed)

    # Return response with result
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "input_text": report_text
    })