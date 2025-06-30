from fastapi.responses import FileResponse
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from datetime import datetime
import os

def generate_report_pdf(data: dict, output_path="output/report.pdf"):
    env = Environment(loader=FileSystemLoader("app/templates"))
    template = env.get_template("report_template.html")

    # Extract coordinates
    latitude = data.get("latitude") or (data.get("coordinates")[0] if data.get("coordinates") else None)
    longitude = data.get("longitude") or (data.get("coordinates")[1] if data.get("coordinates") else None)
    coordinates = f"{latitude}, {longitude}" if latitude and longitude else "N/A"

    # Timestamp fallback
    timestamp = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Resolve image path (uploaded file or external URL)
    image_url = data.get("image_url")
    if image_url and image_url.startswith("uploaded://"):
        filename = image_url.replace("uploaded://", "")
        image_path = os.path.join("uploads", filename)
        # Ensure WeasyPrint can read it: use a relative path
        image_url = f"/{image_path}" if os.path.exists(image_path) else None

    html_out = template.render(
        date=timestamp,
        location=data.get("location", "Unknown"),
        coordinates=coordinates,
        hazards=", ".join(data.get("hazards", [])),
        severity=data.get("severity", "N/A"),
        notes=data.get("notes", ""),
        image_url=image_url,
        checklist=data.get("checklist", [])
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    HTML(string=html_out, base_url=".").write_pdf(output_path)
    return output_path