from fastapi.responses import FileResponse
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from datetime import datetime
import os

def generate_report_pdf(data: dict, output_path="output/report.pdf"):
    env = Environment(loader=FileSystemLoader("app/templates"))
    template = env.get_template("report_template.html")

    # Extract coordinates if provided
    latitude = data.get("latitude") or (data.get("coordinates")[0] if data.get("coordinates") else None)
    longitude = data.get("longitude") or (data.get("coordinates")[1] if data.get("coordinates") else None)
    coordinates = f"{latitude}, {longitude}" if latitude and longitude else "N/A"

    # Fallbacks and formatting
    timestamp = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    html_out = template.render(
        date=timestamp,
        location=data.get("location", "Unknown"),
        coordinates=coordinates,
        hazards=", ".join(data.get("hazards", [])),
        severity=data.get("severity", "N/A"),
        notes=data.get("notes", ""),
        image_url=data.get("image_url", None),
        checklist=data.get("checklist", [])
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    HTML(string=html_out).write_pdf(output_path)
    return output_path