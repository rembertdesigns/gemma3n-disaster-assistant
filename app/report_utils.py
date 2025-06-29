from fastapi.responses import FileResponse
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from datetime import datetime
import os

def generate_report_pdf(data: dict, output_path="output/report.pdf"):
    env = Environment(loader=FileSystemLoader("app/templates"))
    template = env.get_template("report_template.html")

    # Extract values safely
    location = data.get("location", "Unknown")
    latitude = data.get("latitude", None)
    longitude = data.get("longitude", None)
    coordinates = f"{latitude}, {longitude}" if latitude and longitude else "N/A"

    timestamp = data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_out = template.render(
        date=timestamp,
        location=location,
        coordinates=coordinates,
        hazards=", ".join(data.get("hazards", [])),
        severity=data.get("severity", "N/A"),
        notes=data.get("notes", ""),
        image_url=data.get("image_url", None)
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    HTML(string=html_out).write_pdf(output_path)
    return output_path