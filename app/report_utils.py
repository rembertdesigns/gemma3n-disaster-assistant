from fastapi.responses import FileResponse
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from datetime import datetime
import os

def generate_report_pdf(data: dict, output_path="output/report.pdf"):
    env = Environment(loader=FileSystemLoader("app/templates"))
    template = env.get_template("report_template.html")
    
    html_out = template.render(
        date=data.get("date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        location=data.get("location", "Unknown"),
        hazards=", ".join(data.get("hazards", [])),
        severity=data.get("severity", "N/A"),
        notes=data.get("notes", ""),
        image_url=data.get("image_url", None)
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    HTML(string=html_out).write_pdf(output_path)
    return output_path