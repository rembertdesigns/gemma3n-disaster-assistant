# app/map_snapshot.py
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from io import BytesIO
import os

env = Environment(loader=FileSystemLoader("templates"))

def generate_map_image(lat: float, lon: float, report_id: str) -> BytesIO:
    template = env.get_template("map_snapshot.html")
    html_content = template.render(latitude=lat, longitude=lon, report_id=report_id)
    pdf_io = BytesIO()
    HTML(string=html_content, base_url=os.getcwd()).write_pdf(pdf_io)
    pdf_io.seek(0)
    return pdf_io