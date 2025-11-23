from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import datetime
from google.cloud import storage

def generate_pdf_report(pdf_filename):

    c = canvas.Canvas(pdf_filename, pagesize=A4)
    width, height = A4
    y = height - 40

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, y, "GATE Swing Analysis Report")
    y -= 25

    # Date
    c.setFont("Helvetica", 10)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    c.drawString(40, y, f"Date: {date_str}")
    y -= 40

    # Overview
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "01. OVERVIEW")
    y -= 15
    c.setFont("Helvetica", 9)
    c.drawString(40, y, "スイング解析レポート（仮データ）")
    y -= 40

    # Placeholder Images
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "02. KEY IMAGES (Top & Impact)")
    y -= 180
    c.rect(40, y, 200, 150)
    c.rect(260, y, 200, 150)
    y -= 180

    # Placeholder Sections
    def section(title):
        nonlocal y
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, title)
        y -= 40

    section("03. HEAD STABILITY")
    section("04. SHOULDER ROTATION")
    section("05. HIP ROTATION")
    section("06. WRIST MECHANICS")
    section("07. SWING PATH")
    section("08. KEY DIAGNOSIS")
    section("09. IMPROVEMENT STRATEGY")

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(40, 30, "GATE - Swing Intelligence Platform")

    c.save()
    return pdf_filename


def upload_to_gcs(local_pdf_path, bucket_name, dest_filename):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_filename)
    blob.upload_from_filename(local_pdf_path)
    blob.make_public()
    return blob.public_url
