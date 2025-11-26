from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import datetime
from google.cloud import storage

def generate_pdf_report(pdf_filename):
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    width, height = A4
    y = height - 40

    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, y, "GATE Swing Analysis Report")
    y -= 25

    c.setFont("Helvetica", 10)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    c.drawString(40, y, f"Date: {date_str}")
    y -= 40

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "スイング解析レポート（仮データ）")
    y -= 200

    c.drawString(40, y, "※ここに解析結果が入ります（本番はAI結果を差し込み）")
    c.save()
    return pdf_filename


def upload_to_gcs(local_pdf_path, bucket_name, dest_filename):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_filename)
    blob.upload_from_filename(local_pdf_path)
    blob.make_public()
    return blob.public_url
