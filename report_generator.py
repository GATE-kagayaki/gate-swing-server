# report_generator.py
from google.cloud import storage

def generate_pdf_report(output_path: str, video_url: str) -> str:
    # video_url を使って解析し、output_path に PDF を生成する処理を書く
    # ここではダミー実装の例
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(output_path, pagesize=A4)
    c.drawString(100, 800, "動画解析レポート")
    c.drawString(100, 780, f"動画URL: {video_url}")
    c.showPage()
    c.save()

    return output_path


def upload_to_gcs(local_path: str, bucket_name: str, dest_blob_name: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(local_path)

    # 必要に応じて署名付きURLに変更可能
    url = blob.generate_signed_url(version="v4", expiration=3600, method="GET")
    return url


