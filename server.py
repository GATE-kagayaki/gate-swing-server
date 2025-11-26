from flask import Flask, request, jsonify
import os
import requests
from google.cloud import storage

# PDF生成用（あなたの report_generator.py を想定）
from report_generator import generate_pdf_report, upload_to_gcs

app = Flask(__name__)

LINE_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# GCS
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


# ---------------------------------------------------
# LINE 返信
# ---------------------------------------------------
def reply_text(reply_token, text):
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }
    body = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text}]
    }
    requests.post(url, headers=headers, json=body)


# ---------------------------------------------------
# 動画ダウンロード (stream)
# ---------------------------------------------------
def download_video_to_gcs(message_id):
    content_url = f"https://api.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_ACCESS_TOKEN}"}

    file_name = f"video_{message_id}.mp4"
    blob = bucket.blob(file_name)

    # ストリーミングで GCS に書き込み
    with requests.get(content_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with blob.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    blob.make_public()
    return blob.public_url


# ---------------------------------------------------
# Webhook
# ---------------------------------------------------
@app.route("/callback", methods=["POST"])
def callback():
    try:
        body = request.get_json()
        events = body.get("events", [])

        print("RAW EVENT:", body)  # デバッグログ

        for event in events:
            if event.get("type") != "message":
                continue

            msg = event["message"]
            reply_token = event["replyToken"]

            # テキスト
            if msg["type"] == "text":
                reply_text(reply_token, f"受信: {msg['text']}")
                continue

            # 動画
            if msg["type"] == "video":
                reply_text(reply_token, "動画を受信しました。解析レポートを作成中です…")

                message_id = msg["id"]

                # 動画を GCS に保存
                video_url = download_video_to_gcs(message_id)

                # PDF生成
                pdf_path = generate_pdf_report("/tmp/report.pdf")

                # PDFをGCSにアップロード
                pdf_url = upload_to_gcs(pdf_path, BUCKET_NAME, f"reports/{message_id}.pdf")

                reply_text(reply_token, f"レポートが完成しました👇\n{pdf_url}")

        return "OK", 200

    except Exception as e:
        print("ERROR in callback:", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------
# Cloud Run 起動
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
