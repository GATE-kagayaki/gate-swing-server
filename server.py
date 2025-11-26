from flask import Flask, request, jsonify
import os
import requests
from google.cloud import storage

from report_generator import generate_pdf_report, upload_to_gcs

app = Flask(__name__)

# --- Environment Variables ---
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# --- GCS Setup ---
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)


# --- Reply to LINE ---
def reply(reply_token, message):
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    data = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": message}]
    }
    requests.post(url, headers=headers, json=data)


# --- Save Video to GCS ---
def save_video_to_gcs_stream(content_url, file_name):
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    blob = bucket.blob(file_name)

    with requests.get(content_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with blob.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # Cloud Run は uniform bucket のため make_public は不要
    return blob.public_url


# --- Webhook Handler ---
@app.route("/callback", methods=["POST"])
def callback():
    try:
        body = request.get_json()
        print("RAW EVENT:", body)  # デバッグログ

        events = body.get("events", [])
        if not events:
            return "OK", 200

        event = events[0]

        # --- Message Event ---
        if event.get("type") == "message":
            msg_type = event["message"]["type"]
            reply_token = event["replyToken"]

            # --- Text Message ---
            if msg_type == "text":
                reply(reply_token, "テキストを受信しました！")
                return "OK", 200

            # --- Video Message ---
            if msg_type == "video":
                reply(reply_token, "動画を受け取りました！レポート作成中です…")

                message_id = event["message"]["id"]
                content_url = f"https://api.line.me/v2/bot/message/{message_id}/content"

                # 1. Save video
                file_name = f"video_{message_id}.mp4"
                video_url = save_video_to_gcs_stream(content_url, file_name)

                # 2. Create PDF
                pdf_local = "/tmp/report.pdf"
                generate_pdf_report(pdf_local)

                # 3. Upload PDF
                pdf_url = upload_to_gcs(pdf_local, GCS_BUCKET_NAME, f"reports/{message_id}.pdf")

                # 4. Reply with link
                reply(reply_token, f"レポートが完成しました👇\n{pdf_url}")

                return "OK", 200

        return "OK", 200

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# --- Run App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
