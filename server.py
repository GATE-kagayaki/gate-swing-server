import os
import json
import time
import math
import shutil
import traceback
import tempfile
from typing import Any, Dict

from flask import Flask, request, abort, jsonify

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent, VideoMessage, FileMessage,
    TextSendMessage
)

from google.cloud import firestore, tasks_v2

# ==================================================
# ENV
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")
TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")

app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
db = firestore.Client(project=GCP_PROJECT_ID)
tasks_client = tasks_v2.CloudTasksClient()
queue_path = tasks_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)

# ==================================================
# Helpers
# ==================================================
def firestore_safe_set(report_id: str, data: Dict[str, Any]) -> None:
    try:
        db.collection("reports").document(report_id).set(data, merge=True)
    except Exception:
        print(traceback.format_exc())

def firestore_safe_update(report_id: str, patch: Dict[str, Any]) -> None:
    try:
        db.collection("reports").document(report_id).update(patch)
    except Exception:
        print(traceback.format_exc())

def safe_line_reply(reply_token: str, text: str) -> None:
    try:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text))
    except LineBotApiError:
        print(traceback.format_exc())

def safe_line_push(user_id: str, text: str) -> None:
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text=text))
    except LineBotApiError:
        print(traceback.format_exc())

def make_initial_reply(report_id: str) -> str:
    return (
        "✅ 動画を受信しました。\n"
        "AIによるスイング数値計測を開始します。\n\n"
        "完了まで1〜3分ほどお待ちください。\n"
        "完了すると自動で通知が届きます。\n\n"
        "【現在のステータス確認】\n"
        f"{SERVICE_HOST_URL}/report/{report_id}"
    )

def make_done_push(report_id: str) -> str:
    return (
        "🎉 スイング計測が完了しました！\n\n"
        "以下のリンクから診断レポートを確認できます。\n\n"
        f"{SERVICE_HOST_URL}/report/{report_id}"
    )

def create_cloud_task(report_id: str, user_id: str, message_id: str) -> None:
    payload = json.dumps({"report_id": report_id, "user_id": user_id, "message_id": message_id}).encode("utf-8")
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{SERVICE_HOST_URL}/worker/process_video",
            "headers": {"Content-Type": "application/json"},
            "body": payload,
            "oidc_token": {
                "service_account_email": TASK_SA_EMAIL,
                "audience": SERVICE_HOST_URL,
            },
        }
    }
    tasks_client.create_task(parent=queue_path, task=task)

# ==================================================
# ROUTES
# ==================================================
@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent)
def handle_any(event: MessageEvent):
    msg = event.message
    user_id = event.source.user_id

    if isinstance(msg, (VideoMessage, FileMessage)):
        report_id = f"{user_id}_{msg.id}"
        firestore_safe_set(report_id, {"user_id": user_id, "status": "PROCESSING"})
        create_cloud_task(report_id, user_id, msg.id)
        safe_line_reply(event.reply_token, make_initial_reply(report_id))
    else:
        safe_line_reply(event.reply_token, "🎥 解析したいスイング動画を送信してください。")

@app.route("/worker/process_video", methods=["POST"])
def worker():
    payload = request.get_json()
    report_id = payload.get("report_id")
    message_id = payload.get("message_id")

    if not report_id or not message_id:
        return jsonify({"error": "invalid payload"}), 400

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, f"{message_id}.mp4")

    try:
        message_content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for chunk in message_content.iter_content():
                f.write(chunk)

        from swing_analysis import analyze_swing
        raw_data, report_text = analyze_swing(video_path)

        firestore_safe_update(report_id, {
            "status": "COMPLETED",
            "raw_data": raw_data,
            "report_text": report_text,
        })

        doc = db.collection("reports").document(report_id).get()
        if doc.exists:
            user_id = doc.to_dict().get("user_id")
            safe_line_push(user_id, make_done_push(report_id))

    except Exception as e:
        print(traceback.format_exc())
        firestore_safe_update(report_id, {"status": "FAILED", "error": str(e)})
        return jsonify({"status": "failed", "error": str(e)}), 200

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return jsonify({"ok": True})

@app.route("/api/report_data/<report_id>")
def api_report_data(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    d = doc.to_dict()
    return jsonify({
        "status": d.get("status"),
        "mediapipe_data": d.get("raw_data", {}),
        "report_text": d.get("report_text", ""),
    })

@app.route("/report/<report_id>")
def report_view(report_id):
    with open("report_template.html", encoding="utf-8") as f:
        html = f.read().replace("__REPORT_ID__", report_id)
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


