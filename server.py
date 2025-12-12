# ====== server.py 完成版 ======

import os
import json
import time
import traceback
from typing import Dict, Any

from flask import Flask, request, abort, jsonify

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore, tasks_v2
from google import genai

# ==================================================
# ENV
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
SERVICE_HOST_URL = os.environ["SERVICE_HOST_URL"].rstrip("/")
TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")
TASK_SA_EMAIL = os.environ["TASK_SA_EMAIL"]

FORCE_PREMIUM = True  # テスト中は常に有料版

# ==================================================
# INIT
# ==================================================
app = Flask(__name__)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

db = firestore.Client(project=GCP_PROJECT_ID)
tasks_client = tasks_v2.CloudTasksClient()
queue_path = tasks_client.queue_path(
    GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME
)

genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ==================================================
# UTIL
# ==================================================
def extract_json_object(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("JSON not found")
        return json.loads(s[start:end + 1])


def send_line_reply(token: str, text: str):
    try:
        line_bot_api.reply_message(token, TextSendMessage(text=text))
    except Exception:
        print(traceback.format_exc())


def send_line_push(user_id: str, text: str):
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text=text))
    except Exception:
        print(traceback.format_exc())


# ==================================================
# ANALYSIS (MediaPipe Stub)
# ==================================================
def analyze_swing() -> Dict[str, Any]:
    return {
        "frame_count": 73,
        "max_shoulder_rotation": -23.8,
        "min_hip_rotation": -179.9,
        "max_wrist_cock": 179.6,
        "max_head_sway": 0.0264,
        "max_knee_sway": 0.0375
    }


# ==================================================
# GEMINI
# ==================================================
def generate_report_json(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
あなたはゴルフスイング解析AIです。
以下のJSONスキーマと完全一致するJSONのみを出力してください。
文章や説明は一切不要です。

【骨格分析データ】
{json.dumps(raw_data, ensure_ascii=False)}

【出力JSONスキーマ】
{{
  "section02": {{ "title": "", "analysis": [] }},
  "section03": {{ "title": "", "analysis": [] }},
  "section04": {{ "title": "", "analysis": [] }},
  "section05": {{ "title": "", "analysis": [] }},
  "section06": {{ "title": "", "analysis": [] }},
  "section07": {{
    "stable_points": [],
    "improvement_points": []
  }},
  "section08": {{
    "drills": [
      {{ "name": "", "howto": [] }}
    ]
  }},
  "section09": {{
    "table": [
      {{ "item": "", "recommendation": "", "reason": "" }}
    ],
    "disclaimer": ""
  }},
  "section10": {{
    "text": ""
  }}
}}
"""
    res = genai_client.models.generate_content(
        model="gemini-1.5-pro",
        contents=prompt
    )
    return extract_json_object(res.text)


# ==================================================
# ROUTES
# ==================================================
@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event):
    user_id = event.source.user_id
    report_id = f"{user_id}_{event.message.id}"

    db.collection("reports").document(report_id).set({
        "status": "PROCESSING",
        "created_at": firestore.SERVER_TIMESTAMP
    })

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{SERVICE_HOST_URL}/worker",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "report_id": report_id,
                "user_id": user_id
            }).encode(),
            "oidc_token": {
                "service_account_email": TASK_SA_EMAIL,
                "audience": SERVICE_HOST_URL
            }
        }
    }
    tasks_client.create_task(parent=queue_path, task=task)

    send_line_reply(
        event.reply_token,
        "✅ 動画を受信しました。解析を開始します。\n完了後に通知します。"
    )


@app.route("/worker", methods=["POST"])
def worker():
    payload = request.json
    report_id = payload["report_id"]
    user_id = payload["user_id"]

    raw = analyze_swing()
    report_json = generate_report_json(raw)

    db.collection("reports").document(report_id).update({
        "status": "COMPLETED",
        "raw_data": raw,
        "report": report_json
    })

    send_line_push(
        user_id,
        f"🎉 AIスイング診断が完了しました！\n{SERVICE_HOST_URL}/report/{report_id}"
    )

    return jsonify({"ok": True})


@app.route("/api/report_data/<report_id>")
def api_report(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    return jsonify(doc.to_dict())


@app.route("/report/<report_id>")
def report_view(report_id):
    return open("report_template.html", encoding="utf-8").read()


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


