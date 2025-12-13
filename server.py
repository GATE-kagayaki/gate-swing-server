import os
import json
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from flask import Flask, request, abort, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore, tasks_v2
from google.api_core.exceptions import NotFound, PermissionDenied
from google import genai
from google.genai import errors as genai_errors

# ==================================================
# ENV
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")
TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")

FORCE_PREMIUM = os.environ.get("FORCE_PREMIUM", "true").lower() in (
    "1", "true", "yes", "on"
)

# ==================================================
# App init
# ==================================================
app = Flask(__name__)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None

db = firestore.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else None
tasks_client = tasks_v2.CloudTasksClient() if GCP_PROJECT_ID else None

queue_path = None
if tasks_client and GCP_PROJECT_ID:
    queue_path = tasks_client.queue_path(
        GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME
    )

# ==================================================
# Utils
# ==================================================
def now_ts() -> float:
    return time.time()


def firestore_safe_set(report_id: str, data: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("reports").document(report_id).set(data, merge=True)
    except Exception:
        print("[Firestore set error]")
        print(traceback.format_exc())


def firestore_safe_update(report_id: str, patch: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("reports").document(report_id).update(patch)
    except Exception:
        print("[Firestore update error]")
        print(traceback.format_exc())


def safe_line_reply(reply_token: str, text: str) -> None:
    if not line_bot_api:
        return
    try:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text))
    except LineBotApiError:
        print("[LINE reply error]")
        print(traceback.format_exc())


def safe_line_push(user_id: str, text: str) -> None:
    if not line_bot_api:
        return
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text=text))
    except LineBotApiError:
        print("[LINE push error]")
        print(traceback.format_exc())


# ==================================================
# Message text
# ==================================================
def make_initial_reply(report_id: str) -> str:
    return (
        "✅ 動画を受信しました。解析を開始します！\n"
        "（モード：全機能プレビュー）\n\n"
        "AIによるスイング診断には最大3分ほどかかります。\n"
        "【処理状況確認URL】\n"
        f"{SERVICE_HOST_URL}/report/{report_id}\n\n"
        "【料金プラン】\n"
        "・都度契約：500円／1回\n"
        "・回数券　：1,980円／5回券\n"
        "・月額契約：4,980円／月"
    )


def make_done_push(report_id: str) -> str:
    return (
        "🎉 AIスイング診断が完了しました！\n\n"
        "【診断レポートURL】\n"
        f"{SERVICE_HOST_URL}/report/{report_id}"
    )


# ==================================================
# Analysis (stub)
# ==================================================
def analyze_swing_stub() -> Dict[str, Any]:
    return {
        "frame_count": 72,
        "max_shoulder_rotation": 46.1,
        "min_hip_rotation": 26.3,
        "max_wrist_cock": 94.8,
        "max_head_drift_x": 0.019,
        "max_knee_sway_x": 0.032,
    }


# ==================================================
# Gemini
# ==================================================
def run_gemini_report(raw: Dict[str, Any]) -> str:
    if not GEMINI_API_KEY:
        return "AI診断レポートを生成できませんでした（APIキー未設定）。"

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        "あなたはプロのゴルフスイングコーチです。\n"
        "以下の骨格データをもとに、読みやすく実践的な日本語の診断レポートを作成してください。\n"
        "最初にポジティブな評価を入れ、その後で最重要課題を1つ示してください。\n\n"
        f"{json.dumps(raw, ensure_ascii=False, indent=2)}"
    )

    try:
        resp = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt,
        )
        return resp.text or ""
    except Exception as e:
        print("[Gemini error]", e)
        return "AI診断中にエラーが発生しました。"


# ==================================================
# Cloud Tasks
# ==================================================
def create_cloud_task(report_id: str, user_id: str) -> str:
    payload = json.dumps(
        {"report_id": report_id, "user_id": user_id}
    ).encode("utf-8")

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
    resp = tasks_client.create_task(parent=queue_path, task=task)
    return resp.name


# ==================================================
# Routes
# ==================================================
@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/webhook", methods=["POST"])
def webhook():
    if not handler:
        abort(500)

    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event: MessageEvent):
    user_id = event.source.user_id
    report_id = f"{user_id}_{event.message.id}"

    firestore_safe_set(
        report_id,
        {
            "user_id": user_id,
            "status": "PROCESSING",
            "created_at": firestore.SERVER_TIMESTAMP,
        },
    )

    try:
        create_cloud_task(report_id, user_id)
    except Exception as e:
        firestore_safe_update(
            report_id,
            {"status": "TASK_ERROR", "summary": str(e)},
        )
        safe_line_reply(event.reply_token, "システムエラーが発生しました。")
        return

    safe_line_reply(event.reply_token, make_initial_reply(report_id))


@app.route("/worker/process_video", methods=["POST"])
def worker():
    started = now_ts()
    payload = request.get_json() or {}
    report_id = payload.get("report_id")
    user_id = payload.get("user_id")

    try:
        raw = analyze_swing_stub()
        ai_report = run_gemini_report(raw)

        firestore_safe_update(
            report_id,
            {
                "status": "COMPLETED",
                "raw_data": raw,
                "ai_report": ai_report,
                "elapsed_sec": round(now_ts() - started, 2),
            },
        )
        safe_line_push(user_id, make_done_push(report_id))
    except Exception as e:
        firestore_safe_update(
            report_id,
            {"status": "FAILED", "summary": str(e)},
        )

    return jsonify({"ok": True})


@app.route("/api/report_data/<report_id>")
def api_report_data(report_id: str):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    data = doc.to_dict() or {}
    return jsonify(
        {
            "summary": data.get("summary", ""),
            "mediapipe_data": data.get("raw_data", {}),
            "ai_report_text": data.get("ai_report", ""),
        }
    )


@app.route("/report/<report_id>")
def report_view(report_id: str):
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>GATE AIスイングドクター</title>
<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100">
<div class="max-w-4xl mx-auto p-6">
<div class="bg-white p-6 rounded shadow">
<h1 class="text-2xl font-bold text-emerald-600">GATE AIスイングドクター</h1>
<div id="summary" class="mt-4"></div>
<div id="metrics" class="grid grid-cols-2 gap-3 mt-6"></div>
<div id="report" class="mt-6"></div>
</div>
</div>

<script>
const id = location.pathname.split("/").pop();
fetch("/api/report_data/" + id)
.then(r => r.json())
.then(d => {
  document.getElementById("summary").innerText = d.summary || "";
  const m = d.mediapipe_data || {};
  document.getElementById("metrics").innerHTML =
    `<div>肩回旋: ${m.max_shoulder_rotation}</div>
     <div>腰回旋: ${m.min_hip_rotation}</div>
     <div>コック角: ${m.max_wrist_cock}</div>`;
  document.getElementById("report").innerText = d.ai_report_text || "";
});
</script>
</body>
</html>
"""


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

