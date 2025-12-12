import os
import json
import tempfile
import shutil
from flask import Flask, request, abort, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore, tasks_v2
from google import genai

# =====================
# 環境変数
# =====================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast1")
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL")

# =====================
# 初期化
# =====================
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

db = firestore.Client(project=GCP_PROJECT_ID)
tasks_client = tasks_v2.CloudTasksClient()
queue_path = tasks_client.queue_path(
    GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME
)

# =====================
# Webhook
# =====================
@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


# =====================
# LINE 動画受信
# =====================
@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event):
    user_id = event.source.user_id
    message_id = event.message.id
    report_id = f"{user_id}_{message_id}"

    db.collection("reports").document(report_id).set({
        "user_id": user_id,
        "status": "PROCESSING",
        "created_at": firestore.SERVER_TIMESTAMP
    })

    payload = json.dumps({
        "report_id": report_id,
        "user_id": user_id
    }).encode()

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{SERVICE_HOST_URL}/worker/process_video",
            "headers": {"Content-Type": "application/json"},
            "body": payload,
        }
    }

    tasks_client.create_task(parent=queue_path, task=task)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="動画を受信しました。解析を開始します。")
    )


# =====================
# Worker（解析）
# =====================
@app.route("/worker/process_video", methods=["POST"])
def process_video():
    data = request.get_json()
    report_id = data["report_id"]
    user_id = data["user_id"]

    # ダミー解析結果
    analysis = {
        "frame_count": 73,
        "max_shoulder_rotation": -23.8,
        "min_hip_rotation": -179.9,
        "max_wrist_cock": 179.6,
        "max_head_drift_x": 0.0264,
        "max_knee_sway_x": 0.0375
    }

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
以下のゴルフスイング骨格データを元に、
日本語で簡潔な診断コメントを作成してください。

{json.dumps(analysis, ensure_ascii=False, indent=2)}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    db.collection("reports").document(report_id).update({
        "status": "COMPLETED",
        "analysis": analysis,
        "ai_report": response.text
    })

    report_url = f"{SERVICE_HOST_URL}/report/{report_id}"

    line_bot_api.push_message(
        user_id,
        TextSendMessage(
            text=f"解析が完了しました。\nレポートはこちら:\n{report_url}"
        )
    )

    return jsonify({"status": "ok"})


# =====================
# JSON API
# =====================
@app.route("/api/report_data/<report_id>")
def api_report(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404

    data = doc.to_dict()
    return jsonify({
        "status": data.get("status"),
        "mediapipe_data": data.get("analysis"),
        "ai_report_text": data.get("ai_report")
    })


# =====================
# Web レポート表示
# =====================
@app.route("/report/<report_id>")
def report_view(report_id):
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>GATE AIスイングドクター</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100">
<div class="max-w-4xl mx-auto p-6 bg-white mt-6 shadow rounded">
<h1 class="text-3xl font-bold text-center mb-4">GATE AIスイングドクター</h1>
<p class="text-center text-sm text-gray-500 mb-6">レポートID: {report_id}</p>

<div id="content">読み込み中...</div>
</div>

<script>
fetch("/api/report_data/{report_id}")
.then(r => r.json())
.then(d => {{
  if (d.error) {{
    document.getElementById("content").innerText = "レポートが見つかりません";
    return;
  }}

  const m = d.mediapipe_data;
  let html = "<h2 class='text-xl font-bold mb-2'>骨格データ</h2><ul>";
  for (const k in m) {{
    html += `<li>${{k}} : ${{m[k]}}</li>`;
  }}
  html += "</ul><h2 class='text-xl font-bold mt-4 mb-2'>診断コメント</h2>";
  html += "<p>" + d.ai_report_text.replace(/\\n/g,"<br>") + "</p>";
  document.getElementById("content").innerHTML = html;
});
</script>
</body>
</html>
"""


# =====================
# 起動
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
