import os
import json
import math
import tempfile
import shutil
from datetime import datetime, timezone

from flask import Flask, request, jsonify, abort, send_from_directory
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, VideoMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError
from google.cloud import firestore, tasks_v2

import cv2
import mediapipe as mp

# ======================
# 基本設定
# ======================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
SERVICE_HOST_URL = os.environ["SERVICE_HOST_URL"].rstrip("/")
PROJECT_ID = os.environ["GCP_PROJECT_ID"]
QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")
QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_SA_EMAIL = os.environ["TASK_SA_EMAIL"]

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
db = firestore.Client()
tasks_client = tasks_v2.CloudTasksClient()

# ======================
# ユーティリティ
# ======================
def reply(token, text):
    line_bot_api.reply_message(token, TextSendMessage(text=text))

def push(user_id, text):
    line_bot_api.push_message(user_id, TextSendMessage(text=text))

# ======================
# MediaPipe解析
# ======================
def analyze_video(path):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(path)

    frame_count = 0
    max_shoulder = 0
    min_hip = 999
    max_wrist = 0
    max_head = 0
    max_knee = 0

    def angle(a,b,c):
        ax, ay = a[0]-b[0], a[1]-b[1]
        cx, cy = c[0]-b[0], c[1]-b[1]
        dot = ax*cx + ay*cy
        na = math.hypot(ax,ay)
        nc = math.hypot(cx,cy)
        if na*nc == 0: return 0
        return math.degrees(math.acos(max(-1,min(1,dot/(na*nc)))))

    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            frame_count += 1
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks: continue

            lm = res.pose_landmarks.landmark
            def xy(i): return (lm[i].x, lm[i].y)

            LS, RS = 11, 12
            LH, RH, LK = 23, 24, 25
            LE, LW, LI = 13, 15, 19
            NO = 0

            max_shoulder = max(max_shoulder, angle(xy(LS), xy(RS), xy(RH)))
            min_hip = min(min_hip, angle(xy(LH), xy(RH), xy(LK)))
            max_wrist = max(max_wrist, angle(xy(LE), xy(LW), xy(LI)))
            max_head = max(max_head, abs(xy(NO)[0]-0.5))
            max_knee = max(max_knee, abs(xy(LK)[0]-0.5))

    cap.release()
    return {
        "frame_count": frame_count,
        "max_shoulder": round(max_shoulder,2),
        "min_hip": round(min_hip,2),
        "max_wrist": round(max_wrist,2),
        "head_sway": round(max_head,4),
        "knee_sway": round(max_knee,4)
    }

# ======================
# Cloud Tasks
# ======================
def enqueue_task(report_id, user_id, message_id):
    parent = tasks_client.queue_path(PROJECT_ID, QUEUE_LOCATION, QUEUE_NAME)
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{SERVICE_HOST_URL}/task-handler",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "report_id": report_id,
                "user_id": user_id,
                "message_id": message_id
            }).encode(),
            "oidc_token": {
                "service_account_email": TASK_SA_EMAIL,
                "audience": SERVICE_HOST_URL
            }
        }
    }
    tasks_client.create_task(parent=parent, task=task)

# ======================
# Webhook
# ======================
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
def on_video(event):
    report_id = f"{event.source.user_id}_{event.message.id}"
    db.collection("reports").document(report_id).set({
        "status": "PROCESSING",
        "is_premium": False,
        "created_at": datetime.now(timezone.utc).isoformat()
    })

    enqueue_task(report_id, event.source.user_id, event.message.id)

    reply(event.reply_token,
          "動画を受信しました。\nAIによるスイング解析を開始します。\n\n"
          f"【進行状況の確認】\n{SERVICE_HOST_URL}/report/{report_id}\n\n"
          "【料金プラン】\n"
          "① 都度会員 500円/回\n"
          "② 回数券 1,980円/5回\n"
          "③ 月額会員 4,980円/月\n"
          "※無料版でも基本解析はご利用いただけます。")

# ======================
# Task Handler
# ======================
@app.route("/task-handler", methods=["POST"])
def task_handler():
    data = request.json
    report_id = data["report_id"]
    message_id = data["message_id"]
    user_id = data["user_id"]

    tmp = tempfile.mkdtemp()
    video_path = os.path.join(tmp, "video.mp4")

    content = line_bot_api.get_message_content(message_id)
    with open(video_path, "wb") as f:
        for c in content.iter_content():
            f.write(c)

    raw = analyze_video(video_path)

    # ---- 01
    analysis = {
        "01": {
            "title": "骨格計測データ",
            "data": {
                "解析フレーム数": raw["frame_count"],
                "最大肩回転角（°）": raw["max_shoulder"],
                "最小腰回転角（°）": raw["min_hip"],
                "最大手首コック角（°）": raw["max_wrist"],
                "最大頭部ブレ": raw["head_sway"],
                "最大膝ブレ": raw["knee_sway"]
            }
        },
        "07": {
            "title": "総合評価",
            "text": [
                "全体として安定性の高いスイングです。",
                "下半身の安定を活かすことで再現性向上が期待できます。"
            ]
        }
    }

    db.collection("reports").document(report_id).update({
        "status": "COMPLETED",
        "analysis": analysis,
        "raw": raw
    })

    push(user_id, f"🎉 解析が完了しました\n{SERVICE_HOST_URL}/report/{report_id}")
    shutil.rmtree(tmp, ignore_errors=True)
    return "OK"

# ======================
# 表示API
# ======================
@app.route("/report/<report_id>")
def report_page(report_id):
    return send_from_directory("templates", "report.html")

@app.route("/api/report_data/<report_id>")
def api_data(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({})
    d = doc.to_dict()
    return jsonify({
        "analysis": d.get("analysis"),
        "is_premium": d.get("is_premium", False)
    })

# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
