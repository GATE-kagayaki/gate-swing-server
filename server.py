import os
import json
import math
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from typing import Dict, Any

import cv2
import mediapipe as mp
from flask import Flask, request, jsonify, abort, render_template

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore
from google.cloud import tasks_v2


# ======================
# 基本設定
# ======================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
SERVICE_HOST_URL = os.environ["SERVICE_HOST_URL"].rstrip("/")
TASK_SA_EMAIL = os.environ["TASK_SA_EMAIL"]
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")

QUEUE_NAME = "video-analysis-queue"
QUEUE_LOCATION = "asia-northeast2"

db = firestore.Client()
tasks_client = tasks_v2.CloudTasksClient()

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# ======================
# MediaPipe解析
# ======================
def analyze(video_path: str) -> Dict[str, Any]:
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    max_shoulder = 0.0
    min_hip = 999.0
    max_wrist = 0.0
    max_head = 0.0
    max_knee = 0.0

    def angle(a, b, c):
        ax, ay = a[0] - b[0], a[1] - b[1]
        bx, by = c[0] - b[0], c[1] - b[1]
        dot = ax * bx + ay * by
        na = math.hypot(ax, ay)
        nb = math.hypot(bx, by)
        if na * nb == 0:
            return 0
        return math.degrees(math.acos(max(-1, min(1, dot / (na * nb)))))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            def xy(i): return (lm[i].x, lm[i].y)

            max_shoulder = max(max_shoulder, angle(xy(11), xy(12), xy(24)))
            min_hip = min(min_hip, angle(xy(23), xy(24), xy(25)))
            max_wrist = max(max_wrist, angle(xy(13), xy(15), xy(19)))
            max_head = max(max_head, abs(xy(0)[0] - 0.5))
            max_knee = max(max_knee, abs(xy(25)[0] - 0.5))

    cap.release()

    return {
        "解析フレーム数": frame_count,
        "最大肩回転角": round(max_shoulder, 2),
        "最小腰回転角": round(min_hip, 2),
        "最大コック角": round(max_wrist, 2),
        "最大頭ブレ量": round(max_head, 4),
        "最大膝ブレ量": round(max_knee, 4),
    }


# ======================
# Cloud Tasks作成
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
                "audience": SERVICE_HOST_URL,
            },
        }
    }
    tasks_client.create_task(parent=parent, task=task)


# ======================
# LINE Webhook
# ======================
@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    handler.handle(body, signature)
    return "OK"


@handler.add(MessageEvent, message=VideoMessage)
def on_video(event):
    report_id = f"{event.source.user_id}_{event.message.id}"
    db.collection("reports").document(report_id).set({
        "status": "PROCESSING",
        "created_at": datetime.now(timezone.utc).isoformat()
    })
    enqueue_task(report_id, event.source.user_id, event.message.id)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=f"動画を受信しました。\n解析を開始します。\n{SERVICE_HOST_URL}/report/{report_id}")
    )


# ======================
# task-handler
# ======================
@app.route("/task-handler", methods=["POST"])
def task_handler():
    d = request.json
    report_id = d["report_id"]
    message_id = d["message_id"]
    user_id = d["user_id"]

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "video.mp4")

    try:
        content = line_bot_api.get_message_content(message_id)
        with open(path, "wb") as f:
            for c in content.iter_content():
                f.write(c)

        raw = analyze(path)

        analysis = {
            "01": {
                "title": "骨格計測データ（AI解析）",
                "data": raw
            },
            "07": {
                "title": "総合診断",
                "text": [
                    "スイング軸が安定しており、再現性の高い動きが確認できます。",
                    "上半身の捻転量を増やすことで飛距離向上が期待できます。"
                ]
            },
            "08": {
                "title": "改善ドリル",
                "drills": [
                    {"ドリル名": "クロスアームターン", "目的": "捻転向上", "やり方": "胸の前で腕を組み回旋"}
                ]
            },
            "10": {
                "title": "まとめ",
                "text": ["現状は非常に良好です。継続的な練習を行いましょう。"]
            }
        }

        db.collection("reports").document(report_id).update({
            "status": "COMPLETED",
            "analysis": analysis
        })

        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=f"解析が完了しました。\n{SERVICE_HOST_URL}/report/{report_id}")
        )

        return "OK"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ======================
# 表示/API
# ======================
@app.route("/report/<report_id>")
def report_page(report_id):
    return render_template("report.html", report_id=report_id)


@app.route("/api/report_data/<report_id>")
def api_data(report_id):
    doc = db.collection("reports").document(report_id).get()
    return jsonify(doc.to_dict())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
