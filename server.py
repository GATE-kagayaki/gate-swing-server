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
from linebot.models import MessageEvent, VideoMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError, LineBotApiError

from google.cloud import firestore, tasks_v2
from google.api_core.exceptions import PermissionDenied, NotFound


# ==================================================
# 必須環境変数（ここで止める）
# ==================================================
PROJECT_ID = (
    os.environ.get("GCP_PROJECT_ID")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
)
if not PROJECT_ID:
    raise RuntimeError("GCP_PROJECT_ID が設定されていません")

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL")

QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")
QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")

if not all([LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET, SERVICE_HOST_URL, TASK_SA_EMAIL]):
    raise RuntimeError("必須環境変数が不足しています")

TASK_HANDLER_URL = f"{SERVICE_HOST_URL}/task-handler"


# ==================================================
# 初期化
# ==================================================
app = Flask(__name__, template_folder="templates")
app.config["JSON_AS_ASCII"] = False

db = firestore.Client(project=PROJECT_ID)
tasks_client = tasks_v2.CloudTasksClient()

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# ==================================================
# 共通ユーティリティ
# ==================================================
def safe_reply(token: str, text: str):
    try:
        line_bot_api.reply_message(token, TextSendMessage(text=text))
    except LineBotApiError:
        print(traceback.format_exc())


def safe_push(user_id: str, text: str):
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text=text))
    except LineBotApiError:
        print(traceback.format_exc())


# ==================================================
# Cloud Tasks enqueue
# ==================================================
def enqueue_task(report_id: str, user_id: str, message_id: str):
    parent = tasks_client.queue_path(
        PROJECT_ID, QUEUE_LOCATION, QUEUE_NAME
    )

    payload = json.dumps({
        "report_id": report_id,
        "user_id": user_id,
        "message_id": message_id,
    }, ensure_ascii=False).encode("utf-8")

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": TASK_HANDLER_URL,
            "headers": {"Content-Type": "application/json"},
            "body": payload,
            "oidc_token": {
                "service_account_email": TASK_SA_EMAIL,
                "audience": SERVICE_HOST_URL,
            },
        }
    }

    tasks_client.create_task(parent=parent, task=task)


# ==================================================
# MediaPipe 解析
# ==================================================
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
            xy = lambda i: (lm[i].x, lm[i].y)

            max_shoulder = max(max_shoulder, angle(xy(11), xy(12), xy(24)))
            min_hip = min(min_hip, angle(xy(23), xy(24), xy(25)))
            max_wrist = max(max_wrist, angle(xy(13), xy(15), xy(19)))
            max_head = max(max_head, abs(xy(0)[0] - 0.5))
            max_knee = max(max_knee, abs(xy(25)[0] - 0.5))

    cap.release()

    if frame_count < 10:
        raise RuntimeError("動画が短すぎます")

    return {
        "解析フレーム数": frame_count,
        "最大肩回転角": round(max_shoulder, 2),
        "最小腰回転角": round(min_hip, 2),
        "最大手首コック角": round(max_wrist, 2),
        "最大頭部ブレ": round(max_head, 4),
        "最大膝ブレ": round(max_knee, 4),
    }


# ==================================================
# Webhook
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


@handler.add(MessageEvent, message=VideoMessage)
def on_video(event: MessageEvent):
    user_id = event.source.user_id
    message_id = event.message.id
    report_id = f"{user_id}_{message_id}"

    db.collection("reports").document(report_id).set({
        "status": "PROCESSING",
        "created_at": datetime.now(timezone.utc).isoformat()
    })

    enqueue_task(report_id, user_id, message_id)

    safe_reply(
        event.reply_token,
        f"✅ 動画を受信しました。\n解析を開始します。\n\n{SERVICE_HOST_URL}/report/{report_id}"
    )


# ==================================================
# Task handler
# ==================================================
@app.route("/task-handler", methods=["POST"])
def task_handler():
    d = request.get_json()
    report_id = d["report_id"]
    user_id = d["user_id"]
    message_id = d["message_id"]

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, "video.mp4")

    try:
        content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        raw = analyze(video_path)

        analysis = {
            "01": {
                "title": "骨格計測データ（AIが測定）",
                "data": raw
            }
        }

        db.collection("reports").document(report_id).update({
            "status": "COMPLETED",
            "analysis": analysis
        })

        safe_push(
            user_id,
            f"🎉 スイング解析が完了しました！\n{SERVICE_HOST_URL}/report/{report_id}"
        )

        return jsonify({"ok": True})

    except Exception as e:
        print(traceback.format_exc())
        db.collection("reports").document(report_id).update({
            "status": "FAILED",
            "error": str(e)
        })
        safe_push(user_id, "解析中にエラーが発生しました")
        return "error", 500

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==================================================
# レポート表示
# ==================================================
@app.route("/report/<report_id>")
def report_page(report_id):
    return render_template("report.html", report_id=report_id)


@app.route("/api/report_data/<report_id>")
def report_data(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    return jsonify(doc.to_dict())


# ==================================================
# 起動
# ==================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
