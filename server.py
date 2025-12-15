import os
import json
import math
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from typing import Dict, Any

from flask import Flask, request, jsonify, abort, render_template
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore
from google.cloud import tasks_v2

import cv2
import mediapipe as mp


# ==================================================
# CONFIG
# ==================================================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")
QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")

TASK_HANDLER_PATH = "/task-handler"
TASK_HANDLER_URL = f"{SERVICE_HOST_URL}{TASK_HANDLER_PATH}"

db = firestore.Client()
tasks_client = tasks_v2.CloudTasksClient()

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# ==================================================
# LINE文言
# ==================================================
def make_initial_reply(report_id: str) -> str:
    return (
        "動画を受信しました。\n"
        "AIによるスイング解析を開始します。\n\n"
        "解析完了まで、1〜3分ほどお待ちください。\n"
        "完了次第、結果をお知らせします。\n\n"
        "【進行状況の確認】\n"
        "以下のURLから、解析の進行状況や\n"
        "レポートの準備状況を確認できます。\n"
        f"{SERVICE_HOST_URL}/report/{report_id}\n\n"
        "【料金プラン（プロ評価付きフルレポート）】\n"
        "① 都度会員　500円／1回\n"
        "② 回数券　1,980円／5回\n"
        "③ 月額会員　4,980円／月\n\n"
        "※無料版でも骨格解析と総合評価はご利用いただけます。"
    )


def make_done_push(report_id: str) -> str:
    return (
        "🎉 スイング計測が完了しました！\n\n"
        "以下のリンクから診断レポートを確認できます。\n\n"
        f"{SERVICE_HOST_URL}/report/{report_id}"
    )


# ==================================================
# Cloud Tasks
# ==================================================
def enqueue_task(report_id: str, user_id: str, message_id: str):
    parent = tasks_client.queue_path(PROJECT_ID, QUEUE_LOCATION, QUEUE_NAME)
    payload = json.dumps(
        {"report_id": report_id, "user_id": user_id, "message_id": message_id},
        ensure_ascii=False,
    ).encode("utf-8")

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
# MediaPipe解析
# ==================================================
def analyze_video(video_path: str) -> Dict[str, Any]:
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    max_shoulder = 0.0
    min_hip = 999.0
    max_wrist = 0.0
    max_head = 0.0
    max_knee = 0.0

    def angle(p1, p2, p3):
        ax, ay = p1[0] - p2[0], p1[1] - p2[1]
        bx, by = p3[0] - p2[0], p3[1] - p2[1]
        dot = ax * bx + ay * by
        na = math.hypot(ax, ay)
        nb = math.hypot(bx, by)
        if na * nb == 0:
            return 0.0
        return math.degrees(math.acos(max(-1, min(1, dot / (na * nb)))))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
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

            LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LH = mp_pose.PoseLandmark.LEFT_HIP.value
            RH = mp_pose.PoseLandmark.RIGHT_HIP.value
            LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
            LW = mp_pose.PoseLandmark.LEFT_WRIST.value
            LI = mp_pose.PoseLandmark.LEFT_INDEX.value
            NO = mp_pose.PoseLandmark.NOSE.value
            LK = mp_pose.PoseLandmark.LEFT_KNEE.value

            max_shoulder = max(max_shoulder, angle(xy(LS), xy(RS), xy(RH)))
            min_hip = min(min_hip, angle(xy(LH), xy(RH), xy(LK)))
            max_wrist = max(max_wrist, angle(xy(LE), xy(LW), xy(LI)))
            max_head = max(max_head, abs(xy(NO)[0] - 0.5))
            max_knee = max(max_knee, abs(xy(LK)[0] - 0.5))

    cap.release()

    return {
        "解析フレーム数": frame_count,
        "最大肩回転角": round(max_shoulder, 2),
        "最小腰回転角": round(min_hip, 2),
        "最大手首コック角": round(max_wrist, 2),
        "最大頭部ブレ": round(max_head, 4),
        "最大膝ブレ": round(max_knee, 4),
    }


# ==================================================
# Routes
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
def on_video(event):
    user_id = event.source.user_id
    message_id = event.message.id
    report_id = f"{user_id}_{message_id}"

    db.collection("reports").document(report_id).set({
        "status": "PROCESSING",
        "created_at": datetime.now(timezone.utc).isoformat()
    })

    enqueue_task(report_id, user_id, message_id)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=make_initial_reply(report_id))
    )


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

        result = analyze_video(video_path)

        db.collection("reports").document(report_id).update({
            "status": "COMPLETED",
            "analysis": result,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

        line_bot_api.push_message(
            user_id, TextSendMessage(text=make_done_push(report_id))
        )
        return jsonify({"ok": True})

    except Exception:
        traceback.print_exc()
        db.collection("reports").document(report_id).update({
            "status": "FAILED"
        })
        return "error", 500

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route("/report/<report_id>")
def report_page(report_id):
    return render_template("report.html", report_id=report_id)


@app.route("/api/report_data/<report_id>")
def report_data(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    return jsonify(doc.to_dict())
