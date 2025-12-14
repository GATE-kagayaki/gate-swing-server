import os
import json
import math
import shutil
import tempfile
import traceback
from typing import Dict, Any

from flask import Flask, request, abort, jsonify, send_from_directory
from google.cloud import firestore, tasks_v2
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (MessageEvent, VideoMessage, FileMessage, TextMessage,
                            TextSendMessage)

# ========== ENV ========== #
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")
TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")

# ========== INIT ========== #
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
db = firestore.Client(project=GCP_PROJECT_ID)
tasks_client = tasks_v2.CloudTasksClient()
queue_path = tasks_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)

# ========== HELPERS ========== #
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
    except Exception:
        print(traceback.format_exc())

def safe_line_push(user_id: str, text: str) -> None:
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text=text))
    except Exception:
        print(traceback.format_exc())

# ========== MESSAGE TEMPLATES ========== #
def make_initial_reply(report_id: str) -> str:
    return (
        "✅ 動画を受信しました。\n"
        "AIによるスイング数値計測を開始します。\n\n"
        "完了まで1〜3分ほどお待ちください。\n"
        "完了すると自動で通知が届きます。\n\n"
        f"【現在のステータス確認】\n{SERVICE_HOST_URL}/report/{report_id}"
    )

def make_done_push(report_id: str) -> str:
    return (
        "🎉 スイング診断が完了しました！\n\n"
        "レポートを以下のリンクからご確認いただけます。\n\n"
        f"{SERVICE_HOST_URL}/report/{report_id}"
    )

# ========== LINE WEBHOOK ========== #
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
def handle_message(event: MessageEvent):
    msg = event.message
    user_id = event.source.user_id

    if isinstance(msg, (VideoMessage, FileMessage)):
        report_id = f"{user_id}_{msg.id}"
        firestore_safe_set(report_id, {
            "user_id": user_id,
            "status": "PROCESSING",
            "created_at": firestore.SERVER_TIMESTAMP
        })
        create_cloud_task(report_id, user_id, msg.id)
        safe_line_reply(event.reply_token, make_initial_reply(report_id))
    else:
        safe_line_reply(event.reply_token, "🎥 スイング動画を送ってください。")

# ========== TASK ========== #
def create_cloud_task(report_id: str, user_id: str, message_id: str):
    payload = json.dumps({"report_id": report_id, "user_id": user_id, "message_id": message_id}).encode("utf-8")
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{SERVICE_HOST_URL}/worker/process_video",
            "headers": {"Content-Type": "application/json"},
            "body": payload,
            "oidc_token": {
                "service_account_email": TASK_SA_EMAIL,
                "audience": SERVICE_HOST_URL
            }
        }
    }
    tasks_client.create_task(parent=queue_path, task=task)

# ========== VIDEO ANALYSIS ========== #
def analyze_swing_with_mediapipe(video_path: str) -> Dict[str, Any]:
    import cv2
    import mediapipe as mp

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
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (na * nb)))))

    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
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
        "frame_count": frame_count,
        "max_shoulder_rotation": round(max_shoulder, 2),
        "min_hip_rotation": round(min_hip, 2),
        "max_wrist_cock": round(max_wrist, 2),
        "max_head_drift_x": round(max_head, 4),
        "max_knee_sway_x": round(max_knee, 4)
    }

@app.route("/worker/process_video", methods=["POST"])
def worker():
    payload = request.get_json()
    report_id = payload.get("report_id")
    message_id = payload.get("message_id")

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, f"{message_id}.mp4")
    try:
        content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)
        data = analyze_swing_with_mediapipe(video_path)
        firestore_safe_update(report_id, {
            "status": "COMPLETED",
            "raw_data": data,
            "completed_at": firestore.SERVER_TIMESTAMP
        })
        doc = db.collection("reports").document(report_id).get()
        if doc.exists:
            user_id = doc.to_dict().get("user_id")
            safe_line_push(user_id, make_done_push(report_id))
    except Exception as e:
        firestore_safe_update(report_id, {"status": "FAILED", "error": str(e)})
        return jsonify({"status": "failed", "error": str(e)}), 200
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return jsonify({"ok": True})

# HTML レポート出力とデータ取得APIは別ファイルに移植可能。ここでは続きとして記述可。
# templates/report.html を使ったビジュアルなHTMLテンプレートも必要です。

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

