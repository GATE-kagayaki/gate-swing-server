import os
import json
import math
import shutil
import traceback
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict

from flask import Flask, request, jsonify, abort, send_from_directory

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore


# ==================================================
# CONFIG
# ==================================================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")

db = firestore.Client()

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


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
        "完了すると自動で通知が届きます。\n\n"
        "【現在のステータス確認】\n"
        f"{SERVICE_HOST_URL}/report/{report_id}"
    )


def make_done_push(report_id: str) -> str:
    return (
        "🎉 スイング計測が完了しました！\n\n"
        "以下のリンクから診断レポートを確認できます。\n\n"
        "【診断レポートを見る】\n"
        f"{SERVICE_HOST_URL}/report/{report_id}"
    )


# ==================================================
# MediaPipe Analysis（構文安全版）
# ==================================================
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
        c = max(-1.0, min(1.0, dot / (na * nb)))
        return math.degrees(math.acos(c))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
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

    finally:
        cap.release()
        pose.close()

    if frame_count < 10:
        raise RuntimeError("解析に必要なフレーム数が不足しています。")

    return {
        "frame_count": frame_count,
        "max_shoulder_rotation": round(max_shoulder, 2),
        "min_hip_rotation": round(min_hip, 2),
        "max_wrist_cock": round(max_wrist, 2),
        "max_head_drift_x": round(max_head, 4),
        "max_knee_sway_x": round(max_knee, 4),
    }


# ==================================================
# 直列解析実行（Cloud Tasks 不使用）
# ==================================================
def run_analysis(report_id: str, user_id: str, message_id: str):
    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, f"{message_id}.mp4")
    doc_ref = db.collection("reports").document(report_id)

    try:
        doc_ref.update({"status": "IN_PROGRESS"})

        content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        raw_data = analyze_swing_with_mediapipe(video_path)

        analysis = {
            "01": {
                "title": "骨格計測データ（AIが測った数値）",
                "data": {
                    "解析フレーム数": raw_data["frame_count"],
                    "最大肩回転": raw_data["max_shoulder_rotation"],
                    "最小腰回転": raw_data["min_hip_rotation"],
                    "最大コック角": raw_data["max_wrist_cock"],
                    "最大頭ブレ": raw_data["max_head_drift_x"],
                    "最大膝ブレ": raw_data["max_knee_sway_x"],
                },
            }
        }

        doc_ref.update({
            "status": "COMPLETED",
            "raw_data": raw_data,
            "analysis": analysis,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

        safe_line_push(user_id, make_done_push(report_id))

    except Exception as e:
        doc_ref.update({"status": "FAILED", "error": str(e)})
        safe_line_push(user_id, "解析中にエラーが発生しました。")
        print(traceback.format_exc())

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==================================================
# Routes
# ==================================================
@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception:
        print(traceback.format_exc())
    return "OK"


@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event: MessageEvent):
    user_id = event.source.user_id
    msg = event.message
    report_id = f"{user_id}_{msg.id}"

    firestore_safe_set(
        report_id,
        {
            "user_id": user_id,
            "status": "PROCESSING",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    safe_line_reply(event.reply_token, make_initial_reply(report_id))

    # ★ 直列実行
    run_analysis(report_id, user_id, msg.id)


@app.route("/report/<report_id>")
def serve_report(report_id):
    return send_from_directory("templates", "report.html")


@app.route("/api/report_data/<report_id>")
def api_report_data(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    return jsonify(doc.to_dict())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
