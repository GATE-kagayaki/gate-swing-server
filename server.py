import os
import json
import math
import shutil
import traceback
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# --- 修正: cv2, mp のインポートを追記 ---
import numpy as np
import cv2 
import mediapipe as mp 
# ------------------------------------

from flask import Flask, request, jsonify, abort, send_from_directory

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore
from google.cloud import tasks_v2
from google.api_core.exceptions import NotFound, PermissionDenied
from google.cloud.firestore import SERVER_TIMESTAMP


# ==================================================
# CONFIG
# ==================================================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")

PROJECT_ID = (
    os.environ.get("GCP_PROJECT_ID")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("GCP_PROJECT")
    or ""
)

QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")
QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")

SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")

TASK_HANDLER_PATH = "/task-handler"
TASK_HANDLER_URL = f"{SERVICE_HOST_URL}{TASK_HANDLER_PATH}"

db = firestore.Client()
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
tasks_client = tasks_v2.CloudTasksClient()


# ==================================================
# Helpers (省略 - 変更なし)
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
        "【診断レポートを見る】\n"
        f"{SERVICE_HOST_URL}/report/{report_id}"
    )


def is_premium_user(user_id: str) -> bool:
    return True


# ==================================================
# Cloud Tasks (省略 - 変更なし)
# ==================================================
def create_cloud_task(report_id: str, user_id: str, message_id: str) -> str:
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID is empty. Set GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT.")
    if not SERVICE_HOST_URL:
        raise RuntimeError("SERVICE_HOST_URL is empty.")
    if not TASK_SA_EMAIL:
        raise RuntimeError("TASK_SA_EMAIL is empty.")

    queue_path = tasks_client.queue_path(PROJECT_ID, QUEUE_LOCATION, QUEUE_NAME)

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

    resp = tasks_client.create_task(parent=queue_path, task=task)
    return resp.name


# ==================================================
# MediaPipe analysis (変更なし)
# ==================================================
def analyze_swing_with_mediapipe(video_path: str) -> Dict[str, Any]:
    # mp_pose, cap, angle, xy, LS, RS, ... は global な cv2, mp を利用します
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

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            lm = res.pose.landmarks.landmark
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

    if frame_count < 10:
        raise RuntimeError("解析に必要なフレーム数が不足しています。もう少し長めの動画でお試しください。")

    return {
        "frame_count": frame_count,
        "max_shoulder_rotation": round(max_shoulder, 2),
        "min_hip_rotation": round(min_hip, 2),
        "max_wrist_cock": round(max_wrist, 2),
        "max_head_drift_x": round(max_head, 4),
        "max_knee_sway_x": round(max_knee, 4),
    }


# ==================================================
# Routes (変更なし)
# ==================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "project_id": PROJECT_ID,
        "queue_location": QUEUE_LOCATION,
        "queue_name": QUEUE_NAME,
        "service_host_url": SERVICE_HOST_URL,
        "task_handler_url": TASK_HANDLER_URL,
        "task_sa_email_set": bool(TASK_SA_EMAIL),
    })


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
def handle_video(event: MessageEvent):
    user_id = event.source.user_id
    msg = event.message
    report_id = f"{user_id}_{msg.id}"

    firestore_safe_set(
        report_id,
        {
            "user_id": user_id,
            "status": "PROCESSING",
            "is_premium": is_premium_user(user_id),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    try:
        task_name = create_cloud_task(report_id, user_id, msg.id)
        firestore_safe_update(report_id, {"task_name": task_name})
        safe_line_reply(event.reply_token, make_initial_reply(report_id))
    except (NotFound, PermissionDenied) as e:
        firestore_safe_update(report_id, {"status": "TASK_FAILED", "error": str(e)})
        safe_line_reply(event.reply_token, "システムエラーが発生しました。時間を置いて再度お試しください。")
    except Exception as e:
        firestore_safe_update(report_id, {"status": "TASK_FAILED", "error": str(e)})
        print("Failed to create task:", traceback.format_exc())
        safe_line_reply(event.reply_token, "システムエラーが発生しました。時間を置いて再度お試しください。")


@app.route("/task-handler", methods=["POST"])
def task_handler():
    d = request.get_json(silent=True) or {}
    report_id = d.get("report_id")
    message_id = d.get("message_id")
    user_id = d.get("user_id")

    if not report_id or not message_id or not user_id:
        return "Invalid payload", 400

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, f"{message_id}.mp4")

    doc_ref = db.collection("reports").document(report_id)

    try:
        doc_ref.update({"status": "IN_PROGRESS"})
        
        # 1) download
        content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        # 2) analyze
        raw_data = analyze_swing_with_mediapipe(video_path)

        # 3) analysisデータの構築
        analysis = {
            "01": {
                "title": "骨格計測データ（AIが測った数値）",
                "data": {
                    "解析フレーム数": raw_data["frame_count"],
                    "最大肩回転": str(raw_data["max_shoulder_rotation"]),
                    "最小腰回転": str(raw_data["min_hip_rotation"]),
                    "最大コック角": str(raw_data["max_wrist_cock"]),
                    "最大頭ブレ（Sway）": str(raw_data["max_head_drift_x"]),
                    "最大膝ブレ（Sway）": str(raw_data["max_knee_sway_x"]),
                },
            },
            "07": {
                "title": "総合診断",
                "text": [
                    "**安定している点**",
                    "頭と下半身のブレが少なく、スイング軸が安定しています",
                    "再現性の高いスイングを構築しやすい土台を備えています",
                    "",
                    "**改善が期待される点**",
                    "上半身の捻転量が不足している場合、パワー効率が活かし切れません",
                    "手首主導になりやすい場合、タイミングのズレが出やすくなります",
                ],
            },
            "08": {
                "title": "改善戦略とドリル",
                "drills": [
                    { "ドリル名": "クロスアームターン", "目的": "上半身の捻転量を増やす", "やり方": "①胸の前で腕を軽く組む\n②下半身をできるだけ動かさず胸を回す\n③“胸が回る感覚”を保ったまま左右交互に行う", },
                    { "ドリル名": "L to L スイング", "目的": "手首の使いすぎを抑える", "やり方": "①腰〜腰の振り幅で構える\n②体の回転でクラブを動かす\n③リズムを一定にして手先で調整しない", },
                    { "ドリル名": "ウォールターン", "目的": "軸を保った回旋を習得する", "やり方": "①壁を背にしてアドレス\n②頭の位置をなるべく固定して肩を回す\n③壁との距離が変わらないか確認する", },
                ],
            },
            "09": {
                "title": "スイング傾向補正型フィッティング（ドライバーのみ）",
                "fitting": {
                    "シャフト重量": "50g台後半", "フレックス": "SR〜S", "キックポイント": "先中調子", "トルク": "3.8〜4.5", 
                    "備考": "下半身の安定性を活かしつつ、捻転不足を補うシャフト挙動を推奨。"
                },
            },
            "10": {
                "title": "まとめ（次のステップ）",
                "text": [
                    "安定した下半身と軸を活かしながら、上半身の捻転量を高めていくことが最優先課題です。",
                    "体全体を使ったスイングに近づくことで、飛距離と方向性の両立が期待できます。",
                    "",
                    "お客様のゴルフライフが、より充実したものになることを切に願っています。",
                ],
            },
        }


        doc_ref.update(
            {
                "status": "COMPLETED",
                "raw_data": raw_data,
                "analysis": analysis,
                "updated_at": SERVER_TIMESTAMP,
            }
        )

        safe_line_push(user_id, make_done_push(report_id))
        return jsonify({"ok": True}), 200

    except Exception as e:
        print(traceback.format_exc())
        doc_ref.update({"status": "FAILED", "error": str(e)})
        safe_line_push(user_id, "システムエラーが発生し、解析を完了できませんでした。")
        return "Internal Error", 500

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route("/report/<report_id>")
def serve_report(report_id):
    """レポートHTMLファイルを提供する"""
    return send_from_directory("templates", "report.html")


@app.route("/api/report_data/<report_id>")
def api_report_data(report_id):
    """フロントエンドへレポートデータをJSONで提供する"""
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    d = doc.to_dict() or {}
    return jsonify(
        {
            "status": d.get("status"),
            "analysis": d.get("analysis", {}),
            "raw_data": d.get("raw_data", {}),
            "error": d.get("error"),
            "created_at": d.get("created_at"),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

