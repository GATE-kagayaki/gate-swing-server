import os
import json
import time
import math
import shutil
import traceback
import tempfile
from typing import Any, Dict, Optional

from flask import Flask, request, abort, jsonify

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent,
    VideoMessage,
    TextMessage,
    ImageMessage,
    StickerMessage,
    FileMessage,
    TextSendMessage,
)

from google.cloud import firestore, tasks_v2

# ==================================================
# ENV
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")

# Cloud Tasks Config
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")
# 修正: キー名を正しく修正しました
TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")

# ==================================================
# App init
# ==================================================
app = Flask(__name__)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Firestore & Cloud Tasks Clients
db = firestore.Client(project=GCP_PROJECT_ID)
tasks_client = tasks_v2.CloudTasksClient()
queue_path = tasks_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)

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

# ==================================================
# Messages
# ==================================================
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

# ==================================================
# Cloud Tasks
# ==================================================
def create_cloud_task(report_id: str, user_id: str, message_id: str) -> None:
    payload = json.dumps(
        {"report_id": report_id, "user_id": user_id, "message_id": message_id}
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
    tasks_client.create_task(parent=queue_path, task=task)

# ==================================================
# MediaPipe Analysis
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

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            # 高速化のため、すべてのフレームを処理せず間引くことも検討可能だが、一旦全フレーム処理
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

            # 簡易ロジック
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
        "max_knee_sway_x": round(max_knee, 4),
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

@handler.add(MessageEvent)
def handle_any(event: MessageEvent):
    msg = event.message
    user_id = event.source.user_id

    # 動画またはファイルの場合のみ処理
    if isinstance(msg, (VideoMessage, FileMessage)):
        report_id = f"{user_id}_{msg.id}"
        firestore_safe_set(
            report_id,
            {
                "user_id": user_id,
                "status": "PROCESSING",
                "created_at": firestore.SERVER_TIMESTAMP,
            },
        )
        create_cloud_task(report_id, user_id, msg.id)
        safe_line_reply(event.reply_token, make_initial_reply(report_id))
        return

    # それ以外は案内
    safe_line_reply(event.reply_token, "🎥 解析したいスイング動画を送信してください。")

@app.route("/worker/process_video", methods=["POST"])
def worker():
    payload = request.get_json()
    report_id = payload.get("report_id")
    message_id = payload.get("message_id")

    if not report_id or not message_id:
        return jsonify({"error": "invalid payload"}), 400

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, f"{message_id}.mp4")

    try:
        # LINEから動画を取得
        message_content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for chunk in message_content.iter_content():
                f.write(chunk)

        # MediaPipe解析
        raw_data = analyze_swing_with_mediapipe(video_path)

        # 完了ステータス更新
        firestore_safe_update(
            report_id,
            {
                "status": "COMPLETED",
                "raw_data": raw_data,
                "completed_at": firestore.SERVER_TIMESTAMP,
            },
        )

        # 完了通知
        doc = db.collection("reports").document(report_id).get()
        if doc.exists:
            data = doc.to_dict()
            safe_line_push(data.get("user_id"), make_done_push(report_id))

    except Exception as e:
        print(f"Error: {e}")
        firestore_safe_update(report_id, {"status": "FAILED", "error": str(e)})
        return jsonify({"status": "failed", "error": str(e)}), 200

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return jsonify({"ok": True})

@app.route("/api/report_data/<report_id>")
def api_report_data(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    d = doc.to_dict()
    return jsonify({
        "status": d.get("status"),
        "mediapipe_data": d.get("raw_data", {}),
        "created_at": d.get("created_at")
    })

# Word風デザインのHTMLテンプレート
REPORT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>スイング診断レポート</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;700&display=swap" rel="stylesheet">
<style>
    body { font-family: 'Noto Serif JP', serif; background-color: #f3f4f6; color: #333; }
    .a4-paper {
        background: white;
        width: 100%;
        max-width: 210mm;
        min-height: 297mm;
        margin: 20px auto;
        padding: 40px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    @media (max-width: 640px) {
        .a4-paper { margin: 0; padding: 20px; min-height: 100vh; box-shadow: none; }
    }
    .metric-row { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e5e7eb; padding: 12px 0; }
    .metric-name { font-weight: bold; color: #4b5563; }
    .metric-value { font-size: 1.25rem; font-weight: bold; color: #059669; }
</style>
</head>
<body>

<div class="a4-paper">
    <div class="border-b-2 border-emerald-800 pb-4 mb-8 flex justify-between items-end">
        <div>
            <h1 class="text-2xl font-bold text-emerald-900">SWING ANALYSIS REPORT</h1>
            <p class="text-sm text-gray-500 mt-1">AIスイング診断書</p>
        </div>
        <div class="text-right">
            <p class="text-xs text-gray-400">REPORT ID</p>
            <p class="font-mono text-xs text-gray-500" id="reportIdDisplay">---</p>
        </div>
    </div>

    <div id="loading" class="text-center py-20">
        <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-emerald-700 mx-auto"></div>
        <p class="mt-4 text-gray-500 text-sm">データを取得中...</p>
    </div>

    <div id="error" class="hidden text-center py-20 bg-red-50 text-red-700 rounded mb-4">
        <p class="font-bold">データ取得エラー</p>
        <p class="text-sm" id="errorMsg">URLを確認してください。</p>
    </div>

    <div id="content" class="hidden">
        <div class="mb-8">
            <h2 class="text-lg font-bold text-emerald-800 border-l-4 border-emerald-600 pl-3 mb-4">測定結果 (Measurements)</h2>
            
            <div class="space-y-2">
                <div class="metric-row">
                    <span class="metric-name">肩の捻転 (Shoulder Rotation)</span>
                    <span class="metric-value"><span id="val_shoulder">-</span><span class="text-sm text-gray-500 ml-1">°</span></span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">腰の回転 (Hip Rotation)</span>
                    <span class="metric-value"><span id="val_hip">-</span><span class="text-sm text-gray-500 ml-1">°</span></span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">コック角 (Wrist Cock)</span>
                    <span class="metric-value"><span id="val_wrist">-</span><span class="text-sm text-gray-500 ml-1">°</span></span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">頭のブレ (Head Drift)</span>
                    <span class="metric-value"><span id="val_head">-</span><span class="text-sm text-gray-500 ml-1">px</span></span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">膝の揺れ (Knee Sway)</span>
                    <span class="metric-value"><span id="val_knee">-</span><span class="text-sm text-gray-500 ml-1">px</span></span>
                </div>
            </div>
        </div>

        <div class="bg-gray-50 p-6 rounded-lg border border-gray-200 text-sm text-gray-600">
            <h3 class="font-bold text-gray-800 mb-2">💡 分析ノート</h3>
            <ul class="list-disc ml-5 space-y-1">
                <li>肩の捻転は90度以上、腰は45度程度が理想的な捻転差を生みます。</li>
                <li>頭のブレが大きい場合、軸が安定していない可能性があります。</li>
                <li>数値はカメラのアングルや距離によって変動します。定

