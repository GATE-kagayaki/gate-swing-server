import os
import json
import math
import tempfile
import shutil
import traceback
from datetime import datetime, timezone

from flask import Flask, request, abort, jsonify, render_template
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, VideoMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError

from google.cloud import firestore, tasks_v2

# ==================================================
# App / Config
# ==================================================
app = Flask(__name__, template_folder="templates")

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]

PROJECT_ID = os.environ["PROJECT_ID"]
QUEUE_NAME = os.environ["TASK_QUEUE_NAME"]
QUEUE_LOCATION = os.environ["TASK_QUEUE_LOCATION"]
SERVICE_HOST_URL = os.environ["SERVICE_HOST_URL"]
TASK_SA_EMAIL = os.environ["TASK_SA_EMAIL"]

db = firestore.Client()
tasks_client = tasks_v2.CloudTasksClient()

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ==================================================
# ★ テスト切替（ここだけ触る）
# ==================================================
def user_is_premium(user_id: str) -> bool:
    # False → 無料版テスト
    # True  → 有料版テスト
    return False

# ==================================================
# MediaPipe analysis（簡略）
# ==================================================
def analyze(video_path: str):
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    max_shoulder = 0
    min_hip = 999
    max_wrist = 0
    max_head = 0
    max_knee = 0

    def angle(a, b, c):
        ax, ay = a[0]-b[0], a[1]-b[1]
        cx, cy = c[0]-b[0], c[1]-b[1]
        dot = ax*cx + ay*cy
        na = math.hypot(ax, ay)
        nc = math.hypot(cx, cy)
        if na*nc == 0:
            return 0
        return math.degrees(math.acos(dot/(na*nc)))

    with mp_pose.Pose() as pose:
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
            RH = mp_pose.PoseLandmark.RIGHT_HIP.value
            LH = mp_pose.PoseLandmark.LEFT_HIP.value
            LK = mp_pose.PoseLandmark.LEFT_KNEE.value
            LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
            LW = mp_pose.PoseLandmark.LEFT_WRIST.value
            LI = mp_pose.PoseLandmark.LEFT_INDEX.value
            NO = mp_pose.PoseLandmark.NOSE.value

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
        "max_head_drift": round(max_head, 4),
        "max_knee_sway": round(max_knee, 4),
    }

# ==================================================
# Analysis JSON 生成（最終確定）
# ==================================================
def build_analysis(raw, is_premium):
    analysis = {
        "01": {
            "title": "骨格計測データ（AIが測定）",
            "items": [
                {"name": "解析フレーム数", "value": raw["frame_count"], "guide": "150～300"},
                {"name": "最大肩回転角（°）", "value": raw["max_shoulder_rotation"], "guide": "80～110"},
                {"name": "最小腰回転角（°）", "value": raw["min_hip_rotation"], "guide": "35～45"},
                {"name": "最大手首コック角（°）", "value": raw["max_wrist_cock"], "guide": "120～150"},
                {"name": "最大頭部ブレ", "value": raw["max_head_drift"], "guide": "0.05～0.15"},
                {"name": "最大膝ブレ", "value": raw["max_knee_sway"], "guide": "0.05～0.20"},
            ],
        },
        "07": {
            "title": "総合評価",
            "text": [
                "骨格データからスイング全体の傾向を評価しました。",
                "安定性と回転量のバランスを整えることで、再現性の向上が期待できます。",
                "",
                "より詳しい分析をご希望の方へ",
                "本レポートではスイング全体の傾向を評価しています。",
                "ご自身のスイングを深く理解したい方は、ぜひフルレポートをご活用ください。",
            ],
        },
    }

    if not is_premium:
        return analysis

    # --- 有料版のみ ---
    analysis.update({
        "02": {"title": "Shoulder Rotation（肩回転）", "good": ["回転量は十分"], "bad": ["回し過ぎの傾向"]},
        "03": {"title": "Hip Rotation（腰回転）", "good": ["下半身は安定"], "bad": ["回転が浅くなりやすい"]},
        "04": {"title": "Wrist Cock（コック角）", "good": ["パワーを作れる"], "bad": ["手首主導になりやすい"]},
        "05": {"title": "Head Stability（頭部安定）", "good": ["大きな上下動なし"], "bad": ["左右ブレあり"]},
        "06": {"title": "Knee Stability（膝安定）", "good": ["踏ん張れている"], "bad": ["流れやすい"]},
        "08": {"title": "練習ドリル", "drills": []},
        "09": {"title": "シャフトフィッティング指針", "table": []},
        "10": {
            "title": "まとめ",
            "text": [
                "今回の解析ではスイングの土台は十分に整っています。",
                "体の回転とクラブ動作の同調が今後の課題です。",
                "練習と調整を重ねることで安定性はさらに向上します。",
                "",
                "あなたのゴルフライフが、より充実したものになることを切に願っています。",
            ],
        },
    })

    return analysis

# ==================================================
# Webhook
# ==================================================
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
    user_id = event.source.user_id
    msg_id = event.message.id
    report_id = f"{user_id}_{msg_id}"

    db.collection("reports").document(report_id).set({
        "status": "PROCESSING",
        "created_at": datetime.now(timezone.utc).isoformat()
    })

    reply = (
        "動画を受信しました。\n"
        "AIによるスイング解析を開始します。\n\n"
        "【進行状況の確認】\n"
        f"{SERVICE_HOST_URL}/report/{report_id}\n\n"
        "【料金プラン】\n"
        "① 500円／1回\n② 1,980円／5回\n③ 4,980円／月"
    )
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

# ==================================================
# Task handler（直実行版）
# ==================================================
@app.route("/task-handler", methods=["POST"])
def task_handler():
    d = request.get_json()
    report_id = d["report_id"]
    user_id = d["user_id"]
    msg_id = d["message_id"]

    tmp = tempfile.mkdtemp()
    video_path = os.path.join(tmp, "video.mp4")

    try:
        content = line_bot_api.get_message_content(msg_id)
        with open(video_path, "wb") as f:
            for c in content.iter_content():
                f.write(c)

        raw = analyze(video_path)
        is_premium = user_is_premium(user_id)
        analysis = build_analysis(raw, is_premium)

        db.collection("reports").document(report_id).update({
            "status": "COMPLETED",
            "analysis": analysis
        })

        line_bot_api.push_message(
            user_id,
            TextSendMessage(
                text="🎉 スイング計測が完了しました！\n"
                     f"{SERVICE_HOST_URL}/report/{report_id}"
            )
        )
        return jsonify(ok=True)

    except Exception as e:
        traceback.print_exc()
        return "error", 500
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ==================================================
# Pages
# ==================================================
@app.route("/report/<report_id>")
def report_page(report_id):
    return render_template("report.html", report_id=report_id)

@app.route("/api/report_data/<report_id>")
def report_data(report_id):
    doc = db.collection("reports").document(report_id).get()
    return jsonify(doc.to_dict() if doc.exists else {})
