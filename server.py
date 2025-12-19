import os
import json
import math
import shutil
import traceback
import tempfile
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter, defaultdict

from flask import Flask, request, jsonify, abort, render_template

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore
from google.cloud import tasks_v2
from google.api_core.exceptions import NotFound, PermissionDenied

# ==================================================
# CONFIG
# ==================================================
app = Flask(__name__, template_folder="templates")
app.config["JSON_AS_ASCII"] = False

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")

PROJECT_ID = (
    os.environ.get("PROJECT_ID")
    or os.environ.get("GCP_PROJECT_ID")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
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
# Utility
# ==================================================
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

# ==================================================
# MediaPipe Analysis（③完全版）
# ==================================================
def analyze_swing_with_mediapipe(video_path: str) -> Dict[str, Any]:
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCVがビデオを読み込めませんでした。")

    angles = defaultdict(list)

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

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
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

            angles["shoulder"].append(angle(xy(LS), xy(RS), xy(RH)))
            angles["hip"].append(angle(xy(LH), xy(RH), xy(LK)))
            angles["wrist"].append(angle(xy(LE), xy(LW), xy(LI)))
            angles["head"].append(abs(xy(NO)[0] - 0.5))
            angles["knee"].append(abs(xy(LK)[0] - 0.5))

    cap.release()

    if len(angles["shoulder"]) < 20:
        raise RuntimeError("解析フレーム不足")

    def pack(xs):
        return {
            "max": round(max(xs), 2),
            "mean": round(mean(xs), 2),
            "std": round(std(xs), 3),
        }

    return {
        "frame_count": len(angles["shoulder"]),
        "shoulder": pack(angles["shoulder"]),
        "hip": pack(angles["hip"]),
        "wrist": pack(angles["wrist"]),
        "head": pack(angles["head"]),
        "knee": pack(angles["knee"]),
    }

# ==================================================
# 3×3×3 Judge Core
# ==================================================
def judge_3x3x3(value: Dict[str, float], good_range: Tuple[float, float]) -> Dict[str, str]:
    lo, hi = good_range

    # 量
    if value["mean"] < lo:
        main = "low"
    elif value["mean"] > hi:
        main = "high"
    else:
        main = "mid"

    # 安定性
    quality = "stable" if value["std"] < (hi - lo) * 0.15 else "unstable"

    return {
        "main": main,
        "quality": quality,
    }

# ==================================================
# Section Builders（02〜06 共通思想）
# ==================================================
def build_section(title: str, value: Dict[str, float], good_range, seed):
    judge = judge_3x3x3(value, good_range)

    good, bad = [], []

    if judge["main"] == "mid":
        good.append("平均値が目安レンジ内で安定しています。")
    else:
        bad.append("平均値が目安レンジから外れています。")

    if judge["quality"] == "stable":
        good.append("動きのばらつきが小さく、再現性があります。")
    else:
        bad.append("ばらつきが大きく、安定性に課題があります。")

    pro = (
        f"平均{value['mean']}、最大{value['max']}、ばらつき{value['std']}です。"
        "ピークではなく“普段の動き”が評価を左右しています。"
    )

    return {
        "title": title,
        "value": value["mean"],
        "good": good[:3] or ["致命的な破綻は見られません。"],
        "bad": bad[:3] or ["現状は安定しています。"],
        "pro_comment": pro,
    }

# ==================================================
# Analysis Builder（③完全版）
# ==================================================
def build_analysis(raw: Dict[str, Any], report_id: str) -> Dict[str, Any]:
    analysis = {}

    analysis["02"] = build_section(
        "02. Shoulder Rotation（肩回転）",
        raw["shoulder"],
        (85, 105),
        report_id,
    )
    analysis["03"] = build_section(
        "03. Hip Rotation（腰回転）",
        raw["hip"],
        (36, 50),
        report_id,
    )
    analysis["04"] = build_section(
        "04. Wrist Cock（手首）",
        raw["wrist"],
        (70, 90),
        report_id,
    )
    analysis["05"] = build_section(
        "05. Head Stability（頭部）",
        raw["head"],
        (0.06, 0.15),
        report_id,
    )
    analysis["06"] = build_section(
        "06. Knee Stability（膝）",
        raw["knee"],
        (0.10, 0.20),
        report_id,
    )

    return analysis

# ==================================================
# Routes（実戦用）
# ==================================================
@app.route("/task-handler", methods=["POST"])
def task_handler():
    d = request.get_json()
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

        raw = analyze_swing_with_mediapipe(path)
        analysis = build_analysis(raw, report_id)

        db.collection("reports").document(report_id).set({
            "status": "COMPLETED",
            "raw_data": raw,
            "analysis": analysis,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=f"解析完了\n{SERVICE_HOST_URL}/report/{report_id}")
        )
        return "OK", 200

    except Exception:
        traceback.print_exc()
        return "NG", 500
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

@app.route("/report/<report_id>")
def report(report_id):
    return render_template("report.html", report_id=report_id)

@app.route("/api/report_data/<report_id>")
def api_report(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    return jsonify(doc.to_dict())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
