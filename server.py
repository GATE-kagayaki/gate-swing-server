import os
import json
import math
import shutil
import traceback
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

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
# Utilities
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


def is_premium_user(user_id: str) -> bool:
    # Stripe 連携後に置き換え
    return True


# ==================================================
# Category helpers
# ==================================================
def cat3_by_range(value: float, lo: float, hi: float) -> str:
    if value < lo:
        return "low"
    if value > hi:
        return "high"
    return "mid"


def cat3_sway(value: float, lo: float, hi: float) -> str:
    if value < lo:
        return "good"
    if value > hi:
        return "bad"
    return "mid"


# ==================================================
# 02–06 判定＋文章生成
# ==================================================
def build_paid_02_to_06(raw: Dict[str, Any]) -> Dict[str, Any]:
    shoulder = cat3_by_range(raw["max_shoulder_rotation"], 80, 110)
    hip = cat3_by_range(raw["min_hip_rotation"], 35, 45)
    wrist = cat3_by_range(raw["max_wrist_cock"], 120, 150)

    head = cat3_sway(raw["max_head_drift"], 0.05, 0.15)
    knee = cat3_sway(raw["max_knee_sway"], 0.05, 0.20)

    judgements = {
        "shoulder": {"amount": shoulder},
        "hip": {"amount": hip},
        "wrist": {"amount": wrist},
        "head": {"stability": head},
        "knee": {"stability": knee},
    }

    def bullets(main, head, knee):
        good, bad = [], []
        if main == "high":
            good.append("回転量が大きく、パワーを出せるポテンシャルがあります。")
        if main == "low":
            bad.append("回転量が少なく、エネルギー効率が下がりやすいです。")
        if head == "bad":
            bad.append("頭部ブレがあり、再現性が下がりやすいです。")
        if knee == "bad":
            bad.append("下半身が流れやすく、軸が不安定です。")
        return good[:3] or ["全体として安定しています。"], bad[:3] or ["大きな崩れは見られません。"]

    g2, b2 = bullets(shoulder, head, knee)
    g3, b3 = bullets(hip, head, knee)
    g4, b4 = bullets(wrist, head, knee)
    g5, b5 = bullets(head, head, knee)
    g6, b6 = bullets(knee, head, knee)

    return {
        "_judgements": judgements,
        "02": {"title": "02. Shoulder Rotation", "value": raw["max_shoulder_rotation"], "good": g2, "bad": b2},
        "03": {"title": "03. Hip Rotation", "value": raw["min_hip_rotation"], "good": g3, "bad": b3},
        "04": {"title": "04. Wrist Cock", "value": raw["max_wrist_cock"], "good": g4, "bad": b4},
        "05": {"title": "05. Head Stability", "value": raw["max_head_drift"], "good": g5, "bad": b5},
        "06": {"title": "06. Knee Stability", "value": raw["max_knee_sway"], "good": g6, "bad": b6},
    }


# ==================================================
# 08 ドリル自動選択（15本）
# ==================================================
def build_paid_08(analysis_02_to_06: Dict[str, Any]) -> Dict[str, Any]:
    j = analysis_02_to_06["_judgements"]
    needs = set()

    if j["shoulder"]["amount"] == "high":
        needs.add("SHOULDER_HIGH")
    if j["shoulder"]["amount"] == "low":
        needs.add("SHOULDER_LOW")
    if j["hip"]["amount"] == "low":
        needs.add("HIP_LOW")
    if j["wrist"]["amount"] == "high":
        needs.add("WRIST_HIGH")
    if j["head"]["stability"] == "bad":
        needs.add("HEAD_BAD")
    if j["knee"]["stability"] == "bad":
        needs.add("KNEE_BAD")

    drills = [
        {"name": "頭固定ドリル", "tags": {"HEAD_BAD"}, "purpose": "頭部ブレを抑える", "how": "壁の前で頭を固定"},
        {"name": "膝タオルドリル", "tags": {"KNEE_BAD"}, "purpose": "下半身安定", "how": "膝にタオルを挟む"},
        {"name": "クロスアーム同調", "tags": {"SHOULDER_HIGH"}, "purpose": "同調改善", "how": "腕を組んで回転"},
        {"name": "壁ドリル（腰）", "tags": {"HIP_LOW"}, "purpose": "腰回転改善", "how": "尻を壁に当てる"},
        {"name": "L to L スイング", "tags": {"WRIST_HIGH"}, "purpose": "手首主導抑制", "how": "腰〜腰の振り幅"},
    ]

    scored = []
    for d in drills:
        score = len(d["tags"] & needs)
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = [d for _, d in scored[:3]] or [drills[0]]

    return {"title": "08. Training Drills", "drills": selected}


# ==================================================
# 09 シャフトフィッティング
# ==================================================
def build_paid_09(analysis_02_to_06: Dict[str, Any], inputs: Dict[str, Any] | None = None) -> Dict[str, Any]:
    inputs = inputs or {}
    j = analysis_02_to_06["_judgements"]

    hs = inputs.get("head_speed")
    miss = inputs.get("miss")

    rows = []

    if hs:
        if hs < 33:
            w = "40–50g"
        elif hs < 38:
            w = "50–60g"
        else:
            w = "60–70g"
        rows.append({"item": "重量", "guide": w, "reason": "ヘッドスピード基準"})
    else:
        rows.append({"item": "重量", "guide": "50–60g", "reason": "安定性重視"})

    kp = "中調子"
    if miss in ("slice", "push"):
        kp = "先調子寄り"
    if j["wrist"]["amount"] == "high":
        kp = "中元調子"
    rows.append({"item": "キックポイント", "guide": kp, "reason": "タイミング重視"})

    rows.append({"item": "フレックス", "guide": "R–SR–S", "reason": "試打推奨"})
    rows.append({"item": "トルク", "guide": "3.5–4.5", "reason": "方向安定"})

    return {
        "title": "09. Shaft Fitting Guide",
        "table": rows,
        "note": "本結果は指標のため、購入時は必ず試打を行ってください。",
    }


# ==================================================
# Analysis Builder
# ==================================================
def build_analysis(raw: Dict[str, Any], premium: bool) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {}

    if premium:
        sec02to06 = build_paid_02_to_06(raw)
        analysis.update(sec02to06)
        analysis["08"] = build_paid_08(sec02to06)
        analysis["09"] = build_paid_09(sec02to06)
    return analysis


# ==================================================
# API: 09 更新
# ==================================================
@app.route("/api/update_fitting", methods=["POST"])
def api_update_fitting():
    data = request.get_json()
    report_id = data["report_id"]
    inputs = data.get("inputs", {})

    ref = db.collection("reports").document(report_id)
    doc = ref.get()
    analysis = doc.to_dict()["analysis"]

    sec02to06 = { "_judgements": analysis["_judgements"] }
    new09 = build_paid_09(sec02to06, inputs)

    ref.update({"analysis.09": new09})
    return jsonify({"ok": True, "analysis_09": new09})
