import os
import json
import math
import shutil
import traceback
import tempfile
import random
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter

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
    url = f"{SERVICE_HOST_URL}/report/{report_id}"
    return (
        "✅ 動画を受信しました。\n"
        "AIによるスイング解析を開始します。\n\n"
        "解析完了まで、1〜3分ほどお待ちください。\n"
        "完了次第、結果をお知らせします。\n\n"
        "【進行状況の確認】\n"
        f"{url}"
    )


def make_done_push(report_id: str) -> str:
    url = f"{SERVICE_HOST_URL}/report/{report_id}"
    return (
        "🎉 スイング計測が完了しました！\n\n"
        "以下のリンクから診断レポートを確認できます。\n\n"
        f"{url}"
    )


# ==================================================
# Premium判定（本番は決済と連携でOK）
# ==================================================
def is_premium_user(user_id: str) -> bool:
    # Stripe連携後に置き換え
    return True


# ==================================================
# Cloud Tasks
# ==================================================
def create_cloud_task(report_id: str, user_id: str, message_id: str) -> str:
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID is empty. Set PROJECT_ID or GCP_PROJECT_ID.")
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
# Reference averages (fixed inside product)
# ==================================================
REF_AVG = {
    "shoulder": 95.0,
    "hip": 42.0,
    "wrist": 80.0,
    "head": 0.10,
    "knee": 0.15,
}


def _safe_pstdev(values: List[float]) -> float:
    if not values or len(values) < 2:
        return 0.0
    try:
        return float(statistics.pstdev(values))
    except Exception:
        return 0.0


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    try:
        return float(statistics.mean(values))
    except Exception:
        return 0.0


def _fmt_deg(x: float) -> str:
    return f"{x:.1f}°"


def _fmt_sway(x: float) -> str:
    return f"{x:.3f}"


def _diff_phrase(value: float, ref: float, unit: str, tol_ratio: float = 0.03) -> str:
    # ±3%以内は「平均付近」
    if ref == 0:
        return "平均値の基準が設定されていません。"
    diff = value - ref
    if abs(diff) <= abs(ref) * tol_ratio:
        return "一般的な平均付近です。"
    if diff > 0:
        return f"平均より{abs(diff):.1f}{unit}大きめです。"
    return f"平均より{abs(diff):.1f}{unit}小さめです。"


def _stability_phrase(std: float, kind: str) -> str:
    # kind: "deg" or "sway"
    # ざっくり閾値（軽量・実務向け）
    if kind == "deg":
        if std <= 2.0:
            return "動きのばらつきは小さく、再現性が高い状態です。"
        if std <= 4.5:
            return "動きは平均的なばらつきで、改善を積み上げやすい状態です。"
        return "動きのばらつきが大きく、タイミングがズレやすい状態です。"
    else:
        if std <= 0.010:
            return "ブレのばらつきは小さく、軸が揃っています。"
        if std <= 0.020:
            return "ブレは平均的で、安定性は許容範囲です。"
        return "ブレのばらつきが大きく、安定性が崩れやすい状態です。"


def _confidence_prefix(conf: str) -> str:
    if conf == "high":
        return "この数値から明確に言えます。"
    if conf == "mid":
        return "傾向として見られます。"
    return "参考値として捉えてください。"


def _pro_3lines(conf: str, line1: str, line2: str, line3: str) -> str:
    # HTML側は innerHTML なので <br> で3行固定
    p = _confidence_prefix(conf)
    return f"{p} {line1}<br>{line2}<br>{line3}"


# ==================================================
# MediaPipe analysis（max + 平均 + ばらつき）
# ==================================================
def analyze_swing_with_mediapipe(video_path: str) -> Dict[str, Any]:
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCVがビデオを読み込めませんでした。")

    frame_count = 0

    max_shoulder = 0.0
    min_hip = 999.0
    max_wrist = 0.0
    max_head = 0.0
    max_knee = 0.0

    shoulder_values: List[float] = []
    hip_values: List[float] = []
    wrist_values: List[float] = []
    head_values: List[float] = []
    knee_values: List[float] = []

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

            lm = res.pose_landmarks.landmark

            def xy(i):
                return (lm[i].x, lm[i].y)

            LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LH = mp_pose.PoseLandmark.LEFT_HIP.value
            RH = mp_pose.PoseLandmark.RIGHT_HIP.value
            LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
            LW = mp_pose.PoseLandmark.LEFT_WRIST.value
            LI = mp_pose.PoseLandmark.LEFT_INDEX.value
            NO = mp_pose.PoseLandmark.NOSE.value
            LK = mp_pose.PoseLandmark.LEFT_KNEE.value

            sh = angle(xy(LS), xy(RS), xy(RH))
            hip = angle(xy(LH), xy(RH), xy(LK))
            wr = angle(xy(LE), xy(LW), xy(LI))
            hd = abs(xy(NO)[0] - 0.5)
            kn = abs(xy(LK)[0] - 0.5)

            shoulder_values.append(float(sh))
            hip_values.append(float(hip))
            wrist_values.append(float(wr))
            head_values.append(float(hd))
            knee_values.append(float(kn))

            max_shoulder = max(max_shoulder, sh)
            min_hip = min(min_hip, hip)
            max_wrist = max(max_wrist, wr)
            max_head = max(max_head, hd)
            max_knee = max(max_knee, kn)

    cap.release()

    if frame_count < 10:
        raise RuntimeError("解析に必要なフレーム数が不足しています。")

    # 平均（hipは仕様上「min_hip」を使っているが、平均も持つ）
    sh_avg = _safe_mean(shoulder_values)
    hip_avg = _safe_mean(hip_values)
    wr_avg = _safe_mean(wrist_values)
    hd_avg = _safe_mean(head_values)
    kn_avg = _safe_mean(knee_values)

    # ばらつき（標準偏差）
    sh_std = _safe_pstdev(shoulder_values)
    hip_std = _safe_pstdev(hip_values)
    wr_std = _safe_pstdev(wrist_values)
    hd_std = _safe_pstdev(head_values)
    kn_std = _safe_pstdev(knee_values)

    return {
        "frame_count": int(frame_count),

        # 既存キー（互換維持）
        "max_shoulder_rotation": round(float(max_shoulder), 2),
        "min_hip_rotation": round(float(min_hip), 2),
        "max_wrist_cock": round(float(max_wrist), 2),
        "max_head_drift": round(float(max_head), 4),
        "max_knee_sway": round(float(max_knee), 4),

        # 追加キー（差別化の核）
        "avg_shoulder_rotation": round(float(sh_avg), 2),
        "avg_hip_rotation": round(float(hip_avg), 2),
        "avg_wrist_cock": round(float(wr_avg), 2),
        "avg_head_drift": round(float(hd_avg), 4),
        "avg_knee_sway": round(float(kn_avg), 4),

        "std_shoulder_rotation": round(float(sh_std), 3),
        "std_hip_rotation": round(float(hip_std), 3),
        "std_wrist_cock": round(float(wr_std), 3),
        "std_head_drift": round(float(hd_std), 4),
        "std_knee_sway": round(float(kn_std), 4),
    }


# ==================================================
# Section 01
# ==================================================
def build_section_01(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "01. 骨格計測データ（AIが測定）",
        "items": [
            {
                "name": "解析フレーム数",
                "value": raw["frame_count"],
                "description": "動画から解析できたフレーム数です。数が多いほど、動作全体を安定して解析できています。",
                "guide": "150〜300 フレーム",
            },
            {
                "name": "最大肩回転角（°）",
                "value": raw["max_shoulder_rotation"],
                "description": "スイング中に肩がどれだけ回転したかを示す角度です。上半身の回旋量の指標になります。",
                "guide": "85〜105°",
            },
            {
                "name": "最小腰回転角（°）",
                "value": raw["min_hip_rotation"],
                "description": "スイング中に腰が最も回転した瞬間の角度です。下半身の回旋量を表します。",
                "guide": "36〜50°（目安）",
            },
            {
                "name": "最大手首コック角（°）",
                "value": raw["max_wrist_cock"],
                "description": "スイング中に手首が最も折れた角度です。クラブの“溜め”の指標になります。",
                "guide": "70〜90°（本計測仕様の目安）",
            },
            {
                "name": "最大頭部ブレ（Sway）",
                "value": raw["max_head_drift"],
                "description": "スイング中に頭の位置が左右にどれだけ動いたかを示します。スイング軸の安定性を表します。",
                "guide": "0.06〜0.15",
            },
            {
                "name": "最大膝ブレ（Sway）",
                "value": raw["max_knee_sway"],
                "description": "スイング中に膝が左右にどれだけ動いたかを示します。下半身の安定性の指標です。",
                "guide": "0.10〜0.20",
            },
        ],
    }


# ==================================================
# 02 肩：3×3×3 判定＋文章（confidence + 差分 + ばらつき）
# ==================================================
def judge_shoulder(raw: Dict[str, Any]) -> Dict[str, Any]:
    shoulder = raw["max_shoulder_rotation"]
    hip = abs(raw["min_hip_rotation"])
    frame = raw["frame_count"]

    if shoulder < 85:
        main = "low"
    elif shoulder > 105:
        main = "high"
    else:
        main = "mid"

    x_factor = shoulder - hip
    if x_factor < 35:
        xf = "low"
    elif x_factor > 55:
        xf = "high"
    else:
        xf = "mid"

    if frame < 80:
        conf = "low"
    elif frame < 180:
        conf = "mid"
    else:
        conf = "high"

    tags: List[str] = []
    if main == "low":
        tags.append("肩回転不足")
    if main == "high":
        tags.append("肩回転過多")
    if xf == "low":
        tags.append("捻転差不足")
    if xf == "high":
        tags.append("捻転差過多")

    return {
        "main": main,
        "x_factor": xf,
        "confidence": conf,
        "x_factor_value": round(x_factor, 1),
        "tags": tags,
    }


def shoulder_good_bad(judge: Dict[str, Any], raw: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    sh = raw["max_shoulder_rotation"]
    xf = judge["x_factor_value"]
    good: List[str] = []
    bad: List[str] = []

    if judge["main"] == "mid":
        good.append(f"肩回転角{sh}°は目安レンジ内で、上半身の回旋が安定しています。")
    if judge["x_factor"] == "mid":
        good.append(f"捻転差{xf}°が確保されており、切り返しでエネルギーを溜められています。")

    if judge["main"] == "low":
        bad.append(f"最大肩回転角が{sh}°と小さく、上半身でパワーを作れていません。")
    if judge["main"] == "high":
        bad.append(f"最大肩回転角が{sh}°と大きく、回転量がブレやすい状態です。")
    if judge["x_factor"] == "low":
        bad.append(f"捻転差が{xf}°と不足しており、肩と腰が同時に動いています。")
    if judge["x_factor"] == "high":
        bad.append(f"捻転差が{xf}°と大きく、腰が止まりすぎて上体が先行しています。")

    if not good:
        good = ["上半身の動きに大きな破綻はなく、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の回旋は安定しており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_shoulder_pro(judge: Dict[str, Any], raw: Dict[str, Any]) -> str:
    conf = judge["confidence"]
    sh = float(raw["max_shoulder_rotation"])
    sh_std = float(raw.get("std_shoulder_rotation", 0.0))
    diff = _diff_phrase(sh, REF_AVG["shoulder"], "°")
    stab = _stability_phrase(sh_std, "deg")

    # 3行固定（矛盾を作らない）
    l1 = f"肩回転は{_fmt_deg(sh)}で、{diff}"
    l2 = f"{stab}（ばらつきσ={sh_std:.1f}°）"
    l3 = "量を増やす/減らすより、同じ幅とテンポを揃える意識を優先してください。"
    return _pro_3lines(conf, l1, l2, l3)


def build_paid_02_shoulder(raw: Dict[str, Any]) -> Dict[str, Any]:
    judge = judge_shoulder(raw)
    good, bad = shoulder_good_bad(judge, raw)
    pro = generate_shoulder_pro(judge, raw)
    return {
        "title": "02. Shoulder Rotation（肩回転）",
        "value": raw["max_shoulder_rotation"],
        "judge": judge,
        "tags": judge["tags"],
        "good": good,
        "bad": bad,
        "pro_comment": pro,
    }


# ==================================================
# 03 腰：3×3×3 判定＋文章（confidence + 差分 + ばらつき）
# ==================================================
def judge_hip(raw: Dict[str, Any]) -> Dict[str, Any]:
    hip = abs(raw["min_hip_rotation"])
    shoulder = raw["max_shoulder_rotation"]
    frame = raw["frame_count"]

    if hip < 35:
        main = "low"
    elif hip > 50:
        main = "high"
    else:
        main = "mid"

    x_factor = shoulder - hip
    if x_factor < 35:
        xf = "low"
    elif x_factor > 55:
        xf = "high"
    else:
        xf = "mid"

    if frame < 80:
        conf = "low"
    elif frame < 180:
        conf = "mid"
    else:
        conf = "high"

    tags: List[str] = []
    if main == "low":
        tags.append("腰回転不足")
    if main == "high":
        tags.append("腰回転過多")
    if xf == "low":
        tags.append("捻転差不足")
    if xf == "high":
        tags.append("捻転差過多")

    return {
        "main": main,
        "x_factor": xf,
        "confidence": conf,
        "x_factor_value": round(x_factor, 1),
        "tags": tags,
    }


def hip_good_bad(judge: Dict[str, Any], raw: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    hip = abs(raw["min_hip_rotation"])
    xf = judge["x_factor_value"]
    good: List[str] = []
    bad: List[str] = []

    if judge["main"] == "mid":
        good.append(f"腰回転量{hip}°は目安レンジ内で、下半身の土台が安定しています。")
    if judge["x_factor"] == "mid":
        good.append(f"捻転差{xf}°が確保されており、切り返しで溜めが作れています。")

    if judge["main"] == "low":
        bad.append(f"腰回転量が{hip}°と小さく、下半身の推進力を活かし切れていません。")
    if judge["main"] == "high":
        bad.append(f"腰回転量が{hip}°と大きく、上体が先に開きやすい状態です。")
    if judge["x_factor"] == "low":
        bad.append(f"捻転差が{xf}°と不足しており、肩と腰が同時に動いています。")
    if judge["x_factor"] == "high":
        bad.append(f"捻転差が{xf}°と大きく、腰が止まり上体が先行しています。")

    if not good:
        good = ["下半身の動きに大きな破綻はなく、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の下半身は安定しており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_hip_pro(judge: Dict[str, Any], raw: Dict[str, Any]) -> str:
    conf = judge["confidence"]
    hip_min = float(abs(raw["min_hip_rotation"]))
    hip_std = float(raw.get("std_hip_rotation", 0.0))
    diff = _diff_phrase(hip_min, REF_AVG["hip"], "°")
    stab = _stability_phrase(hip_std, "deg")

    l1 = f"腰回転は{_fmt_deg(hip_min)}で、{diff}"
    l2 = f"{stab}（ばらつきσ={hip_std:.1f}°）"
    l3 = "切り返しで腰が先行し過ぎないよう、下半身→上半身の順番を固定してください。"
    return _pro_3lines(conf, l1, l2, l3)


def build_paid_03_hip(raw: Dict[str, Any]) -> Dict[str, Any]:
    judge = judge_hip(raw)
    good, bad = hip_good_bad(judge, raw)
    pro = generate_hip_pro(judge, raw)
    return {
        "title": "03. Hip Rotation（腰回転）",
        "value": raw["min_hip_rotation"],
        "judge": judge,
        "tags": judge["tags"],
        "good": good,
        "bad": bad,
        "pro_comment": pro,
    }


# ==================================================
# 04〜06（同一思想：主指標＋関連指標＋信頼度）
#  + 文章は「confidence + 差分 + ばらつき」で3行固定
# ==================================================
def judge_wrist(raw: Dict[str, Any]) -> Dict[str, Any]:
    wrist = raw["max_wrist_cock"]
    shoulder = raw["max_shoulder_rotation"]
    hip = abs(raw["min_hip_rotation"])
    frame = raw["frame_count"]

    if wrist < 70:
        main = "low"
    elif wrist > 90:
        main = "high"
    else:
        main = "mid"

    x_factor = shoulder - hip
    if x_factor < 35:
        rel = "low"
    elif x_factor > 55:
        rel = "high"
    else:
        rel = "mid"

    if frame < 80:
        conf = "low"
    elif frame < 180:
        conf = "mid"
    else:
        conf = "high"

    tags: List[str] = []
    if main == "low":
        tags.append("コック不足")
    if main == "high":
        tags.append("コック過多")
    if rel == "low":
        tags.append("体幹主導不足")

    return {
        "main": main,
        "related": rel,
        "confidence": conf,
        "x_factor_value": round(x_factor, 1),
        "tags": tags,
    }


def wrist_good_bad(judge: Dict[str, Any], raw: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    w = raw["max_wrist_cock"]
    xf = judge["x_factor_value"]
    good: List[str] = []
    bad: List[str] = []

    if judge["main"] == "mid":
        good.append(f"手首コック角{w}°は目安レンジ内で、再現性の高い形です。")
    if judge["related"] == "mid":
        good.append(f"捻転差{xf}°があり、体の回転と連動しています。")

    if judge["main"] == "low":
        bad.append(f"コック角{w}°が小さく、溜めを作れていません。")
    if judge["main"] == "high":
        bad.append(f"コック角{w}°が大きく、手首主導になっています。")
    if judge["related"] == "low":
        bad.append(f"捻転差{xf}°が小さく、体幹より手先が先行しています。")

    if not good:
        good = ["手首の動きに大きな破綻はなく、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の手首操作は安定しており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_wrist_pro(judge: Dict[str, Any], raw: Dict[str, Any]) -> str:
    conf = judge["confidence"]
    w = float(raw["max_wrist_cock"])
    w_std = float(raw.get("std_wrist_cock", 0.0))
    diff = _diff_phrase(w, REF_AVG["wrist"], "°")
    stab = _stability_phrase(w_std, "deg")

    # “矛盾しない”書き方：手首主導＝致命傷ではないが、再現性低下の原因になる、に統一
    l1 = f"手首コックは{_fmt_deg(w)}で、{diff}"
    l2 = f"{stab}（ばらつきσ={w_std:.1f}°）"
    l3 = "手首は致命傷ではありませんが、主導になると再現性が落ちるため“体幹主導”に戻すのが最短です。"
    return _pro_3lines(conf, l1, l2, l3)


def build_paid_04_wrist(raw: Dict[str, Any]) -> Dict[str, Any]:
    judge = judge_wrist(raw)
    good, bad = wrist_good_bad(judge, raw)
    pro = generate_wrist_pro(judge, raw)
    return {
        "title": "04. Wrist Cock（手首コック）",
        "value": raw["max_wrist_cock"],
        "judge": judge,
        "tags": judge["tags"],
        "good": good,
        "bad": bad,
        "pro_comment": pro,
    }


def judge_head(raw: Dict[str, Any]) -> Dict[str, Any]:
    h = raw["max_head_drift"]
    knee = raw["max_knee_sway"]
    frame = raw["frame_count"]

    if h < 0.06:
        main = "low"   # 良
    elif h > 0.15:
        main = "high"  # 悪
    else:
        main = "mid"

    if knee < 0.10:
        rel = "low"
    elif knee > 0.20:
        rel = "high"
    else:
        rel = "mid"

    if frame < 80:
        conf = "low"
    elif frame < 180:
        conf = "mid"
    else:
        conf = "high"

    tags: List[str] = []
    if main == "high":
        tags.append("頭部ブレ大")
    if rel == "high":
        tags.append("下半身不安定")

    return {"main": main, "related": rel, "confidence": conf, "tags": tags}


def head_good_bad(judge: Dict[str, Any], raw: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    h = raw["max_head_drift"]
    good: List[str] = []
    bad: List[str] = []

    if judge["main"] == "low":
        good.append(f"頭部ブレ{h}は小さく、スイング軸が安定しています。")
    if judge["main"] == "mid":
        good.append(f"頭部ブレ{h}は平均的で、大きく崩れる動きは見られません。")
    if judge["main"] == "high":
        bad.append(f"頭部ブレ{h}が大きく、ミート率が落ちています。")

    if judge["related"] == "high":
        bad.append("膝の安定性が低く、頭部ブレを助長しています。")

    if not good:
        good = ["頭部の位置は大きく崩れておらず、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["頭部の安定は保てており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_head_pro(judge: Dict[str, Any], raw: Dict[str, Any]) -> str:
    conf = judge["confidence"]
    h = float(raw["max_head_drift"])
    h_std = float(raw.get("std_head_drift", 0.0))
    diff = _diff_phrase(h, REF_AVG["head"], "")
    stab = _stability_phrase(h_std, "sway")

    l1 = f"頭部ブレは{_fmt_sway(h)}で、{diff}"
    l2 = f"{stab}（ばらつきσ={h_std:.3f}）"
    l3 = "頭の左右移動を止めるだけでミート率が上がりやすいので、まず“頭の位置固定”を最優先にしてください。"
    return _pro_3lines(conf, l1, l2, l3)


def build_paid_05_head(raw: Dict[str, Any]) -> Dict[str, Any]:
    judge = judge_head(raw)
    good, bad = head_good_bad(judge, raw)
    pro = generate_head_pro(judge, raw)
    return {
        "title": "05. Head Stability（頭部）",
        "value": raw["max_head_drift"],
        "judge": judge,
        "tags": judge["tags"],
        "good": good,
        "bad": bad,
        "pro_comment": pro,
    }


def judge_knee(raw: Dict[str, Any]) -> Dict[str, Any]:
    k = raw["max_knee_sway"]
    head = raw["max_head_drift"]
    frame = raw["frame_count"]

    if k < 0.10:
        main = "low"
    elif k > 0.20:
        main = "high"
    else:
        main = "mid"

    if head < 0.06:
        rel = "low"
    elif head > 0.15:
        rel = "high"
    else:
        rel = "mid"

    if frame < 80:
        conf = "low"
    elif frame < 180:
        conf = "mid"
    else:
        conf = "high"

    tags: List[str] = []
    if main == "high":
        tags.append("膝ブレ大")
    if rel == "high":
        tags.append("上半身不安定")

    return {"main": main, "related": rel, "confidence": conf, "tags": tags}


def knee_good_bad(judge: Dict[str, Any], raw: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    k = raw["max_knee_sway"]
    good: List[str] = []
    bad: List[str] = []

    if judge["main"] == "low":
        good.append(f"膝ブレ{k}は小さく、下半身が安定しています。")
    if judge["main"] == "mid":
        good.append(f"膝ブレ{k}は平均的で、土台は大きく崩れていません。")
    if judge["main"] == "high":
        bad.append(f"膝ブレ{k}が大きく、体重移動が横流れになっています。")

    if judge["related"] == "high":
        bad.append("上半身の動きが膝ブレを助長しています。")

    if not good:
        good = ["下半身の土台は大きく崩れておらず、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["下半身の安定は保てており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_knee_pro(judge: Dict[str, Any], raw: Dict[str, Any]) -> str:
    conf = judge["confidence"]
    k = float(raw["max_knee_sway"])
    k_std = float(raw.get("std_knee_sway", 0.0))
    diff = _diff_phrase(k, REF_AVG["knee"], "")
    stab = _stability_phrase(k_std, "sway")

    l1 = f"膝ブレは{_fmt_sway(k)}で、{diff}"
    l2 = f"{stab}（ばらつきσ={k_std:.3f}）"
    l3 = "下半身の横流れを止めると全体が一気に安定するので、膝幅固定→縦の体重移動の順で整えてください。"
    return _pro_3lines(conf, l1, l2, l3)


def build_paid_06_knee(raw: Dict[str, Any]) -> Dict[str, Any]:
    judge = judge_knee(raw)
    good, bad = knee_good_bad(judge, raw)
    pro = generate_knee_pro(judge, raw)
    return {
        "title": "06. Knee Stability（膝）",
        "value": raw["max_knee_sway"],
        "judge": judge,
        "tags": judge["tags"],
        "good": good,
        "bad": bad,
        "pro_comment": pro,
    }


# ==================================================
# 07 有料：tags要約（02〜06→優先順位→08/09へ接続）
# ==================================================
def collect_tag_counter(analysis: Dict[str, Any]) -> Counter:
    tags: List[str] = []
    for k in ["02", "03", "04", "05", "06"]:
        sec = analysis.get(k) or {}
        tags.extend(sec.get("tags", []) or [])
    return Counter(tags)


def judge_swing_type(tag_counter: Counter) -> str:
    if tag_counter["捻転差不足"] >= 2:
        return "体幹パワー不足型"
    if tag_counter["膝ブレ大"] + tag_counter["頭部ブレ大"] >= 2:
        return "安定性不足型"
    if tag_counter["肩回転過多"] + tag_counter["コック過多"] >= 2:
        return "操作過多型"
    return "バランス型"


def extract_priorities(tag_counter: Counter, max_items: int = 2) -> List[str]:
    order = [
        "捻転差不足",
        "膝ブレ大",
        "頭部ブレ大",
        "コック過多",
        "腰回転不足",
        "肩回転過多",
        "肩回転不足",
        "コック不足",
        "捻転差過多",
        "腰回転過多",
    ]
    result: List[str] = []
    for t in order:
        if tag_counter.get(t, 0) > 0:
            result.append(t)
        if len(result) >= max_items:
            break
    return result


def build_paid_07_from_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    c = collect_tag_counter(analysis)
    swing_type = judge_swing_type(c)
    priorities = extract_priorities(c, 2)

    lines: List[str] = []
    lines.append(f"今回のスイングは「{swing_type}」です。")

    if priorities:
        if len(priorities) == 1:
            lines.append(f"数値上、最も優先すべき改善点は「{priorities[0]}」です。")
        else:
            lines.append("数値上、最も優先すべき改善点は「" + "／".join(priorities) + "」の2点です。")
    else:
        lines.append("数値上、大きな改善テーマは見られません。")

    lines.append("そのため08では、この優先テーマに直結する練習ドリルを選択しています。")
    lines.append("また09では、動きを安定させやすいシャフト特性を指針として提示しています。")

    return {
        "title": "07. 総合評価（プロ要約）",
        "text": lines,
        "meta": {
            "swing_type": swing_type,
            "priorities": priorities,
            "tag_summary": dict(c),
        },
    }


# ==================================================
# 08 ドリル：全定義＋tagsスコアリングで最大3つ
# ==================================================
DRILL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "x_factor_turn",
        "name": "捻転差づくりドリル（肩先行ターン）",
        "category": "体幹",
        "tags": ["捻転差不足"],
        "purpose": "肩と腰の回転差を作り、切り返しでエネルギーを溜める",
        "how": "①トップで肩を深く入れる\n②腰は一拍遅らせる\n③素振りで10回×2セット",
    },
    {
        "id": "shoulder_control",
        "name": "肩回転コントロールドリル",
        "category": "上半身",
        "tags": ["肩回転過多"],
        "purpose": "回し過ぎを抑え、再現性を高める",
        "how": "①ハーフスイング\n②肩の回し幅を一定に\n③10球×2セット",
    },
    {
        "id": "hip_drive",
        "name": "腰主導ターンドリル",
        "category": "下半身",
        "tags": ["腰回転不足"],
        "purpose": "下半身から動く感覚を身につける",
        "how": "①腰から切り返す\n②上体は我慢\n③素振り15回",
    },
    {
        "id": "late_hit",
        "name": "レイトヒットドリル",
        "category": "手首",
        "tags": ["コック不足"],
        "purpose": "タメを作り、インパクト効率を上げる",
        "how": "①トップで静止\n②体の回転で振る\n③連続素振り10回",
    },
    {
        "id": "release_control",
        "name": "リリース抑制ドリル（LtoL）",
        "category": "手首",
        "tags": ["コック過多"],
        "purpose": "手首主導を抑え、体幹主導に戻す",
        "how": "①腰〜腰の振り幅\n②フェース管理重視\n③20回",
    },
    {
        "id": "head_still",
        "name": "頭固定ドリル（壁チェック）",
        "category": "安定性",
        "tags": ["頭部ブレ大"],
        "purpose": "スイング軸を安定させる",
        "how": "①壁の前で構える\n②頭の位置を保つ\n③素振り10回",
    },
    {
        "id": "knee_stable",
        "name": "膝ブレ抑制ドリル",
        "category": "下半身",
        "tags": ["膝ブレ大"],
        "purpose": "下半身の横流れを抑える",
        "how": "①膝幅を固定\n②体重移動を縦意識\n③10回×2",
    },
    {
        "id": "sync_turn",
        "name": "全身同調ターンドリル（クロスアーム）",
        "category": "体幹",
        "tags": ["体幹主導不足", "捻転差不足"],
        "purpose": "体全体で回る感覚を作る",
        "how": "①腕を胸の前でクロス\n②胸と腰を同時に回す\n③左右10回",
    },
    {
        "id": "tempo",
        "name": "テンポ安定ドリル（メトロノーム）",
        "category": "リズム",
        "tags": ["再現性不足"],
        "purpose": "タイミングを一定にする",
        "how": "①一定テンポで素振り\n②10回\n③その後ボール10球",
    },
    {
        "id": "balance",
        "name": "バランスチェックドリル",
        "category": "安定性",
        "tags": ["下半身不安定", "上半身不安定"],
        "purpose": "軸と体重配分を整える",
        "how": "①片足立ち\n②ゆっくり素振り\n③左右5回",
    },
]


def collect_all_tags(analysis: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    for k in ["02", "03", "04", "05", "06"]:
        sec = analysis.get(k)
        if sec and "tags" in sec:
            tags.extend(sec["tags"] or [])
    return tags


def select_drills_by_tags(tags: List[str], max_drills: int = 3) -> List[Dict[str, str]]:
    tagset = set(tags)
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for d in DRILL_DEFINITIONS:
        score = len(set(d["tags"]) & tagset)
        if score > 0:
            scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[Dict[str, Any]] = []
    used_categories: set = set()

    for score, d in scored:
        if d["category"] in used_categories:
            continue
        selected.append(d)
        used_categories.add(d["category"])
        if len(selected) >= max_drills:
            break

    if not selected:
        for d in DRILL_DEFINITIONS:
            if d["id"] == "tempo":
                selected = [d]
                break

    return [{"name": d["name"], "purpose": d["purpose"], "how": d["how"]} for d in selected]


def build_paid_08(analysis: Dict[str, Any]) -> Dict[str, Any]:
    tags = collect_all_tags(analysis)
    drills = select_drills_by_tags(tags, 3)
    return {"title": "08. Training Drills（練習ドリル）", "drills": drills}


# ==================================================
# 09 フィッティング：指数＋任意入力連動
# ==================================================
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _norm_range(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    return _clamp01((v - lo) / (hi - lo))


def _norm_inverse(v: float, lo: float, hi: float) -> float:
    return 1.0 - _norm_range(v, lo, hi)


def calc_power_idx(raw: Dict[str, Any]) -> int:
    sh = raw["max_shoulder_rotation"]          # 85..105
    hip = abs(raw["min_hip_rotation"])         # 36..50
    wrist = raw["max_wrist_cock"]              # 70..90（本仕様）
    xf = sh - hip                              # 36..55

    a = _norm_range(sh, 85, 105)
    b = _norm_range(hip, 36, 50)
    c = _norm_range(wrist, 70, 90)
    d = _norm_range(xf, 36, 55)
    return int(round((a + b + c + d) / 4.0 * 100))


def calc_stability_idx(raw: Dict[str, Any]) -> int:
    head = raw["max_head_drift"]               # 0.06..0.15（小さいほど良）
    knee = raw["max_knee_sway"]                # 0.10..0.20（小さいほど良）

    a = _norm_inverse(head, 0.06, 0.15)
    b = _norm_inverse(knee, 0.10, 0.20)
    return int(round((a + b) / 2.0 * 100))


def _to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _norm_miss(miss: Any) -> str:
    s = (str(miss).strip() if miss is not None else "")
    if any(k in s for k in ["スライス", "プッシュ", "右"]):
        return "right"
    if any(k in s for k in ["フック", "引っかけ", "左"]):
        return "left"
    return "none"


def _norm_gender(g: Any) -> str:
    s = (str(g).strip().lower() if g is not None else "")
    if s in ["male", "man", "m", "男性"]:
        return "male"
    if s in ["female", "woman", "f", "女性"]:
        return "female"
    return "none"


def infer_hs_band(power_idx: int) -> str:
    if power_idx <= 33:
        return "low"
    if power_idx <= 66:
        return "mid"
    return "high"


def build_paid_09(raw: Dict[str, Any], user_inputs: Dict[str, Any]) -> Dict[str, Any]:
    power_idx = calc_power_idx(raw)
    stability_idx = calc_stability_idx(raw)

    hs = _to_float_or_none(user_inputs.get("head_speed"))
    miss = _norm_miss(user_inputs.get("miss_tendency"))
    gender = _norm_gender(user_inputs.get("gender"))

    rows: List[Dict[str, str]] = []

    # 重量
    if hs is not None:
        if hs < 35:
            weight = "40〜50g"
            reason = f"ヘッドスピード{hs:.1f}m/sでは、軽めが振り切りに直結します。"
        elif hs < 40:
            weight = "50g前後"
            reason = f"ヘッドスピード{hs:.1f}m/sでは、50g前後が基準です。"
        elif hs < 45:
            weight = "50〜60g"
            reason = f"ヘッドスピード{hs:.1f}m/sでは、50〜60gが最も安定します。"
        else:
            weight = "60〜70g"
            reason = f"ヘッドスピード{hs:.1f}m/sでは、60g以上が当たり負けを抑えます。"
    else:
        band = infer_hs_band(power_idx)
        if band == "low":
            weight = "40〜50g"
            reason = f"入力が無いため指数で判定します。パワー指数{power_idx}では軽めが最適です。"
        elif band == "mid":
            weight = "50〜60g"
            reason = f"入力が無いため指数で判定します。パワー指数{power_idx}では標準帯が最適です。"
        else:
            weight = "60〜70g"
            reason = f"入力が無いため指数で判定します。パワー指数{power_idx}では重めが安定します。"

    if stability_idx <= 40 and "40〜50g" in weight:
        weight = "50g前後"
        reason += f" 安定性指数{stability_idx}のため、軽すぎはブレを増やすので避けます。"

    rows.append({"item": "重量", "guide": weight, "reason": reason})

    # フレックス
    if hs is not None:
        if hs < 33:
            flex = "L〜A"
        elif hs < 38:
            flex = "A〜R"
        elif hs < 42:
            flex = "R〜SR"
        elif hs < 46:
            flex = "SR〜S"
        elif hs < 50:
            flex = "S〜X"
        else:
            flex = "X"
        reason = f"ヘッドスピード{hs:.1f}m/sに対して、しなり戻りが遅れない範囲で設定します。"
    else:
        band = infer_hs_band(power_idx)
        if band == "low":
            flex = "A〜R"
        elif band == "mid":
            flex = "R〜SR"
        else:
            flex = "SR〜S"
        reason = f"入力が無いため指数で判定します。パワー指数{power_idx}に対して適正帯です。"

    if gender == "female" and flex in ["SR〜S", "S〜X", "S", "X"]:
        flex = "R〜SR"
        reason += " 性別入力に基づき、振りやすさと再現性を優先して1段柔らかめに寄せます。"

    rows.append({"item": "フレックス", "guide": flex, "reason": reason})

    # キックポイント
    if miss == "right":
        kp = "先〜中"
        reason = "右へのミス傾向は、つかまり側（先〜中）が結果を整えます。"
    elif miss == "left":
        kp = "中〜元"
        reason = "左へのミス傾向は、つかまり過ぎを抑える（中〜元）が結果を整えます。"
    else:
        wrist_high = raw["max_wrist_cock"] > 90
        head_bad = raw["max_head_drift"] > 0.15
        if wrist_high or head_bad or stability_idx <= 40:
            kp = "中〜元"
            reason = f"入力が無いため数値で判定します。安定性指数{stability_idx}のため元寄りで挙動を抑えます。"
        else:
            kp = "中"
            reason = "入力が無いため一般的指針を採用します。中調子が基準です。"

    rows.append({"item": "キックポイント", "guide": kp, "reason": reason})

    # トルク
    if stability_idx <= 40:
        tq = "3.0〜4.0"
        reason = f"安定性指数{stability_idx}のため、低トルクでフェース挙動を抑えます。"
    elif stability_idx <= 70:
        tq = "3.5〜5.0"
        reason = f"安定性指数{stability_idx}のため、標準帯でバランスを取ります。"
    else:
        tq = "4.0〜6.0"
        reason = f"安定性指数{stability_idx}のため、高めのトルクでも再現性が崩れません。"

    if miss == "left" and tq == "4.0〜6.0":
        tq = "3.0〜4.5"
        reason += " 左ミス補正としてトルクを下げ、つかまり過ぎを抑えます。"
    if miss == "right" and tq == "3.0〜4.0":
        tq = "4.0〜5.5"
        reason += " 右ミス補正としてトルクを上げ、つかまりを補います。"

    rows.append({"item": "トルク", "guide": tq, "reason": reason})

    return {
        "title": "09. Shaft Fitting Guide（推奨）",
        "table": rows,
        "note": "本結果は指標のため、購入時は試打を推奨します。",
        "meta": {
            "power_idx": power_idx,
            "stability_idx": stability_idx,
            "head_speed": hs,
            "miss_tendency": user_inputs.get("miss_tendency"),
            "gender": user_inputs.get("gender"),
        },
    }


# ==================================================
# 10 まとめ（有料）
# ==================================================
def build_paid_10(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "10. Summary（まとめ）",
        "text": [
            "今回の解析では、回転量を活かせる土台が確認できました。",
            "次のステップは「優先テーマを2点に絞って改善すること」です。",
            "08のドリルと09の指針を使い、同じ幅・同じテンポを作っていきましょう。",
            "",
            "あなたのゴルフライフが、より充実したものになることを願っています。",
        ],
    }


# ==================================================
# 無料 07
# ==================================================
def build_free_07(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "07. 総合評価",
        "text": [
            "本レポートでは、スイング全体の傾向を骨格データに基づいて評価しています。",
            "有料版では、部位別評価・練習ドリル・フィッティング指針まで含めて提示します。",
        ],
    }


# ==================================================
# Analysis builder（完成版）
# ==================================================
def build_analysis(raw: Dict[str, Any], premium: bool, report_id: str, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {"01": build_section_01(raw)}

    if not premium:
        analysis["07"] = build_free_07(raw)
        return analysis

    analysis["02"] = build_paid_02_shoulder(raw)
    analysis["03"] = build_paid_03_hip(raw)
    analysis["04"] = build_paid_04_wrist(raw)
    analysis["05"] = build_paid_05_head(raw)
    analysis["06"] = build_paid_06_knee(raw)

    analysis["07"] = build_paid_07_from_analysis(analysis)
    analysis["08"] = build_paid_08(analysis)
    analysis["09"] = build_paid_09(raw, user_inputs or {})
    analysis["10"] = build_paid_10(raw)
    return analysis


# ==================================================
# Routes
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

    premium = is_premium_user(user_id)

    firestore_safe_set(
        report_id,
        {
            "user_id": user_id,
            "status": "PROCESSING",
            "is_premium": premium,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_inputs": {},
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

        content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        raw_data = analyze_swing_with_mediapipe(video_path)

        doc = doc_ref.get()
        docd = doc.to_dict() or {}
        premium = bool(docd.get("is_premium", False))
        user_inputs = docd.get("user_inputs", {}) or {}

        analysis = build_analysis(raw_data, premium, report_id, user_inputs)

        doc_ref.update({
            "status": "COMPLETED",
            "raw_data": raw_data,
            "analysis": analysis,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

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
def report_page(report_id):
    return render_template("report.html", report_id=report_id)


@app.route("/api/report_data/<report_id>")
def api_report_data(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    d = doc.to_dict() or {}
    return jsonify({
        "status": d.get("status"),
        "analysis": d.get("analysis", {}),
        "raw_data": d.get("raw_data", {}),
        "is_premium": d.get("is_premium", False),
        "error": d.get("error"),
        "created_at": d.get("created_at"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
