import os
import json
import math
import shutil
import traceback
import tempfile
import random
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
                "audience": SERVICE_HOST_URL,  # Cloud Run URL
            },
        }
    }

    resp = tasks_client.create_task(parent=queue_path, task=task)
    return resp.name


# ==================================================
# Stats helpers (max + mean + std)
# ==================================================
def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _std(xs: List[float]) -> float:
    # population std
    if not xs:
        return 0.0
    m = _mean(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return float(math.sqrt(v))


def _round(x: float, n: int = 2) -> float:
    return round(float(x), n)


def _conf_tier(valid_frames: int, total_frames: int) -> Tuple[str, float]:
    """
    confidence = 0..1
    目安:
      - フレーム数
      - ランドマーク取得率
    """
    if total_frames <= 0:
        return ("low", 0.0)
    ratio = valid_frames / total_frames

    # 基本スコア（フレーム数）
    if valid_frames < 60:
        base = 0.35
    elif valid_frames < 140:
        base = 0.60
    else:
        base = 0.85

    conf = max(0.0, min(1.0, base * 0.7 + ratio * 0.3))

    if conf < 0.55:
        tier = "low"
    elif conf < 0.80:
        tier = "mid"
    else:
        tier = "high"
    return (tier, conf)


# ==================================================
# MediaPipe analysis (③: per-frame series -> max/mean/std/confidence)
# ==================================================
def analyze_swing_with_mediapipe(video_path: str) -> Dict[str, Any]:
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCVがビデオを読み込めませんでした。")

    total_frames = 0
    valid_frames = 0

    shoulder_series: List[float] = []
    hip_series: List[float] = []
    wrist_series: List[float] = []
    head_series: List[float] = []
    knee_series: List[float] = []

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
            total_frames += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            valid_frames += 1
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
            w = angle(xy(LE), xy(LW), xy(LI))
            head = abs(xy(NO)[0] - 0.5)
            knee = abs(xy(LK)[0] - 0.5)

            shoulder_series.append(float(sh))
            hip_series.append(float(hip))
            wrist_series.append(float(w))
            head_series.append(float(head))
            knee_series.append(float(knee))

    cap.release()

    if valid_frames < 10:
        raise RuntimeError("解析に必要なフレーム数が不足しています。")

    conf_tier, conf_value = _conf_tier(valid_frames, total_frames)

    # NOTE: hipは「最小」ではなく、シリーズ統計を取るため平均等を持つ。
    # 互換のため "min_hip_rotation" は残しつつ、mean/stdも出す。
    hip_min = min(hip_series) if hip_series else 0.0

    return {
        # frame meta
        "frame_count": int(valid_frames),
        "total_frames": int(total_frames),
        "valid_ratio": _round(valid_frames / total_frames, 4) if total_frames else 0.0,
        "confidence_tier": conf_tier,
        "confidence_value": _round(conf_value, 3),

        # shoulder
        "max_shoulder_rotation": _round(max(shoulder_series), 2),
        "mean_shoulder_rotation": _round(_mean(shoulder_series), 2),
        "std_shoulder_rotation": _round(_std(shoulder_series), 2),

        # hip (compat key + stats)
        "min_hip_rotation": _round(hip_min, 2),
        "mean_hip_rotation": _round(_mean(hip_series), 2),
        "std_hip_rotation": _round(_std(hip_series), 2),

        # wrist
        "max_wrist_cock": _round(max(wrist_series), 2),
        "mean_wrist_cock": _round(_mean(wrist_series), 2),
        "std_wrist_cock": _round(_std(wrist_series), 2),

        # head
        "max_head_drift": _round(max(head_series), 4),
        "mean_head_drift": _round(_mean(head_series), 4),
        "std_head_drift": _round(_std(head_series), 4),

        # knee
        "max_knee_sway": _round(max(knee_series), 4),
        "mean_knee_sway": _round(_mean(knee_series), 4),
        "std_knee_sway": _round(_std(knee_series), 4),
    }


# ==================================================
# Section 01 (表示は既存フロントを壊さないため value を文字列で)
# ==================================================
def _fmt_stat(maxv: float, meanv: float, stdv: float, unit: str = "") -> str:
    u = unit
    # 例: "max 102.3° / mean 95.1° / σ 4.2°"
    return f"max {maxv}{u} / mean {meanv}{u} / σ {stdv}{u}"


def build_section_01(raw: Dict[str, Any]) -> Dict[str, Any]:
    conf_line = f'{raw.get("confidence_tier","")} ({raw.get("confidence_value",0)})'
    return {
        "title": "01. 骨格計測データ（AIが測定）",
        "items": [
            {
                "name": "解析フレーム数",
                "value": f'{raw["frame_count"]} / total {raw.get("total_frames",0)} / ratio {raw.get("valid_ratio",0)}',
                "description": "解析できたフレーム数と取得率です。数と取得率が高いほど、評価の信頼度が上がります。",
                "guide": "150〜300 フレーム（目安）",
            },
            {
                "name": "解析信頼度（confidence）",
                "value": conf_line,
                "description": "フレーム数とランドマーク取得率から算出した信頼度です。低い場合は断定表現を抑えます。",
                "guide": "mid〜high 推奨",
            },
            {
                "name": "肩回転（°）",
                "value": _fmt_stat(raw["max_shoulder_rotation"], raw["mean_shoulder_rotation"], raw["std_shoulder_rotation"], "°"),
                "description": "上半身の回旋量（max）と、平均的な回旋傾向（mean）、再現性（σ）を示します。",
                "guide": "mean 85〜105°（目安）",
            },
            {
                "name": "腰回転（°）",
                "value": _fmt_stat(raw.get("min_hip_rotation", 0.0), raw["mean_hip_rotation"], raw["std_hip_rotation"], "°"),
                "description": "下半身の回旋量です。minは互換表示、判定はmean/σを重視します。",
                "guide": "mean 36〜50°（目安）",
            },
            {
                "name": "手首コック（°）",
                "value": _fmt_stat(raw["max_wrist_cock"], raw["mean_wrist_cock"], raw["std_wrist_cock"], "°"),
                "description": "溜めの量（mean）と、手首操作の暴れ（σ）を見ます。",
                "guide": "mean 70〜90°（本計測仕様の目安）",
            },
            {
                "name": "頭部ブレ（Sway）",
                "value": _fmt_stat(raw["max_head_drift"], raw["mean_head_drift"], raw["std_head_drift"]),
                "description": "軸の安定性（mean）と、動きのブレ幅（σ）を示します。",
                "guide": "mean 0.06〜0.15",
            },
            {
                "name": "膝ブレ（Sway）",
                "value": _fmt_stat(raw["max_knee_sway"], raw["mean_knee_sway"], raw["std_knee_sway"]),
                "description": "下半身の横流れ傾向（mean）と、再現性（σ）を示します。",
                "guide": "mean 0.10〜0.20",
            },
        ],
    }


# ==================================================
# ②〜⑥：共通 “プロ版” 評価ロジック（mean + std + confidence）
# ==================================================
def _tier3_by_range(v: float, lo: float, hi: float) -> str:
    if v < lo:
        return "low"
    if v > hi:
        return "high"
    return "mid"


def _repro_tier(std: float, ok: float, bad: float) -> str:
    """
    stdが小さいほど再現性が高い
      - std <= ok: good
      - ok < std <= bad: mid
      - std > bad: bad
    """
    if std <= ok:
        return "good"
    if std <= bad:
        return "mid"
    return "bad"


def _confidence_from_raw(raw: Dict[str, Any]) -> Tuple[str, float]:
    return (raw.get("confidence_tier", "low"), float(raw.get("confidence_value", 0.0)))


def _soften_if_low_conf(text: str, conf_tier: str) -> str:
    if conf_tier == "low":
        # 断定を弱める
        return text.replace("です。", "傾向があります。").replace("最優先", "優先")
    return text


# ==================================================
# 02 Shoulder (mean+std+confidence + 3×3×3)
# ==================================================
def judge_shoulder(raw: Dict[str, Any]) -> Dict[str, Any]:
    sh_mean = float(raw["mean_shoulder_rotation"])
    sh_std = float(raw["std_shoulder_rotation"])
    hip_mean = float(raw["mean_hip_rotation"])

    conf_tier, conf_value = _confidence_from_raw(raw)

    # 主指標：肩回転（平均）
    main = _tier3_by_range(sh_mean, 85, 105)

    # 関連：捻転差（平均）
    x_factor = sh_mean - hip_mean
    rel = _tier3_by_range(x_factor, 35, 55)

    # 質：再現性（σ）
    quality = _repro_tier(sh_std, ok=4.0, bad=8.0)

    tags: List[str] = []
    if main == "low":
        tags.append("肩回転不足")
    if main == "high":
        tags.append("肩回転過多")
    if rel == "low":
        tags.append("捻転差不足")
    if rel == "high":
        tags.append("捻転差過多")
    if quality == "bad":
        tags.append("再現性不足")

    return {
        "main": main,
        "related": rel,
        "quality": quality,
        "confidence": conf_tier,
        "confidence_value": round(conf_value, 3),
        "x_factor_value": round(x_factor, 1),
        "tags": tags,
        "mean": round(sh_mean, 2),
        "std": round(sh_std, 2),
        "max": float(raw["max_shoulder_rotation"]),
    }


def shoulder_good_bad(j: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    good: List[str] = []
    bad: List[str] = []

    if j["main"] == "mid":
        good.append(f"肩回転（mean {j['mean']}°）は目安レンジ内で、回旋量の土台は安定しています。")
    if j["related"] == "mid":
        good.append(f"捻転差（{j['x_factor_value']}°）が確保されており、切り返しで溜めを作れています。")
    if j["quality"] == "good":
        good.append(f"回旋のばらつき（σ {j['std']}°）が小さく、再現性が高い動きです。")

    if j["main"] == "low":
        bad.append(f"肩回転（mean {j['mean']}°）が少なく、上半身のエネルギーが不足しています。")
    if j["main"] == "high":
        bad.append(f"肩回転（mean {j['mean']}°）が大きく、回し過ぎでタイミングがズレやすいです。")
    if j["related"] == "low":
        bad.append(f"捻転差（{j['x_factor_value']}°）が不足しており、肩と腰が同調し過ぎています。")
    if j["related"] == "high":
        bad.append(f"捻転差（{j['x_factor_value']}°）が大きく、腰が止まり上体が先行しやすいです。")
    if j["quality"] == "bad":
        bad.append(f"回旋のばらつき（σ {j['std']}°）が大きく、同じスイング幅を作れていません。")

    if not good:
        good = ["上半身の動きに大きな破綻はなく、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の回旋は安定しており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_shoulder_pro(j: Dict[str, Any], seed: str) -> str:
    rnd = random.Random(seed + "_shoulder_pro")

    lines: List[str] = []
    # 3行程度 = 2〜3文に固定
    lines.append(f"肩は mean {j['mean']}° / σ {j['std']}° で、傾向と再現性を見ています。")

    if j["quality"] == "bad":
        lines.append("量そのものよりも「同じ幅で回る」ことが最優先です。")
    elif j["main"] == "high":
        lines.append("回し過ぎが出やすいので、トップの深さを揃える意識が効きます。")
    elif j["main"] == "low":
        lines.append("肩を回す意識より、捻転差を作る動きで自然に回旋量を増やしてください。")
    else:
        lines.append("ここは維持でOKです。次は他部位の優先テーマに集中しましょう。")

    # confidence低なら断定を弱める
    text = " ".join(lines)
    return _soften_if_low_conf(text, j["confidence"])


def build_paid_02_shoulder(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_shoulder(raw)
    good, bad = shoulder_good_bad(j)
    pro = generate_shoulder_pro(j, seed)
    value = f"max {j['max']}° / mean {j['mean']}° / σ {j['std']}°"
    return {
        "title": "02. Shoulder Rotation（肩回転）",
        "value": value,
        "judge": j,
        "tags": j["tags"],
        "good": good,
        "bad": bad,
        "pro_comment": pro,
    }


# ==================================================
# 03 Hip
# ==================================================
def judge_hip(raw: Dict[str, Any]) -> Dict[str, Any]:
    hip_mean = float(raw["mean_hip_rotation"])
    hip_std = float(raw["std_hip_rotation"])
    sh_mean = float(raw["mean_shoulder_rotation"])

    conf_tier, conf_value = _confidence_from_raw(raw)

    main = _tier3_by_range(hip_mean, 36, 50)
    x_factor = sh_mean - hip_mean
    rel = _tier3_by_range(x_factor, 35, 55)

    quality = _repro_tier(hip_std, ok=3.5, bad=7.0)

    tags: List[str] = []
    if main == "low":
        tags.append("腰回転不足")
    if main == "high":
        tags.append("腰回転過多")
    if rel == "low":
        tags.append("捻転差不足")
    if rel == "high":
        tags.append("捻転差過多")
    if quality == "bad":
        tags.append("再現性不足")

    return {
        "main": main,
        "related": rel,
        "quality": quality,
        "confidence": conf_tier,
        "confidence_value": round(conf_value, 3),
        "x_factor_value": round(x_factor, 1),
        "tags": tags,
        "mean": round(hip_mean, 2),
        "std": round(hip_std, 2),
        "min": float(raw.get("min_hip_rotation", hip_mean)),
    }


def hip_good_bad(j: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    good: List[str] = []
    bad: List[str] = []

    if j["main"] == "mid":
        good.append(f"腰回転（mean {j['mean']}°）は目安レンジ内で、下半身主導の土台があります。")
    if j["related"] == "mid":
        good.append(f"捻転差（{j['x_factor_value']}°）が確保され、切り返しで溜めを作れています。")
    if j["quality"] == "good":
        good.append(f"腰回転のばらつき（σ {j['std']}°）が小さく、下半身が安定しています。")

    if j["main"] == "low":
        bad.append(f"腰回転（mean {j['mean']}°）が少なく、地面反力を活かし切れていません。")
    if j["main"] == "high":
        bad.append(f"腰回転（mean {j['mean']}°）が大きく、上体が先に開きやすいです。")
    if j["related"] == "low":
        bad.append(f"捻転差（{j['x_factor_value']}°）が不足し、肩と腰が同時に動いています。")
    if j["related"] == "high":
        bad.append(f"捻転差（{j['x_factor_value']}°）が大きく、腰が止まり上体が先行しています。")
    if j["quality"] == "bad":
        bad.append(f"腰回転のばらつき（σ {j['std']}°）が大きく、テンポが揃っていません。")

    if not good:
        good = ["下半身の動きに大きな破綻はなく、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の下半身は安定しており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_hip_pro(j: Dict[str, Any], seed: str) -> str:
    lines: List[str] = []
    lines.append(f"腰は mean {j['mean']}° / σ {j['std']}° を基準に、下半身主導の質を見ています。")

    if j["quality"] == "bad":
        lines.append("まずはテンポを落として、同じ幅で回る感覚を作ると一気に安定します。")
    elif j["main"] == "low":
        lines.append("回す量を増やすというより、切り返しで腰が先に動く順序を作ってください。")
    elif j["main"] == "high":
        lines.append("回り過ぎは上体の突っ込みを誘発します。腰の回し幅を一定に揃えましょう。")
    else:
        lines.append("ここは良い状態です。優先テーマは他部位に寄せてOKです。")

    text = " ".join(lines)
    return _soften_if_low_conf(text, j["confidence"])


def build_paid_03_hip(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_hip(raw)
    good, bad = hip_good_bad(j)
    pro = generate_hip_pro(j, seed)
    value = f"min {j['min']}° / mean {j['mean']}° / σ {j['std']}°"
    return {
        "title": "03. Hip Rotation（腰回転）",
        "value": value,
        "judge": j,
        "tags": j["tags"],
        "good": good,
        "bad": bad,
        "pro_comment": pro,
    }


# ==================================================
# 04 Wrist
# ==================================================
def judge_wrist(raw: Dict[str, Any]) -> Dict[str, Any]:
    w_mean = float(raw["mean_wrist_cock"])
    w_std = float(raw["std_wrist_cock"])
    sh_mean = float(raw["mean_shoulder_rotation"])
    hip_mean = float(raw["mean_hip_rotation"])

    conf_tier, conf_value = _confidence_from_raw(raw)

    main = _tier3_by_range(w_mean, 70, 90)  # mean基準
    x_factor = sh_mean - hip_mean
    rel = _tier3_by_range(x_factor, 35, 55)

    # 手首はブレが出やすいので閾値はやや厳しめ
    quality = _repro_tier(w_std, ok=5.0, bad=10.0)

    tags: List[str] = []
    if main == "low":
        tags.append("コック不足")
    if main == "high":
        tags.append("コック過多")
    if rel == "low":
        tags.append("体幹主導不足")
    if quality == "bad":
        tags.append("再現性不足")

    # “手首主導”は「平均が高い」または「stdが大きい」で付与
    if main == "high" or quality == "bad":
        tags.append("手首主導")

    return {
        "main": main,
        "related": rel,
        "quality": quality,
        "confidence": conf_tier,
        "confidence_value": round(conf_value, 3),
        "x_factor_value": round(x_factor, 1),
        "tags": tags,
        "mean": round(w_mean, 2),
        "std": round(w_std, 2),
        "max": float(raw["max_wrist_cock"]),
    }


def wrist_good_bad(j: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    good: List[str] = []
    bad: List[str] = []

    if j["main"] == "mid":
        good.append(f"手首コック（mean {j['mean']}°）は目安レンジ内で、溜めの量は適正です。")
    if j["quality"] == "good":
        good.append(f"コックのばらつき（σ {j['std']}°）が小さく、手元が安定しています。")
    if j["related"] == "mid":
        good.append(f"捻転差（{j['x_factor_value']}°）があり、体の回転と連動しています。")

    if j["main"] == "low":
        bad.append(f"コック（mean {j['mean']}°）が少なく、溜めを作れていません。")
    if j["main"] == "high":
        bad.append(f"コック（mean {j['mean']}°）が大きく、手首主導が出ています。")
    if j["related"] == "low":
        bad.append(f"捻転差（{j['x_factor_value']}°）が小さく、体幹より手先が先行しています。")
    if j["quality"] == "bad":
        bad.append(f"コックのばらつき（σ {j['std']}°）が大きく、同じ形を再現できていません。")

    if not good:
        good = ["手首の動きに大きな破綻はなく、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の手首操作は安定しており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_wrist_pro(j: Dict[str, Any], seed: str) -> str:
    lines: List[str] = []
    lines.append(f"手首は mean {j['mean']}° / σ {j['std']}° で「溜めの量」と「操作の暴れ」を分けて見ます。")

    if j["quality"] == "bad":
        lines.append("今は“手首で合わせる”動きが混ざりやすいので、LtoLなど小さい振り幅で形を固定してください。")
    elif j["main"] == "high":
        lines.append("コック量が強めなので、体の回転で振る意識に戻すと再現性が上がります。")
    elif j["main"] == "low":
        lines.append("コックを作る意識より、回転で自然に入る形を優先すると改善が早いです。")
    else:
        lines.append("ここは良い状態です。維持しつつ他の優先テーマに集中しましょう。")

    text = " ".join(lines)
    return _soften_if_low_conf(text, j["confidence"])


def build_paid_04_wrist(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_wrist(raw)
    good, bad = wrist_good_bad(j)
    pro = generate_wrist_pro(j, seed)
    value = f"max {j['max']}° / mean {j['mean']}° / σ {j['std']}°"
    return {
        "title": "04. Wrist Cock（手首コック）",
        "value": value,
        "judge": j,
        "tags": j["tags"],
        "good": good,
        "bad": bad,
        "pro_comment": pro,
    }


# ==================================================
# 05 Head
# ==================================================
def judge_head(raw: Dict[str, Any]) -> Dict[str, Any]:
    h_mean = float(raw["mean_head_drift"])
    h_std = float(raw["std_head_drift"])
    k_mean = float(raw["mean_knee_sway"])

    conf_tier, conf_value = _confidence_from_raw(raw)

    # head は小さいほど良いが、UI整合のため tier は「low=良 / high=悪」のまま使う
    if h_mean < 0.06:
        main = "low"   # 良
    elif h_mean > 0.15:
        main = "high"  # 悪
    else:
        main = "mid"

    # 関連：膝が大きいと頭も流れやすい
    if k_mean < 0.10:
        rel = "low"
    elif k_mean > 0.20:
        rel = "high"
    else:
        rel = "mid"

    quality = _repro_tier(h_std, ok=0.015, bad=0.035)

    tags: List[str] = []
    if main == "high":
        tags.append("頭部ブレ大")
    if rel == "high":
        tags.append("下半身不安定")
    if quality == "bad":
        tags.append("再現性不足")

    return {
        "main": main,
        "related": rel,
        "quality": quality,
        "confidence": conf_tier,
        "confidence_value": round(conf_value, 3),
        "tags": tags,
        "mean": round(h_mean, 4),
        "std": round(h_std, 4),
        "max": float(raw["max_head_drift"]),
    }


def head_good_bad(j: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    good: List[str] = []
    bad: List[str] = []

    if j["main"] == "low":
        good.append(f"頭部ブレ（mean {j['mean']}）が小さく、軸が安定しています。")
    if j["main"] == "mid":
        good.append(f"頭部ブレ（mean {j['mean']}）は平均的で、大崩れは見られません。")
    if j["quality"] == "good":
        good.append(f"ばらつき（σ {j['std']}）が小さく、同じ軸で動けています。")

    if j["main"] == "high":
        bad.append(f"頭部ブレ（mean {j['mean']}）が大きく、ミート率が落ちやすいです。")
    if j["related"] == "high":
        bad.append("膝の不安定が頭部ブレを助長しています。")
    if j["quality"] == "bad":
        bad.append(f"ばらつき（σ {j['std']}）が大きく、毎回の軸が揃っていません。")

    if not good:
        good = ["頭部の位置は大きく崩れておらず、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["頭部の安定は保てており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_head_pro(j: Dict[str, Any], seed: str) -> str:
    lines: List[str] = []
    lines.append(f"頭部は mean {j['mean']} / σ {j['std']} で、軸の“平均”と“揺れ幅”を評価します。")
    if j["main"] == "high" or j["quality"] == "bad":
        lines.append("まずは頭の位置を固定し、下半身の横流れを止めると改善が早いです。")
    else:
        lines.append("軸は良い状態です。下半身側の安定を揃えると完成度が上がります。")
    text = " ".join(lines)
    return _soften_if_low_conf(text, j["confidence"])


def build_paid_05_head(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_head(raw)
    good, bad = head_good_bad(j)
    pro = generate_head_pro(j, seed)
    value = f"max {j['max']} / mean {j['mean']} / σ {j['std']}"
    return {
        "title": "05. Head Stability（頭部）",
        "value": value,
        "judge": j,
        "tags": j["tags"],
        "good": good,
        "bad": bad,
        "pro_comment": pro,
    }


# ==================================================
# 06 Knee
# ==================================================
def judge_knee(raw: Dict[str, Any]) -> Dict[str, Any]:
    k_mean = float(raw["mean_knee_sway"])
    k_std = float(raw["std_knee_sway"])
    h_mean = float(raw["mean_head_drift"])

    conf_tier, conf_value = _confidence_from_raw(raw)

    main = _tier3_by_range(k_mean, 0.10, 0.20)

    # 関連：頭部が大きいと膝も崩れやすい（相互）
    if h_mean < 0.06:
        rel = "low"
    elif h_mean > 0.15:
        rel = "high"
    else:
        rel = "mid"

    quality = _repro_tier(k_std, ok=0.02, bad=0.05)

    tags: List[str] = []
    if main == "high":
        tags.append("膝ブレ大")
    if rel == "high":
        tags.append("上半身不安定")
    if quality == "bad":
        tags.append("再現性不足")

    return {
        "main": main,
        "related": rel,
        "quality": quality,
        "confidence": conf_tier,
        "confidence_value": round(conf_value, 3),
        "tags": tags,
        "mean": round(k_mean, 4),
        "std": round(k_std, 4),
        "max": float(raw["max_knee_sway"]),
    }


def knee_good_bad(j: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    good: List[str] = []
    bad: List[str] = []

    if j["main"] == "mid":
        good.append(f"膝ブレ（mean {j['mean']}）は平均的で、土台は大きく崩れていません。")
    if j["main"] == "low":
        good.append(f"膝ブレ（mean {j['mean']}）が小さく、下半身が安定しています。")
    if j["quality"] == "good":
        good.append(f"ばらつき（σ {j['std']}）が小さく、同じ動きを再現できています。")

    if j["main"] == "high":
        bad.append(f"膝ブレ（mean {j['mean']}）が大きく、体重移動が横流れになっています。")
    if j["related"] == "high":
        bad.append("上半身の不安定が膝ブレを助長しています。")
    if j["quality"] == "bad":
        bad.append(f"ばらつき（σ {j['std']}）が大きく、毎回の土台が揃っていません。")

    if not good:
        good = ["下半身の土台は大きく崩れておらず、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["下半身の安定は保てており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_knee_pro(j: Dict[str, Any], seed: str) -> str:
    lines: List[str] = []
    lines.append(f"膝は mean {j['mean']} / σ {j['std']} で、横流れの“傾向”と“再現性”を見ます。")
    if j["main"] == "high" or j["quality"] == "bad":
        lines.append("まずは膝幅固定＋ゆっくり素振りで、横流れを止めるのが最短です。")
    else:
        lines.append("下半身は良い状態です。頭部側の安定を揃えると完成度が上がります。")
    text = " ".join(lines)
    return _soften_if_low_conf(text, j["confidence"])


def build_paid_06_knee(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_knee(raw)
    good, bad = knee_good_bad(j)
    pro = generate_knee_pro(j, seed)
    value = f"max {j['max']} / mean {j['mean']} / σ {j['std']}"
    return {
        "title": "06. Knee Stability（膝）",
        "value": value,
        "judge": j,
        "tags": j["tags"],
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
    if tag_counter["肩回転過多"] + tag_counter["手首主導"] >= 2:
        return "操作過多型"
    if tag_counter["再現性不足"] >= 2:
        return "再現性不足型"
    return "バランス型"


def extract_priorities(tag_counter: Counter, max_items: int = 2) -> List[str]:
    order = [
        "再現性不足",
        "捻転差不足",
        "膝ブレ大",
        "頭部ブレ大",
        "手首主導",
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
        "id": "release_control",
        "name": "リリース抑制ドリル（LtoL）",
        "category": "手首",
        "tags": ["手首主導", "コック過多"],
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
        "id": "tempo",
        "name": "テンポ安定ドリル（メトロノーム）",
        "category": "リズム",
        "tags": ["再現性不足"],
        "purpose": "タイミングを一定にする",
        "how": "①一定テンポで素振り\n②10回\n③その後ボール10球",
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
# 09 フィッティング（既存ロジック維持）
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
    sh = float(raw["mean_shoulder_rotation"])       # mean
    hip = float(raw["mean_hip_rotation"])           # mean
    wrist = float(raw["mean_wrist_cock"])           # mean
    xf = sh - hip

    a = _norm_range(sh, 85, 105)
    b = _norm_range(hip, 36, 50)
    c = _norm_range(wrist, 70, 90)
    d = _norm_range(xf, 36, 55)
    return int(round((a + b + c + d) / 4.0 * 100))


def calc_stability_idx(raw: Dict[str, Any]) -> int:
    head = float(raw["mean_head_drift"])
    knee = float(raw["mean_knee_sway"])

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
        wrist_high = float(raw["mean_wrist_cock"]) > 90
        head_bad = float(raw["mean_head_drift"]) > 0.15
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
# 10 Summary
# ==================================================
def build_paid_10(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "10. Summary（まとめ）",
        "text": [
            "今回の解析では、max（瞬間値）だけでなく mean（傾向）と σ（再現性）まで評価しています。",
            "次のステップは「優先テーマを2点に絞って改善すること」です。",
            "08のドリルと09の指針を使い、同じ幅・同じテンポを作っていきましょう。",
            "",
            "あなたのゴルフライフが、より充実したものになることを願っています。",
        ],
    }


# ==================================================
# Free 07
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
# Analysis builder（③ 完全プロ版）
# ==================================================
def build_analysis(raw: Dict[str, Any], premium: bool, report_id: str, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {"01": build_section_01(raw)}

    if not premium:
        analysis["07"] = build_free_07(raw)
        return analysis

    analysis["02"] = build_paid_02_shoulder(raw, seed=report_id)
    analysis["03"] = build_paid_03_hip(raw, seed=report_id)
    analysis["04"] = build_paid_04_wrist(raw, seed=report_id)
    analysis["05"] = build_paid_05_head(raw, seed=report_id)
    analysis["06"] = build_paid_06_knee(raw, seed=report_id)

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


def _handle_line_webhook():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


# ★ 404対策：/webhook 以外に / と /callback も受ける（LINE設定がどれでも落ちない）
@app.route("/", methods=["POST"])
def webhook_root():
    return _handle_line_webhook()


@app.route("/callback", methods=["POST"])
def webhook_callback():
    return _handle_line_webhook()


@app.route("/webhook", methods=["POST"])
def webhook():
    return _handle_line_webhook()


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
