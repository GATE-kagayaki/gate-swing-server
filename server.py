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

from flask import Flask, request, jsonify, abort, render_template, Response

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
# Premium判定（本番は決済と連携）
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
# Math / Stats
# ==================================================
def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    m = _mean(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(v)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm01(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    return _clamp((x - lo) / (hi - lo), 0.0, 1.0)


def _angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    ax, ay = p1[0] - p2[0], p1[1] - p2[1]
    bx, by = p3[0] - p2[0], p3[1] - p2[1]
    dot = ax * bx + ay * by
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na * nb == 0:
        return 0.0
    c = _clamp(dot / (na * nb), -1.0, 1.0)
    return math.degrees(math.acos(c))


def _cat3(value: float, lo: float, hi: float) -> str:
    # low / mid / high
    if value < lo:
        return "low"
    if value > hi:
        return "high"
    return "mid"


def _cat3_small_is_good(value: float, lo: float, hi: float) -> str:
    # sway系：小さいほど良い => good / mid / bad
    if value < lo:
        return "good"
    if value > hi:
        return "bad"
    return "mid"


def _conf_cat(c: float) -> str:
    # confidence: low/mid/high
    if c < 0.45:
        return "low"
    if c < 0.75:
        return "mid"
    return "high"


def _stability_cat(std: float, lo: float, hi: float) -> str:
    # 小さいほど安定
    if std < lo:
        return "stable"
    if std > hi:
        return "unstable"
    return "mid"


# ==================================================
# MediaPipe analysis (Top〜Impact segment, max/mean/std/conf)
# ==================================================
def analyze_swing_with_mediapipe(video_path: str) -> Dict[str, Any]:
    """
    - MediaPipe Poseでフレーム列を取得
    - トップ〜インパクト区間をヒューリスティックに抽出
    - 各指標について max / mean / std を算出
    - confidence は「有効フレーム数 × ランドマーク可視性」で 0..1 推定
    """
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCVがビデオを読み込めませんでした。")

    # per-frame series
    sh_series: List[float] = []
    hip_series: List[float] = []
    wrist_series: List[float] = []
    head_series: List[float] = []
    knee_series: List[float] = []
    vis_series: List[float] = []

    frame_count = 0
    valid_count = 0

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

            def xy(i: int) -> Tuple[float, float]:
                return (lm[i].x, lm[i].y)

            def vis(i: int) -> float:
                # landmark.visibility は 0..1
                v = getattr(lm[i], "visibility", 0.0)
                try:
                    return float(v)
                except Exception:
                    return 0.0

            LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LH = mp_pose.PoseLandmark.LEFT_HIP.value
            RH = mp_pose.PoseLandmark.RIGHT_HIP.value
            LK = mp_pose.PoseLandmark.LEFT_KNEE.value
            LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
            LW = mp_pose.PoseLandmark.LEFT_WRIST.value
            LI = mp_pose.PoseLandmark.LEFT_INDEX.value
            NO = mp_pose.PoseLandmark.NOSE.value

            # 指標（すべて2D正規化座標上の角度/変位）
            sh = _angle(xy(LS), xy(RS), xy(RH))                  # shoulder "rotation" proxy
            hip = _angle(xy(LH), xy(RH), xy(LK))                 # hip "rotation" proxy
            wrist = _angle(xy(LE), xy(LW), xy(LI))               # wrist cock proxy
            head = abs(xy(NO)[0] - 0.5)                          # head sway
            knee = abs(xy(LK)[0] - 0.5)                          # knee sway

            # 可視性（主要点の平均）
            v = _mean([vis(LS), vis(RS), vis(LH), vis(RH), vis(LK), vis(LE), vis(LW), vis(NO)])

            sh_series.append(float(sh))
            hip_series.append(float(hip))
            wrist_series.append(float(wrist))
            head_series.append(float(head))
            knee_series.append(float(knee))
            vis_series.append(float(v))
            valid_count += 1

    cap.release()

    if frame_count < 10 or valid_count < 10:
        raise RuntimeError("解析に必要なフレーム数が不足しています。")

    # ---- Top〜Impact の切り出し（ヒューリスティック）
    # top: 肩回転（proxy）の最大点
    top_idx = int(max(range(len(sh_series)), key=lambda i: sh_series[i]))

    # impact: top以降で「wristが最小になる点」を探す（リリース終盤のproxy）
    # 探索範囲は top_idx+1 .. top_idx + 45%（長すぎる誤検知回避）
    end_search = min(len(wrist_series) - 1, top_idx + max(8, int(len(wrist_series) * 0.45)))
    if top_idx + 1 <= end_search:
        impact_idx = int(min(range(top_idx + 1, end_search + 1), key=lambda i: wrist_series[i]))
    else:
        impact_idx = min(len(wrist_series) - 1, top_idx + 1)

    if impact_idx <= top_idx:
        impact_idx = min(len(wrist_series) - 1, top_idx + 1)

    seg = slice(top_idx, impact_idx + 1)

    seg_sh = sh_series[seg]
    seg_hip = hip_series[seg]
    seg_wrist = wrist_series[seg]
    seg_head = head_series[seg]
    seg_knee = knee_series[seg]
    seg_vis = vis_series[seg]

    seg_len = len(seg_sh)

    # confidence（0..1）
    # - 区間有効フレーム数（短いほど低い）
    # - landmark可視性平均（低いほど低い）
    len_score = _norm01(seg_len, 8, 60)          # 8fで0、60fで1近傍
    vis_score = _clamp(_mean(seg_vis), 0.0, 1.0)
    conf = float(_clamp(0.55 * len_score + 0.45 * vis_score, 0.0, 1.0))

    # 統計
    def pack(xs: List[float]) -> Dict[str, float]:
        return {
            "max": round(float(max(xs)), 2),
            "mean": round(float(_mean(xs)), 2),
            "std": round(float(_std(xs)), 2),
        }

    raw = {
        "frame_count_total": int(frame_count),
        "frame_count_valid": int(valid_count),
        "segment_top_index": int(top_idx),
        "segment_impact_index": int(impact_idx),
        "segment_frame_count": int(seg_len),
        "confidence": round(conf, 3),

        "shoulder_rotation": pack(seg_sh),
        "hip_rotation": pack(seg_hip),
        "wrist_cock": pack(seg_wrist),
        "head_sway": {
            "max": round(float(max(seg_head)), 4),
            "mean": round(float(_mean(seg_head)), 4),
            "std": round(float(_std(seg_head)), 4),
        },
        "knee_sway": {
            "max": round(float(max(seg_knee)), 4),
            "mean": round(float(_mean(seg_knee)), 4),
            "std": round(float(_std(seg_knee)), 4),
        },
    }

    return raw


# ==================================================
# Section 01 (表示：max/mean/std/conf)
# ==================================================
def build_section_01(raw: Dict[str, Any]) -> Dict[str, Any]:
    conf = raw.get("confidence", 0.0)
    seg_n = raw.get("segment_frame_count", 0)

    def vline(name: str, d: Dict[str, Any], guide: str, desc: str) -> Dict[str, Any]:
        return {
            "name": name,
            "value": f"max {d['max']} / mean {d['mean']} / σ {d['std']}",
            "description": desc,
            "guide": guide,
        }

    items = [
        {
            "name": "解析フレーム（全体 / 有効）",
            "value": f"{raw.get('frame_count_total')} / {raw.get('frame_count_valid')}",
            "description": "動画全体のフレーム数と、骨格推定が成立したフレーム数です。",
            "guide": "有効フレームが多いほど安定",
        },
        {
            "name": "解析区間（トップ〜インパクト）",
            "value": f"{seg_n} frames",
            "description": "本レポートはトップ〜インパクト区間のみを抽出して評価しています。",
            "guide": "8〜60 frames 目安",
        },
        {
            "name": "信頼度（confidence）",
            "value": conf,
            "description": "区間フレーム数とランドマーク可視性から推定した信頼度（0〜1）です。",
            "guide": "0.75以上：高 / 0.45〜0.74：中 / 0.44以下：低",
        },
        vline(
            "肩回転（°）",
            raw["shoulder_rotation"],
            "mean：85〜105°（目安）",
            "トップ〜インパクト区間の肩の回旋量（proxy）です。",
        ),
        vline(
            "腰回転（°）",
            raw["hip_rotation"],
            "mean：36〜50°（目安）",
            "トップ〜インパクト区間の腰の回旋量（proxy）です。",
        ),
        vline(
            "手首コック（°）",
            raw["wrist_cock"],
            "mean：70〜90°（本計測仕様の目安）",
            "トップ〜インパクト区間の手首角度（proxy）です。",
        ),
        {
            "name": "頭部ブレ（Sway）",
            "value": f"max {raw['head_sway']['max']} / mean {raw['head_sway']['mean']} / σ {raw['head_sway']['std']}",
            "description": "頭の左右移動量（x方向）です。小さいほど軸が安定します。",
            "guide": "mean：0.06〜0.15（目安）",
        },
        {
            "name": "膝ブレ（Sway）",
            "value": f"max {raw['knee_sway']['max']} / mean {raw['knee_sway']['mean']} / σ {raw['knee_sway']['std']}",
            "description": "膝の左右移動量（x方向）です。小さいほど下半身が安定します。",
            "guide": "mean：0.10〜0.20（目安）",
        },
    ]

    return {"title": "01. 骨格計測データ（トップ〜インパクト区間）", "items": items}


# ==================================================
# 02〜06：3×3×3（値×安定性×信頼度） + プロ目線（3行）
# ==================================================
def _pro3(lines: List[str]) -> str:
    # フロントが innerHTML なので <br> を使って3行化
    lines = [l.strip() for l in lines if l.strip()]
    lines = lines[:3]
    while len(lines) < 3:
        lines.append("—")
    return "<br>".join(lines)


def _seeded_choice(seed: str, bucket: List[str], salt: str) -> str:
    rnd = random.Random(f"{seed}:{salt}")
    return rnd.choice(bucket) if bucket else ""


def build_02_shoulder(raw: Dict[str, Any], report_id: str) -> Dict[str, Any]:
    sh = raw["shoulder_rotation"]
    hip = raw["hip_rotation"]
    xf_mean = round(float(sh["mean"] - hip["mean"]), 2)
    conf = float(raw.get("confidence", 0.0))

    # 3x3x3
    main = _cat3(sh["mean"], 85, 105)
    stab = _stability_cat(sh["std"], 6, 14)  # 目安（小さいほど良い）
    confc = _conf_cat(conf)

    tags: List[str] = []
    if main == "low":
        tags.append("肩回転不足")
    if main == "high":
        tags.append("肩回転過多")
    if xf_mean < 35:
        tags.append("捻転差不足")
    if xf_mean > 55:
        tags.append("捻転差過多")
    if stab == "unstable":
        tags.append("肩回転ばらつき")

    good: List[str] = []
    bad: List[str] = []

    if main == "mid":
        good.append(f"肩は mean {sh['mean']}°（σ {sh['std']}°）で、回旋量は目安レンジ内です。")
    if stab == "stable":
        good.append(f"肩のばらつき（σ {sh['std']}°）が小さく、再現性を作りやすい状態です。")
    if 35 <= xf_mean <= 55:
        good.append(f"捻転差は mean {xf_mean}° で、肩と腰の差は適正帯です。")

    if main == "low":
        bad.append(f"肩の回旋量が mean {sh['mean']}° と少なめで、上半身のエネルギーが出にくい可能性があります。")
    if main == "high":
        bad.append(f"肩の回旋量が mean {sh['mean']}° と大きく、量が増えるほどタイミングがズレやすくなります。")
    if xf_mean < 35:
        bad.append(f"捻転差が mean {xf_mean}° と小さく、肩と腰が同時に動きやすい状態です。")
    if xf_mean > 55:
        bad.append(f"捻転差が mean {xf_mean}° と大きく、腰が止まりやすく上体先行になりやすい状態です。")
    if stab == "unstable":
        bad.append(f"肩回転のばらつき（σ {sh['std']}°）が大きく、同じ幅で回りにくい状態です。")

    if not good:
        good = ["上半身の動きに大きな破綻は見られず、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の回旋は安定しており、再現性を維持しやすい状態です。"]

    # プロ目線（3行・具体）
    pro = _pro3([
        f"肩は mean {sh['mean']}° / σ {sh['std']}°、捻転差は mean {xf_mean}° を基準に評価します。",
        f"この区間での課題は「量」より「ばらつき（σ）」です。σが大きいほどインパクト再現性が落ちます。",
        f"対策は“肩を回す”ではなく、トップ位置を固定して同じ幅で戻す（肩と腰の差を崩さない）ことです。",
    ])

    return {
        "title": "02. Shoulder Rotation（肩回転）",
        "value": f"max {sh['max']} / mean {sh['mean']} / σ {sh['std']}（conf {raw.get('confidence')}）",
        "judge": {"main": main, "stability": stab, "confidence": confc, "x_factor_mean": xf_mean},
        "tags": tags,
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro,
    }


def build_03_hip(raw: Dict[str, Any], report_id: str) -> Dict[str, Any]:
    hip = raw["hip_rotation"]
    sh = raw["shoulder_rotation"]
    conf = float(raw.get("confidence", 0.0))

    xf_mean = round(float(sh["mean"] - hip["mean"]), 2)

    main = _cat3(hip["mean"], 36, 50)
    stab = _stability_cat(hip["std"], 6, 14)
    confc = _conf_cat(conf)

    tags: List[str] = []
    if main == "low":
        tags.append("腰回転不足")
    if main == "high":
        tags.append("腰回転過多")
    if xf_mean < 35:
        tags.append("捻転差不足")
    if xf_mean > 55:
        tags.append("捻転差過多")
    if stab == "unstable":
        tags.append("腰回転ばらつき")

    good: List[str] = []
    bad: List[str] = []

    if main == "mid":
        good.append(f"腰は mean {hip['mean']}°（σ {hip['std']}°）で、下半身の回旋量は目安レンジ内です。")
    if stab == "stable":
        good.append(f"腰のばらつき（σ {hip['std']}°）が小さく、下半身主導の再現性が作りやすい状態です。")
    if 35 <= xf_mean <= 55:
        good.append(f"捻転差は mean {xf_mean}° で、上半身に対して腰が先行しすぎていません。")

    if main == "low":
        bad.append(f"腰の回旋量が mean {hip['mean']}° と少なめで、下半身の推進力が出にくい可能性があります。")
    if main == "high":
        bad.append(f"腰の回旋量が mean {hip['mean']}° と大きく、上体がつられて開きやすくなります。")
    if xf_mean < 35:
        bad.append(f"捻転差が mean {xf_mean}° と小さく、腰と肩が同時に動いて“溜め”が作りにくい状態です。")
    if xf_mean > 55:
        bad.append(f"捻転差が mean {xf_mean}° と大きく、腰が止まりやすく上体先行が出やすい状態です。")
    if stab == "unstable":
        bad.append(f"腰回転のばらつき（σ {hip['std']}°）が大きく、切り返し前後のタイミング差が出やすい状態です。")

    if not good:
        good = ["下半身の動きに大きな破綻は見られず、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の下半身は安定しており、再現性を維持しやすい状態です。"]

    # プロ目線（3行・具体化：本動画内、σと捻転差で“質”を見る）
    pro = _pro3([
        f"腰は mean {hip['mean']}° / σ {hip['std']}° を基準に、下半身主導の“質（揃い方）”を見ています。",
        f"σが大きい場合は、トップ〜切り返し直後の腰の回旋量が区間内で揃っていないサインです（本動画内）。",
        f"改善は「腰を速く回す」ではなく、トップで一度“同じ形”を作り、捻転差（mean {xf_mean}°）を崩さず戻すことです。",
    ])

    return {
        "title": "03. Hip Rotation（腰回転）",
        "value": f"max {hip['max']} / mean {hip['mean']} / σ {hip['std']}（conf {raw.get('confidence')}）",
        "judge": {"main": main, "stability": stab, "confidence": confc, "x_factor_mean": xf_mean},
        "tags": tags,
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro,
    }


def build_04_wrist(raw: Dict[str, Any], report_id: str) -> Dict[str, Any]:
    w = raw["wrist_cock"]
    sh = raw["shoulder_rotation"]
    hip = raw["hip_rotation"]
    conf = float(raw.get("confidence", 0.0))

    xf_mean = round(float(sh["mean"] - hip["mean"]), 2)

    # 計測仕様の目安（mean: 70-90）
    main = _cat3(w["mean"], 70, 90)
    stab = _stability_cat(w["std"], 5, 12)
    confc = _conf_cat(conf)

    tags: List[str] = []
    if main == "low":
        tags.append("コック不足")
    if main == "high":
        tags.append("コック過多")
    if xf_mean < 35:
        tags.append("体幹主導不足")
    if stab == "unstable":
        tags.append("手首ばらつき")

    good: List[str] = []
    bad: List[str] = []

    if main == "mid":
        good.append(f"手首は mean {w['mean']}°（σ {w['std']}°）で、コック量は目安レンジ内です。")
    if stab == "stable":
        good.append(f"手首角度のばらつき（σ {w['std']}°）が小さく、リリースが揃いやすい状態です。")
    if xf_mean >= 35:
        good.append(f"捻転差は mean {xf_mean}° あり、体幹との連動を作りやすい土台があります。")

    if main == "low":
        bad.append(f"コック量が mean {w['mean']}° と小さく、“溜め”が作りにくい可能性があります。")
    if main == "high":
        bad.append(f"コック量が mean {w['mean']}° と大きく、手首主導（操作）が出やすい状態です。")
    if xf_mean < 35:
        bad.append(f"捻転差が mean {xf_mean}° と小さく、体幹より先に手元が動きやすい状態です。")
    if stab == "unstable":
        bad.append(f"手首角度のばらつき（σ {w['std']}°）が大きく、インパクト付近の当たりが揃いにくい可能性があります。")

    if not good:
        good = ["手首の動きに大きな破綻は見られず、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の手首操作は安定しており、再現性を維持しやすい状態です。"]

    pro = _pro3([
        f"手首は mean {w['mean']}° / σ {w['std']}° を基準に、操作量と揃い方を見ています（本動画内）。",
        f"mean が高い場合は“コックを作る”方向に寄りやすく、σ が大きい場合はリリースタイミングが区間内で揃っていない可能性があります。",
        f"対策は「手首を固める」ではなく、捻転差（mean {xf_mean}°）を保ったまま体の回転で下ろし、LtoLの幅を一定にすることです。",
    ])

    return {
        "title": "04. Wrist Cock（手首コック）",
        "value": f"max {w['max']} / mean {w['mean']} / σ {w['std']}（conf {raw.get('confidence')}）",
        "judge": {"main": main, "stability": stab, "confidence": confc, "x_factor_mean": xf_mean},
        "tags": tags,
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro,
    }


def build_05_head(raw: Dict[str, Any], report_id: str) -> Dict[str, Any]:
    h = raw["head_sway"]
    k = raw["knee_sway"]
    conf = float(raw.get("confidence", 0.0))

    # swayは小さいほど良い（mean基準）
    main = _cat3_small_is_good(h["mean"], 0.06, 0.15)  # good/mid/bad
    stab = _stability_cat(h["std"], 0.020, 0.050)      # 目安
    confc = _conf_cat(conf)

    tags: List[str] = []
    if main == "bad":
        tags.append("頭部ブレ大")
    if k["mean"] > 0.20:
        tags.append("下半身不安定")
    if stab == "unstable":
        tags.append("頭部ばらつき")

    good: List[str] = []
    bad: List[str] = []

    if main == "good":
        good.append(f"頭部は mean {h['mean']}（σ {h['std']}）で、左右ブレが小さく軸が安定しています。")
    if main == "mid":
        good.append(f"頭部は mean {h['mean']}（σ {h['std']}）で、平均的なブレ幅です。")
    if main == "bad":
        bad.append(f"頭部は mean {h['mean']}（σ {h['std']}）で、左右移動が大きくミート率が落ちやすい状態です。")
    if stab == "unstable":
        bad.append(f"頭部ブレのばらつき（σ {h['std']}）が大きく、インパクト位置が揃いにくい可能性があります。")
    if k["mean"] > 0.20:
        bad.append("膝側の横流れが大きく、頭部ブレを助長している可能性があります（本動画内）。")

    if not good:
        good = ["頭部の位置は大きく崩れておらず、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["頭部の安定は保てており、再現性を維持しやすい状態です。"]

    pro = _pro3([
        f"頭部は mean {h['mean']} / σ {h['std']} を基準に、軸の“移動量”と“揃い方”を見ています。",
        f"mean が高いほど軸が横に逃げやすく、σ が高いほどトップ〜インパクトの頭位置が区間内で揃っていません（本動画内）。",
        f"対策は頭を固定する意識ではなく、膝（mean {k['mean']}）の横流れを抑えて体の回転軸を作ることです。",
    ])

    return {
        "title": "05. Head Stability（頭部）",
        "value": f"max {h['max']} / mean {h['mean']} / σ {h['std']}（conf {raw.get('confidence')}）",
        "judge": {"main": main, "stability": stab, "confidence": confc},
        "tags": tags,
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro,
    }


def build_06_knee(raw: Dict[str, Any], report_id: str) -> Dict[str, Any]:
    k = raw["knee_sway"]
    h = raw["head_sway"]
    conf = float(raw.get("confidence", 0.0))

    main = _cat3_small_is_good(k["mean"], 0.10, 0.20)   # good/mid/bad
    stab = _stability_cat(k["std"], 0.025, 0.060)
    confc = _conf_cat(conf)

    tags: List[str] = []
    if main == "bad":
        tags.append("膝ブレ大")
    if h["mean"] > 0.15:
        tags.append("上半身不安定")
    if stab == "unstable":
        tags.append("下半身ばらつき")

    good: List[str] = []
    bad: List[str] = []

    if main == "good":
        good.append(f"膝は mean {k['mean']}（σ {k['std']}）で、横流れが小さく土台が安定しています。")
    if main == "mid":
        good.append(f"膝は mean {k['mean']}（σ {k['std']}）で、平均的なブレ幅です。")
    if main == "bad":
        bad.append(f"膝は mean {k['mean']}（σ {k['std']}）で、横流れが大きく体重移動が流れやすい状態です。")
    if stab == "unstable":
        bad.append(f"膝ブレのばらつき（σ {k['std']}）が大きく、踏み替えが揃いにくい可能性があります。")
    if h["mean"] > 0.15:
        bad.append("頭部側のブレが大きく、膝の安定を崩している可能性があります（本動画内）。")

    if not good:
        good = ["下半身の土台は大きく崩れておらず、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["下半身の安定は保てており、再現性を維持しやすい状態です。"]

    pro = _pro3([
        f"膝は mean {k['mean']} / σ {k['std']} を基準に、土台の横流れと揃い方を見ています。",
        f"mean が高い場合は横移動が強く、σ が高い場合はトップ〜インパクトの踏み替えが区間内で揃っていません（本動画内）。",
        f"対策は体重を“横に”移すのではなく、膝幅を保って回転で打つ（縦方向の圧を作る）ことです。",
    ])

    return {
        "title": "06. Knee Stability（膝）",
        "value": f"max {k['max']} / mean {k['mean']} / σ {k['std']}（conf {raw.get('confidence')}）",
        "judge": {"main": main, "stability": stab, "confidence": confc},
        "tags": tags,
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro,
    }


# ==================================================
# 07：要約（02〜06のタグ集計→優先度→08/09に接続）
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
        "肩回転ばらつき",
        "腰回転ばらつき",
        "手首ばらつき",
        "頭部ばらつき",
        "下半身ばらつき",
    ]
    result: List[str] = []
    for t in order:
        if tag_counter.get(t, 0) > 0:
            result.append(t)
        if len(result) >= max_items:
            break
    return result


def build_paid_07_from_analysis(analysis: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    c = collect_tag_counter(analysis)
    swing_type = judge_swing_type(c)
    priorities = extract_priorities(c, 2)

    conf = raw.get("confidence", 0.0)
    seg_n = raw.get("segment_frame_count", 0)

    lines: List[str] = []
    lines.append(f"今回のスイングは「{swing_type}」です（confidence {conf} / 区間 {seg_n} frames）。")

    if priorities:
        if len(priorities) == 1:
            lines.append(f"数値上の優先テーマは「{priorities[0]}」です。")
        else:
            lines.append("数値上の優先テーマは「" + "／".join(priorities) + "」の2点です。")
    else:
        lines.append("数値上、大きな改善テーマは見られません。")

    lines.append("08では優先テーマに直結するドリルを選択し、09では動きを安定させやすいシャフト特性を提示します。")

    return {
        "title": "07. 総合評価（プロ要約）",
        "text": lines,
        "meta": {"swing_type": swing_type, "priorities": priorities, "tag_summary": dict(c)},
    }


# ==================================================
# 08：ドリル（タグ一致で最大3つ）
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
        "tags": ["肩回転過多", "肩回転ばらつき"],
        "purpose": "回し過ぎ/ばらつきを抑え、再現性を高める",
        "how": "①ハーフスイング\n②肩の回し幅を一定に\n③10球×2セット",
    },
    {
        "id": "hip_drive",
        "name": "腰主導ターンドリル",
        "category": "下半身",
        "tags": ["腰回転不足", "腰回転ばらつき"],
        "purpose": "下半身主導の形とタイミングを揃える",
        "how": "①腰から切り返す\n②上体は我慢\n③素振り15回",
    },
    {
        "id": "release_control",
        "name": "リリース抑制ドリル（LtoL）",
        "category": "手首",
        "tags": ["コック過多", "手首ばらつき"],
        "purpose": "手首主導を抑え、体幹主導に戻す",
        "how": "①腰〜腰の振り幅\n②フェース管理重視\n③20回",
    },
    {
        "id": "head_still",
        "name": "頭固定ドリル（壁チェック）",
        "category": "安定性",
        "tags": ["頭部ブレ大", "頭部ばらつき"],
        "purpose": "スイング軸を安定させる",
        "how": "①壁の前で構える\n②頭の位置を保つ\n③素振り10回",
    },
    {
        "id": "knee_stable",
        "name": "膝ブレ抑制ドリル",
        "category": "下半身安定",
        "tags": ["膝ブレ大", "下半身ばらつき"],
        "purpose": "下半身の横流れを抑える",
        "how": "①膝幅を固定\n②体重移動を縦意識\n③10回×2",
    },
]


def collect_all_tags(analysis: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    for k in ["02", "03", "04", "05", "06"]:
        sec = analysis.get(k)
        if sec:
            tags.extend(sec.get("tags", []) or [])
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
        # fallback
        selected = [{
            "name": "テンポ安定ドリル（メトロノーム）",
            "purpose": "タイミングを一定にする",
            "how": "①一定テンポで素振り\n②10回\n③その後ボール10球",
        }]
        return selected

    return [{"name": d["name"], "purpose": d["purpose"], "how": d["how"]} for d in selected]


def build_paid_08(analysis: Dict[str, Any]) -> Dict[str, Any]:
    tags = collect_all_tags(analysis)
    drills = select_drills_by_tags(tags, 3)
    return {"title": "08. Training Drills（練習ドリル）", "drills": drills}


# ==================================================
# 09：フィッティング（指数＋任意入力）
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
    sh = raw["shoulder_rotation"]["mean"]       # 85..105
    hip = raw["hip_rotation"]["mean"]           # 36..50
    wrist = raw["wrist_cock"]["mean"]           # 70..90
    xf = sh - hip                               # 35..55

    a = _norm_range(sh, 85, 105)
    b = _norm_range(hip, 36, 50)
    c = _norm_range(wrist, 70, 90)
    d = _norm_range(xf, 35, 55)
    return int(round((a + b + c + d) / 4.0 * 100))


def calc_stability_idx(raw: Dict[str, Any]) -> int:
    head = raw["head_sway"]["mean"]             # 小さいほど良
    knee = raw["knee_sway"]["mean"]             # 小さいほど良

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

    hs = _to_float_or_none((user_inputs or {}).get("head_speed"))
    miss = _norm_miss((user_inputs or {}).get("miss_tendency"))
    gender = _norm_gender((user_inputs or {}).get("gender"))

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
        wrist_high = raw["wrist_cock"]["mean"] > 90
        head_bad = raw["head_sway"]["mean"] > 0.15
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
            "miss_tendency": (user_inputs or {}).get("miss_tendency"),
            "gender": (user_inputs or {}).get("gender"),
        },
    }


# ==================================================
# 10 Summary（有料）
# ==================================================
def build_paid_10(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "10. Summary（まとめ）",
        "text": [
            "今回の解析はトップ〜インパクト区間に限定し、max/mean/σとconfidenceで“量と質”を評価しています。",
            "次のステップは、優先テーマを2点に絞り「同じ幅・同じタイミング」を作ることです。",
            "08のドリルと09の指針を使い、再現性を段階的に上げていきましょう。",
        ],
    }


# ==================================================
# 無料 07
# ==================================================
def build_free_07(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "07. 総合評価",
        "text": [
            "本レポートでは、トップ〜インパクト区間の骨格データに基づいて評価しています。",
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

    analysis["02"] = build_02_shoulder(raw, report_id)
    analysis["03"] = build_03_hip(raw, report_id)
    analysis["04"] = build_04_wrist(raw, report_id)
    analysis["05"] = build_05_head(raw, report_id)
    analysis["06"] = build_06_knee(raw, report_id)

    analysis["07"] = build_paid_07_from_analysis(analysis, raw)
    analysis["08"] = build_paid_08(analysis)
    analysis["09"] = build_paid_09(raw, user_inputs or {})
    analysis["10"] = build_paid_10(raw)
    return analysis


# ==================================================
# Routes
# ==================================================
@app.route("/favicon.ico")
def favicon():
    # 404ノイズ削減
    return Response(status=204)


@app.route("/", methods=["GET", "POST"])
def root():
    # LINE webhook URL が "/" になっているケースでも落ちないように受ける
    if request.method == "POST":
        return webhook()
    return jsonify({"ok": True, "message": "GATE Swing Doctor API"})


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
            # 将来フォーム入力などで入る想定。無ければ空でOK。
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
