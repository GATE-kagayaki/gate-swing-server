import os
import json
import math
import shutil
import traceback
import tempfile
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


def _safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _safe_std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    m = _safe_mean(xs)
    v = sum((x - m) ** 2 for x in xs) / max(1, (len(xs) - 1))
    return float(math.sqrt(v))


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
# MediaPipe analysis（max/mean/std/conf + 順序判定用 series）
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

    shoulders: List[float] = []
    hips: List[float] = []
    wrists: List[float] = []
    heads: List[float] = []
    knees: List[float] = []
    x_factors: List[float] = []

    # 順序判定用（フレーム系列）
    sh_series: List[float] = []
    hip_series: List[float] = []

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

            lm = res.pose_landmarks.landmark
            valid_frames += 1

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

            sh = float(angle(xy(LS), xy(RS), xy(RH)))
            hip = float(angle(xy(LH), xy(RH), xy(LK)))
            wr = float(angle(xy(LE), xy(LW), xy(LI)))
            hd = float(abs(xy(NO)[0] - 0.5))
            kn = float(abs(xy(LK)[0] - 0.5))

            shoulders.append(sh)
            hips.append(hip)
            wrists.append(wr)
            heads.append(hd)
            knees.append(kn)
            x_factors.append(float(sh - abs(hip)))

            sh_series.append(sh)
            hip_series.append(abs(hip))

    cap.release()

    if total_frames < 10 or valid_frames < 5:
        raise RuntimeError("解析に必要なフレーム数が不足しています。")

    conf = float(valid_frames) / float(total_frames)

    def pack(xs: List[float], nd: int = 2) -> Dict[str, float]:
        if not xs:
            return {"max": 0.0, "mean": 0.0, "std": 0.0}
        return {
            "max": round(float(max(xs)), nd),
            "mean": round(float(_safe_mean(xs)), nd),
            "std": round(float(_safe_std(xs)), nd),
        }

    def _peak_velocity_index(series: List[float]) -> int:
        # 速度 = 隣接差分。最大の変化点（絶対値）を「動き出しが強い箇所」として扱う
        if len(series) < 3:
            return 0
        v = [abs(series[i] - series[i - 1]) for i in range(1, len(series))]
        # ごく端のノイズ回避：最初と最後の1/10を除外して探索（最低限のガード）
        n = len(v)
        lo = max(0, int(n * 0.1))
        hi = min(n, int(n * 0.9))
        if hi - lo < 5:
            lo, hi = 0, n
        best_i = lo
        best_val = -1.0
        for i in range(lo, hi):
            if v[i] > best_val:
                best_val = v[i]
                best_i = i
        return best_i  # vのindex（= series側では best_i+1）

    sh_i = _peak_velocity_index(sh_series)
    hip_i = _peak_velocity_index(hip_series)
    # 何フレーム差で「先行」とみなすか（短い動画でも破綻しにくい閾値）
    lead_thr = 3

    if hip_i + lead_thr < sh_i:
        sequence = "hip_first"
    elif sh_i + lead_thr < hip_i:
        sequence = "shoulder_first"
    else:
        sequence = "sync"

    return {
        "frame_count": int(total_frames),
        "valid_frames": int(valid_frames),
        "confidence": round(conf, 3),

        "shoulder": pack(shoulders, 2),
        "hip": pack(hips, 2),
        "wrist": pack(wrists, 2),
        "head": pack(heads, 4),
        "knee": pack(knees, 4),
        "x_factor": pack(x_factors, 2),

        # 07用メタ
        "sequence": {
            "type": sequence,          # hip_first / shoulder_first / sync
            "shoulder_peak_i": int(sh_i),
            "hip_peak_i": int(hip_i),
            "threshold_frames": int(lead_thr),
        },
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
                "value": raw.get("frame_count", 0),
                "description": "動画から解析できたフレーム数です。",
                "guide": "150〜300 フレーム",
            },
            {
                "name": "有効フレーム数 / 信頼度",
                "value": f'{raw.get("valid_frames", 0)} / {raw.get("confidence", 0):.3f}',
                "description": "骨格推定が取れているフレーム数と、その比率です。",
                "guide": "conf 0.70以上が目安",
            },
            {
                "name": "肩回転（°）",
                "value": f'max {raw["shoulder"]["max"]} / mean {raw["shoulder"]["mean"]} / σ {raw["shoulder"]["std"]}',
                "description": "上半身の回旋量です（本動画内の統計）。",
                "guide": "比較は同条件で",
            },
            {
                "name": "腰回転（°）",
                "value": f'max {raw["hip"]["max"]} / mean {raw["hip"]["mean"]} / σ {raw["hip"]["std"]}',
                "description": "下半身の回旋量です（本動画内の統計）。",
                "guide": "比較は同条件で",
            },
            {
                "name": "手首コック（°）",
                "value": f'max {raw["wrist"]["max"]} / mean {raw["wrist"]["mean"]} / σ {raw["wrist"]["std"]}',
                "description": "手首角の統計です（本動画内）。",
                "guide": "比較は同条件で",
            },
            {
                "name": "頭部ブレ（Sway）",
                "value": f'max {raw["head"]["max"]} / mean {raw["head"]["mean"]} / σ {raw["head"]["std"]}',
                "description": "頭の左右ブレ量です（本動画内）。",
                "guide": "小さいほど安定",
            },
            {
                "name": "膝ブレ（Sway）",
                "value": f'max {raw["knee"]["max"]} / mean {raw["knee"]["mean"]} / σ {raw["knee"]["std"]}',
                "description": "膝の左右ブレ量です（本動画内）。",
                "guide": "小さいほど安定",
            },
        ],
    }


# ==================================================
# 02〜06：良い点／改善点（無ければ「特にありません」）
# ＋プロ目線（矛盾なし／数値の繰り返しは最小限／言語化を厚く）
# ==================================================
def _conf(raw: Dict[str, Any]) -> float:
    return float(raw.get("confidence", 0.0))


def _frames(raw: Dict[str, Any]) -> int:
    return int(raw.get("valid_frames", 0))


def _value_line(maxv: float, meanv: float, stdv: float, conf: float) -> str:
    return f"max {maxv} / mean {meanv} / σ {stdv}（conf {conf:.3f}）"


def judge_shoulder(raw: Dict[str, Any]) -> Dict[str, Any]:
    sh = raw["shoulder"]
    xf = raw["x_factor"]

    tags: List[str] = []
    if sh["mean"] < 85:
        tags.append("肩回転不足")
    elif sh["mean"] > 105:
        tags.append("肩回転過多")

    if xf["mean"] < 35:
        tags.append("捻転差不足")
    elif xf["mean"] > 55:
        tags.append("捻転差過多")

    if sh["std"] > 15:
        tags.append("肩回転バラつき大")

    return {"tags": tags}


def build_paid_02_shoulder(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_shoulder(raw)
    sh = raw["shoulder"]
    xf = raw["x_factor"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    if 85 <= sh["mean"] <= 105:
        good.append(f"肩回転は mean {sh['mean']}°で、量は基準レンジです。")
    if sh["std"] <= 12:
        good.append(f"肩回転のばらつき（σ {sh['std']}°）は抑えられており、上半身の回旋は揃っています。")
    if xf["mean"] >= 35:
        good.append(f"捻転差は mean {xf['mean']}°で、肩と腰の差は確保されています。")

    if sh["mean"] < 85:
        bad.append(f"肩回転は mean {sh['mean']}°で不足です。")
    if sh["mean"] > 105:
        bad.append(f"肩回転は mean {sh['mean']}°で過多です。")
    if xf["mean"] < 35:
        bad.append(f"捻転差は mean {xf['mean']}°で不足です。")
    if sh["std"] > 15:
        bad.append(f"肩回転のばらつき（σ {sh['std']}°）が大きく、回旋量が揃っていません。")

    if not good:
        good = ["良い点は特にありません。"]
    if not bad:
        bad = ["改善点は特にありません。"]

    pro_comment = (
        "上半身は回り幅そのものより、回した量を同じ幅で再現できているかが評価軸です。 "
        "捻転差が不足している場合は、肩が回っているのに“溜め”が残らず、切り返しで加速の材料が作れません。 "
        "ばらつきが大きい場合は、同じトップを作れていないため、インパクトの再現性が落ちます。"
        if xf["mean"] < 35 or sh["std"] > 15
        else
        "上半身は基準レンジで、回し過ぎ・不足のどちらにも寄っていません。 "
        "捻転差が確保できているため、切り返しで“溜め”を作る土台があります。 "
        "この区間では、上半身が主因でスイングが崩れる状態ではありません。"
    )

    return {
        "title": "02. Shoulder Rotation（肩回転）",
        "value": _value_line(sh["max"], sh["mean"], sh["std"], conf),
        "tags": j["tags"],
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro_comment,
    }


def judge_hip(raw: Dict[str, Any]) -> Dict[str, Any]:
    hip = raw["hip"]
    xf = raw["x_factor"]

    tags: List[str] = []
    if hip["mean"] < 36:
        tags.append("腰回転不足")
    elif hip["mean"] > 50:
        tags.append("腰回転過多")

    if xf["mean"] < 35:
        tags.append("捻転差不足")
    elif xf["mean"] > 55:
        tags.append("捻転差過多")

    if hip["std"] > 15:
        tags.append("腰回転バラつき大")

    return {"tags": tags}


def build_paid_03_hip(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_hip(raw)
    hip = raw["hip"]
    xf = raw["x_factor"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    if 36 <= hip["mean"] <= 50:
        good.append(f"腰回転は mean {hip['mean']}°で、量は基準レンジです。")
    if hip["std"] <= 12:
        good.append(f"腰回転のばらつき（σ {hip['std']}°）は抑えられており、下半身の回旋は揃っています。")

    if hip["mean"] > 50:
        bad.append(f"腰回転は mean {hip['mean']}°で過多です。")
    if hip["mean"] < 36:
        bad.append(f"腰回転は mean {hip['mean']}°で不足です。")
    if xf["mean"] < 35:
        bad.append(f"捻転差は mean {xf['mean']}°で不足です。")
    if hip["std"] > 15:
        bad.append(f"腰回転のばらつき（σ {hip['std']}°）が大きく、回旋量が揃っていません。")

    if not good:
        good = ["良い点は特にありません。"]
    if not bad:
        bad = ["改善点は特にありません。"]

    # 「毎回」禁止 → 「本動画内」表現に統一
    if hip["mean"] > 50 or hip["std"] > 15:
        pro_comment = (
            "腰は回転量そのものより、切り返し前後で“同じ回し幅”を保てているかが質の評価になります。 "
            "ばらつきが大きい場合は、本動画内で腰の回し始め・回し幅が一定になっておらず、上体が先にほどける原因になります。 "
            "捻転差が不足している場合は、腰と肩が同じタイミングで動いてしまい、下半身主導の形が作れません。"
        )
    else:
        pro_comment = (
            "腰回転は基準レンジで、下半身主導の土台があります。 "
            "ばらつきが抑えられているため、本動画内で回し幅の再現性も確保できています。 "
            "この区間では、腰の動きが原因で大きく崩れる状態ではありません。"
        )

    return {
        "title": "03. Hip Rotation（腰回転）",
        "value": _value_line(hip["max"], hip["mean"], hip["std"], conf),
        "tags": j["tags"],
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro_comment,
    }


def judge_wrist(raw: Dict[str, Any]) -> Dict[str, Any]:
    w = raw["wrist"]
    tags: List[str] = []
    if w["mean"] < 70:
        tags.append("コック不足")
    elif w["mean"] > 90:
        tags.append("コック過多")
    if w["std"] > 15:
        tags.append("手首バラつき大")
    return {"tags": tags}


def build_paid_04_wrist(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_wrist(raw)
    w = raw["wrist"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    if 70 <= w["mean"] <= 90:
        good.append(f"手首コックは mean {w['mean']}°で、量は基準レンジです。")
    if w["std"] <= 12:
        good.append(f"手首コックのばらつき（σ {w['std']}°）は抑えられており、形は揃っています。")

    if w["mean"] < 70:
        bad.append(f"手首コックは mean {w['mean']}°で不足です。")
    if w["mean"] > 90:
        bad.append(f"手首コックは mean {w['mean']}°で過多です。")
    if w["std"] > 15:
        bad.append(f"手首コックのばらつき（σ {w['std']}°）が大きく、形が揃っていません。")

    if not good:
        good = ["良い点は特にありません。"]
    if not bad:
        bad = ["改善点は特にありません。"]

    if w["mean"] > 90 or w["std"] > 15:
        pro_comment = (
            "手首が主役になると、体の回転よりもフェース操作で当てにいく割合が増えます。 "
            "ばらつきが大きい場合は、本動画内で“同じ手首の形”を作れていないため、インパクトの再現性が落ちます。 "
            "手首は量を増やすより、トップで作った角度を崩さずに体の回転で運べているかが評価ポイントです。"
        )
    else:
        pro_comment = (
            "手首の量は基準レンジで、形も揃っています。 "
            "この区間では、手首操作が原因でミスを増やす状態ではありません。 "
            "手首は現状のまま、体幹と下半身の動きに優先度を置けます。"
        )

    return {
        "title": "04. Wrist Cock（手首コック）",
        "value": _value_line(w["max"], w["mean"], w["std"], conf),
        "tags": j["tags"],
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro_comment,
    }


def judge_head(raw: Dict[str, Any]) -> Dict[str, Any]:
    h = raw["head"]
    tags: List[str] = []
    if h["mean"] > 0.15:
        tags.append("頭部ブレ大")
    if h["std"] > 0.05:
        tags.append("頭位置バラつき大")
    return {"tags": tags}


def build_paid_05_head(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_head(raw)
    h = raw["head"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    if h["mean"] <= 0.10 and h["std"] <= 0.03:
        good.append(f"頭部ブレは mean {h['mean']}で小さく、軸は安定しています。")

    if h["mean"] > 0.15:
        bad.append(f"頭部ブレは mean {h['mean']}で大きく、軸が崩れています。")
    if h["std"] > 0.05:
        bad.append(f"頭部ブレのばらつき（σ {h['std']}）が大きく、位置が揃っていません。")

    if not good:
        good = ["良い点は特にありません。"]
    if not bad:
        bad = ["改善点は特にありません。"]

    if h["mean"] > 0.15 or h["std"] > 0.05:
        pro_comment = (
            "頭は“動かないこと”が正解ではなく、動いたとしても同じ量・同じ方向に収まることが安定の条件です。 "
            "本動画内でブレ量が大きい場合は、回転や体重移動の中で軸が逃げており、打点・入射角が一定になりません。 "
            "頭部は結果の指標で、原因は膝や骨盤の横流れにあることが多いため、同時に下半身の安定も確認します。"
        )
    else:
        pro_comment = (
            "頭部は安定しており、軸がスイング中に大きく逃げていません。 "
            "この区間では、頭の動きが直接ミスを増やす状態ではありません。 "
            "安定している指標なので、他の優先テーマに集中できます。"
        )

    return {
        "title": "05. Head Stability（頭部）",
        "value": _value_line(h["max"], h["mean"], h["std"], conf),
        "tags": j["tags"],
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro_comment,
    }


def judge_knee(raw: Dict[str, Any]) -> Dict[str, Any]:
    k = raw["knee"]
    tags: List[str] = []
    if k["mean"] > 0.20:
        tags.append("膝ブレ大")
    if k["std"] > 0.06:
        tags.append("膝位置バラつき大")
    return {"tags": tags}


def build_paid_06_knee(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_knee(raw)
    k = raw["knee"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    if k["mean"] <= 0.12 and k["std"] <= 0.04:
        good.append(f"膝ブレは mean {k['mean']}で小さく、下半身は安定しています。")

    if k["mean"] > 0.20:
        bad.append(f"膝ブレは mean {k['mean']}で大きく、土台が崩れています。")
    if k["std"] > 0.06:
        bad.append(f"膝ブレのばらつき（σ {k['std']}）が大きく、位置が揃っていません。")

    if not good:
        good = ["良い点は特にありません。"]
    if not bad:
        bad = ["改善点は特にありません。"]

    if k["mean"] > 0.20 or k["std"] > 0.06:
        pro_comment = (
            "膝は“回すための支点”で、ここが横に流れると腰の回転が回転ではなくスライドになります。 "
            "本動画内でブレが大きい場合は、回転の順序以前に土台が崩れており、再現性が落ちる原因になります。 "
            "膝の安定が出るだけで、腰→肩の順序も作りやすくなります。"
        )
    else:
        pro_comment = (
            "膝の安定が確保できており、土台が崩れてスイングが破綻する状態ではありません。 "
            "下半身が安定しているため、回転の順序と捻転差を作る作業に移れます。 "
            "この区間では、膝は強みとして扱えます。"
        )

    return {
        "title": "06. Knee Stability（膝）",
        "value": _value_line(k["max"], k["mean"], k["std"], conf),
        "tags": j["tags"],
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro_comment,
    }


# ==================================================
# 07：総合評価（5パターン以上 + 順序（sequence）反映 + 具体性）
# ==================================================
def collect_tag_counter(analysis: Dict[str, Any]) -> Counter:
    tags: List[str] = []
    for k in ["02", "03", "04", "05", "06"]:
        sec = analysis.get(k) or {}
        tags.extend(sec.get("tags", []) or [])
    return Counter(tags)


def _sequence_label(raw: Dict[str, Any]) -> str:
    seq = (raw.get("sequence") or {}).get("type")
    if seq == "hip_first":
        return "順序：腰→肩（下半身先行）"
    if seq == "shoulder_first":
        return "順序：肩→腰（上半身先行）"
    return "順序：同調（同時に動く傾向）"


def judge_swing_type_v2(tag_counter: Counter, raw: Dict[str, Any]) -> str:
    """
    5パターン以上に拡張。
    ※タグ（結果）と sequence（順序）を両方使って分類する。
    """
    seq = (raw.get("sequence") or {}).get("type")

    # 安定性
    if tag_counter["膝ブレ大"] + tag_counter["頭部ブレ大"] >= 1:
        if tag_counter["膝ブレ大"] + tag_counter["頭部ブレ大"] >= 2:
            return "安定性不足型"
        return "安定性注意型"

    # 操作系（手首/肩の過多とバラつき）
    if tag_counter["コック過多"] + tag_counter["手首バラつき大"] >= 2:
        return "手首主導（操作過多）型"
    if tag_counter["肩回転過多"] + tag_counter["肩回転バラつき大"] >= 2:
        return "上半身先行（開き）型"

    # 体幹パワー（捻転差）
    if tag_counter["捻転差不足"] >= 1:
        # 順序が同調/肩先行なら「溜め不足」に寄せる
        if seq in ["sync", "shoulder_first"]:
            return "体幹パワー不足型"
        # 腰先行でも捻転差不足なら「下半身が回るが溜めが残らない」
        return "下半身先行だが溜め不足型"

    # 下半身不足
    if tag_counter["腰回転不足"] >= 1:
        return "下半身主導不足型"

    # デフォルト
    if seq == "shoulder_first":
        return "上半身先行（開き）型"
    if seq == "sync":
        return "同調回転型"
    return "バランス型"


def extract_priorities(tag_counter: Counter, max_items: int = 2) -> List[str]:
    order = [
        "捻転差不足",
        "膝ブレ大",
        "頭部ブレ大",
        "コック過多",
        "手首バラつき大",
        "腰回転不足",
        "腰回転過多",
        "肩回転過多",
        "肩回転不足",
        "肩回転バラつき大",
        "捻転差過多",
        "膝位置バラつき大",
        "頭位置バラつき大",
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
    swing_type = judge_swing_type_v2(c, raw)
    priorities = extract_priorities(c, 2)

    conf = _conf(raw)
    frames = _frames(raw)
    seq_text = _sequence_label(raw)

    # 具体性：タイプ別に「何が起きているか」を1〜2行で言語化（毎回NG → 本動画内）
    type_detail = ""
    if swing_type == "安定性不足型":
        type_detail = "本動画内では土台（膝・軸）のブレが大きく、回転の良し悪し以前に再現性が落ちています。"
    elif swing_type == "安定性注意型":
        type_detail = "本動画内では安定性指標に弱点があり、回転の順序を作っても結果が揺れやすい状態です。"
    elif swing_type == "手首主導（操作過多）型":
        type_detail = "本動画内では体の回転より手首の形で当てにいく割合が高く、フェース挙動が結果を左右します。"
    elif swing_type == "上半身先行（開き）型":
        type_detail = "本動画内では肩が先に動きやすく、切り返しで上体がほどけて球が散る方向に寄ります。"
    elif swing_type == "体幹パワー不足型":
        type_detail = "本動画内では捻転差が残らず、切り返しで加速の材料（溜め）を作れていません。"
    elif swing_type == "下半身先行だが溜め不足型":
        type_detail = "腰が先に動けても、肩との差が残らないため“下半身主導の質”が完成していません。"
    elif swing_type == "下半身主導不足型":
        type_detail = "下半身の回転量が不足し、上半身と手元でスイングを成立させる比率が高い状態です。"
    elif swing_type == "同調回転型":
        type_detail = "肩と腰が同時に動きやすく、溜めを作るよりも一体で回って当てるタイプです。"
    else:
        type_detail = "大きな破綻は少なく、優先テーマを絞ると伸びやすい状態です。"

    lines: List[str] = []
    lines.append(f"今回のスイングは「{swing_type}」です（confidence {conf:.3f} / 区間 {frames} frames）。")
    lines.append(seq_text)
    lines.append(type_detail)
    lines.append("")
    if priorities:
        if len(priorities) == 1:
            lines.append(f"優先テーマは「{priorities[0]}」です。")
        else:
            lines.append("優先テーマは「" + "／".join(priorities) + "」の2点です。")
    else:
        lines.append("優先テーマはありません。")
    lines.append("")
    lines.append("08では優先テーマに直結するドリルを選択し、09では動きを安定させやすいシャフト特性を提示します。")

    return {
        "title": "07. 総合評価（プロ要約）",
        "text": lines,
        "meta": {
            "swing_type": swing_type,
            "priorities": priorities,
            "tag_summary": dict(c),
            "confidence": conf,
            "frames": frames,
            "sequence": raw.get("sequence", {}),
        },
    }


# ==================================================
# 08 ドリル（現状維持）
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
        "tags": ["捻転差不足"],
        "purpose": "体全体で回る感覚を作る",
        "how": "①腕を胸の前でクロス\n②胸と腰を同時に回す\n③左右10回",
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
        selected = [DRILL_DEFINITIONS[0]]

    return [{"name": d["name"], "purpose": d["purpose"], "how": d["how"]} for d in selected]


def build_paid_08(analysis: Dict[str, Any]) -> Dict[str, Any]:
    tags = collect_all_tags(analysis)
    drills = select_drills_by_tags(tags, 3)
    return {"title": "08. Training Drills（練習ドリル）", "drills": drills}


# ==================================================
# 09 フィッティング（現状維持）
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
    sh = float(raw["shoulder"]["mean"])
    hip = float(abs(raw["hip"]["mean"]))
    wrist = float(raw["wrist"]["mean"])
    xf = float(raw["x_factor"]["mean"])

    a = _norm_range(sh, 85, 105)
    b = _norm_range(hip, 36, 50)
    c = _norm_range(wrist, 70, 90)
    d = _norm_range(xf, 36, 55)
    return int(round((a + b + c + d) / 4.0 * 100))


def calc_stability_idx(raw: Dict[str, Any]) -> int:
    head = float(raw["head"]["mean"])
    knee = float(raw["knee"]["mean"])

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
        wrist_high = float(raw["wrist"]["mean"]) > 90
        head_bad = float(raw["head"]["mean"]) > 0.15
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
# 10 まとめ（現状維持）
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


def build_free_07(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "07. 総合評価",
        "text": [
            "本レポートでは、スイング全体の傾向を骨格データに基づいて評価しています。",
            "有料版では、部位別評価・練習ドリル・フィッティング指針まで含めて提示します。",
        ],
    }


# ==================================================
# Analysis builder
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

    analysis["07"] = build_paid_07_from_analysis(analysis, raw)
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


@app.route("/", methods=["POST"])
def webhook_root_alias():
    return webhook()


@app.route("/callback", methods=["POST"])
def webhook_callback_alias():
    return webhook()


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
    except Exception:
        firestore_safe_update(report_id, {"status": "TASK_FAILED", "error": "create_task_failed"})
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
