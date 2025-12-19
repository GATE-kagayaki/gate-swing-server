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


def format_3lines(lines: List[str]) -> str:
    # report.html 側は innerHTML で差し込まれるので <br> で改行できる
    lines = [x.strip() for x in lines if str(x).strip()]
    if len(lines) >= 3:
        return "<br>".join(lines[:3])
    if len(lines) == 2:
        return "<br>".join(lines + ["次の1点だけ絞って直すと、結果が最短で変わります。"])
    if len(lines) == 1:
        return "<br>".join(lines + ["この数値は“癖”ではなく“傾向”です。", "まずは同じ幅・同じテンポを優先してください。"])
    return "数値は安定しています。<br>大きな修正は不要です。<br>同じ幅・同じテンポの維持が最優先です。"


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
# MediaPipe analysis
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

            max_shoulder = max(max_shoulder, angle(xy(LS), xy(RS), xy(RH)))
            min_hip = min(min_hip, angle(xy(LH), xy(RH), xy(LK)))
            max_wrist = max(max_wrist, angle(xy(LE), xy(LW), xy(LI)))
            max_head = max(max_head, abs(xy(NO)[0] - 0.5))
            max_knee = max(max_knee, abs(xy(LK)[0] - 0.5))

    cap.release()

    if frame_count < 10:
        raise RuntimeError("解析に必要なフレーム数が不足しています。")

    return {
        "frame_count": int(frame_count),
        "max_shoulder_rotation": round(float(max_shoulder), 2),
        "min_hip_rotation": round(float(min_hip), 2),
        "max_wrist_cock": round(float(max_wrist), 2),
        "max_head_drift": round(float(max_head), 4),
        "max_knee_sway": round(float(max_knee), 4),
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
                "description": "スイング中に手首が最も折れた角度です。手先の介入量（主導の強さ）の指標になります。",
                "guide": "120〜150°（目安）",
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
# 02 肩：3×3×3 判定＋非定型文
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


SHOULDER_PRO_TEXT: Dict[Tuple[str, str], List[List[str]]] = {
    ("low", "low"): [[
        "最大肩回転角は{sh}°、捻転差は{xf}°です。",
        "上半身でエネルギーを作れておらず、切り返しで溜めが残りません。",
        "肩を回す意識ではなく「腰との差を作る」動きを最優先してください。",
    ]],
    ("low", "mid"): [[
        "肩回転量は{sh}°と控えめですが、捻転差{xf}°は確保されています。",
        "量を増やすより、トップの“止まり”を作って回転タイミングを揃える方が結果が安定します。",
        "狙いは回転量アップではなく、毎回同じ幅を出すことです。",
    ]],
    ("low", "high"): [[
        "肩回転{sh}°が少ない一方で捻転差{xf}°が大きい状態です。",
        "腰が止まりすぎて、肩だけで帳尻を合わせています。",
        "腰の回転を“自然に入れる”だけで同期が取れ、ミスが減ります。",
    ]],
    ("mid", "low"): [[
        "肩回転量{sh}°は目安内ですが、捻転差{xf}°が不足しています。",
        "肩と腰が同時に動き、切り返しで溜めが消えています。",
        "腰を一拍遅らせて差を作ると、同じ力でも飛びと方向が揃います。",
    ]],
    ("mid", "mid"): [[
        "肩回転{sh}°と捻転差{xf}°はいずれも目安レンジ内です。",
        "上半身の回旋は完成度が高く、ここは“変えないこと”が正解です。",
        "余計な意識を入れず、テンポ固定で再現性を伸ばしてください。",
    ]],
    ("mid", "high"): [[
        "肩回転{sh}°は目安内ですが、捻転差{xf}°が大きい状態です。",
        "腰が止まり、上体だけが深く入って突っ込みを作りやすくなります。",
        "腰を止めずに回して差を適正化すると、当たり方が一段安定します。",
    ]],
    ("high", "low"): [[
        "肩回転{sh}°は大きいのに、捻転差{xf}°が小さい状態です。",
        "腰も同時に回り、回転が“量だけ”になってタイミングがズレやすいです。",
        "切り返しで腰を一拍遅らせて差を作ると、同じ回転量でも曲がりが減ります。",
    ]],
    ("high", "mid"): [[
        "肩回転{sh}°は大きく、パワーを出せる状態です。",
        "ただし回し過ぎは再現性を落とすので、狙いは“量を増やす”ではありません。",
        "毎回同じ回し幅に揃えるだけで、結果が一気にまとまります。",
    ]],
    ("high", "high"): [[
        "肩回転{sh}°と捻転差{xf}°がどちらも大きく、出力は十分です。",
        "一方で回し過ぎはタイミングズレを生み、ミスの幅が広がります。",
        "量より“同じ幅”を優先すると、強さを残したまま安定します。",
    ]],
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
        bad.append(f"最大肩回転角{sh}°が小さく、上半身で出力を作れていません。")
    if judge["main"] == "high":
        bad.append(f"最大肩回転角{sh}°が大きく、回転量がブレやすい状態です。")
    if judge["x_factor"] == "low":
        bad.append(f"捻転差{xf}°が不足しており、肩と腰が同時に動いています。")
    if judge["x_factor"] == "high":
        bad.append(f"捻転差{xf}°が大きく、腰が止まりすぎて上体が先行しています。")

    if not good:
        good = ["上半身の動きに大きな破綻はなく、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の回旋は安定しており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_shoulder_pro(judge: Dict[str, Any], raw: Dict[str, Any], seed: str) -> str:
    key = (judge["main"], judge["x_factor"])
    blocks = SHOULDER_PRO_TEXT.get(key) or [[
        "肩の回旋は大きな問題は見られません。",
        "数値のブレが少ない状態です。",
        "テンポ固定で維持してください。",
    ]]
    rnd = random.Random(seed + "_shoulder")
    lines = rnd.choice(blocks)
    return format_3lines([x.format(sh=raw["max_shoulder_rotation"], xf=judge["x_factor_value"]) for x in lines])


def build_paid_02_shoulder(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    judge = judge_shoulder(raw)
    good, bad = shoulder_good_bad(judge, raw)
    pro = generate_shoulder_pro(judge, raw, seed)
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
# 03 腰：3×3×3 判定＋非定型文
# ==================================================
def judge_hip(raw: Dict[str, Any]) -> Dict[str, Any]:
    hip = abs(raw["min_hip_rotation"])
    shoulder = raw["max_shoulder_rotation"]
    frame = raw["frame_count"]

    if hip < 36:
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


HIP_PRO_TEXT: Dict[Tuple[str, str], List[List[str]]] = {
    ("low", "low"): [[
        "腰回転量{hip}°、捻転差{xf}°です。",
        "下半身が使えておらず、切り返しで溜めも作れていません。",
        "腰の回転量を確保することが最優先です。",
    ]],
    ("low", "mid"): [[
        "腰回転{hip}°は控えめですが、捻転差{xf}°は確保されています。",
        "腰を止めすぎているだけなので、自然に回すだけで再現性が上がります。",
        "狙いは“腰を速く回す”ではなく“止めない”ことです。",
    ]],
    ("low", "high"): [[
        "腰回転{hip}°が少ないのに捻転差{xf}°が大きい状態です。",
        "腰が止まり、上体だけで合わせているので突っ込みが出やすいです。",
        "腰の回転を入れて同期を取ると、ミート率が安定します。",
    ]],
    ("mid", "low"): [[
        "腰回転{hip}°は適正ですが、捻転差{xf}°が不足しています。",
        "腰と肩が同調しすぎており、切り返しで“タメ”が残りません。",
        "腰を一拍遅らせるだけで、同じ力でも飛距離効率が上がります。",
    ]],
    ("mid", "mid"): [[
        "腰回転{hip}°と捻転差{xf}°はともに目安レンジ内です。",
        "下半身主導の形ができており、大きな修正は不要です。",
        "今の土台を崩さず、テンポと幅の固定に集中してください。",
    ]],
    ("mid", "high"): [[
        "腰回転{hip}°は適正ですが、捻転差{xf}°が大きい状態です。",
        "腰が止まって上体が先行し、被りや突っ込みを作ります。",
        "腰を止めずに回して差を整えると、方向性が揃います。",
    ]],
    ("high", "low"): [[
        "腰回転{hip}°が大きいのに捻転差{xf}°が小さい状態です。",
        "肩も同時に動いており、溜めが作れず“回るだけ”になっています。",
        "切り返しで腰を一拍遅らせると、同じ回転でも安定します。",
    ]],
    ("high", "mid"): [[
        "腰回転{hip}°は大きく、下半身主導は作れています。",
        "ただし回り過ぎは上体の開きを誘発し、ミスの幅が広がります。",
        "回転量は増やさず“同じ幅”に揃えることが正解です。",
    ]],
    ("high", "high"): [[
        "腰回転{hip}°と捻転差{xf}°がどちらも大きく、出力は十分です。",
        "回り過ぎはタイミングズレを生み、当たり方が散ります。",
        "量より“同じ幅”を優先すると、強さを残したまま安定します。",
    ]],
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
        bad.append(f"腰回転量{hip}°が小さく、下半身の推進力を活かし切れていません。")
    if judge["main"] == "high":
        bad.append(f"腰回転量{hip}°が大きく、上体が先に開きやすい状態です。")
    if judge["x_factor"] == "low":
        bad.append(f"捻転差{xf}°が不足しており、肩と腰が同時に動いています。")
    if judge["x_factor"] == "high":
        bad.append(f"捻転差{xf}°が大きく、腰が止まり上体が先行しています。")

    if not good:
        good = ["下半身の動きに大きな破綻はなく、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の下半身は安定しており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_hip_pro(judge: Dict[str, Any], raw: Dict[str, Any], seed: str) -> str:
    key = (judge["main"], judge["x_factor"])
    blocks = HIP_PRO_TEXT.get(key) or [[
        "腰の回転動作に大きな問題は見られません。",
        "下半身の数値は安定しています。",
        "テンポと幅の固定を優先してください。",
    ]]
    rnd = random.Random(seed + "_hip")
    lines = rnd.choice(blocks)
    return format_3lines([x.format(hip=abs(raw["min_hip_rotation"]), xf=judge["x_factor_value"]) for x in lines])


def build_paid_03_hip(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    judge = judge_hip(raw)
    good, bad = hip_good_bad(judge, raw)
    pro = generate_hip_pro(judge, raw, seed)
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
# 04 手首：主指標＋関連指標＋信頼度（矛盾なし）
# ==================================================
def judge_wrist(raw: Dict[str, Any]) -> Dict[str, Any]:
    wrist = raw["max_wrist_cock"]               # 実測は 0〜180 近くまで出る
    shoulder = raw["max_shoulder_rotation"]
    hip = abs(raw["min_hip_rotation"])
    frame = raw["frame_count"]

    if wrist < 120:
        main = "low"
    elif wrist > 150:
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


WRIST_PRO_TEXT: Dict[Tuple[str, str], List[List[str]]] = {
    ("low", "low"): [[
        "コック角{w}°、捻転差{xf}°です。",
        "体幹で溜めを作れておらず、手元で合わせる動きになっています。",
        "最優先は捻転差の確保で、手首は“作る”ではなく“入る”形に戻してください。",
    ]],
    ("low", "mid"): [[
        "コック角{w}°は少なめですが、捻転差{xf}°は確保されています。",
        "手首を増やす意識は不要で、回転で自然に入る形にするほど再現性が上がります。",
        "狙いはコック量アップではなく、トップ〜切り返しのテンポ固定です。",
    ]],
    ("low", "high"): [[
        "コック角{w}°が少ない一方、捻転差{xf}°は大きい状態です。",
        "腰が止まりすぎて上体が深く入り、手首が入り切らずに当たりが薄くなります。",
        "腰を止めずに回して同期を取ると、手首は勝手に収まります。",
    ]],
    ("mid", "low"): [[
        "コック角{w}°は目安内ですが、捻転差{xf}°が不足しています。",
        "体幹が使えていないため、インパクトで手首が暴れやすい土台です。",
        "捻転差を作る動きに戻すと、手首は“操作しなくても”安定します。",
    ]],
    ("mid", "mid"): [[
        "コック角{w}°と捻転差{xf}°はいずれも目安レンジ内です。",
        "手首は余計な意識を入れない方が良く、現状維持が正解です。",
        "テンポ固定だけで、当たりと方向がさらに揃います。",
    ]],
    ("mid", "high"): [[
        "コック角{w}°は目安内ですが、捻転差{xf}°が大きい状態です。",
        "腰が止まって上体が先行し、結果的に手首で合わせる場面が増えます。",
        "腰を止めずに回して差を整えると、手首の介入が減ります。",
    ]],
    ("high", "low"): [[
        "コック角{w}°が大きく、捻転差{xf}°も不足しています。",
        "体幹ではなく手先でスピードを作っており、再現性が崩れます。",
        "手首を抑えるより先に、捻転差を作って体幹主導に戻してください。",
    ]],
    ("high", "mid"): [[
        "コック角{w}°が大きく、手首主導が数値に出ています。",
        "この状態はタイミング依存になり、ミスが日替わりで出ます。",
        "狙いは“手首を止める”ではなく、回転で振って手首の介入を減らすことです。",
    ]],
    ("high", "high"): [[
        "コック角{w}°と捻転差{xf}°がどちらも大きい状態です。",
        "出力は出せますが、手先が入りやすくタイミングズレの幅が大きくなります。",
        "回し幅を揃えて“同じトップ”を作ると、手首の暴れが一気に減ります。",
    ]],
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
        bad.append(f"コック角{w}°が小さく、溜めが作れていません。")
    if judge["main"] == "high":
        bad.append(f"コック角{w}°が大きく、手首主導が数値として出ています。")
    if judge["related"] == "low":
        bad.append(f"捻転差{xf}°が不足しており、体幹より手先が先行しています。")
    if judge["related"] == "high":
        bad.append(f"捻転差{xf}°が大きく、腰が止まりやすく手首で合わせやすい土台です。")

    if not good:
        good = ["手首の動きに大きな破綻はなく、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の手首操作は安定しており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_wrist_pro(judge: Dict[str, Any], raw: Dict[str, Any], seed: str) -> str:
    key = (judge["main"], judge["related"])
    blocks = WRIST_PRO_TEXT.get(key) or [[
        "手首の数値は安定しています。",
        "大きな操作は見られません。",
        "テンポと幅の固定を優先してください。",
    ]]
    rnd = random.Random(seed + "_wrist")
    lines = rnd.choice(blocks)
    return format_3lines([x.format(w=raw["max_wrist_cock"], xf=judge["x_factor_value"]) for x in lines])


def build_paid_04_wrist(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    judge = judge_wrist(raw)
    good, bad = wrist_good_bad(judge, raw)
    pro = generate_wrist_pro(judge, raw, seed)
    return {
        "title": "04. Wrist Cock（手首コック）",
        "value": raw["max_wrist_cock"],
        "judge": judge,
        "tags": judge["tags"],
        "good": good,
        "bad": bad,
        "pro_comment": pro,
    }


# ==================================================
# 05 頭部：主指標＋関連指標＋信頼度（3行）
# ==================================================
def judge_head(raw: Dict[str, Any]) -> Dict[str, Any]:
    h = raw["max_head_drift"]
    knee = raw["max_knee_sway"]
    frame = raw["frame_count"]

    # 小さいほど良い：low=良 / mid=普通 / high=悪
    if h < 0.06:
        main = "low"
    elif h > 0.15:
        main = "high"
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


HEAD_PRO_TEXT: Dict[Tuple[str, str], List[List[str]]] = {
    ("low", "low"): [[
        "頭部ブレは小さく、軸は安定しています。",
        "この状態なら当たり負けが起きにくく、方向性が揃います。",
        "余計な意識を入れず、テンポ固定で維持してください。",
    ]],
    ("low", "mid"): [[
        "頭部は安定しています。",
        "次に揃えるべきは下半身で、そこが整うとミスの幅がさらに縮みます。",
        "頭はそのまま、膝の横流れだけを止めてください。",
    ]],
    ("mid", "high"): [[
        "頭部ブレは平均域ですが、膝の流れが頭を引っ張っています。",
        "下半身が横に流れると、上体は必ず追従して軸がズレます。",
        "膝の横流れを止めるだけで、頭は自然に落ち着きます。",
    ]],
    ("high", "mid"): [[
        "頭部ブレが大きく、ミート率が落ちる数値です。",
        "膝は崩れていないので、原因は上体の左右移動に絞れます。",
        "頭の位置を固定し、回転で振る形に戻してください。",
    ]],
    ("high", "high"): [[
        "頭と膝が同時に流れています。",
        "この組み合わせは軸が毎回ズレるので、当たりも方向も散ります。",
        "最優先は下半身の横流れを止めて、頭を同じ位置に残すことです。",
    ]],
}


def head_good_bad(judge: Dict[str, Any], raw: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    h = raw["max_head_drift"]
    good: List[str] = []
    bad: List[str] = []

    if judge["main"] == "low":
        good.append(f"頭部ブレ{h}は小さく、スイング軸が安定しています。")
    if judge["main"] == "mid":
        good.append(f"頭部ブレ{h}は平均的で、大きく崩れる動きは見られません。")
    if judge["main"] == "high":
        bad.append(f"頭部ブレ{h}が大きく、インパクトの再現性が落ちています。")

    if judge["related"] == "high":
        bad.append("膝の安定性が低く、頭部ブレを助長しています。")
    if judge["related"] == "low":
        good.append("下半身の土台が安定しているため、頭の安定を作りやすい状態です。")

    if not good:
        good = ["頭部の位置は大きく崩れておらず、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["頭部の安定は保てており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_head_pro(judge: Dict[str, Any], raw: Dict[str, Any], seed: str) -> str:
    key = (judge["main"], judge["related"])
    blocks = HEAD_PRO_TEXT.get(key) or [[
        "頭部の動きは概ね安定しています。",
        "大きな崩れは見られません。",
        "テンポと幅の固定を優先してください。",
    ]]
    rnd = random.Random(seed + "_head")
    lines = rnd.choice(blocks)
    return format_3lines([x.format(h=raw["max_head_drift"], k=raw["max_knee_sway"]) for x in lines])


def build_paid_05_head(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    judge = judge_head(raw)
    good, bad = head_good_bad(judge, raw)
    pro = generate_head_pro(judge, raw, seed)
    return {
        "title": "05. Head Stability（頭部）",
        "value": raw["max_head_drift"],
        "judge": judge,
        "tags": judge["tags"],
        "good": good,
        "bad": bad,
        "pro_comment": pro,
    }


# ==================================================
# 06 膝：主指標＋関連指標＋信頼度（3行）
# ==================================================
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


KNEE_PRO_TEXT: Dict[Tuple[str, str], List[List[str]]] = {
    ("low", "low"): [[
        "膝ブレが小さく、下半身の土台は完成度が高いです。",
        "この土台があると、上体の回転が素直に乗って方向性が揃います。",
        "今は“強くする”より“同じ幅”の維持を優先してください。",
    ]],
    ("low", "mid"): [[
        "膝は安定しています。",
        "頭部のブレを揃えると、ミート率と方向性がさらにまとまります。",
        "膝は維持し、頭の位置だけを同じ場所に残してください。",
    ]],
    ("mid", "high"): [[
        "膝ブレは平均域ですが、頭の流れが膝を引っ張っています。",
        "上体が左右に動くと、下半身も連動して横流れが増えます。",
        "頭の左右移動を止めるだけで、膝も自然に安定します。",
    ]],
    ("high", "mid"): [[
        "膝ブレが大きく、体重移動が横流れになっています。",
        "この状態は回転が止まりやすく、手先で合わせる場面が増えます。",
        "最優先は膝幅の固定で、縦の踏み替えに戻してください。",
    ]],
    ("high", "high"): [[
        "膝と頭が同時に流れています。",
        "軸が毎回ズレるので、当たりも方向も散る数値です。",
        "まず膝の横流れを止め、頭を同じ位置に残すことが最短です。",
    ]],
}


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
        bad.append("上半身の左右移動が膝ブレを助長しています。")
    if judge["related"] == "low":
        good.append("頭部が安定しているため、膝の安定を作りやすい状態です。")

    if not good:
        good = ["下半身の土台は大きく崩れておらず、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["下半身の安定は保てており、再現性を維持しやすい状態です。"]

    return good[:3], bad[:3]


def generate_knee_pro(judge: Dict[str, Any], raw: Dict[str, Any], seed: str) -> str:
    key = (judge["main"], judge["related"])
    blocks = KNEE_PRO_TEXT.get(key) or [[
        "膝の安定性は概ね保てています。",
        "大きな崩れは見られません。",
        "テンポと幅の固定を優先してください。",
    ]]
    rnd = random.Random(seed + "_knee")
    lines = rnd.choice(blocks)
    return format_3lines([x.format(k=raw["max_knee_sway"], h=raw["max_head_drift"]) for x in lines])


def build_paid_06_knee(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    judge = judge_knee(raw)
    good, bad = knee_good_bad(judge, raw)
    pro = generate_knee_pro(judge, raw, seed)
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
            lines.append(f"数値上、最優先の改善点は「{priorities[0]}」です。")
        else:
            lines.append("数値上、最優先の改善点は「" + "／".join(priorities) + "」の2点です。")
    else:
        lines.append("数値上、大きな改善テーマは見られません。")
    lines.append("08はこの優先テーマに直結するドリルだけを選択しています。")
    lines.append("09はこの動きを“安定させやすい”シャフト特性を指針として提示しています。")

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
        "tags": ["コック過多"],
        "purpose": "手首主導を抑え、体幹主導に戻す",
        "how": "①腰〜腰の振り幅\n②手先で合わせず回転で動かす\n③一定リズムで20回",
    },
    {
        "id": "late_hit",
        "name": "レイトヒットドリル（タメづくり）",
        "category": "手首",
        "tags": ["コック不足"],
        "purpose": "タメを作り、インパクト効率を上げる",
        "how": "①トップで一瞬止める\n②体の回転で振る\n③連続素振り10回",
    },
    {
        "id": "head_still",
        "name": "頭固定ドリル（壁チェック）",
        "category": "安定性",
        "tags": ["頭部ブレ大"],
        "purpose": "頭の左右移動を止め、軸を安定させる",
        "how": "①壁の前でアドレス\n②頭と壁の距離を一定に\n③素振り10回",
    },
    {
        "id": "knee_stable",
        "name": "膝ブレ抑制ドリル",
        "category": "下半身",
        "tags": ["膝ブレ大"],
        "purpose": "下半身の横流れを止め、回転の土台を作る",
        "how": "①膝幅を固定\n②踏み替えは縦を意識\n③10回×2セット",
    },
    {
        "id": "sync_turn",
        "name": "全身同調ターンドリル（クロスアーム）",
        "category": "体幹",
        "tags": ["体幹主導不足", "捻転差不足"],
        "purpose": "上半身だけが先行する動きを抑え、体全体で回る感覚を作る",
        "how": "①腕を胸の前でクロス\n②胸と腰を同時に回す\n③左右10回",
    },
    {
        "id": "tempo",
        "name": "テンポ安定ドリル（メトロノーム）",
        "category": "リズム",
        "tags": ["再現性不足"],
        "purpose": "タイミングを一定にして再現性を上げる",
        "how": "①一定テンポで素振り\n②10回\n③その後ボール10球",
    },
    {
        "id": "balance",
        "name": "バランスチェックドリル",
        "category": "安定性",
        "tags": ["下半身不安定", "上半身不安定"],
        "purpose": "軸と体重配分を整える",
        "how": "①片足立ちでゆっくり素振り\n②左右5回\n③倒れるなら強度を下げる",
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
    wrist = raw["max_wrist_cock"]              # 120..150（目安）
    xf = sh - hip                              # 35..55

    a = _norm_range(sh, 85, 105)
    b = _norm_range(hip, 36, 50)
    c = _norm_range(wrist, 120, 150)
    d = _norm_range(xf, 35, 55)
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
            reason = f"入力が無いため指数で判定します。パワー指数{power_idx}では軽めが適正です。"
        elif band == "mid":
            weight = "50〜60g"
            reason = f"入力が無いため指数で判定します。パワー指数{power_idx}では標準帯が適正です。"
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
        wrist_high = raw["max_wrist_cock"] > 150
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
            # 任意入力（将来LINEの別フローで入る想定。無ければ空）
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
