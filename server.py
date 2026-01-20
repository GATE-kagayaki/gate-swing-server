import os
import json
import math
import shutil
import traceback
import tempfile
from datetime import timedelta, datetime, timezone
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter

from linebot.models import (
    MessageEvent, 
    TextMessage, 
    VideoMessage, 
    TextSendMessage,
    QuickReply,
    QuickReplyButton,
    MessageAction
)

import stripe
from flask import Flask, request, jsonify, abort, render_template, render_template_string

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

# Firestore
db = firestore.Client()
users_ref = db.collection("users")


# ==================================================
# Free plan limit（月1回）
# ==================================================
FREE_LIMIT_PER_MONTH = 1  # ←月1回

def _month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")  # 例: "2026-01"

def can_use_free_plan(user_id: str) -> bool:
    """
    free ユーザーが今月あと何回使えるか判定する（副作用なし）
    """
    now = datetime.now(timezone.utc)
    doc_ref = users_ref.document(user_id)
    doc = doc_ref.get()
    data = doc.to_dict() or {}

    # plan が free 以外は対象外（=制限しない）
    plan = data.get("plan", "free")
    if plan != "free":
        return True

    used_month = data.get("free_used_month")
    used_count = int(data.get("free_used_count", 0))

    # 初回 or 月が変わっていたら未使用扱い
    if used_month != _month_key(now):
        used_count = 0

    return used_count < FREE_LIMIT_PER_MONTH

def increment_free_usage(user_id: str) -> None:
    """
    free ユーザーの今月利用回数を +1 する（副作用あり）
    ※ transactionで競合に強くする
    """
    now = datetime.now(timezone.utc)
    month = _month_key(now)
    doc_ref = users_ref.document(user_id)

    @firestore.transactional
    def _txn(txn: firestore.Transaction):
        snap = doc_ref.get(transaction=txn)

        # 未登録なら作って1回消費
        if not snap.exists:
            txn.set(
                doc_ref,
                {
                    "plan": "free",
                    "free_used_month": month,
                    "free_used_count": 1,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
            return

        data = snap.to_dict() or {}

        # free以外は触らない
        if data.get("plan", "free") != "free":
            return

        used_month = data.get("free_used_month")
        used_count = int(data.get("free_used_count", 0))

        # 月が変わっていたらリセット
        if used_month != month:
            used_month = month
            used_count = 0

        txn.set(
            doc_ref,
            {
                "plan": "free",
                "free_used_month": used_month,
                "free_used_count": used_count + 1,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

    txn = db.transaction()
    _txn(txn)



# ==================================================
# 開発者用：常にプレミアム扱いするLINEユーザー
# ==================================================
FORCE_PREMIUM_USER_IDS = {
    "U9b5fd7cc3faa61b33f8705d4265b0dfc",
}

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


def safe_line_reply(reply_token: str, text: str, user_id: str = None) -> None:
    try:
        # まずは通常の「返信（無料）」を試みる
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text))
    except LineBotApiError as e:
        # 返信期限(Invalid reply token)が切れた場合、user_idがあればプッシュ送信で救済
        if e.status_code == 400 and user_id:
            print(f"[INFO] ReplyToken切れのため、PushMessageで代替送信します: {user_id}")
            safe_line_push(user_id, text, force=True)
        else:
            print(f"[ERROR] LINE返信エラー: {traceback.format_exc()}")

def safe_line_push(user_id: str, text: str, force: bool = False) -> None:
    # force=True でない限り、上限対策として送信をスキップ（今まで通り）
    if not force:
        print("[INFO] LINE push skipped (上限対策):", user_id, text[:50])
        return

    try:
        line_bot_api.push_message(user_id, TextSendMessage(text=text))
        print(f"[LOG] Push送信成功: {user_id}")
    except Exception:
        print(f"[ERROR] Push送信失敗: {traceback.format_exc()}")



def make_initial_reply(report_id: str) -> str:
    url = f"{SERVICE_HOST_URL}/report/{report_id}"
    return (
        "✅ 動画を受信しました。\n"
        "AIによるスイング解析を開始します。\n\n"
        "⏱ 解析には【1〜3分程度】かかります。\n"
        "完了通知が届かない場合でも、\n"
        "1〜3分後に下記URLを再度ご確認ください。\n\n"
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


def current_month_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


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
    """
    Firestore の users/{user_id} を参照して premium 判定を行う
    ※ 強制プレミアムIDは常に True
    """
    if user_id in FORCE_PREMIUM_USER_IDS:
        return True

    doc_ref = users_ref.document(user_id)
    doc = doc_ref.get()

    # 未登録ユーザーは free として作成
    if not doc.exists:
        doc_ref.set({
            "plan": "free",
            "ticket_remaining": 0,
            "plan_expire_at": None,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
        return False

    data = doc.to_dict() or {}
    plan = data.get("plan", "free")

    # 単発/回数券
    if plan in ("single", "ticket"):
        return int(data.get("ticket_remaining", 0)) > 0

    # 月額
    if plan == "monthly":
        expire = data.get("plan_expire_at")
        if expire and expire.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc):
            return True
        return False

    # free
    return False
    
def consume_ticket_if_needed(user_id: str, report_id: str) -> None:
    """
    解析完了時に、ticket/single の残数を 1 消費する（冪等）
    - Cloud Tasks の再実行があっても二重消費しない
    - 強制プレミアムは消費しない
    """
    if user_id in FORCE_PREMIUM_USER_IDS:
        # 開発者IDは常にプレミアム扱い。消費しない。
        return

    report_ref = db.collection("reports").document(report_id)
    user_ref = users_ref.document(user_id)

    @firestore.transactional
    def _txn(txn: firestore.Transaction):
        report_snap = report_ref.get(transaction=txn)
        if not report_snap.exists:
            # レポートが無いのは想定外だが、消費はしない
            return

        report = report_snap.to_dict() or {}

        # すでに消費済みなら何もしない（冪等）
        if report.get("entitlement_consumed") is True:
            return

        # このレポートはプレミアムとして処理したか？
        # ※ report.html を触らない前提なので、レポート側の is_premium を正とする
        if not bool(report.get("is_premium", False)):
            # 無料レポートなら消費しない
            txn.set(report_ref, {"entitlement_consumed": True, "entitlement_type": "free"}, merge=True)
            return

        user_snap = user_ref.get(transaction=txn)
        if not user_snap.exists:
            # ユーザー未登録なら消費しない（プレミアム判定の整合は別途）
            txn.set(report_ref, {"entitlement_consumed": True, "entitlement_type": "unknown_user"}, merge=True)
            return

        u = user_snap.to_dict() or {}
        plan = u.get("plan", "free")

        # 月額は消費なし
        if plan == "monthly":
            txn.set(report_ref, {"entitlement_consumed": True, "entitlement_type": "monthly"}, merge=True)
            return

        # 単発/回数券は残数を1消費
        if plan in ("single", "ticket"):
            remaining = int(u.get("ticket_remaining", 0))
            if remaining <= 0:
                # 本来ここに来ない想定だが、二重送信等で起き得る
                # ここでは減らさず、レポート側に記録して冪等化だけは完了させる
                txn.set(
                    report_ref,
                    {
                        "entitlement_consumed": True,
                        "entitlement_type": plan,
                        "entitlement_error": "no_ticket_remaining",
                    },
                    merge=True,
                )
                return

            # 減算（トランザクション内で安全）
            txn.update(user_ref, {
                "ticket_remaining": remaining - 1,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
            txn.set(
                report_ref,
                {
                    "entitlement_consumed": True,
                    "entitlement_type": plan,
                },
                merge=True,
            )
            return

        # free 等は消費なし
        txn.set(report_ref, {"entitlement_consumed": True, "entitlement_type": plan}, merge=True)

    @firestore.transactional
    def _txn(txn: firestore.Transaction):
        print("[DEBUG] entitlement txn start", user_id, report_id)
        ...


    
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
# MediaPipe analysis（max/mean/std/conf）
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

            sh = angle(xy(LS), xy(RS), xy(RH))
            hip = angle(xy(LH), xy(RH), xy(LK))
            wr = angle(xy(LE), xy(LW), xy(LI))
            hd = abs(xy(NO)[0] - 0.5)
            kn = abs(xy(LK)[0] - 0.5)

            shoulders.append(float(sh))
            hips.append(float(hip))
            wrists.append(float(wr))
            heads.append(float(hd))
            knees.append(float(kn))
            x_factors.append(float(sh - abs(hip)))

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
# 02〜06：良い点／改善点
#  - 良い点は最低1行（無い場合は「良い点は特にありません。」）
#  - 改善点は無ければ「改善点は特にありません。」
#  - プロ目線：数値の言い換え中心（過度に数値列挙しない／矛盾しない／「毎回」禁止）
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

    main = "mid"
    if sh["mean"] < 85:
        main = "low"
    elif sh["mean"] > 105:
        main = "high"

    rel = "mid"
    if xf["mean"] < 35:
        rel = "low"
    elif xf["mean"] > 55:
        rel = "high"

    tags: List[str] = []
    if main == "low":
        tags.append("肩回転不足")
    if main == "high":
        tags.append("肩回転過多")
    if rel == "low":
        tags.append("捻転差不足")
    if rel == "high":
        tags.append("捻転差過多")
    return {"main": main, "related": rel, "tags": tags}


def build_paid_02_shoulder(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_shoulder(raw)
    sh = raw["shoulder"]
    xf = raw["x_factor"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    # 良い点（最低1行）
    if sh["std"] <= 10:
        good.append("肩の回し幅は揃っており、上半身の再現性は確保されています。")
    if 85 <= sh["mean"] <= 105:
        good.append("肩の回旋量は基準レンジに収まっています。")
    if xf["mean"] >= 35:
        good.append("肩と腰の差（捻転差）は確保できています。")
    if not good:
        good = ["良い点は特にありません。"]

    # 改善点
    if sh["mean"] < 85:
        bad.append(f"肩回転は mean {sh['mean']}°で不足です。")
    if sh["mean"] > 105:
        bad.append(f"肩回転は mean {sh['mean']}°で過多です。")
    if xf["mean"] < 35:
        bad.append(f"捻転差は mean {xf['mean']}°で不足です。")
    if sh["std"] > 15:
        bad.append(f"肩回転のばらつき（σ {sh['std']}°）が大きく、回旋量が揃っていません。")
    if not bad:
        bad = ["改善点は特にありません。"]

    # プロ目線（言語化）
    pro_lines: List[str] = []
    pro_lines.append("上半身は回り幅そのものより、回した量を同じ幅で再現できているかが評価軸です。")
    if sh["std"] <= 10:
        pro_lines.append("本動画では肩の回旋は同じ幅で安定して再現できています。")
    else:
        pro_lines.append("本動画では肩の回旋幅が一定せず、トップの再現性が取れていません。")

    if xf["mean"] < 35:
        pro_lines.append("捻転差が不足しているため、切り返しでエネルギーが溜まらない状態です。")
    else:
        pro_lines.append("捻転差は確保されており、切り返しに必要な準備はできています。")

    pro_lines.append("このスイングでは、主因は肩と腰の役割分担です。")

    pro_comment = " ".join(pro_lines[:3])

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

    main = "mid"
    if hip["mean"] < 36:
        main = "low"
    elif hip["mean"] > 50:
        main = "high"

    rel = "mid"
    if xf["mean"] < 35:
        rel = "low"
    elif xf["mean"] > 55:
        rel = "high"

    tags: List[str] = []
    if main == "low":
        tags.append("腰回転不足")
    if main == "high":
        tags.append("腰回転過多")
    if rel == "low":
        tags.append("捻転差不足")
    if rel == "high":
        tags.append("捻転差過多")
    return {"main": main, "related": rel, "tags": tags}


def build_paid_03_hip(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_hip(raw)
    hip = raw["hip"]
    xf = raw["x_factor"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    # 良い点（最低1行）
    if hip["std"] <= 10:
        good.append("腰の回し幅は揃っており、下半身の再現性は確保されています。")
    if 36 <= hip["mean"] <= 50:
        good.append("腰の回旋量は基準レンジに収まっています。")
    if not good:
        good = ["良い点は特にありません。"]

    # 改善点
    if hip["mean"] > 50:
        bad.append(f"腰回転は mean {hip['mean']}°で過多です。")
    if hip["mean"] < 36:
        bad.append(f"腰回転は mean {hip['mean']}°で不足です。")
    if xf["mean"] < 35:
        bad.append(f"捻転差は mean {xf['mean']}°で不足です。")
    if hip["std"] > 15:
        bad.append(f"腰回転のばらつき（σ {hip['std']}°）が大きく、回旋量が揃っていません。")
    if not bad:
        bad = ["改善点は特にありません。"]

    # プロ目線（言語化）
    pro_lines: List[str] = []
    pro_lines.append("腰は「回す量」ではなく、「肩との順序」と「回し幅の揃い方」で質が決まります。")
    if hip["mean"] > 50:
        pro_lines.append("本動画では腰が先に回る動きが強く出ています。")
    elif hip["mean"] < 36:
        pro_lines.append("本動画では下半身の回旋量が不足しています。")
    else:
        pro_lines.append("本動画では腰の回旋量は適正範囲に収まっています。")

    if hip["std"] > 15:
        pro_lines.append("腰の回転が一定せず、下半身主導の再現性が取れていません。")
    else:
        pro_lines.append("下半身の回転は安定しており、土台として機能しています。")

    pro_lines.append("このスイングでは、主因は下半身主導のタイミングです。")

    pro_comment = " ".join(pro_lines[:3])

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
    xf = raw["x_factor"]

    main = "mid"
    if w["mean"] < 70:
        main = "low"
    elif w["mean"] > 90:
        main = "high"

    rel = "mid"
    if xf["mean"] < 35:
        rel = "low"
    elif xf["mean"] > 55:
        rel = "high"

    tags: List[str] = []
    if main == "low":
        tags.append("コック不足")
    if main == "high":
        tags.append("コック過多")
    if rel == "low":
        tags.append("捻転差不足")
    return {"main": main, "related": rel, "tags": tags}


def build_paid_04_wrist(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_wrist(raw)
    w = raw["wrist"]
    xf = raw["x_factor"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    # 良い点（最低1行）
    if w["std"] <= 10:
        good.append("手元の角度変化は揃っており、インパクト条件の再現性は確保されています。")
    if 70 <= w["mean"] <= 90:
        good.append("手首コック量は基準レンジに収まっています。")
    if not good:
        good = ["良い点は特にありません。"]

    # 改善点
    if w["mean"] < 70:
        bad.append(f"手首コックは mean {w['mean']}°で不足です。")
    if w["mean"] > 90:
        bad.append(f"手首コックは mean {w['mean']}°で過多です。")
    if w["std"] > 15:
        bad.append(f"手首コックのばらつき（σ {w['std']}°）が大きく、動きが揃っていません。")
    if xf["mean"] < 35:
        bad.append(f"捻転差は mean {xf['mean']}°で不足です。")
    if not bad:
        bad = ["改善点は特にありません。"]

    # プロ目線（言語化）
    pro_lines: List[str] = []
    pro_lines.append("手元は「コック量の大小」より、体の回転に対して手元が介入し過ぎていないかが評価軸です。")
    if w["mean"] > 90:
        pro_lines.append("本動画では手首の動きが主導になっています。")
    elif w["mean"] < 70:
        pro_lines.append("本動画では手首のコック量が不足しています。")
    else:
        pro_lines.append("本動画では手首のコック量は適正です。")

    if w["std"] > 15:
        pro_lines.append("リリースのタイミングが一定せず、インパクト効率が安定していません。")
    else:
        pro_lines.append("手首の使い方は安定しており、動きは揃っています。")

    pro_lines.append("このスイングでは、主因はリリースのタイミングです。")

    pro_comment = " ".join(pro_lines[:3])

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
    k = raw["knee"]

    tags: List[str] = []
    if h["mean"] > 0.15:
        tags.append("頭部ブレ大")
    if k["mean"] > 0.20:
        tags.append("膝ブレ大")  # 07判定の整合のため head側にも付与してよい
    if k["mean"] > 0.20:
        tags.append("下半身不安定")
    return {"tags": tags}


def build_paid_05_head(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_head(raw)
    h = raw["head"]
    k = raw["knee"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    # 良い点（最低1行）：「軸が揃っている」＝stdで拾う
    if h["std"] <= 0.03:
        good.append("頭の位置は揃っており、再現性の土台はあります。")
    if h["mean"] <= 0.10:
        good.append("頭の左右ブレは抑えられており、軸は安定しています。")
    if not good:
        good = ["良い点は特にありません。"]

    # 改善点
    if h["mean"] > 0.15:
        bad.append(f"頭部ブレは mean {h['mean']}で大きく、軸が崩れています。")
    if h["std"] > 0.05:
        bad.append(f"頭部ブレのばらつき（σ {h['std']}）が大きく、位置が揃っていません。")
    if k["mean"] > 0.20:
        bad.append(f"膝ブレは mean {k['mean']}で大きく、頭部ブレを増幅させています。")
    if not bad:
        bad = ["改善点は特にありません。"]

    # プロ目線（言語化）
    pro_lines: List[str] = []
    pro_lines.append("頭部は「動いたかどうか」より、動いても同じ場所に戻れるか（軸の再現性）が評価軸です。")
    if h["mean"] > 0.15:
        pro_lines.append("本動画では頭部の左右移動が大きく出ています。")
    else:
        pro_lines.append("本動画では頭部の位置は比較的安定しています。")

    if h["std"] > 0.05:
        pro_lines.append("頭の位置が一定せず、スイング軸が安定していません。")
    else:
        pro_lines.append("頭の位置は揃っており、軸は一定です。")

    pro_lines.append("このスイングでは、主因は上半身の軸管理です。")

    pro_comment = " ".join(pro_lines[:3])

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
    h = raw["head"]

    tags: List[str] = []
    if k["mean"] > 0.20:
        tags.append("膝ブレ大")
    if h["mean"] > 0.15:
        tags.append("上半身不安定")
    return {"tags": tags}


def build_paid_06_knee(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_knee(raw)
    k = raw["knee"]
    h = raw["head"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    # 良い点（最低1行）：「揃い」をstdで拾う
    if k["std"] <= 0.04:
        good.append("膝の位置は揃っており、下半身の再現性の土台はあります。")
    if k["mean"] <= 0.12:
        good.append("膝の左右ブレは抑えられており、土台は安定しています。")
    if not good:
        good = ["良い点は特にありません。"]

    # 改善点
    if k["mean"] > 0.20:
        bad.append(f"膝ブレは mean {k['mean']}で大きく、土台が崩れています。")
    if k["std"] > 0.06:
        bad.append(f"膝ブレのばらつき（σ {k['std']}）が大きく、位置が揃っていません。")
    if h["mean"] > 0.15:
        bad.append(f"頭部ブレは mean {h['mean']}で大きく、膝ブレと同時に軸が崩れています。")
    if not bad:
        bad = ["改善点は特にありません。"]

    # プロ目線（言語化）
    pro_lines: List[str] = []
    pro_lines.append("下半身は「踏めているか」より、回転中も土台が横に流れないかが評価軸です。")
    if k["mean"] > 0.20:
        pro_lines.append("本動画では下半身の横方向の動きが大きく出ています。")
    else:
        pro_lines.append("本動画では下半身の動きは抑えられています。")

    if k["std"] > 0.06:
        pro_lines.append("膝の位置が一定せず、インパクト時の土台が不安定です。")
    else:
        pro_lines.append("膝の位置は安定しており、下半身は土台として機能しています。")

    pro_lines.append("このスイングでは、主因は下半身の安定性です。")

    pro_comment = " ".join(pro_lines[:3])

    return {
        "title": "06. Knee Stability（膝）",
        "value": _value_line(k["max"], k["mean"], k["std"], conf),
        "tags": j["tags"],
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro_comment,
    }


# ==================================================
# 07：プロ要約（パターンを1〜2増やす／初回ユーザー向けの一文を入れる）
# ==================================================
def collect_tag_counter(analysis: Dict[str, Any]) -> Counter:
    tags: List[str] = []
    for k in ["02", "03", "04", "05", "06"]:
        sec = analysis.get(k) or {}
        tags.extend(sec.get("tags", []) or [])
    return Counter(tags)


def judge_swing_type(tag_counter: Counter) -> str:
    # 追加パターン（おすすめの2つ）
    hand = tag_counter["コック過多"] + tag_counter["コック不足"]
    lower = tag_counter["腰回転過多"] + tag_counter["腰回転不足"] + tag_counter["膝ブレ大"] + tag_counter["下半身不安定"]

    # 既存の主要パターン
    if tag_counter["捻転差不足"] >= 2:
        return "体幹パワー不足型"
    if tag_counter["膝ブレ大"] + tag_counter["頭部ブレ大"] >= 2:
        return "安定性不足型"
    if tag_counter["肩回転過多"] + tag_counter["コック過多"] >= 2:
        return "操作過多型"

    # 新規（条件は控えめに）
    if hand >= 1 and (tag_counter["捻転差不足"] == 0) and (lower == 0):
        return "手元主因型"
    if lower >= 2 and (tag_counter["捻転差不足"] == 0):
        return "下半身主因型"

    return "バランス型"


def extract_priorities(tag_counter: Counter, max_items: int = 2) -> List[str]:
    order = [
        "捻転差不足",
        "膝ブレ大",
        "頭部ブレ大",
        "コック過多",
        "コック不足",
        "腰回転過多",
        "腰回転不足",
        "肩回転過多",
        "肩回転不足",
        "捻転差過多",
    ]
    result: List[str] = []
    for t in order:
        if tag_counter.get(t, 0) > 0:
            if t not in result:
                result.append(t)
        if len(result) >= max_items:
            break
    return result


def _summary_template(swing_type: str) -> List[str]:
    # 07の「型」別テンプレ（短め・具体・余計な主張はしない）
    if swing_type == "体幹パワー不足型":
        return [
            "回転量を増やすことではなく、肩と腰の動き出しの順序が結果を左右しています。",
            "捻転差が小さい状態は、切り返しで“溜め”が残らず、加速が手元に寄りやすくなります。",
        ]
    if swing_type == "安定性不足型":
        return [
            "最大の課題は回転量ではなく、土台と軸が保てているかです。",
            "軸が揺れる状態は、打点とフェース向きの再現性を同時に落とします。",
        ]
    if swing_type == "操作過多型":
        return [
            "スイングの主役が体幹よりも手元側に寄りやすい状態です。",
            "操作が増えると、方向と打点のズレが連動して大きくなります。",
        ]
    if swing_type == "手元主因型":
        return [
            "体の回転よりも、手元の角度変化が結果に強く影響しています。",
            "手元の介入度が高いほど、フェース管理が難しくなりミス幅が広がります。",
        ]
    if swing_type == "下半身主因型":
        return [
            "回転量そのものより、下半身がどの順序で動いているかが質を分けます。",
            "下半身の土台が崩れると、上半身が補正に回り、操作が増えやすくなります。",
        ]
    # バランス型
    return [
        "大きな破綻が少なく、テーマを絞って改善を積み上げやすい状態です。",
        "「最優先テーマ」だけに集中すると、変化が最も出やすくなります。",
    ]


def build_paid_07_from_analysis(analysis: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    c = collect_tag_counter(analysis)
    swing_type = judge_swing_type(c)
    priorities = extract_priorities(c, 2)

    conf = _conf(raw)
    frames = _frames(raw)

    lines: List[str] = []
    lines.append(f"今回のスイングは「{swing_type}」です（confidence {conf:.3f} / 区間 {frames} frames）。")
    lines.append("※ 初回の方は、今回は「最優先テーマ」だけを確認してください。")
    lines.append("")

    # 型の説明（2文）
    lines.extend(_summary_template(swing_type))
    lines.append("")

    # 優先テーマ（最大2つ）
    if priorities:
        if len(priorities) == 1:
            lines.append(f"数値上の最優先テーマは「{priorities[0]}」です。")
        else:
            lines.append("数値上の優先テーマは「" + "／".join(priorities) + "」の2点です。")
    else:
        lines.append("数値上の優先テーマはありません。")

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
        },
    }


def build_free_07(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    無料版の07は「数値に基づく総合評価（プロ目線）」までは出す。
    ただし、02〜06の部位別の深掘り・改善手順・ドリル選定は有料に残す。
    """

    # --- 数値取得 ---
    sh = raw.get("shoulder", {})  # degrees
    hip = raw.get("hip", {})      # degrees
    w = raw.get("wrist", {})      # degrees
    head = raw.get("head", {})    # sway
    knee = raw.get("knee", {})    # sway
    xf = raw.get("x_factor", {})  # degrees
    conf = float(raw.get("confidence", 0.0))
    frames = int(raw.get("valid_frames", 0))

    # --- 無料版用に「タグ」をrawから推定（既存judge_*の閾値と整合） ---
    tags: List[str] = []

    # 肩回転
    sh_mean = float(sh.get("mean", 0.0))
    sh_std = float(sh.get("std", 0.0))
    if sh_mean < 85:
        tags.append("肩回転不足")
    elif sh_mean > 105:
        tags.append("肩回転過多")

    # 腰回転
    hip_mean = float(hip.get("mean", 0.0))
    hip_std = float(hip.get("std", 0.0))
    if hip_mean < 36:
        tags.append("腰回転不足")
    elif hip_mean > 50:
        tags.append("腰回転過多")

    # 手首コック
    w_mean = float(w.get("mean", 0.0))
    w_std = float(w.get("std", 0.0))
    if w_mean < 70:
        tags.append("コック不足")
    elif w_mean > 90:
        tags.append("コック過多")

    # 捻転差
    xf_mean = float(xf.get("mean", 0.0))
    if xf_mean < 35:
        tags.append("捻転差不足")
    elif xf_mean > 55:
        tags.append("捻転差過多")

    # 安定性
    head_mean = float(head.get("mean", 0.0))
    knee_mean = float(knee.get("mean", 0.0))
    if head_mean > 0.15:
        tags.append("頭部ブレ大")
    if knee_mean > 0.20:
        tags.append("膝ブレ大")
        tags.append("下半身不安定")

    # --- 既存の総合ロジックを流用（型分類・優先順位） ---
    c = Counter(tags)
    swing_type = judge_swing_type(c)
    priorities = extract_priorities(c, 2)

    # --- プロ目線文章（無料版の完成形） ---
    lines: List[str] = []
    lines.append(f"今回のスイングは「{swing_type}」です（confidence {conf:.3f} / 区間 {frames} frames）。")

    # 優先テーマ（最大2つ）
    if priorities:
        if len(priorities) == 1:
            lines.append(f"数値上の最優先テーマは「{priorities[0]}」です。")
        else:
            lines.append("数値上の優先テーマは「" + "／".join(priorities) + "」の2点です。")
    else:
        lines.append("数値上の優先テーマはありません。")

    lines.append("")

    # 優先テーマの根拠（数値で断定）
    # ※無料は「原因分解」や「手順」まで言わない。現象と影響だけ言い切る。
    if "頭部ブレ大" in priorities or ("頭部ブレ大" in c and len(priorities) == 0):
        lines.append(f"本動画では頭部ブレが mean {head_mean:.4f} で大きく、軸が安定しにくい状態です。")
    if "膝ブレ大" in priorities or ("膝ブレ大" in c and len(priorities) == 0):
        lines.append(f"本動画では膝ブレが mean {knee_mean:.4f} で大きく、下半身の土台が崩れています。")

    if "捻転差不足" in priorities:
        lines.append(f"本動画では捻転差が mean {xf_mean:.2f}°で小さく、切り返しの準備が不足しています。")
    if "腰回転過多" in priorities:
        lines.append(f"本動画では腰回転が mean {hip_mean:.2f}°で大きく、下半身の主張が強い状態です。")
    if "肩回転過多" in priorities:
        lines.append(f"本動画では肩回転が mean {sh_mean:.2f}°で大きく、上半身が回り過ぎています。")
    if "コック過多" in priorities:
        lines.append(f"本動画では手首コックが mean {w_mean:.2f}°で大きく、手元の介入が強い状態です。")

    lines.append("")

    # できている点（必ず入れる）
    good_points: List[str] = []
    if 85 <= sh_mean <= 105:
        good_points.append("肩の回旋量は基準レンジに収まっています。")
    if sh_std <= 15:
        good_points.append("肩の回し幅は大きく崩れておらず、上半身の再現性の土台はあります。")
    if head_mean <= 0.15:
        good_points.append("頭部ブレは大きくはなく、軸は破綻していません。")
    if knee_mean <= 0.20:
        good_points.append("膝ブレは上限を超えておらず、下半身は大きく流れていません。")
    if xf_mean >= 35:
        good_points.append("捻転差は確保できており、切り返しの準備はできています。")

    if good_points:
        lines.append("良い点： " + " ".join(good_points[:2]))
    else:
        lines.append("良い点： 大きな破綻は見られません。")

    lines.append("")
    lines.append("有料版では、部位別評価（02〜06）で主因を特定し、総合評価の精度を上げた上で、練習ドリルとフィッティング指針まで提示します。")

    return {
        "title": "07. 総合評価（無料版：プロ目線）",
        "text": lines,
        "meta": {
            "swing_type": swing_type,
            "priorities": priorities,
            "tag_summary": dict(c),
            "confidence": conf,
            "frames": frames,
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
    return jsonify(
        {
            "ok": True,
            "project_id": PROJECT_ID,
            "queue_location": QUEUE_LOCATION,
            "queue_name": QUEUE_NAME,
            "service_host_url": SERVICE_HOST_URL,
            "task_handler_url": TASK_HANDLER_URL,
            "task_sa_email_set": bool(TASK_SA_EMAIL),
        }
    )

# ==================================================
# Stripe Checkout 作成
# ==================================================
import stripe
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

# Stripeからコピーした新しい署名シークレットを反映
endpoint_secret = "whsec_dZAi4sELzWVwKECvIAUdZ8Jd8QMQhrsw"

@app.route('/stripe/webhook', methods=['POST'])
def stripe_webhook():
    # 生データを確実に取得するため get_data() を使用します
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        # ここで「本物のStripeからの通知か」を署名検証します
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except stripe.error.SignatureVerificationError as e:
        print(f"⚠️ 署名検証に失敗しました: {e}")
        return 'Invalid signature', 400
    except Exception as e:
        print(f"⚠️ エラーが発生しました: {e}")
        return 'Error', 400

    # 支払い完了イベント（checkout.session.completed）の処理
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        
        # ★【最重要】ホームページから引き継がれるIDを取得
        line_user_id = session.get('client_reference_id')

        if line_user_id:
            # 1. Firestoreのユーザー情報を更新（チケット付与）
            user_ref = db.collection('users').document(line_user_id)
            user_ref.set({
                'ticket_remaining': firestore.Increment(1),
                'last_payment_date': firestore.SERVER_TIMESTAMP
            }, merge=True)
            print(f"✅ Firestore更新成功: {line_user_id}")

            # 2. 決済した本人にLINEでお礼メッセージを送信
            try:
                line_bot_api.push_message(
                    line_user_id,
                    TextSendMessage(text="決済を確認しました！⛳️\nこのままスイング動画を送ってください。AI解析を開始します。")
                )
                print(f"✅ LINEメッセージ送信成功: {line_user_id}")
            except Exception as e:
                print(f"⚠️ LINE送信失敗: {e}")
        else:
            print("⚠️ 警告: client_reference_id が空です")

    return jsonify(success=True)
    
def handle_successful_payment(user_id: str, plan: str):
    """
    Firestoreのユーザー権限をプランに応じて更新する
    """
    doc_ref = db.collection("users").document(user_id)
    now = datetime.now(timezone.utc)

    if plan == "single":
        # 1回券：残り回数を +1
        doc_ref.update({
            "plan": "single",
            "ticket_remaining": firestore.Increment(1),
            "updated_at": firestore.SERVER_TIMESTAMP
        })
    elif plan == "ticket":
        # 5回券：残り回数を +5
        doc_ref.update({
            "plan": "ticket",
            "ticket_remaining": firestore.Increment(5),
            "updated_at": firestore.SERVER_TIMESTAMP
        })
    elif plan == "monthly":
        # 月額プラン：期限を30日後に設定
        from datetime import timedelta
        expire_at = now + timedelta(days=30)
        doc_ref.update({
            "plan": "monthly",
            "plan_expire_at": expire_at,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
    print(f"[DB_UPDATE] User {user_id} の権限を {plan} に更新しました。")

@app.route("/stripe/checkout", methods=["POST"])
def stripe_checkout():
    data = request.get_json(silent=True) or {}

    line_user_id = data.get("line_user_id")
    plan = data.get("plan")  # "single" / "ticket" / "monthly"

    # 1. バリデーション
    if not stripe.api_key:
        return jsonify({"error": "STRIPE_SECRET_KEY is not set"}), 500
    if not line_user_id or plan not in ("single", "ticket", "monthly"):
        return jsonify({"error": "invalid request"}), 400

    # 2. 価格IDの取得（前後スペースを除去する .strip() を追加して安全性を向上）
    price_map = {
        "single": os.environ.get("STRIPE_PRICE_SINGLE", "").strip(),
        "ticket": os.environ.get("STRIPE_PRICE_TICKET", "").strip(),
        "monthly": os.environ.get("STRIPE_PRICE_MONTHLY", "").strip(),
    }
    price_id = price_map.get(plan, "")
    
    if not price_id:
        return jsonify({"error": f"price_id not set for plan={plan}"}), 500

    # 3. 支払いモードの判定（重要！）
    # 月額プランなら 'subscription'、それ以外（単発・回数券）なら 'payment'
    checkout_mode = "subscription" if plan == "monthly" else "payment"

    success_url = os.environ.get("STRIPE_SUCCESS_URL", SERVICE_HOST_URL)
    cancel_url = os.environ.get("STRIPE_CANCEL_URL", SERVICE_HOST_URL)

    # 4. Stripe セッション作成
    try:
        session = stripe.checkout.Session.create(
        mode=checkout_mode,
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        client_reference_id=line_user_id, # LINE ID
        # --- ここを追加：Webhookでプランを判別するために必須 ---
        metadata={
            "plan": plan,             # "single", "ticket", "monthly"
            "line_user_id": line_user_id
        },
        # --------------------------------------------------
        success_url=success_url,
        cancel_url=cancel_url,
    )
        return jsonify({"checkout_url": session.url}), 200

    except Exception as e:
        print(f"[ERROR] Stripe Session Create Failed: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# LINEのWebhook URLが /webhook 以外でも落ちないように受け口を複数用意
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

    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()
    user_data = user_doc.to_dict() if user_doc.exists else {}
    tickets = user_data.get('ticket_remaining', 0)

    force_paid_report = is_premium_user(user_id) or tickets > 0
    if not is_premium_user(user_id) and tickets > 0:
        user_ref.update({'ticket_remaining': firestore.Increment(-1)})

    # 【重要】URLエラーを防ぐため、先に保存を完了させる
    firestore_safe_set(report_id, {
        "user_id": user_id,
        "status": "PROCESSING",
        "is_premium": force_paid_report,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_inputs": {},
    })
   
    try:
        # メッセージを組み立て
        base_message = (
            "動画を正常に受け付けました！⛳️\n"
            "AI解析を開始します。1～3分ほどで完了します。\n"
            f"解析状況はこちら：\nhttps://gate-golf.com/mypage/?id={report_id}"
        )

        # 解析タスクの作成（これが失敗しても返信は届くように try 内の最後に置くか検討）
        task_name = create_cloud_task(report_id, user_id, msg.id)
        firestore_safe_update(report_id, {"task_name": task_name})

        if force_paid_report:
            fitting_intro = "\n\n09フィッティング解析のため、現在の「ヘッドスピード」「主なミスの傾向」「性別（任意）」を教えてください。"
            instruction = "\n\n【1/3】まずは「ヘッドスピード」を数字（例：42）だけで送ってください。"
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=f"{base_message}{fitting_intro}{instruction}")
            )
        else:
            increment_free_usage(user_id)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=base_message))

    except Exception as e:
        print(f"[ERROR] {traceback.format_exc()}")
        # エラーが起きてもユーザーに状況を伝える
        safe_line_reply(event.reply_token, "動画は受け取りましたが、解析の予約に失敗しました。事務局へお問い合わせください。", user_id=user_id)


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    # 文字の整理と「料金プラン」の優先判定（リッチメニュー対策）
    text = event.message.text.strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    user_id = event.source.user_id

    if "料金プラン" in text:
        plan_text = (
            "【GATE 料金プラン】⛳️\n\n"
            "🔹1回券: 500円(税込)\nhttps://buy.stripe.com/00w28sdezc5A8lR2ej18c00\n\n"
            "🔹回数券: 1,980円(税込)\nhttps://buy.stripe.com/fZucN66QbfhM6dJ7yD18c03\n\n"
            "🔹月額プラン: 4,980円(税込)\nhttps://buy.stripe.com/3cIfZi2zVd9E1XtdX118c05"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=plan_text))
        return

    # インデックスエラーを回避するため、まず全取得してから最新の1件を特定
    docs = db.collection('reports').where('user_id', '==', user_id).get()

    if docs:
        # 作成日時が一番新しいレポートを選ぶ
        latest_report = max(docs, key=lambda d: d.to_dict().get('created_at', ''))
        report_ref = latest_report.reference
        
        # 数字（HS）の保存
        if text.isdigit():
            val = int(text)
            if 10 <= val <= 70:
                report_ref.update({"user_inputs.head_speed": val})
                items = [
                    QuickReplyButton(action=MessageAction(label="スライス/右", text="ミス：スライス")),
                    QuickReplyButton(action=MessageAction(label="フック/左", text="ミス：フック")),
                    QuickReplyButton(action=MessageAction(label="特に無し", text="ミス：無し")),
                ]
                line_bot_api.reply_message(
                    event.reply_token, 
                    TextSendMessage(text=f"HS {val}m/s で保存しました。\n\n【2/3】次に「主なミスの傾向」を選択してください。", quick_reply=QuickReply(items=items))
                )
                return

        # ミスの傾向
        elif "ミス：" in text:
            val = text.replace("ミス：", "")
            report_ref.update({"user_inputs.miss_tendency": val})
            items = [
                QuickReplyButton(action=MessageAction(label="男性", text="性別：男性")),
                QuickReplyButton(action=MessageAction(label="女性", text="性別：女性")),
                QuickReplyButton(action=MessageAction(label="回答しない", text="性別：none"))
            ]
            line_bot_api.reply_message(
                event.reply_token, 
                TextSendMessage(text="【3/3】最後に「性別」を教えてください（任意）。", quick_reply=QuickReply(items=items))
            )
            return

        # 性別
        elif "性別：" in text:
            val = text.replace("性別：", "")
            report_ref.update({"user_inputs.gender": val})
            line_bot_api.reply_message(
                event.reply_token, 
                TextSendMessage(text="ありがとうございます。情報を解析に反映します！完成まで今しばらくお待ちください。⛳️")
            )
            return
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
