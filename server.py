import os
os.environ["MP_DEVICE"] = "cpu"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import json
import math
import shutil
import traceback
import tempfile
from datetime import timedelta, datetime, timezone
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter
import logging

from linebot.models import (
    MessageEvent, 
    TextMessage,     
    VideoMessage, 
    TextSendMessage,
    QuickReply,
    QuickReplyButton,
    MessageAction
)

from flask import Flask, request, jsonify, abort, render_template, render_template_string, redirect

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError

from google.cloud import firestore
from google.cloud import tasks_v2
from google.api_core.exceptions import NotFound, PermissionDenied

from google.cloud import storage   # ★追加（動画アップロード用）
import google.auth
from google.auth.transport.requests import Request

import stripe
# --- Stripe設定 ---
# 本番環境では環境変数から取得することを推奨します
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
def call_llm(prompt: str) -> str:
    try:
        res = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {
                    "role": "system",
                    "content": "あなたはプロのゴルフコーチです。寄り添い型で、初心者にも分かる言葉で説明してください。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
        )
        return res.choices[0].message.content
    except Exception as e:
        # エラーが起きたらログに残し、ユーザーには定型文を返す
        logging.error(f"LLM Error: {e}")
        return "（現在、詳細なコメントを生成できません。ドリルを確認して練習を進めてみましょう！）"

def get_cancel_portal_url(customer_id: str):
    """
    ユーザー専用の解約・管理ポータルURLを生成
    """
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url="https://gate-golf.com/mypage" 
        )
        return session.url
    except Exception as e:
        logging.error(f"Portal Error: {e}")
        return None

app = Flask(__name__, template_folder="templates")
app.config["JSON_AS_ASCII"] = False

line_bot_api = LineBotApi(os.environ.get('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.environ.get('LINE_CHANNEL_SECRET'))

def get_stripe_secrets():
    stripe_key = os.environ.get("STRIPE_SECRET_KEY")
    endpoint_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

    if not stripe_key:
        raise RuntimeError("STRIPE_SECRET_KEY is not set")

    if not endpoint_secret:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET is not set")

    stripe.api_key = stripe_key
    return stripe_key, endpoint_secret



@app.route("/webhook", methods=['POST'])
def callback():
    import json, traceback
    from flask import request

    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)

    # ★ここ：printで必ず出す
    try:
        payload = json.loads(body)
        events = payload.get("events", [])
        print(f"[LINE] events_count={len(events)} types={[e.get('type') for e in events]}", flush=True)
        print(f"[LINE] event_ids={[e.get('webhookEventId') or (e.get('message') or {}).get('id') for e in events]}", flush=True)
    except Exception as e:
        print(f"[LINE] payload parse error: {e}", flush=True)

    
    try:
        handler.handle(body, signature)
    except Exception as e:
        print(f"!!! Webhook Error !!!: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return 'Internal Error', 500

    return 'OK'



db = firestore.Client()
users_ref = db.collection("users")

def reply_quick_start(reply_token: str):
    line_bot_api.reply_message(
        reply_token,
        TextSendMessage(
            text="【任意】分かる範囲で選んでください（スキップ可）",
            quick_reply=QuickReply(items=[
                QuickReplyButton(action=MessageAction(label="HS", text="HS")),
                QuickReplyButton(action=MessageAction(label="ミス傾向", text="ミス傾向")),
                QuickReplyButton(action=MessageAction(label="性別", text="性別")),
                QuickReplyButton(action=MessageAction(label="スキップ", text="スキップ")),
            ])
        )
    )




# ==================================================
# CONFIG
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")


QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")
QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")

SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")

TASK_HANDLER_PATH = "/task-handler"
TASK_HANDLER_URL = f"{SERVICE_HOST_URL}{TASK_HANDLER_PATH}"

# Firestore
FIRESTORE_DB = os.environ.get("FIRESTORE_DB", "(default)")

from google.cloud import firestore

db = firestore.Client()
users_ref = db.collection("users")


print(
    f"[BOOT] GOOGLE_CLOUD_PROJECT={os.environ.get('GOOGLE_CLOUD_PROJECT')} "
    f"PROJECT_ID={PROJECT_ID} firestore_db={FIRESTORE_DB}",
    flush=True
)


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
    "Ud8a58146e64e232efe3f94681d17d8ff",  
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


def safe_line_reply(reply_token: str, text: str, user_id: str = None, msgid: str = None) -> None:
    try:
        print(f"[SEND] reply msgid={msgid} reply={reply_token} text_head={text[:12]!r}", flush=True)
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text))
    except LineBotApiError as e:
        print(f"[ERR ] reply failed msgid={msgid} status={getattr(e,'status_code',None)} text_head={text[:12]!r}", flush=True)

        # Invalid reply token のときだけ push で救済
        if getattr(e, "status_code", None) == 400 and user_id:
            print(f"[SEND] fallback-push msgid={msgid} user={user_id} text_head={text[:12]!r}", flush=True)
            safe_line_push(user_id, text, force=True)


def safe_line_push(user_id: str, text: str, force: bool = False) -> None:
    if not force:
        print(f"[INFO] push skipped user={user_id} text_head={text[:12]!r}", flush=True)
        return
    try:
        print(f"[SEND] push user={user_id} text_head={text[:12]!r}", flush=True)
        line_bot_api.push_message(user_id, TextSendMessage(text=text))
    except Exception:
        print(f"[ERROR] push failed: {traceback.format_exc()}", flush=True)



def make_initial_reply(report_id: str) -> str:
    host = (SERVICE_HOST_URL or "").strip().rstrip("/")

    # スキーム補完（https:// が無ければ付与）
    if host and not host.startswith(("https://", "http://")):
        host = "https://" + host

    # host が空なら壊れたURLを出さない
    if not host:
        return (
            "✅ 動画を受信しました。\n"
            "AIによるスイング解析を開始します。\n\n"
            "⚠️ システム設定エラーのため、URLを生成できませんでした。\n"
            "時間を置いて再度お試しください。"
        )

    url = f"{host}/report/{report_id}"

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
    host = (SERVICE_HOST_URL or "").strip().rstrip("/")

    # スキーム補完（https:// が無ければ付ける）
    if host and not host.startswith(("https://", "http://")):
        host = "https://" + host

    # host が空なら、壊れたURLを出さない
    if not host:
        return "🎉 スイング計測が完了しました！（URL生成に失敗しました）"

    url = f"{host}/report/{report_id}"

    # URLは必ず「単独の1行」にする
    return (
        "🎉 スイング計測が完了しました！\n\n"
        "【診断レポートURL】\n"
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
import os
import logging
from datetime import datetime, timezone
from google.cloud import firestore

def is_premium_user(user_id: str) -> bool:
    disable = os.environ.get("DISABLE_FORCE_PREMIUM")
    in_force = user_id in FORCE_PREMIUM_USER_IDS
    logging.warning("[PREMIUM] disable=%r in_force=%s user_id=%s", disable, in_force, user_id)

    # 強制プレミアム（ただし env=1 のときは無効化）
    if disable != "1" and in_force:
        logging.warning("[PREMIUM] FORCE premium applied (disable=%r)", disable)
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
        logging.warning("[PREMIUM] user not found -> created as free")
        return False

    data = doc.to_dict() or {}
    logging.warning(
        "[PREMIUM] plan=%r ticket_remaining=%r plan_expire_at=%r",
        data.get("plan"),
        data.get("ticket_remaining"),
        data.get("plan_expire_at"),
    )

    plan = data.get("plan", "free")

    # 単発/回数券
    if plan in ("single", "ticket"):
        return False

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
        print("[DEBUG] entitlement txn start", user_id, report_id, flush=True)
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
def analyze_swing_with_mediapipe(video_path, overlay_out_path=None, user_id=None):
    # --- [追加箇所] 初級者〜中級者向けの設定（厳しすぎず、上達を楽しめる基準） ---
    
    # 1. デフォルト設定（アイアン：初・中級者が「緑」を出しやすい広めの許容範囲）
    config = {
        "spine_limit": 5.0,           # 前傾維持：5度までは「安定」とみなす
        "head_limit": 8.0,            # 頭のブレ：8cm程度までは許容（MediaPipeのノイズ対策込）
        "knee_limit": 11.0,           # 膝の動き：11cm程度まで
        "shoulder_std_limit": 22.0    # 肩回転のばらつき：初級者の再現性に合わせて広めに
    }
    club_type = "iron"
    spine_error_margin = config["spine_limit"] # 既存の描画ロジック用

    if user_id is None:
        print("user_id が渡されていません。デフォルト値(iron)で解析を続行します。")
    else:
        try:
            from google.cloud import firestore
            db = firestore.Client()
            user_doc = db.collection("users").document(user_id).get()
            if user_doc.exists:
                user_data = user_doc.to_dict() or {}
                prefill = user_data.get("prefill", {})
                club_type = prefill.get("club_type", "iron") # 未設定時はアイアン

            # 2. クラブ別の許容誤差を設定（長いクラブほど遠心力で体が動くため、さらに基準を広げる）
            if club_type == "driver":
                config = {
                    "spine_limit": 6.5,       # ドライバーは大きく振るため6.5度までOK
                    "head_limit": 11.0,       # ビハインド・ザ・ボールを考慮し広めに
                    "knee_limit": 15.0,       # 積極的な体重移動を許容
                    "shoulder_std_limit": 26.0
                }
            elif club_type == "wood_ut":
                config = {
                    "spine_limit": 5.5,
                    "head_limit": 9.5,
                    "knee_limit": 13.0,
                    "shoulder_std_limit": 24.0
                }
            
            # 既存の変数 spine_error_margin を今回の設定値で更新
            spine_error_margin = config["spine_limit"]

        except Exception as e:
            print(f"Firestoreの取得に失敗しました。デフォルト値で続行します: {e}")
    # ---------------------------------------------------------------------

    snaps = []
    import os
    os.environ["MP_DEVICE"] = "cpu"
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

    import logging
    import cv2
    import mediapipe as mp
    import math
    from typing import List, Dict, Any

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils  # ★追加
    mp_styles = mp.solutions.drawing_styles  # ★追加（任意、見た目が安定）

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCVがビデオを読み込めませんでした。")

    # ★overlay writer（必要な時だけ作る）
    writer = None
    tmp_path = None  # ✅ 追加

    if overlay_out_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ✅ 一時ファイルパスを作成
        tmp_path = overlay_out_path.replace(".mp4", "_tmp.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))  # ✅ tmp_pathに書き出し
        logging.warning(f"[DEBUG] overlay_writer_opened={writer.isOpened()} path={tmp_path} fps={fps} size=({w},{h})")

    total_frames = 0
    valid_frames = 0
    start_frame = None
    end_frame = None

    shoulders: List[float] = []
    hips: List[float] = []
    wrists: List[float] = []
    heads: List[float] = []
    knees: List[float] = []
    x_factors: List[float] = []
    spines: List[float] = []

    def angle_3d(p1, p2, p3):
        # ベクトル BA (p1-p2) と BC (p3-p2) を計算
        ax, ay, az = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
        bx, by, bz = p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]
        
        # 3次元の内積: A・B = AxBx + AyBy + AzBz
        dot = ax * bx + ay * by + az * bz
        
        # 3次元のベクトル長（ノルム）: |A| = sqrt(Ax^2 + Ay^2 + Az^2)
        na = math.sqrt(ax**2 + ay**2 + az**2)
        nb = math.sqrt(bx**2 + by**2 + bz**2)
        
        if na * nb == 0:
            return 0.0
        
        # 角度計算: cos(theta) = (A・B) / (|A|*|B|)
        c = max(-1.0, min(1.0, dot / (na * nb)))
        return math.degrees(math.acos(c))

    def draw_overlay_skeleton(frame, lm, mp_pose, color):
        import cv2

        h, w = frame.shape[:2]

        def pt(idx):
            return (int(lm[idx].x * w), int(lm[idx].y * h))

        def ok(idx):
            return lm[idx].visibility >= 0.6

        LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
        RE = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        LW = mp_pose.PoseLandmark.LEFT_WRIST.value
        RW = mp_pose.PoseLandmark.RIGHT_WRIST.value
        LH = mp_pose.PoseLandmark.LEFT_HIP.value
        RH = mp_pose.PoseLandmark.RIGHT_HIP.value
        LK = mp_pose.PoseLandmark.LEFT_KNEE.value
        RK = mp_pose.PoseLandmark.RIGHT_KNEE.value
        NO = mp_pose.PoseLandmark.NOSE.value

        connections = [
            (LS, RS),
            (LS, LE), (LE, LW),
            (RS, RE), (RE, RW),
            (LH, RH),
            (LH, LK),
            (RH, RK),
            (LS, LH),
            (RS, RH),
        ]

        for a, b in connections:
            if ok(a) and ok(b):
                cv2.line(frame, pt(a), pt(b), color, 3)

        for idx in [NO, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK]:
            if ok(idx):
                cv2.circle(frame, pt(idx), 4, color, -1)

    def draw_spine_line(frame, lm, mp_pose, color):
        import cv2

        h, w = frame.shape[:2]

        LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        LH = mp_pose.PoseLandmark.LEFT_HIP.value
        RH = mp_pose.PoseLandmark.RIGHT_HIP.value

        sh_x = int(((lm[LS].x + lm[RS].x) / 2.0) * w)
        sh_y = int(((lm[LS].y + lm[RS].y) / 2.0) * h)

        hip_x = int(((lm[LH].x + lm[RH].x) / 2.0) * w)
        hip_y = int(((lm[LH].y + lm[RH].y) / 2.0) * h)

        cv2.line(frame, (hip_x, hip_y), (sh_x, sh_y), color, 4)
        cv2.circle(frame, (hip_x, hip_y), 5, color, -1)
        cv2.circle(frame, (sh_x, sh_y), 5, color, -1)

    def draw_gaze_line(frame, lm, mp_pose, spine_angle_deg, color=(255, 255, 0)):
        import cv2, math
        h, w = frame.shape[:2]

        LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        LH = mp_pose.PoseLandmark.LEFT_HIP.value
        RH = mp_pose.PoseLandmark.RIGHT_HIP.value

        sh_x = int(((lm[LS].x + lm[RS].x) / 2.0) * w)
        sh_y = int(((lm[LS].y + lm[RS].y) / 2.0) * h)
        hip_x = int(((lm[LH].x + lm[RH].x) / 2.0) * w)
        hip_y = int(((lm[LH].y + lm[RH].y) / 2.0) * h)

        vec_x = sh_x - hip_x
        vec_y = sh_y - hip_y
        length = math.sqrt(vec_x**2 + vec_y**2)
        if length < 1e-6:
            return

        extend = length * 0.6
        norm_x = vec_x / length
        norm_y = vec_y / length
        end_x = int(sh_x + norm_x * extend)
        end_y = int(sh_y + norm_y * extend)

        cv2.arrowedLine(frame, (sh_x, sh_y), (end_x, end_y), color, 2, tipLength=0.3)
        cv2.putText(frame, f"Tilt: {spine_angle_deg:.1f}",
                    (sh_x + 10, sh_y - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # model_complexity=1 はCPU環境で速度と精度のバランスが最も良い設定です。
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        address_buffer = []
        analysis_started = False
        analysis_finished = False
        start_frame = None
        end_frame = None
        finish_buffer = []

        base_spine_angle = None
        spine_shoulder_history = []
        spine_hip_history = []
        base_nose = None
        base_lknee = None
        pos_history = []
        is_analyzing = False
        swing_ended = False
        has_reached_top = False

        top_wrist_y = 999.0
        top_spine_angle = None
        impact_spine_angle = None
        impact_detected = False

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            total_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                if writer is not None and frame is not None:
                    writer.write(frame)
                continue

            lm = res.pose_landmarks.landmark
            valid_frames += 1

            def xyz_stable(i):
                return (lm[i].x, lm[i].y, lm[i].z * 0.5)

            def avg_point(history):
                n = len(history)
                if n == 0:
                    return None
                return (
                    sum(p[0] for p in history) / n,
                    sum(p[1] for p in history) / n,
                    sum(p[2] for p in history) / n,
                )

            def dist_3d(p, base):
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, base)))

            LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LH = mp_pose.PoseLandmark.LEFT_HIP.value
            RH = mp_pose.PoseLandmark.RIGHT_HIP.value
            LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
            LW = mp_pose.PoseLandmark.LEFT_WRIST.value
            LI = mp_pose.PoseLandmark.LEFT_INDEX.value
            NO = mp_pose.PoseLandmark.NOSE.value
            LK = mp_pose.PoseLandmark.LEFT_KNEE.value

            curr_nose = xyz_stable(NO)
            curr_lknee = xyz_stable(LK)
            curr_lwrist = xyz_stable(LW)
            nose_y = lm[NO].y

            # --- C. 角度計算（前傾はここで1回だけ） ---
            sh = angle_3d(xyz_stable(LS), xyz_stable(RS), xyz_stable(RH))
            hip = angle_3d(xyz_stable(LH), xyz_stable(RH), xyz_stable(LK))
            wr = 180.0 - angle_3d(xyz_stable(LE), xyz_stable(LW), xyz_stable(LI))
            spine_angle = 0.0

            if (
                lm[LS].visibility >= 0.7 and
                lm[RS].visibility >= 0.7 and
                lm[LH].visibility >= 0.7 and
                lm[RH].visibility >= 0.7
            ):
                ls = xyz_stable(LS)
                rs = xyz_stable(RS)
                lh = xyz_stable(LH)
                rh = xyz_stable(RH)

                shoulder_mid = (
                    (ls[0] + rs[0]) / 2.0,
                    (ls[1] + rs[1]) / 2.0,
                    (ls[2] + rs[2]) / 2.0,
                )
                hip_mid = (
                    (lh[0] + rh[0]) / 2.0,
                    (lh[1] + rh[1]) / 2.0,
                    (lh[2] + rh[2]) / 2.0,
                )

                spine_shoulder_history.append(shoulder_mid)
                spine_hip_history.append(hip_mid)

                if len(spine_shoulder_history) > 3:
                    spine_shoulder_history.pop(0)
                if len(spine_hip_history) > 3:
                    spine_hip_history.pop(0)

                smooth_shoulder_mid = avg_point(spine_shoulder_history)
                smooth_hip_mid = avg_point(spine_hip_history)

                if smooth_shoulder_mid is not None and smooth_hip_mid is not None:
                    dx = smooth_shoulder_mid[0] - smooth_hip_mid[0]
                    dy = smooth_shoulder_mid[1] - smooth_hip_mid[1]
                    spine_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

            # --- A. アドレス安定区間を検出してベース確保 ---

            is_pose_visible = (
                lm[LS].visibility >= 0.7 and
                lm[RS].visibility >= 0.7 and
                lm[LH].visibility >= 0.7 and
                lm[RH].visibility >= 0.7 and
                lm[NO].visibility >= 0.7 and
                lm[LK].visibility >= 0.7
            )

            # 前傾角に依存せず、姿勢が見えていればバッファ
            if not analysis_started and is_pose_visible:

                address_buffer.append({
                    "nose": curr_nose,
                    "lknee": curr_lknee,
                    "spine": spine_angle,
                    "lwrist_y": curr_lwrist[1],
                })

                if len(address_buffer) > 8:
                    address_buffer.pop(0)

            # バッファが溜まったら開始判定
            if not analysis_started and len(address_buffer) >= 5:

                wrist_move = abs(
                    address_buffer[-1]["lwrist_y"]
                    - address_buffer[0]["lwrist_y"]
                )

                # 手元が動き始めたら開始
                if wrist_move > 0.0015:

                    base_nose = (
                        sum(f["nose"][0] for f in address_buffer) / len(address_buffer),
                        sum(f["nose"][1] for f in address_buffer) / len(address_buffer),
                        sum(f["nose"][2] for f in address_buffer) / len(address_buffer),
                    )

                    base_lknee = (
                        sum(f["lknee"][0] for f in address_buffer) / len(address_buffer),
                        sum(f["lknee"][1] for f in address_buffer) / len(address_buffer),
                        sum(f["lknee"][2] for f in address_buffer) / len(address_buffer),
                    )

                    base_spine_angle = (
                        sum(f["spine"] for f in address_buffer) / len(address_buffer)
                    )

                    for f in address_buffer:
                        if float(f["spine"]) > 0:
                            spines.append(float(f["spine"]))

                    analysis_started = True
                    start_frame = total_frames
                    
            # 解析開始前でも、姿勢が見えていればラインは表示する
            if not analysis_started:
                if writer is not None and frame is not None:
                    out = frame.copy()

                    color = (0, 255, 0)

                    draw_overlay_skeleton(out, lm, mp_pose, color)
                    draw_spine_line(out, lm, mp_pose, color)

                    if spine_angle > 0:
                        draw_gaze_line(out, lm, mp_pose, spine_angle)

                    writer.write(out)
                continue
            
            # --- B. トップ・終了候補判定 ---
            if curr_lwrist[1] < nose_y:
                has_reached_top = True

            is_finish_candidate = has_reached_top and curr_lwrist[1] > (nose_y + 0.1)

            # --- B2. フィニッシュ安定判定 ---
            if analysis_started and not analysis_finished:
                if is_finish_candidate:
                    finish_buffer.append({
                        "nose": curr_nose,
                        "lknee": curr_lknee,
                        "lwrist_y": curr_lwrist[1],
                    })
                    if len(finish_buffer) > 6:
                        finish_buffer.pop(0)

                    if len(finish_buffer) >= 4:
                        nose_move = abs(finish_buffer[-1]["nose"][1] - finish_buffer[0]["nose"][1])
                        knee_move = abs(finish_buffer[-1]["lknee"][1] - finish_buffer[0]["lknee"][1])
                        wrist_move = abs(finish_buffer[-1]["lwrist_y"] - finish_buffer[0]["lwrist_y"])

                        # フィニッシュ後の動きが小さくなったら終了
                        if nose_move < 0.015 and knee_move < 0.02 and wrist_move < 0.03:
                            analysis_finished = True
                            swing_ended = True
                            end_frame = total_frames
                else:
                    finish_buffer.clear()

            # 終了後はループを抜ける
            if analysis_finished:
                if writer is not None and frame is not None:
                    writer.write(frame)
                break
                
            # --- D. データ保存 ---
            hd = dist_3d(curr_nose, base_nose) * 100
            kn = dist_3d(curr_lknee, base_lknee) * 100

            shoulders.append(float(sh))
            hips.append(float(hip))
            wrists.append(float(wr))
            heads.append(float(hd))
            knees.append(float(kn))
            x_factors.append(float(sh - abs(hip)))

            if spine_angle > 10:
                spines.append(float(spine_angle))

            if spine_angle > 0:
                if curr_lwrist[1] < top_wrist_y:
                    top_wrist_y = curr_lwrist[1]
                    top_spine_angle = float(spine_angle)

                if has_reached_top and (not impact_detected) and curr_lwrist[1] > (nose_y + 0.05):
                    impact_spine_angle = float(spine_angle)
                    impact_detected = True

            # --- E. overlay描画 ---
            if writer is not None and frame is not None:
                out = frame.copy()

                color = (0, 255, 0)
                if base_spine_angle is not None and spine_angle > 0:
                    delta_spine = abs(spine_angle - base_spine_angle)

                    # 3 や 6 という固定値を、上記で設定した spine_error_margin に置き換え
                    if delta_spine <= spine_error_margin:
                        color = (0, 255, 0)
                    elif delta_spine <= (spine_error_margin * 2):
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)
                        
                draw_overlay_skeleton(out, lm, mp_pose, color)
                draw_spine_line(out, lm, mp_pose, color)
                
                if spine_angle > 0:
                    draw_gaze_line(out, lm, mp_pose, spine_angle)
                
                writer.write(out)

    # whileループ終了後
    cap.release()

    final_overlay_path = None

    if writer is not None:
        writer.release()
        logging.warning("[DEBUG] overlay writer released")

        import os
        import subprocess

        logging.warning(f"[DEBUG] tmp_path={tmp_path}")
        logging.warning(f"[DEBUG] overlay_out_path={overlay_out_path}")
        logging.warning(f"[DEBUG] tmp_exists_before={os.path.exists(tmp_path) if tmp_path else False}")

        if tmp_path and overlay_out_path and os.path.exists(tmp_path):
            try:
                result = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", tmp_path,
                        "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-crf", "23",
                        overlay_out_path
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logging.warning("[DEBUG] FFmpeg変換完了")

                if os.path.exists(overlay_out_path):
                    final_overlay_path = overlay_out_path
                    logging.warning(f"[DEBUG] final_overlay_path={final_overlay_path}")
                else:
                    logging.warning("[DEBUG] ffmpeg成功扱いだが overlay_out_path が存在しない")

            except Exception as e:
                logging.warning(f"[DEBUG] FFmpeg変換失敗: {e}")

                try:
                    os.rename(tmp_path, overlay_out_path)
                    if os.path.exists(overlay_out_path):
                        final_overlay_path = overlay_out_path
                        logging.warning("[DEBUG] rename fallback 成功")
                    else:
                        logging.warning("[DEBUG] rename後も overlay_out_path が存在しない")
                except Exception as rename_e:
                    logging.warning(f"[DEBUG] rename失敗: {rename_e}")

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    logging.warning("[DEBUG] tmp_path削除完了")

    if final_overlay_path:
        overlay_out_path = final_overlay_path
    
    # --- ヘルパー関数の定義 ---
    def _safe_mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def _safe_std(xs):
        if not xs: return 0.0
        m = _safe_mean(xs)
        return math.sqrt(sum((x - m)**2 for x in xs) / len(xs))

    def _bench_delta(val, base_val):
        return float(val - base_val)

    def dist_3d(p, base):
        return math.sqrt(sum((a - b)**2 for a, b in zip(p, base)))

    def pack(xs: List[float], nd: int = 2) -> Dict[str, float]:
        if not xs:
            return {"max": 0.0, "min": 0.0, "mean": 0.0, "std": 0.0}
        return {
            "max": round(float(max(xs)), nd),
            "min": round(float(min(xs)), nd),
            "mean": round(float(_safe_mean(xs)), nd),
            "std": round(float(_safe_std(xs)), nd),
        }

    # --- 解析結果のバリデーション ---
    if total_frames < 10 or valid_frames < 5:
        raise RuntimeError("解析に必要なフレーム数が不足しています。")

    # --- データの集計・計算 ---
    conf = float(valid_frames) / float(total_frames)

    # spine保険
    if len(spines) == 0:
        spines.append(0.0)

    # (ここに必要な計算ロジックが入ります)
    # (中略：解析処理)

    # 最終的な結果を返す
    result = {
        "club_type": club_type,
        "thresholds": config,  # この値を 08セクションのタグ判定に使う
        "frame_count": int(total_frames),
        "valid_frames": int(valid_frames),
        "confidence": round(conf, 3),
        "shoulder": pack(shoulders, 2),
        "hip": pack(hips, 2),
        "wrist": pack(wrists, 2),
        "head": pack(heads, 4),
        "knee": pack(knees, 4),
        "x_factor": pack(x_factors, 2),
        "spine": pack(spines, 2),
        "spine_raw": [round(float(x), 2) for x in spines],
        "spine_top": round(float(top_spine_angle), 2) if top_spine_angle is not None else 0.0,
        "spine_impact": round(float(impact_spine_angle), 2) if impact_spine_angle is not None else 0.0,
        "base_spine_angle": round(float(base_spine_angle), 2) if base_spine_angle is not None else 0.0,
        "snaps": snaps
    }

    if overlay_out_path:
        result["overlay_path"] = overlay_out_path

    return result
  
# ==================================================
# Section 01: 修正版（3D・％単位対応・診断クラブ追加・クラブ別目安対応）
# ==================================================
def build_section_01(raw: Dict[str, Any], club_type: str) -> Dict[str, Any]:
    # クラブ名の日本語表示用マッピング
    club_name_jp = {
        "driver": "ドライバー",
        "iron": "アイアン",
        "wood": "ウッド",
        "utility": "ユーティリティ"
    }.get(club_type, club_type)

    # クラブ別の目安（guide）設定
    if club_type == "driver":
        guides = {
            "shoulder": "maxで90°〜120°",  # ドライバーは回旋が大きくなるため
            "hip": "maxで35°〜65°",
            "wrist": "meanで40°〜80°",
            "head": "meanで8.0%以下",     # 頭のブレもアイアンより許容する
            "knee": "meanで12.0%以下"     # 膝のブレもアイアンより許容する
        }
    else:
        # アイアン等の基準（元の固定値をベースに設定）
        guides = {
            "shoulder": "maxで80°〜115°",
            "hip": "maxで30°〜60°",
            "wrist": "meanで40°〜80°",
            "head": "meanで6.5%以下",
            "knee": "meanで10.0%以下"
        }

    return {
        "title": "01. 骨格計測データ（AIが測定）",
        "items": [
            {
                "name": "診断クラブ",
                "value": club_name_jp,
                "description": "今回解析を行ったクラブの種類です。",
                "guide": "-",
            },
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
                "value": f'max {raw["shoulder"]["max"]:.1f} / mean {raw["shoulder"]["mean"]:.1f} / σ {raw["shoulder"]["std"]:.1f}',
                "description": "3D空間での上半身の回旋量です。",
                "guide": guides["shoulder"],
            },
            {
                "name": "腰回転（°）",
                "value": f'max {raw["hip"]["max"]:.1f} / mean {raw["hip"]["mean"]:.1f} / σ {raw["hip"]["std"]:.1f}',
                "description": "3D空間での下半身の回旋量です。",
                "guide": guides["hip"],
            },
            {
                "name": "手首コック（°）",
                "value": f'max {raw["wrist"]["max"]:.1f} / mean {raw["wrist"]["mean"]:.1f} / σ {raw["wrist"]["std"]:.1f}',
                "description": "手首のタメの角度（3D）です。",
                "guide": guides["wrist"],
            },
            {
                "name": "頭部ブレ（%）",
                "value": f'max {raw["head"]["max"]:.1f} / mean {raw["head"]["mean"]:.1f} / σ {raw["head"]["std"]:.1f}',
                "description": "アドレス時からの頭部の移動量です（画面幅比）。",
                "guide": guides["head"],
            },
            {
                "name": "膝ブレ（%）",
                "value": f'max {raw["knee"]["max"]:.1f} / mean {raw["knee"]["mean"]:.1f} / σ {raw["knee"]["std"]:.1f}',
                "description": "アドレス時からの膝の移動量です（画面幅比）。",
                "guide": guides["knee"],
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
    # {v:.1f} を加えることで、18.632... を 18.6 に整えます
    return f"max {maxv:.1f} / mean {meanv:.1f} / σ {stdv:.1f}（conf {conf:.3f}）"


def judge_shoulder(raw: Dict[str, Any]) -> Dict[str, Any]:
    sh = raw["shoulder"]
    xf = raw["x_factor"]

    sh_val = float(sh["max"])
    xf_val = float(xf["max"])

    main = "mid"
    if sh_val < 80:
        main = "low"
    elif sh_val > 115:
        main = "high"

    rel = "mid"
    if xf_val < 30:
        rel = "low"
    elif xf_val > 70:
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

def _range_ideal(lo: float, hi: float, unit: str) -> dict:
    return {
        "type": "range",
        "min": float(lo),
        "max": float(hi),
        "unit": unit,
    }

def _le_ideal(th: float, unit: str) -> dict:
    return {
        "type": "le",
        "value": float(th),
        "unit": unit,
    }

def _ge_ideal(th: float, unit: str) -> dict:
    return {
        "type": "ge",
        "value": float(th),
        "unit": unit,
    }

    
def _bench_line(label: str, unit: str, stat: str, ideal: dict, *, current) -> dict:
    try:
        cur = float(current)
    except Exception:
        cur = None

    return {
        "label": label,
        "unit": unit,
        "stat": stat,
        "ideal": ideal,
        "current": cur,
    }


import json
import logging
from typing import Dict, Any, List

def generate_llm_comment_02(payload: Dict[str, Any]) -> str:
    prompt = f"""
あなたはプロのゴルフコーチです。
寄り添い型で、前向きな文脈で説明してください。

以下の肩の回転に関する計測データと個別評価を元に、ユーザーに対する「総合評価」を生成してください。

【計測データと個別評価】
・肩の回転量: {payload['sh_max']:.1f}° -> 評価: {payload['rotation_eval']} ({payload['rotation_comment']})
・回転の安定性(ばらつき): {payload['sh_std']:.1f}° -> 評価: {payload['stability_eval']} ({payload['stability_comment']})
・肩と腰の捻転差: {payload['xf_max']:.1f}° -> 評価: {payload['xfactor_eval']} ({payload['xfactor_comment']})

【意識すること】
・良い点があるからこそ改善すると伸びる、という前向きな文脈にする
・専門用語をそのまま使いすぎず、イメージしやすい言葉に言い換える
・出力はJSON形式のみとする

【出力形式】
以下の2つのキーを持つJSON形式のテキストのみを出力してください。Markdownの```jsonなどの装飾は不要です。
{{
    "overall_eval": "良好 または やや改善余地あり または 改善余地あり",
    "overall_comment": "上記のデータを踏まえた、ユーザーへの総合的なアドバイスを簡潔に（50文字〜80文字程度）"
}}
"""
    return call_llm(prompt)


def build_paid_02_shoulder(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_shoulder(raw)
    sh = raw["shoulder"]
    xf = raw["x_factor"]
    conf = _conf(raw)

    # ==========================================================
    # 新しい構成：評価の内訳ロジック（ルールベースで事実を確定）
    # ==========================================================
    # 1. 回転量（max）
    rotation_eval = ""
    rotation_comment = ""
    if sh["max"] < 80:
        rotation_eval = "浅め"
        rotation_comment = "やや捻転が浅く、飛距離を生むためのエネルギーが不足しがちです"
    elif sh["max"] > 115:
        rotation_eval = "大きめ"
        rotation_comment = "回転量が非常に大きく、オーバースイングによる打点のブレが出やすい傾向があります"
    else:
        rotation_eval = "適正"
        rotation_comment = "十分な回転量があり、上半身のエネルギーは作れています"

    # 2. 安定性（std / σ）
    stability_eval = ""
    stability_comment = ""
    if sh["std"] <= 12:
        stability_eval = "安定"
        stability_comment = "トップの位置が揃っており、スイングの再現性が高く安定しています"
    else:
        stability_eval = "やや不安定"
        stability_comment = "トップの位置にばらつきがあり、再現性にややムラがあります"

    # 3. 捻転差（x_factor max）
    xfactor_eval = ""
    xfactor_comment = ""
    if xf["max"] < 30:
        xfactor_eval = "小さめ"
        xfactor_comment = "肩と腰の回転差が小さく、ダウンスイングで力を溜める余地がまだあります"
    elif xf["max"] > 70:
        xfactor_eval = "大きめ"
        xfactor_comment = "肩と腰の回転差が大きく、連動のタイミングがズレやすい状態です"
    else:
        xfactor_eval = "適正"
        xfactor_comment = "適正な捻転差が確保されており、切り返しから出力への準備が整っています"

    # ==========================================================
    # 4. 総合評価の判定（LLMを使用）
    # ==========================================================
    llm_payload = {
        "sh_max": float(sh["max"]),
        "sh_std": float(sh["std"]),
        "xf_max": float(xf["max"]),
        "rotation_eval": rotation_eval,
        "rotation_comment": rotation_comment,
        "stability_eval": stability_eval,
        "stability_comment": stability_comment,
        "xfactor_eval": xfactor_eval,
        "xfactor_comment": xfactor_comment
    }

    overall_eval = "判定中"
    overall_comment = "AIコーチが評価をまとめています…"

    try:
        print("LLM CALL START (02_shoulder)")
        response_text = generate_llm_comment_02(llm_payload)
        print("LLM CALL END (02_shoulder)")

        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        llm_data = json.loads(clean_text)
        
        overall_eval = llm_data.get("overall_eval", overall_eval)
        overall_comment = llm_data.get("overall_comment", overall_comment)

    except Exception as e:
        logging.exception("LLM JSON parse error in 02_shoulder: %s", e)
        overall_eval = "やや改善余地あり"
        overall_comment = "各項目の結果を参考に、連動性と安定性を高める練習を行いましょう。"

    # ==========================================================
    # 数値サマリーと戻り値の構築
    # ==========================================================
    summary_text = f"最大{sh['max']:.1f}° / 平均{sh['mean']:.1f}° / ばらつき{sh['std']:.1f}° / 捻転差{xf['max']:.1f}°"

    # ★修正箇所：内訳の最後に総合評価（LLMコメント）を結合
    new_layout_text = (
        f"回転量：{rotation_eval}\n"
        f"→ {rotation_comment}\n\n"
        f"安定性：{stability_eval}\n"
        f"→ {stability_comment}\n\n"
        f"捻転差：{xfactor_eval}\n"
        f"→ {xfactor_comment}\n\n"
        f"総合：{overall_eval}\n"
        f"→ {overall_comment}"
    )

    good: List[str] = []
    bad: List[str] = []

    bench = [
        _bench_line("肩回転(°)", "°", "max", _range_ideal(80, 115, "°"), current=float(sh["max"])),
        _bench_line("肩回転の安定(°)", "°", "σ", _le_ideal(12.0, "°"), current=float(sh["std"])),
        _bench_line("捻転差(°)", "°", "max", _range_ideal(30, 70, "°"), current=float(xf["max"])),
    ]

    return {
        "title": "02. Shoulder Rotation（肩回転）",
        "value": summary_text,
        "tags": j["tags"],
        "bench": bench,
        "good": good,
        "bad": bad,
        "pro_comment": new_layout_text,
        "new_eval_data": {
            "overall_eval": overall_eval,
            "summary_text": summary_text,
            "rotation": {"eval": rotation_eval, "comment": rotation_comment},
            "stability": {"eval": stability_eval, "comment": stability_comment},
            "xfactor": {"eval": xfactor_eval, "comment": xfactor_comment},
        }
    }
    
def judge_hip(raw: Dict[str, Any]) -> Dict[str, Any]:
    hip = raw["hip"]
    xf = raw["x_factor"]

    # 最大回旋量ベースで判定
    hip_val = hip["max"]
    xf_val = xf["max"]

    main = "mid"
    if hip_val < 30:
        main = "low"
    elif hip_val > 60:
        main = "high"

    rel = "mid"
    if xf_val < 30:
        rel = "low"
    elif xf_val > 70:
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

import json
import logging
from typing import Dict, Any, List

def generate_llm_comment_03(payload: Dict[str, Any]) -> str:
    prompt = f"""
あなたはプロのゴルフコーチです。
寄り添い型で、前向きな文脈で説明してください。

以下の腰の回転に関する計測データと個別評価を元に、ユーザーに対する「総合評価」を生成してください。

【計測データと個別評価】
・腰の回転量: {payload['hip_max']:.1f}° -> 評価: {payload['rotation_eval']} ({payload['rotation_comment']})
・回転の安定性(ばらつき): {payload['hip_std']:.1f}° -> 評価: {payload['stability_eval']} ({payload['stability_comment']})
・肩と腰の捻転差: {payload['xf_max']:.1f}° -> 評価: {payload['xfactor_eval']} ({payload['xfactor_comment']})

【意識すること】
・良い点があるからこそ改善すると伸びる、という前向きな文脈にする
・専門用語をそのまま使いすぎず、イメージしやすい言葉に言い換える
・出力はJSON形式のみとする

【出力形式】
以下の2つのキーを持つJSON形式のテキストのみを出力してください。Markdownの```jsonなどの装飾は不要です。
{{
    "overall_eval": "良好 または やや改善余地あり または 改善余地あり",
    "overall_comment": "上記のデータを踏まえた、ユーザーへの総合的なアドバイスを簡潔に（50文字〜80文字程度）"
}}
"""
    return call_llm(prompt)


def build_paid_03_hip(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_hip(raw)
    hip = raw["hip"]
    xf = raw["x_factor"]
    conf = _conf(raw)

    # ==========================================================
    # 新しい構成：評価の内訳ロジック（ルールベースで事実を確定）
    # ==========================================================
    # 1. 回転量（max）
    rotation_eval = ""
    rotation_comment = ""
    if hip["max"] < 30:
        rotation_eval = "浅め"
        rotation_comment = "下半身の回旋がやや控えめで、下半身主導の動きをさらに高める余地があります"
    elif hip["max"] > 60:
        rotation_eval = "大きめ"
        rotation_comment = "腰の回転量がやや大きく、軸のブレに繋がりやすい状態です"
    else:
        rotation_eval = "適正"
        rotation_comment = "腰の回旋量は適正範囲で、安定した軸回転とスイングの土台が作れています"

    # 2. 安定性（std / σ）
    stability_eval = ""
    stability_comment = ""
    if hip["std"] <= 12:
        stability_eval = "安定"
        stability_comment = "腰の回し幅が揃っており、インパクトの再現性を支える下半身の動きが安定しています"
    else:
        stability_eval = "やや不安定"
        stability_comment = "腰の回転角度にばらつきがあり、ミート率の再現性に影響しやすい状態です"

    # 3. 捻転差（x_factor max）
    xfactor_eval = ""
    xfactor_comment = ""
    if xf["max"] < 30:
        xfactor_eval = "小さめ"
        xfactor_comment = "肩と腰の回転差が小さく、上半身との連動で出力を高める余地があります"
    else:
        xfactor_eval = "適正"
        xfactor_comment = "肩と腰の捻転差が確保されており、力を逃さずインパクトへ繋げる準備ができています"

    # ==========================================================
    # 4. 総合評価の判定（LLMを使用）
    # ==========================================================
    llm_payload = {
        "hip_max": float(hip["max"]),
        "hip_std": float(hip["std"]),
        "xf_max": float(xf["max"]),
        "rotation_eval": rotation_eval,
        "rotation_comment": rotation_comment,
        "stability_eval": stability_eval,
        "stability_comment": stability_comment,
        "xfactor_eval": xfactor_eval,
        "xfactor_comment": xfactor_comment
    }

    overall_eval = "判定中"
    overall_comment = "AIコーチが評価をまとめています…"

    try:
        print("LLM CALL START (03_hip)")
        response_text = generate_llm_comment_03(llm_payload)
        print("LLM CALL END (03_hip)")

        # 万が一LLMが ```json などのMarkdown記号をつけて返してきた場合の安全対策
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        llm_data = json.loads(clean_text)
        
        overall_eval = llm_data.get("overall_eval", overall_eval)
        overall_comment = llm_data.get("overall_comment", overall_comment)

    except Exception as e:
        logging.exception("LLM JSON parse error in 03_hip: %s", e)
        # エラー時は無難な総合評価をフォールバックとして設定
        overall_eval = "やや改善余地あり"
        overall_comment = "各項目の結果を参考に、下半身の安定感と上半身との連動性を意識した練習を行いましょう。"

    # ==========================================================
    # 数値サマリーと戻り値の構築
    # ==========================================================
    summary_text = f"最大{hip['max']:.1f}° / 平均{hip['mean']:.1f}° / ばらつき{hip['std']:.1f}° / 捻転差{xf['max']:.1f}°"

    new_layout_text = (
        f"回転量：{rotation_eval}\n"
        f"→ {rotation_comment}\n\n"
        f"安定性：{stability_eval}\n"
        f"→ {stability_comment}\n\n"
        f"捻転差：{xfactor_eval}\n"
        f"→ {xfactor_comment}\n\n"
        f"総合：{overall_eval}\n"
        f"→ {overall_comment}"
    )

    good: List[str] = []
    bad: List[str] = []

    bench = [
        _bench_line("腰回転(°)", "°", "max", _range_ideal(30, 60, "°"), current=float(hip["max"])),
        _bench_line("腰回転の安定(°)", "°", "σ", _le_ideal(12.0, "°"), current=float(hip["std"])),
        _bench_line("捻転差(°)", "°", "max", _ge_ideal(30.0, "°"), current=float(xf["max"])),
    ]

    return {
        "title": "03. Hip Rotation（腰回転）",
        "value": summary_text,
        "tags": j["tags"],
        "bench": bench,
        "good": good,
        "bad": bad,
        "pro_comment": new_layout_text,
        "new_eval_data": {
            "overall_eval": overall_eval,
            "summary_text": summary_text,
            "rotation": {"eval": rotation_eval, "comment": rotation_comment},
            "stability": {"eval": stability_eval, "comment": stability_comment},
            "xfactor": {"eval": xfactor_eval, "comment": xfactor_comment},
        }
    }
    
def judge_wrist(raw: Dict[str, Any]) -> Dict[str, Any]:
    # バックエンドで反転済みの数値をそのまま使用
    w_mean = float(raw["wrist"]["mean"])
    xf_mean = float(raw["x_factor"]["mean"])

    main = "mid"
    if w_mean < 40:
        main = "low"
    elif w_mean > 80:
        main = "high"

    rel = "mid"
    if xf_mean < 30:
        rel = "low"

    tags: List[str] = []
    if main == "low":
        tags.append("コック不足")
    if main == "high":
        tags.append("コック過多")
    if rel == "low":
        tags.append("捻転差不足")

    return {"main": main, "related": rel, "tags": tags}


import json
import logging
from typing import Dict, Any, List

def generate_llm_comment_04(payload: Dict[str, Any]) -> str:
    prompt = f"""
あなたはプロのゴルフコーチです。
寄り添い型で、前向きな文脈で説明してください。

以下の手首の挙動に関する計測データと個別評価を元に、ユーザーに対する「総合評価」を生成してください。

【計測データと個別評価】
・最大コック量: {payload['w_max']:.1f}° -> 評価: {payload['amount_eval']} ({payload['amount_comment']})
・手首の安定性(ばらつき): {payload['w_std']:.1f}° -> 評価: {payload['stability_eval']} ({payload['stability_comment']})
・タメの維持(平均値): {payload['w_mean']:.1f}° -> 評価: {payload['impact_eval']} ({payload['impact_comment']})

【意識すること】
・良い点があるからこそ改善すると伸びる、という前向きな文脈にする
・専門用語をそのまま使いすぎず、イメージしやすい言葉に言い換える
・出力はJSON形式のみとする

【出力形式】
以下の2つのキーを持つJSON形式のテキストのみを出力してください。Markdownの```jsonなどの装飾は不要です。
{{
    "overall_eval": "良好 または やや改善余地あり または 改善余地あり",
    "overall_comment": "上記のデータを踏まえた、ユーザーへの総合的なアドバイスを簡潔に（50文字〜80文字程度）"
}}
"""
    return call_llm(prompt)


def build_paid_04_wrist(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    w_raw = raw["wrist"]

    # 解析側で反転済み
    w_mean = float(w_raw["mean"])
    w_max  = float(w_raw["max"])
    w_std  = float(w_raw["std"])

    j = judge_wrist(raw)
    conf = _conf(raw)

    # ==========================================================
    # 新しい構成：評価の内訳ロジック（ルールベースで事実を確定）
    # ==========================================================
    # 1. コック量（max）
    amount_eval = ""
    amount_comment = ""
    if w_max < 35:
        amount_eval = "浅め"
        amount_comment = "バックスイングでのコック量がやや不足しており、力を溜める余地があります"
    elif w_max > 95:
        amount_eval = "深め"
        amount_comment = "コックが深く入っており、飛距離のポテンシャルは高いですが、振り遅れに注意が必要です"
    else:
        amount_eval = "適正"
        amount_comment = "トップでは十分なコックが入っており、ヘッドスピードにつながる土台があります"

    # 2. 安定性（std / σ）
    stability_eval = ""
    stability_comment = ""
    if w_std <= 12:
        stability_eval = "安定"
        stability_comment = "手首の角度変化は一定しており、インパクトでのフェース管理は概ね安定しています"
    else:
        stability_eval = "やや不安定"
        stability_comment = "手首の挙動にばらつきがあり、打点の再現性に影響が出やすい状態です"

    # 3. タメの維持（mean）
    impact_eval = ""
    impact_comment = ""
    if w_mean < 40:
        impact_eval = "早めのリリース"
        impact_comment = "タメが解けるのが早く、インパクトで効率的にパワーを伝えきれていない可能性があります"
    elif w_mean > 80:
        impact_eval = "タメが深め"
        impact_comment = "タメは深いですが、その分リリースのタイミング管理がシビアになりやすい状態です"
    else:
        impact_eval = "良好"
        impact_comment = "手首のタメ（平均角度）は基準内で、効率的なパワー伝達ができています"

    # ==========================================================
    # 4. 総合評価の判定（LLMを使用）
    # ==========================================================
    llm_payload = {
        "w_max": w_max,
        "w_mean": w_mean,
        "w_std": w_std,
        "amount_eval": amount_eval,
        "amount_comment": amount_comment,
        "stability_eval": stability_eval,
        "stability_comment": stability_comment,
        "impact_eval": impact_eval,
        "impact_comment": impact_comment
    }

    overall_eval = "判定中"
    overall_comment = "AIコーチが評価をまとめています…"

    try:
        print("LLM CALL START (04_wrist)")
        response_text = generate_llm_comment_04(llm_payload)
        print("LLM CALL END (04_wrist)")

        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        llm_data = json.loads(clean_text)
        
        overall_eval = llm_data.get("overall_eval", overall_eval)
        overall_comment = llm_data.get("overall_comment", overall_comment)

    except Exception as e:
        logging.exception("LLM JSON parse error in 04_wrist: %s", e)
        overall_eval = "やや改善余地あり"
        overall_comment = "手首のタメとリリースのタイミングを整えることで、さらなる飛距離アップが期待できます。"

    # ==========================================================
    # 数値サマリーと戻り値の構築
    # ==========================================================
    # ユーザー希望の数値サマリー形式：Max / Mean / ばらつき
    summary_text = f"最大{w_max:.1f}° / 平均{w_mean:.1f}° / ばらつき{w_std:.1f}°"

    new_layout_text = (
        f"コック量：{amount_eval}\n"
        f"→ {amount_comment}\n\n"
        f"安定性：{stability_eval}\n"
        f"→ {stability_comment}\n\n"
        f"タメの維持：{impact_eval}\n"
        f"→ {impact_comment}\n\n"
        f"総合：{overall_eval}\n"
        f"→ {overall_comment}"
    )

    good: List[str] = []
    bad: List[str] = []

    bench = [
        _bench_line("手首コック(°)", "°", "mean", _range_ideal(40, 80, "°"), current=float(w_mean)),
        _bench_line("手首コックの上限(°)", "°", "max", _ge_ideal(75.0, "°"), current=float(w_max)),
        _bench_line("手首の再現性(°)", "°", "σ", _le_ideal(15.0, "°"), current=float(w_std)),
    ]

    return {
        "title": "04. Wrist Cock（手首コック）",
        "value": summary_text,
        "tags": j["tags"],
        "bench": bench,
        "good": good,
        "bad": bad,
        "pro_comment": new_layout_text,
        "new_eval_data": {
            "overall_eval": overall_eval,
            "summary_text": summary_text,
            "amount": {"eval": amount_eval, "comment": amount_comment},
            "stability": {"eval": stability_eval, "comment": stability_comment},
            "impact": {"eval": impact_eval, "comment": impact_comment},
        }
    }
    
def judge_head(raw: Dict[str, Any]) -> Dict[str, Any]:
    h = raw["head"]
    k = raw["knee"]

    tags: List[str] = []

    if h["mean"] > 6.0:
        tags.append("頭部ブレ大")

    if k["mean"] > 9.0:
        tags.append("膝ブレ大")

    if k["mean"] > 11.0:
        tags.append("下半身不安定")

    return {"tags": tags}

import json
import logging
from typing import Dict, Any, List

def generate_llm_comment_05(payload: Dict[str, Any]) -> str:
    prompt = f"""
あなたはプロのゴルフコーチです。
寄り添い型で、前向きな文脈で説明してください。

以下の頭部の安定性に関する計測データと個別評価を元に、ユーザーに対する「総合評価」を生成してください。

【計測データと個別評価】
・頭部の移動量(mean): {payload['h_mean']:.1f}% -> 評価: {payload['move_eval']} ({payload['move_comment']})
・頭部位置の安定性(std): {payload['h_std']:.1f}% -> 評価: {payload['stability_eval']} ({payload['stability_comment']})
・下半身・前傾の影響: 評価: {payload['connection_eval']} ({payload['connection_comment']})

【意識すること】
・良い点があるからこそ改善すると伸びる、という前向きな文脈にする
・専門用語をそのまま使いすぎず、イメージしやすい言葉に言い換える
・出力はJSON形式のみとする

【出力形式】
以下の2つのキーを持つJSON形式のテキストのみを出力してください。Markdownの```jsonなどの装飾は不要です。
{{
    "overall_eval": "良好 または やや改善余地あり または 改善余地あり",
    "overall_comment": "上記のデータを踏まえた、ユーザーへの総合的なアドバイスを簡潔に（50文字〜80文字程度）"
}}
"""
    return call_llm(prompt)


def build_paid_05_head(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_head(raw)
    h = raw["head"]
    k = raw["knee"]
    conf = _conf(raw)

    spine_flag = judge_spine_flag(raw)

    # ==========================================================
    # 新しい構成：評価の内訳ロジック（ルールベースで事実を確定）
    # ==========================================================
    # 1. 移動量（mean）
    move_eval = ""
    move_comment = ""
    if h["mean"] <= 4.5:
        move_eval = "良好"
        move_comment = "頭部の位置は全体を通して安定しており、スイング軸の再現性は非常に良好です"
    elif h["mean"] <= 6.0:
        move_eval = "一定範囲内"
        move_comment = "頭部の移動は一定範囲に収まっており、軸の安定性は概ね保たれています"
    else:
        move_eval = "大きめ"
        move_comment = "頭部の移動量がやや大きく、インパクト時の軸の再現性に影響しやすい状態です"

    # 2. 安定性（std / σ）
    stability_eval = ""
    stability_comment = ""
    if h["std"] <= 3.5:
        stability_eval = "安定"
        stability_comment = "各場面での頭の位置が揃っており、スイング全体の再現性が確保されています"
    else:
        stability_eval = "やや不安定"
        stability_comment = "場面によって頭の位置にばらつきがあり、スイングの再現性にムラが出やすい状態です"

    # 3. 連動性（下半身・前傾の影響）
    connection_eval = ""
    connection_comment = ""
    if k["mean"] > 10.0:
        connection_eval = "下半身の影響あり"
        connection_comment = "膝のブレが大きく、それが頭部の安定性（スイング軸）にも影響している可能性があります"
    elif spine_flag == "bad" or spine_flag == "warn":
        connection_eval = "前傾の影響あり"
        connection_comment = "前傾姿勢の維持に課題があり、それが頭部の上下動やブレに繋がっています"
    else:
        connection_eval = "良好"
        connection_comment = "頭部と下半身の連動が整っており、ミート率を支える安定した土台ができています"

    # ==========================================================
    # 4. 総合評価の判定（LLMを使用）
    # ==========================================================
    llm_payload = {
        "h_mean": float(h["mean"]),
        "h_std": float(h["std"]),
        "move_eval": move_eval,
        "move_comment": move_comment,
        "stability_eval": stability_eval,
        "stability_comment": stability_comment,
        "connection_eval": connection_eval,
        "connection_comment": connection_comment
    }

    overall_eval = "判定中"
    overall_comment = "AIコーチが評価をまとめています…"

    try:
        print("LLM CALL START (05_head)")
        response_text = generate_llm_comment_05(llm_payload)
        print("LLM CALL END (05_head)")

        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        llm_data = json.loads(clean_text)
        
        overall_eval = llm_data.get("overall_eval", overall_eval)
        overall_comment = llm_data.get("overall_comment", overall_comment)

    except Exception as e:
        logging.exception("LLM JSON parse error in 05_head: %s", e)
        overall_eval = "やや改善余地あり"
        overall_comment = "頭部の安定性は軸の再現性に直結します。下半身との連動を意識して軸を安定させましょう。"

    # ==========================================================
    # 数値サマリーと戻り値の構築
    # ==========================================================
    summary_text = f"最大{h['max']:.1f}% / 平均{h['mean']:.1f}% / ばらつき{h['std']:.1f}%"

    new_layout_text = (
        f"移動量：{move_eval}\n"
        f"→ {move_comment}\n\n"
        f"安定性：{stability_eval}\n"
        f"→ {stability_comment}\n\n"
        f"連動性：{connection_eval}\n"
        f"→ {connection_comment}\n\n"
        f"総合：{overall_eval}\n"
        f"→ {overall_comment}"
    )

    good: List[str] = []
    bad: List[str] = []

    bench = [
        _bench_line("頭部ブレ(%)", "%", "mean", _le_ideal(6.0, "%"), current=float(h["mean"])),
        _bench_line("頭部ブレの再現性(%)", "%", "σ", _le_ideal(2.5, "%"), current=float(h["std"])),
        _bench_line("頭部ブレの安定目安(%)", "%", "mean", _le_ideal(4.5, "%"), current=float(h["mean"])),
    ]

    return {
        "title": "05. Head Stability（頭部）",
        "value": summary_text,
        "tags": j["tags"],
        "bench": bench,
        "good": good,
        "bad": bad,
        "pro_comment": new_layout_text,
        "new_eval_data": {
            "overall_eval": overall_eval,
            "summary_text": summary_text,
            "move": {"eval": move_eval, "comment": move_comment},
            "stability": {"eval": stability_eval, "comment": stability_comment},
            "connection": {"eval": connection_eval, "comment": connection_comment},
        }
    }
    
def judge_knee(raw: Dict[str, Any]) -> Dict[str, Any]:
    k = raw["knee"]
    h = raw["head"]

    tags: List[str] = []

    if k["mean"] > 9.0:
        tags.append("膝ブレ大")

    if h["mean"] > 6.0:
        tags.append("上半身不安定")

    return {"tags": tags}


import json
import logging
from typing import Dict, Any, List

def generate_llm_comment_06(payload: Dict[str, Any]) -> str:
    prompt = f"""
あなたはプロのゴルフコーチです。
寄り添い型で、前向きな文脈で説明してください。

以下の膝の安定性（スウェー防止）に関する計測データと個別評価を元に、ユーザーに対する「総合評価」を生成してください。

【計測データと個別評価】
・膝の移動量(mean): {payload['k_mean']:.1f}% -> 評価: {payload['move_eval']} ({payload['move_comment']})
・膝位置の安定性(std): {payload['k_std']:.1f}% -> 評価: {payload['stability_eval']} ({payload['stability_comment']})
・上下連動の影響: 評価: {payload['connection_eval']} ({payload['connection_comment']})

【意識すること】
・良い点があるからこそ改善すると伸びる、という前向きな文脈にする
・専門用語をそのまま使いすぎず、イメージしやすい言葉に言い換える
・出力はJSON形式のみとする

【出力形式】
以下の2つのキーを持つJSON形式のテキストのみを出力してください。Markdownの```jsonなどの装飾は不要です。
{{
    "overall_eval": "良好 または やや改善余地あり または 改善余地あり",
    "overall_comment": "上記のデータを踏まえた、ユーザーへの総合的なアドバイスを簡潔に（50文字〜80文字程度）"
}}
"""
    return call_llm(prompt)


def build_paid_06_knee(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_knee(raw)
    k = raw["knee"]
    h = raw["head"]
    conf = _conf(raw)
    spine_flag = judge_spine_flag(raw)

    # ==========================================================
    # 新しい構成：評価の内訳ロジック（ルールベースで事実を確定）
    # ==========================================================
    # 1. 移動量（mean）
    move_eval = ""
    move_comment = ""
    if k["mean"] <= 5.5:
        move_eval = "良好"
        move_comment = "膝の左右ブレは小さく抑えられており、下半身の土台は非常に安定しています"
    elif k["mean"] <= 8.5:
        move_eval = "一定範囲内"
        move_comment = "下半身の動きは許容範囲に収まっており、土台としての安定感は概ね保たれています"
    else:
        move_eval = "大きめ"
        move_comment = "膝の左右への移動量がやや大きく、スイングの軸が流れやすい（スウェー）傾向があります"

    # 2. 安定性（std / σ）
    stability_eval = ""
    stability_comment = ""
    if k["std"] <= 3.5:
        stability_eval = "安定"
        stability_comment = "膝の位置が各場面で揃っており、土台の再現性は高く保たれています"
    else:
        stability_eval = "やや不安定"
        stability_comment = "場面によって膝の位置にばらつきがあり、土台の安定感にムラが出やすい状態です"

    # 3. 連動性（上半身・前傾の影響）
    connection_eval = ""
    connection_comment = ""
    if h["mean"] > 6.5 and k["mean"] > 9.0:
        connection_eval = "上下の連動不足"
        connection_comment = "上半身（頭部）の揺れと連動して下半身も流れており、軸の安定性が活かしにくい状態です"
    elif spine_flag == "bad" or spine_flag == "warn":
        connection_eval = "前傾の影響あり"
        connection_comment = "前傾姿勢の変化が、下半身の安定性や踏み込みの質に一部影響している可能性があります"
    else:
        connection_eval = "良好"
        connection_comment = "下半身と上半身の軸が比較的連動しており、全体の安定感に繋がっています"

    # ==========================================================
    # 4. 総合評価の判定（LLMを使用）
    # ==========================================================
    llm_payload = {
        "k_mean": float(k["mean"]),
        "k_std": float(k["std"]),
        "move_eval": move_eval,
        "move_comment": move_comment,
        "stability_eval": stability_eval,
        "stability_comment": stability_comment,
        "connection_eval": connection_eval,
        "connection_comment": connection_comment
    }

    overall_eval = "判定中"
    overall_comment = "AIコーチが評価をまとめています…"

    try:
        print("LLM CALL START (06_knee)")
        response_text = generate_llm_comment_06(llm_payload)
        print("LLM CALL END (06_knee)")

        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        llm_data = json.loads(clean_text)
        
        overall_eval = llm_data.get("overall_eval", overall_eval)
        overall_comment = llm_data.get("overall_comment", overall_comment)

    except Exception as e:
        logging.exception("LLM JSON parse error in 06_knee: %s", e)
        overall_eval = "やや改善余地あり"
        overall_comment = "下半身の安定は飛距離と方向に直結します。膝のブレを抑え、より強固な土台作りを目指しましょう。"

    # ==========================================================
    # 数値サマリーと戻り値の構築
    # ==========================================================
    summary_text = f"最大{k['max']:.1f}% / 平均{k['mean']:.1f}% / ばらつき{k['std']:.1f}%"

    new_layout_text = (
        f"移動量：{move_eval}\n"
        f"→ {move_comment}\n\n"
        f"安定性：{stability_eval}\n"
        f"→ {stability_comment}\n\n"
        f"連動性：{connection_eval}\n"
        f"→ {connection_comment}\n\n"
        f"総合：{overall_eval}\n"
        f"→ {overall_comment}"
    )

    good: List[str] = []
    bad: List[str] = []

    bench = [
        _bench_line("膝ブレ(%)", "%", "mean", _le_ideal(9.0, "%"), current=float(k["mean"])),
        _bench_line("膝ブレの再現性(%)", "%", "σ", _le_ideal(2.5, "%"), current=float(k["std"])),
        _bench_line("膝ブレの安定目安(%)", "%", "mean", _le_ideal(5.5, "%"), current=float(k["mean"])),
    ]

    return {
        "title": "06. Knee Stability（膝）",
        "value": summary_text,
        "tags": j["tags"],
        "bench": bench,
        "good": good,
        "bad": bad,
        "pro_comment": new_layout_text,
        "new_eval_data": {
            "overall_eval": overall_eval,
            "summary_text": summary_text,
            "move": {"eval": move_eval, "comment": move_comment},
            "stability": {"eval": stability_eval, "comment": stability_comment},
            "connection": {"eval": connection_eval, "comment": connection_comment},
        }
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
    hand = tag_counter["コック過多"] + tag_counter["コック不足"]
    lower = (
        tag_counter["腰回転過多"]
        + tag_counter["腰回転不足"]
        + tag_counter["膝ブレ大"]
        + tag_counter["下半身不安定"]
    )
    stability = (
        tag_counter["膝ブレ大"]
        + tag_counter["頭部ブレ大"]
        + tag_counter["上半身不安定"]
    )

    if tag_counter["捻転差不足"] >= 2:
        return "体幹パワー不足型"

    if stability >= 2:
        return "安定性不足型"

    if tag_counter["肩回転過多"] + tag_counter["コック過多"] >= 2:
        return "操作過多型"

    if hand >= 1 and tag_counter["捻転差不足"] == 0 and lower == 0:
        return "手元主因型"

    if lower >= 2 and tag_counter["捻転差不足"] == 0:
        return "下半身主因型"

    return "バランス型"


def extract_priorities(tag_counter: Counter, max_items: int = 2) -> List[str]:
    order = [
        "膝ブレ大",
        "頭部ブレ大",
        "腰回転過多",
        "腰回転不足",
        "コック過多",
        "コック不足",
        "肩回転過多",
        "肩回転不足",
        "捻転差過多",
    ]

    result: List[str] = []

    # 捻転差不足は「複数箇所で一貫して不足」の時だけ優先課題にする
    if tag_counter.get("捻転差不足", 0) >= 2:
        result.append("捻転差不足")

    for t in order:
        if tag_counter.get(t, 0) > 0 and t not in result:
            result.append(t)
        if len(result) >= max_items:
            break

    return result[:max_items]
    
    
def generate_llm_comment_07(payload: Dict[str, Any]) -> str:
    # 既存のクラブ名取得ロジック
    raw_club = payload.get("club_type", "iron")
    club_map = {
        "driver": "ドライバー",
        "iron": "アイアン",
        "wood": "ウッド",
        "utility": "ユーティリティ"
    }
    club_name = club_map.get(raw_club, "ドライバー")

    # 性別が不明な場合でも自然な呼びかけにする設定
    gender_raw = payload.get("gender", "unknown")
    gender_context = "女性ゴルファー" if gender_raw == "female" else "男性ゴルファー" if gender_raw == "male" else "ゴルファー"
    
    # ==================================================
    # ここから追加：過去比較データ（ステータス差分）の生成
    # ==================================================
    comparison_text = ""
    comp_data = payload.get("comparison_data") or {}
    if comp_data.get("past_count", 0) > 0 and comp_data.get("deltas"):
        d = comp_data["deltas"]
        diffs = []
        if "shoulder_mean" in d: diffs.append(f"肩{d['shoulder_mean']:+.2f}")
        if "hip_mean" in d: diffs.append(f"腰{d['hip_mean']:+.2f}")
        if "head_mean" in d: diffs.append(f"頭{d['head_mean']:+.2f}")
        if "knee_mean" in d: diffs.append(f"膝{d['knee_mean']:+.2f}")
        if "x_factor_mean" in d: diffs.append(f"捻転差{d['x_factor_mean']:+.2f}")
        
        comparison_text = f"\n過去{comp_data['past_count']}回平均とのステータス差分: {'、'.join(diffs)}\n(※プラスは動き/ブレが拡大、マイナスは縮小を意味します。この数値を元に、ゲームのステータスが上がったような「成長点」と、次の「攻略課題」を必ず文章に含めて評価してください。)"
    # ==================================================
    prompt = f"""
あなたはプロのゴルフコーチです。
目の前の{gender_context}に寄り添い、解析データに基づいた「あなただけの分析」を伝えてください。

使用クラブ:
{club_name}

最優先テーマ:
{payload["priority"]}

数値データ:
腰回転 {payload["hip"]} / 捻転差 {payload["x_factor"]} / 前傾 {payload.get("spine", "unknown")} / 頭部 {payload["head"]} / 膝 {payload["knee"]}

{comparison_text}

アドバイス作成の指針:
1. 【称賛】まずは数値から見える「素晴らしい点」を具体的に1つ褒めてください。
2. 【課題】次に、最優先テーマに関連する「もったいないポイント」を指摘してください。
3. 【影響】その課題が、{club_name}特有のミス（例:ドライバーの飛距離ロス、アイアンの打点不安等）にどう繋がっているか、論理的に説明してください。
4. 【期待】最後に、08のドリルに取り組むことで、スイングがどう進化するかを伝え、期待感を高めてください。

文章ルール:
・文字数は120〜180文字程度。
・「1文目は〇〇」といった固定順序は不要です。自然な日本語の語り口を優先してください。
・必ず「{club_name}」のスイングであることを前提とした、具体的かつ前向きな表現を使うこと。
・「{gender_context}ならではの〜」といった視点も、可能であればエッセンスとして加えてください。
・箇条書き、記号、絵文字は禁止。出力は文章のみ。
"""

    return call_llm(prompt)
    
def build_paid_07_from_analysis(
    analysis: Dict[str, Any], 
    raw: Dict[str, Any], 
    comparison: Dict[str, Any] = None  # ←ここを引数に追加
) -> Dict[str, Any]:
    c = collect_tag_counter(analysis)
    swing_type = judge_swing_type(c)
    priorities = extract_priorities(c, 2)

    sh = raw.get("shoulder", {})
    h = raw.get("head", {})
    k = raw.get("knee", {})
    xf = raw.get("x_factor", {})
    conf = _conf(raw)
    frames = _frames(raw)
    spine_flag = judge_spine_flag(raw)

    # LLMに渡すデータセット（比較データ deltas を追加）
    llm_payload = {
        "club_type": raw.get("club_type", "iron"),
        "priority": priorities[0] if priorities else "不明",
        "swing_type": swing_type,

        "shoulder": sh,
        "head": h,
        "knee": k,
        "hip": raw.get("hip", {}),
        "wrist": raw.get("wrist", {}),
        "x_factor": xf,
        "spine_flag": spine_flag,

        "tags": dict(c),

        "confidence": conf,
        "frames": frames,

        "raw_metrics": raw,

        # ★【重要】過去との比較データをペイロードに追加
        "comparison_data": {
            "deltas": comparison.get("deltas", {}) if comparison else {},
            "past_count": comparison.get("past_sessions_count", 0) if comparison else 0
        },

        "coach_style": "empathetic"
    }

    lines: List[str] = []
    lines.append(f"今回のスイングは「{swing_type}」です（confidence {conf:.3f} / 区間 {frames} frames）。")
    
    # 比較データがある場合の注釈
    if comparison and comparison.get("past_sessions_count", 0) > 0:
        lines.append(f"※ 過去{comparison['past_sessions_count']}回の平均データと比較した進捗を分析しました。")
    else:
        lines.append("※ 初回の方は、まずは「最優先テーマ」だけを確認してください。")
    
    lines.append("")
    
    if priorities:
        p_str = "／".join(priorities)
        lines.append(f"数値上の最優先テーマは「{p_str}」です。")

        print("### LLM BLOCK ENTERED ###")

        try:
            print("LLM CALL START")

            # ★ LLMへの命令
            llm_text = generate_llm_comment_07(llm_payload)

            print("LLM CALL END")

            lines.append("")
            lines.append(llm_text)

        except Exception as e:
            import logging
            logging.exception("LLM summary failed: %s", e)

            lines.append("")
            lines.append(f"【システムエラー詳細】{str(e)}")
            
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
            "spine_flag": spine_flag,
            "comparison_info": comparison # メタデータにも含めておく
        },
    }

def build_free_07(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    無料版の07は「数値に基づく総合評価（プロ目線）」までは出す。
    ただし、02〜06の部位別の深掘り・改善手順・ドリル選定は有料に残す。
    """

    # --- 数値取得 ---
    sh = raw.get("shoulder", {})   # degrees
    hip = raw.get("hip", {})       # degrees
    w = raw.get("wrist", {})       # degrees
    head = raw.get("head", {})     # %
    knee = raw.get("knee", {})     # %
    xf = raw.get("x_factor", {})   # degrees
    spine = raw.get("spine", {})   # degrees

    conf = float(raw.get("confidence", 0.0))
    frames = int(raw.get("valid_frames", 0))

    # --- mean/std取得 ---
    sh_mean = float(sh.get("mean", 0.0))
    sh_std = float(sh.get("std", 0.0))

    hip_mean = float(hip.get("mean", 0.0))
    hip_std = float(hip.get("std", 0.0))

    w_mean = float(w.get("mean", 0.0))
    w_std = float(w.get("std", 0.0))

    # 捻転差は無料版も max 基準に統一
    xf_max = float(xf.get("max", 0.0))

    head_mean = float(head.get("mean", 0.0))
    knee_mean = float(knee.get("mean", 0.0))

    spine_mean = float(spine.get("mean", 0.0))

    # --- 前傾評価 ---
    spine_flag = judge_spine_flag(raw)

    # --- 無料版用タグ推定（やや緩和） ---
    tags: List[str] = []

    # 肩回転
    if sh_mean < 80:
        tags.append("肩回転不足")
    elif sh_mean > 110:
        tags.append("肩回転過多")

    # 腰回転
    if hip_mean < 30:
        tags.append("腰回転不足")
    elif hip_mean > 60:
        tags.append("腰回転過多")

    # 手首コック
    if w_mean < 40:
        tags.append("コック不足")
    elif w_mean > 80:
        tags.append("コック過多")

    # 捻転差（max基準）
    if xf_max < 30:
        tags.append("捻転差不足")
    elif xf_max > 70:
        tags.append("捻転差過多")

    # 安定性
    if head_mean > 6.5:
        tags.append("頭部ブレ大")
    if knee_mean > 10.0:
        tags.append("膝ブレ大")
        tags.append("下半身不安定")

    # 前傾維持
    if spine_flag == "warn":
        tags.append("前傾維持にばらつき")
    elif spine_flag == "bad":
        tags.append("前傾維持不安定")

    # --- 総合ロジック ---
    c = Counter(tags)
    swing_type = judge_swing_type(c)
    priorities = extract_priorities(c, 2)

    # --- 本文 ---
    lines: List[str] = []
    lines.append(f"今回のスイングは「{swing_type}」です（confidence {conf:.3f} / 区間 {frames} frames）。")

    if priorities:
        if len(priorities) == 1:
            lines.append(f"数値上の最優先テーマは「{priorities[0]}」です。")
        else:
            lines.append("数値上の優先テーマは「" + "／".join(priorities) + "」の2点です。")
    else:
        lines.append("数値上の優先テーマはありません。")

    lines.append("")

    # --- 優先テーマの根拠（やや柔らかく） ---
    if "頭部ブレ大" in priorities or ("頭部ブレ大" in c and len(priorities) == 0):
        lines.append(f"本動画では頭部ブレが mean {head_mean:.1f}% でやや大きく、軸の再現性に影響しています。")

    if "膝ブレ大" in priorities or ("膝ブレ大" in c and len(priorities) == 0):
        lines.append(f"本動画では膝ブレが mean {knee_mean:.1f}% でやや大きく、下半身の土台の再現性に影響しています。")

    if "捻転差不足" in priorities:
        lines.append(f"本動画では捻転差が max {xf_max:.1f}° でやや小さく、切り返しで力を溜める余地があります。")

    if "捻転差過多" in priorities:
        lines.append(f"本動画では捻転差が max {xf_max:.1f}° で大きめで、肩と腰の連動を整える余地があります。")

    if "腰回転過多" in priorities:
        lines.append(f"本動画では腰回転が mean {hip_mean:.1f}° で大きめで、下半身主導が強く出ています。")

    if "肩回転過多" in priorities:
        lines.append(f"本動画では肩回転が mean {sh_mean:.1f}° で大きめで、上半身主導が強く出ています。")

    if "コック過多" in priorities:
        lines.append(f"本動画では手首コックが mean {w_mean:.1f}° で大きめで、手元の操作量が増えやすい状態です。")

    if "コック不足" in priorities:
        lines.append(f"本動画では手首コックが mean {w_mean:.1f}° でやや小さく、力を溜める余地があります。")

    if "前傾維持不安定" in priorities or ("前傾維持不安定" in c and len(priorities) == 0):
        lines.append(f"本動画では前傾維持の崩れが大きく、上体の起き上がりがスイング軸の安定性に影響しています。（mean {spine_mean:.1f}°）")

    elif "前傾維持にばらつき" in priorities or ("前傾維持にばらつき" in c and len(priorities) == 0):
        lines.append(f"本動画では前傾維持にややばらつきがあり、場面ごとに少しズレが見られます。（mean {spine_mean:.1f}°）")

    lines.append("")

    # --- 良い点（やや緩和） ---
    good_points: List[str] = []

    if 80 <= sh_mean <= 110:
        good_points.append("肩の回旋量は基準レンジに収まっています。")

    if sh_std <= 18:
        good_points.append("肩の回し幅は大きく崩れておらず、上半身の再現性の土台があります。")

    if head_mean <= 6.5:
        good_points.append("頭部ブレは比較的抑えられており、軸は大きく崩れていません。")

    if knee_mean <= 10.0:
        good_points.append("膝ブレは比較的抑えられており、下半身は大きく流れていません。")

    if xf_max >= 30:
        good_points.append("捻転差は確保できており、切り返しの準備はできています。")

    if spine_flag == "ok":
        good_points.append("前傾維持は概ね安定しており、動きはおおむね再現できています。")

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
            "spine_flag": spine_flag,
        },
    }

# ==================================================
# 08 ドリル（07の優先順位連動 ＋ バリエーション拡充版）
# ==================================================
DRILL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "x_factor_turn",
        "name": "捻転差づくりドリル（肩先行ターン）",
        "category": "体幹",
        "tags": ["捻転差不足"],
        "purpose": "● 肩と腰の回転差（X-Factor）を作り、切り返しで力を溜めやすくする\n● 上半身と下半身の順序を整え、スイング全体の連動性を高める",
        "how": "① アドレスを作ったら、下半身は大きく流さずに肩を無理なく深く回す\n② 切り返しでは、下半身からゆるやかに動き始め、上半身が一瞬遅れて動く感覚をつかむ\n③ 10回×2セット、ゆっくりした素振りで行い、肩と腰の時間差を確認する",
    },
    {
        "id": "shoulder_control",
        "name": "肩回転コントロールドリル",
        "category": "上半身",
        "tags": ["肩回転過多"],
        "purpose": "● 肩の回しすぎを抑え、前傾維持を保ったまま回転量を整える\n● 回転の量ではなく、軸を保ちながら再現しやすいトップを作る",
        "how": "① 前傾維持を保ちながら、肩が地面に対して自然な傾きで回るようにハーフスイングする\n② 鏡や動画で、肩の高さや頭の位置が大きく変わっていないか確認する\n③ 10回×2セット、同じトップ位置に収まる感覚を優先して行う",
    },
    {
        "id": "hip_drive",
        "name": "腰主導ターンドリル",
        "category": "下半身",
        "tags": ["腰回転不足"],
        "purpose": "● 下半身から動き出す感覚を身につけ、手打ちを減らす\n● 腰の回転を使って全身の連動を高め、スイングの再現性を整える",
        "how": "① トップ付近で一度止まり、腕ではなく左腰を後ろに引く感覚から始動する\n② 上体はすぐに開かず、下半身が先に動く時間差を意識する\n③ 10回×2セット、素振りでリズムよく行い、腰から始まる動きを確認する",
    },
    {
        "id": "late_hit",
        "name": "レイトヒットドリル",
        "category": "手首",
        "tags": ["コック不足"],
        "purpose": "● 手首のコックを保つ時間を長くし、リリースの早まりを抑える\n● インパクト付近でヘッドを加速しやすい動きを身につける",
        "how": "① トップで1秒止まり、手首の角度を大きく変えずにグリップエンドがボール方向を向くように下ろす\n② 右腰の高さ付近までコックを保ち、体の正面で自然にほどける感覚をつかむ\n③ 10回×2セット、ウェッジか短いクラブでゆっくり行う",
    },
    {
        "id": "release_control",
        "name": "リリースコントロールドリル（LtoL）",
        "category": "手首",
        "tags": ["コック過多"],
        "purpose": "● 手首の使いすぎを抑え、体幹主導でフェースを管理しやすくする\n● 左右のミスを減らし、方向性の安定したインパクトを作る",
        "how": "① 腰から腰までの小さな振り幅で、腕とクラブのL字を保ちながら振る\n② 手先でフェースを返しすぎず、胸の向きでクラブを運ぶ感覚を意識する\n③ 15〜20回、低くまっすぐな球筋をイメージしながら行う",
    },
    {
        "id": "head_stable",
        "name": "頭部安定ドリル（壁チェック）",
        "category": "安定性",
        "tags": ["頭部ブレ大"],
        "purpose": "● 頭の位置関係を大きく崩さず、スイング軸の再現性を高める\n● 視界の安定によって打点と距離感を整えやすくする",
        "how": "① 壁や目印の近くで構え、アドレス時の頭の位置を確認する\n② フィニッシュまで頭の位置が大きく外れすぎないよう、体の回転で振る\n③ 素振り10回×2セット、頭を止めるよりも『軸ごと回る』感覚を意識する",
    },
    {
        "id": "knee_stable",
        "name": "膝ブレ安定ドリル",
        "category": "下半身",
        "tags": ["膝ブレ大"],
        "purpose": "● 膝の横流れを減らし、下半身の土台を安定させる\n● 体重移動と回転のバランスを整え、上半身の再現性も高める",
        "how": "① アドレス時の膝幅を保ちながら、大きく横に流れすぎないようにハーフスイングする\n② 体重移動は横に流すより、足裏で踏み込んで受け止める感覚を意識する\n③ 10回×2セット、膝の向きと骨盤の向きが大きくズレないように行う",
    },
    {
        "id": "sync_turn",
        "name": "全身同調ターンドリル（クロスアーム）",
        "category": "体幹",
        "tags": ["捻転差不足"],
        "purpose": "● 腕と胴体の一体感を高め、体全体で振る感覚を身につける\n● 部位ごとのタイミングのズレを減らし、ショット全体の再現性を高める",
        "how": "① 腕を胸の前でクロスし、手ではなく胸の向きでバックスイングする\n② 胸と腰が極端にバラバラにならず、自然な時間差を保ちながら回る\n③ 左右に10回ずつ、背骨を軸にした回転をゆっくり確認する",
    },
    {
        "id": "step_transition",
        "name": "足踏みステップドリル",
        "category": "下半身",
        "tags": ["腰回転不足", "下半身不安定", "捻転差不足"],
        "purpose": "● 下半身主導の切り返しタイミングを覚え、全身の連動を高める\n● スムーズな重心移動によって、再現性の高いスイングリズムを作る",
        "how": "① 足を狭めて構え、トップ付近で左足を軽く踏み出す\n② 左足の着地をきっかけに、下半身からダウンスイングを始める\n③ 10〜15回、止まらずにリズムよく連続で行い、自然な流れを身につける",
    },
    {
        "id": "tempo_rhythm",
        "name": "テンポ一定ドリル（メトロノーム）",
        "category": "再現性",
        "tags": ["ばらつき大"],
        "purpose": "● スイングのテンポを一定にし、部位ごとのタイミングのズレを減らす\n● 緊張時でも崩れにくい、自分の振りやすいリズムを作る",
        "how": "① 『イチ・ニ・サン』など一定のリズムを声に出しながら素振りする\n② メトロノームを使う場合は、毎回同じテンポでトップからフィニッシュまで振る\n③ 15〜20回、速さよりも同じリズムで振れることを優先して行う",
    },
    {
        "id": "towel_release",
        "name": "タオルスイング（リリース管理）",
        "category": "手首",
        "tags": ["コック過多", "リリースのばらつき大"],
        "purpose": "● リリースのタイミングを整え、手首と体幹の動きを合わせやすくする\n● インパクト付近で加速しやすいタイミングを身につける",
        "how": "① 先端を軽く結んだタオルを持ち、ダウンスイングで音が鳴る位置を確認する\n② できるだけ体の正面から左足前あたりで音が出るように振る\n③ 10回×2〜3セット、毎回同じ位置で音が鳴るかを確認する",
    },
    {
        "id": "spine_posture_keep",
        "name": "前傾維持ドリル",
        "category": "姿勢維持",
        "tags": ["前傾維持不安定", "前傾維持にばらつき"],
        "purpose": "● アドレスで作った姿勢をスイング中も保ち、上体の起き上がりを抑える\n● 頭部と下半身の再現性を高め、安定した回転軸を身につける",
        "how": "① アドレスを作ったら、胸と骨盤の距離感を大きく変えずにハーフスイングする\n② 頭の高さとお尻の位置関係が大きく変わりすぎないように回転する\n③ 10回×2セット、鏡または動画で姿勢変化を確認しながら行う",
    },
    {
        "id": "hip_depth_wall",
        "name": "お尻キープドリル（壁タッチ）",
        "category": "姿勢維持",
        "tags": ["前傾維持不安定", "膝ブレ大", "下半身不安定"],
        "purpose": "● 起き上がりや前後の軸ズレを抑え、前傾維持と下半身安定を同時に高める\n● 切り返し以降も骨盤の位置関係を保ち、回転の再現性を高める",
        "how": "① お尻が壁に軽く触れる位置でアドレスを作る\n② バックスイングからインパクトまで、お尻の接触感が大きく抜けすぎないようにハーフスイングする\n③ 10回×2セット、起き上がりや前への突っ込みが出ていないか確認する",
    },
    {
        "id": "chest_turn_posture",
        "name": "胸回転ドリル（前傾キープ）",
        "category": "体幹",
        "tags": ["前傾維持不安定", "頭部ブレ大", "肩回転過多", "肩回転不足"],
        "purpose": "● 胸郭主導で回転しながら前傾を保ち、頭部の過剰な動きを抑える\n● 肩だけで振る動きを減らし、体幹主体の回転を身につける",
        "how": "① クラブを胸の前に当てて両腕で軽く抱える\n② 前傾を保ったまま、胸の向きだけを右左にゆっくり回す\n③ 頭の高さと胸の傾きが大きく変わりすぎないよう、10回×2セット行う",
    },
]
def collect_all_tags(analysis: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    # 02〜07の各セクションからタグを収集
    for k in ["02", "03", "04", "05", "06", "07"]:
        sec = analysis.get(k)
        if sec and "tags" in sec:
            tags.extend(sec["tags"] or [])
    return tags

def select_drills_with_priority(tags: List[str], priorities: List[str], max_drills: int = 3) -> List[Dict[str, Any]]:
    """
    07で決まった優先課題(priorities)に合致するドリルを最優先で選出し、
    残りの枠を他の検知タグで埋める（カテゴリの重複は避ける）。
    """
    selected: List[Dict[str, Any]] = []
    used_categories: set = set()
    used_drill_ids: set = set()

    # 1. 最優先課題(priorities)に合致するドリルを最優先
    for p_tag in priorities:
        for d in DRILL_DEFINITIONS:
            if p_tag in d["tags"] and d["category"] not in used_categories:
                selected.append(d.copy())
                used_categories.add(d["category"])
                used_drill_ids.add(d["id"])
                break
        if len(selected) >= max_drills:
            break

    # 2. 枠が余っていれば、その他の検知タグ(tags)で補充
    if len(selected) < max_drills:
        tagset = set(tags)
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for d in DRILL_DEFINITIONS:
            if d["id"] in used_drill_ids:
                continue
            # 一致するタグの数でスコアリング
            score = len(set(d["tags"]) & tagset)
            if score > 0:
                scored.append((score, d))
        
        # スコア順（一致タグが多い順）にソート
        scored.sort(key=lambda x: x[0], reverse=True)
        
        for _, d in scored:
            if d["category"] not in used_categories:
                selected.append(d.copy())
                used_categories.add(d["category"])
                used_drill_ids.add(d["id"])
            if len(selected) >= max_drills:
                break

    # 3. フォールバック（何も選ばれない場合）
    if not selected:
        selected = [DRILL_DEFINITIONS[0].copy()]

    return selected

# ==================================================
# ここから追加：08ドリル生成用のLLM呼び出し関数（完全AIオリジナル生成テスト版）
# ==================================================
def generate_llm_drills_08(payload: Dict[str, Any], base_drills: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    priorities = payload.get("priorities", ["不明"])
    p_str = "／".join(priorities)
    club_type = payload.get("club_type", "不明")
    
    # 過去比較データ（ステータス差分）の抽出
    comparison_text = ""
    comp_data = payload.get("comparison_data") or {}
    if comp_data.get("past_count", 0) > 0 and comp_data.get("deltas"):
        d = comp_data["deltas"]
        diffs = []
        if "shoulder_mean" in d: diffs.append(f"肩{d['shoulder_mean']:+.2f}")
        if "hip_mean" in d: diffs.append(f"腰{d['hip_mean']:+.2f}")
        if "head_mean" in d: diffs.append(f"頭{d['head_mean']:+.2f}")
        if "knee_mean" in d: diffs.append(f"膝{d['knee_mean']:+.2f}")
        if "x_factor_mean" in d: diffs.append(f"捻転差{d['x_factor_mean']:+.2f}")
        comparison_text = f"\n過去{comp_data['past_count']}回平均との差分: {'、'.join(diffs)} (※プラスは動き/ブレが拡大、マイナスは縮小)"

    # ★ 既存のベース案をプロンプトから完全に消去し、AIにゼロから考案させます
    # ★ 指示（プロンプト）の書き換え部分
    prompt = f"""
あなたは世界トップクラスのプロゴルフコーチです。
ユーザーの解析データに基づき、最優先で取り組むべき「あなただけの完全オリジナル練習ドリル」を【2つ】考案してください。

使用クラブ: {club_type}
最優先課題: {p_str}
{comparison_text}

アドバイス作成の指針:
1. ユーザーの課題とステータス差分（あれば）を分析し、最も効果的なドリルをゼロから2つ考案してください。
2. 名前は「〇〇攻略ドリル」など、ゲーム感覚でモチベーションが上がるキャッチーな名称にすること。
3. 【目的】は、絶対に長文にならないよう注意し、専門用語を避けた分かりやすい表現で、「どんなエラーが直り、どうレベルアップするのか」を簡潔な2つの箇条書きにしてください（1文は短くスッキリと）。
4. 【手順】は合計3ステップ（①②③）のみ。練習場でスマホを見ながらすぐ実践できるよう、各ステップ【60文字以内】厳守で、理屈は入れすぎず端的なアクションを記述してください。
5. 手順の中にただの作業指示にならないよう、「右足の裏で地面をつかむ感じ」のような『意識する感覚（プロのコツ）』を全体で1つだけ必ず入れてください。
6. 【超重要】完全オリジナルとはいえ、アマチュアが絶対に怪我をしない、一般的なゴルフ理論に基づいた物理的に無理のない安全な動きにしてください。
7. 出力は以下のJSON形式の配列のみを厳守してください。

[
  {{
    "name": "ドリルの名前",
    "purpose": "● 短く分かりやすい目的1\\n● 短く分かりやすい目的2",
    "how": "① 【構え】60文字以内の短いアクション\\n② 【動作】60文字以内のアクション＋『意識する感覚』を1つ\\n③ 【確認】60文字以内の目安回数とチェックポイント"
  }},
  ...
]
"""
    try:
        response_text = call_llm(prompt)

        print("### 08 LLM RESPONSE ###")
        print(response_text)

        import json
        import re
        clean_text = re.sub(r'```json\s*', '', response_text)
        clean_text = re.sub(r'```', '', clean_text).strip()

        drills = json.loads(clean_text)
        return drills

    except Exception as e:
        import logging
        logging.exception("LLM drill generation failed: %s", e)

        return [
            {
                "name": d["name"],
                "purpose": d["purpose"],
                "how": d["how"]
            } for d in base_drills
        ]

def build_paid_08(analysis: Dict[str, Any], raw: Dict[str, Any], comparison: Dict[str, Any] = None) -> Dict[str, Any]:
    sec07 = analysis.get("07") or {}
    meta07 = sec07.get("meta") or {}
    priorities = meta07.get("priorities", [])

    all_tags = collect_all_tags(analysis)
    sh_std = float(raw.get("shoulder", {}).get("std", 0) or 0)
    spine_flag = judge_spine_flag(raw)

    head_mean = float(raw.get("head", {}).get("mean", 0) or 0)
    knee_mean = float(raw.get("knee", {}).get("mean", 0) or 0)

    # ばらつきは強いケースのみ
    if sh_std > 18:
        all_tags.append("ばらつき大")

    # 前傾は他要素にも影響がある場合のみ追加
    if spine_flag == "bad" and (head_mean > 6.0 or knee_mean > 9.0):
        all_tags.append("前傾維持不安定")

    # まずは既存の安全なルールベースでドリルを選出する（AIの参考にするため）
    selected_base_drills = select_drills_with_priority(
        all_tags,
        priorities,
        2
    )

    # LLMに渡すためのペイロードを作成
    llm_payload = {
        "priorities": priorities,
        "club_type": raw.get("club_type", "不明"),
        "comparison_data": {
            "deltas": comparison.get("deltas", {}) if comparison else {},
            "past_count": comparison.get("past_sessions_count", 0) if comparison else 0
        }
    }

    # ★ LLMを使って、ベースドリルを元に無限のバリエーションを生成
    generated_drills = generate_llm_drills_08(llm_payload, [])

    return {
        "title": "08. Training Drills（練習ドリル）",
        "drills": generated_drills
    }
    
# ==================================================
# 09 フィッティング（解析数値による全身統合・逆転ロジック版）
# ==================================================
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _norm_range(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    return _clamp01((v - lo) / (hi - lo))


def _norm_inverse(v: float, lo: float, hi: float) -> float:
    return 1.0 - _norm_range(v, lo, hi)


def calc_power_idx(raw: Dict[str, Any], club_type: str) -> int:
    sh = float(raw["shoulder"]["mean"])
    hip = float(abs(raw["hip"]["mean"]))
    wrist = float(raw["wrist"]["mean"])
    xf = float(raw["x_factor"]["mean"])

    # クラブ別の分岐を追加
    if club_type == "driver":
        a = _norm_range(sh, 90, 120)
        b = _norm_range(hip, 35, 65)
        c = _norm_range(wrist, 70, 90)
        d = _norm_range(xf, 36, 55)
    else:
        a = _norm_range(sh, 85, 105)
        b = _norm_range(hip, 36, 50)
        c = _norm_range(wrist, 70, 90)
        d = _norm_range(xf, 36, 55)

    return int(round((a + b + c + d) / 4.0 * 100))


def calc_stability_idx(raw: Dict[str, Any], club_type: str) -> int:
    head = float(raw["head"]["mean"])
    knee = float(raw["knee"]["mean"])

    # 前傾姿勢（spine）のブレ幅算出を追加
    spines = raw.get("spine_raw") or []
    if spines:
        base = float(spines[0])
        deltas = [abs(float(x) - base) for x in spines if x is not None]
        spine_mean = sum(deltas) / len(deltas) if deltas else 0.0
    else:
        spine_mean = 0.0

    # クラブ別の分岐と前傾(c)の評価を追加
    if club_type == "driver":
        a = _norm_inverse(head, 4.0, 11.0)
        b = _norm_inverse(knee, 5.0, 15.0)
        c = _norm_inverse(spine_mean, 3.0, 8.0)
    else:
        a = _norm_inverse(head, 3.0, 8.0)
        b = _norm_inverse(knee, 4.0, 10.0)
        c = _norm_inverse(spine_mean, 2.0, 6.0)

    # 評価項目が3つになったため、分母を 3.0 に変更
    return int(round((a + b + c) / 3.0 * 100))

# ==================================================
# 新規追加: 総合スコア計算
# ==================================================
def calculate_swing_score(raw: Dict[str, Any], club_type: str) -> int:
    """
    パワーと安定性のインデックスを合算し、100点満点の総合スコアを算出する
    (初心者でも40点程度からスタートできるよう底上げ調整済み)
    """
    power = calc_power_idx(raw, club_type)
    stability = calc_stability_idx(raw, club_type)
    
    # 双方を50%ずつの比率で計算（これが内部的な「素点」）
    raw_total = (power + stability) / 2.0
    
    # --- [修正] 優しさ補正ロジック ---
    # 計算式: 40 + (素点 * 0.6)
    # これにより、0点でも40点から始まり、100点満点は維持されます
    kind_total = 40 + (raw_total * 0.6)
    
    return int(round(kind_total))

# ==================================================
# 新規追加: 有料版向け 総合スコア表示ブロック
# ==================================================
def build_paid_score_block(score: int) -> Dict[str, Any]:
    # スコアに応じたカラーとコメントの出し分け（プロレベルを90点以上に設定）
    if score >= 90:
        color = "#ff3344"  # 赤系（プロ・エクセレント）
        eval_text = "素晴らしいスイングです！プロレベルの高い技術と再現性を兼ね備えています。"
    elif score >= 75:
        color = "#22bb55"  # 緑系（上級・グッド）
        eval_text = "非常に安定感があります。さらなる高み（90点超え）を目指しましょう！"
    elif score >= 55:
        color = "#ffaa00"  # オレンジ系（中級・ステップアップ）
        eval_text = "着実に基礎が身についています。まずは75点クリアを目指しましょう！"

    else:
        # ★ここを追加するだけです！★
        color = "#3b82f6"  # 青系（初級・伸びしろ）
        eval_text = "ここからがスタートです！まずは基本の姿勢を意識して、伸びしろを形にしていきましょう。"

    return {
        "type": "box",
        "layout": "vertical",
        "paddingAll": "lg",
        "backgroundColor": "#f8f9fa",
        "cornerRadius": "md",
        "margin": "md",
        "contents": [
            {
                "type": "text",
                "text": "🎯 総合スイングスコア",
                "weight": "bold",
                "color": "#666666",
                "size": "sm"
            },
            {
                "type": "box",
                "layout": "baseline",
                "margin": "md",
                "contents": [
                    {
                        "type": "text",
                        "text": str(score),
                        "weight": "bold",
                        "size": "4xl",
                        "color": color,
                        "flex": 0
                    },
                    {
                        "type": "text",
                        "text": " / 100点",
                        "weight": "bold",
                        "size": "md",
                        "color": "#999999",
                        "margin": "sm",
                        "flex": 0
                    }
                ]
            },
            {
                "type": "text",
                "text": eval_text,
                "size": "xs",
                "color": "#888888",
                "margin": "sm"
            }
        ]
    }

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

def generate_llm_driver_fitting_ai(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    AIフィッティングエンジン：2026年最新の全ギア（純正・カスタム問わず）から、
    算出された物理スペックに最も合致する1本を特定する。
    """
    hs_val = 0
    try:
        hs_val = float(payload.get("hs", 0))
    except:
        pass
    
    gender = payload.get("gender", "none")
    gender_instruction = ""
    if gender == "female":
        if hs_val <= 33:
            gender_instruction = "【特別条件】対象は女性でHS33以下のため、必ず「レディースクラブ（レディースモデル）」を提案してください。"
        elif hs_val < 36:
            gender_instruction = "【特別条件】対象は女性でHS34〜35のため、「メンズモデル（軽量モデル含む）」と「レディースモデル」を両方合わせて検討し、最適なものを提案してください。"
        else:
            gender_instruction = "【特別条件】対象は女性でHS36以上あるため、しっかり振れるセッティングが必要ですが、シャフトは重すぎず硬すぎない「40g台のカスタムシャフト」や「メンズの軽量モデル」を中心に提案してください。男性用のハードなシャフト（50g台のツアー系など）は避けてください。"

    raw_miss = payload.get("raw_miss", "")
    miss_instruction = f"【特別条件】ユーザーからの実際の悩み・ミス傾向の入力は「{raw_miss}」です。この生の声を深く解釈し、その悩みを最も的確に解決・補正できるヘッド設計やシャフトの組み合わせを優先して選定してください（例：スライスやフックが明らかな場合、安易に直進性モデルを選ぶのではなく、そのミスを相殺できるモデルを選定する等）。"

    prompt = f"""
あなたは世界中のゴルフクラブ・シャフトの剛性分布（EIプロファイル）を学習した「AIフィッティングエンジン」です。
算出された【{payload['kp']}調子】という物理条件は絶対的な正解であり、これと矛盾するシャフト提案はシステムエラーとみなします。

{gender_instruction}
{miss_instruction}

【算出済み物理制約】
■ 推奨重量: {payload['weight']} / 推奨フレックス: {payload['flex']}
■ 推奨調子: {payload['kp']}調子（※この特性を物理的に厳守せよ）

【解析データ】
ユーザーの生の悩み: {payload.get('raw_miss', '')} / HS: {payload['hs']}m/s / 軸ブレ: {payload['stability_val']}% / タメ: {payload['wrist_cock']}度

指令（ミッション）:
1. 2026年4月現在の最新機材（Callaway, Cobra, PING, TaylorMade, Titleist 等を含む国内外の幅広い実在メーカー）から、最適と思われるヘッドを第1候補〜第3候補まで3種類選定してください。架空のクラブ名は厳禁です。
2. シャフトについても、純正・カスタム問わず算出スペックに合致する最新モデルを第1候補〜第3候補まで3種類選定してください。必ず【{payload['kp']}調子】であることを再確認してください。
3. 選定理由は長文を避け、ヘッドは「モデル名」＋「選定理由（例：直進性重視など）を一言で」、シャフトは「モデル名」＋「逆転ロジック等の狙い（例：タメを作るなど）を一言で」簡潔にまとめてください。
4. 以下のJSONフォーマットの配列を厳守して出力してください。

{{
  "proposals": [
    {{
      "rank": "第1候補（Best Fit）",
      "head": "〇〇モデル",
      "head_reason": "高MOIで打点のブレを補正するため",
      "shaft": "〇〇シャフト",
      "shaft_reason": "手元側のしなりで『タメの間』を生成するため"
    }},
    {{
      "rank": "第2候補（Alternative 1）",
      "head": "...",
      "head_reason": "...",
      "shaft": "...",
      "shaft_reason": "..."
    }},
    {{
      "rank": "第3候補（Alternative 2）",
      "head": "...",
      "head_reason": "...",
      "shaft": "...",
      "shaft_reason": "..."
    }}
  ]
}}
"""
    try:
        response_text = call_llm(prompt)
        import json, re
        clean_text = re.sub(r'```json\s*|```', '', response_text).strip()
        return json.loads(clean_text)
    except:
        return {
            "proposals": [
                {
                    "rank": "第1候補（Best Fit）",
                    "head": "2026最新モデル",
                    "head_reason": "高MOI設計で直進性を向上させるため",
                    "shaft": "推奨カスタムシャフト",
                    "shaft_reason": "タメの間を生成しタイミングを安定させるため"
                }
            ]
        }
        
def build_paid_09(raw: Dict[str, Any], user_inputs: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    import logging
    from typing import List, Dict, Any

    logging.warning("[DEBUG] build_paid_09 START (Using aggregate raw data)")

    # --- 1. 基礎データの取得と正規化（完全維持） ---
    hs = _to_float_or_none(user_inputs.get("head_speed"))
    miss = _norm_miss(user_inputs.get("miss_tendency"))
    gender = _norm_gender(user_inputs.get("gender"))
    club_type = user_inputs.get("club_type", "driver")

    try:
        power_idx = int(calc_power_idx(raw, club_type))
    except Exception:
        logging.warning("[DEBUG] calc_power_idx failed; fallback")
        power_idx = 10
    try:
        stability_idx = int(calc_stability_idx(raw, club_type))
    except Exception:
        logging.warning("[DEBUG] calc_stability_idx failed; fallback")
        stability_idx = 10

    # --- 2. 実測値（単数形キー）の抽出（完全維持） ---
    def _f(path: List[str], default: float) -> float:
        cur: Any = raw
        try:
            for k in path:
                if not isinstance(cur, dict): return default
                cur = cur.get(k)
            return float(cur)
        except Exception: return default

    h_mean = _f(["head", "mean"], 10.0)
    k_mean = _f(["knee", "mean"], 10.0)
    stability_val = (h_mean + k_mean) / 2.0
    xf_max = _f(["x_factor", "max"], 0.0)
    wrist_cock = _f(["wrist", "mean"], 0.0)
    max_wrist = _f(["wrist", "max"], wrist_cock)
    wrist_std = _f(["wrist", "std"], 0.0)

    # --- 3. 判定バンド（完全維持） ---
    def _band_stability(val: float) -> str:
        if val < 5.0: return "stable"
        if val < 7.0: return "normal"
        return "unstable"
    def _band_xfactor(val: float) -> str:
        if val < 30.0: return "low"
        if val <= 70.0: return "mid"
        return "high"
    def _band_tame(m_w: float, c_w: float, s_w: float) -> str:
        if m_w < 45.0: return "shallow"
        if m_w < 75.0: return "normal"
        if c_w < 45.0 or s_w >= 15.0: return "unstable_deep"
        return "deep"

    stab_band = _band_stability(stability_val)
    xf_band = _band_xfactor(xf_max)
    tame_band = _band_tame(max_wrist, wrist_cock, wrist_std)

    if hs is not None:
        hs_level = "low" if hs < 38 else ("mid" if hs <= 45 else "high")
    else:
        hs_level = "low" if power_idx < 12 else ("mid" if power_idx <= 18 else "high")
    cock_level = "shallow" if wrist_cock < 45.0 else ("deep" if wrist_cock > 75.0 else "normal")
    cock_label = "浅め" if cock_level == "shallow" else ("深め" if cock_level == "deep" else "標準")

    # --- 4. スペック判定と1行理由の生成（既存ロジック100%維持） ---

    # 【重量】
    if hs is not None:
        if gender == "female":
            weight = "30〜40g" if hs < 35 else ("40〜50g" if hs < 40 else "50g前後")
        else:
            if hs < 35: weight = "40〜50g"
            elif hs < 40: weight = "50g前後"
            elif hs < 45: weight = "50〜60g"
            else: weight = "60〜70g"
        if (hs >= 40 and stability_val > 7.0) or (cock_level == "deep" and stability_val > 5.0):
            weight = "60g前後"
            w_reason = f"HSと深いタメ、軸ブレ{stability_val:.1f}%を考慮し重量で安定化"
        else:
            w_reason = f"HS{hs:.1f}m/sとタメ平均に応じた身体負荷の最適化"
    else:
        weight = {"low": "40〜50g", "mid": "50〜60g", "high": "60〜70g"}[hs_level]
        w_reason = f"パワー指数{power_idx}に基づくAI推奨重量の算出"

    # 【硬さ】
    if hs is not None:
        flex_map = [(33, "L〜A"), (38, "A〜R"), (42, "R〜SR"), (46, "SR〜S"), (50, "S〜X")]
        flex = next((f for h, f in flex_map if hs < h), "X")
        if xf_max > 70.0 or cock_level == "deep":
            flex = "一ランク硬め"
            f_reason = f"強い捻転差(max{xf_max:.1f}°)とタメによる負荷への剛性確保"
        else:
            f_reason = f"HS{hs:.1f}m/sに対する標準的な適正剛性"
    else:
        flex = {"low": "A〜R", "mid": "R〜SR", "high": "SR〜S"}[hs_level]
        f_reason = f"AI判定レベルに基づく標準的な剛性"

    # 【調子】（逆転ロジック維持）
    if miss == "right":
        if wrist_cock < 45.0:
            kp = "元"
            k_reason = "手元側のしなりにより『タメの間』を意図的に作り出し、右ミスを抑制"
        elif tame_band == "unstable_deep" or wrist_std >= 15.0:
            kp = "元"
            k_reason = "タメのばらつきを抑えるため、手元調子でタイミングを安定化"
        elif stability_val > 7.0:
            kp = "中"
            k_reason = f"軸ブレ{stability_val:.1f}%を考慮し、挙動の安定性を優先"
        else:
            kp = "先〜中"
            k_reason = "右ミスに対し、つかまりを助ける先調子系を基準に選定"
    elif miss == "left":
        kp = "中〜元"
        k_reason = "左ミス防止：先端の動きを抑え、つかまり過ぎによるミスを抑制"
    else:
        kp = "中"
        k_reason = "ニュートラルな挙動の中調子で、操作性と安定性のバランスを最適化"

    # 【トルク】
    if stability_val >= 9.0:
        tq = "3.0〜4.0"
        t_reason = f"軸ブレ実測{stability_val:.1f}%を抑え、打点安定性を最優先"
    elif stability_val >= 5.0:
        tq = "3.5〜5.0"
        t_reason = "平均的な軸ブレ量に基づき、標準的なトルク帯を選択"
    else:
        tq = "4.5〜6.0"
        t_reason = "高い安定性を活かし、シャフトの挙動を使いやすく設定"

    # --- 5. AI提案の取得と表示テーブルの構築（3項目構成） ---
    llm_payload = {
        "hs": hs or power_idx, "miss": miss, "weight": weight, "flex": flex, 
        "kp": kp, "wrist_cock": f"{wrist_cock:.1f}", "stability_val": f"{stability_val:.1f}",
        "gender": gender,
        "raw_miss": str(user_inputs.get("miss_tendency", ""))
    }
    ai_fit = generate_llm_driver_fitting_ai(llm_payload)

    final_rows = []

    # 項目1: AI推奨提案（第1〜第3候補）
    proposals = ai_fit.get("proposals", [])
    for p in proposals:
        final_rows.append({
            "item": p.get("rank", "推奨候補"),
            "guide": "AI推奨モデル",
            "reason": (
                f"ヘッド：{p.get('head', '')}（理由：{p.get('head_reason', '')}）<br><br>"
                f"シャフト：{p.get('shaft', '')}（理由：{p.get('shaft_reason', '')}）"
            )
        })

    # 項目2: 診断サマリ
    final_rows.append({
        "item": "診断サマリ",
        "guide": "今回の分析根拠",
        "reason": (
            f"● 軸ブレ：{stability_val:.1f}%（安定性）<br>"
            f"● 捻転差：max {xf_max:.1f}°（パワー）<br>"
            f"● タメ平均：{wrist_cock:.1f}°（リリース）"
        )
    })
    
    # ユーザー要望により「最適シャフトスペック」は削除

    return {
        "title": "09. Driver Fitting Guide（AI推奨）",
        "table": final_rows,
        "note": "※2026年4月現在の最新AIマーケットデータに基づく算出結果です。",
        "meta": {
            "power_idx": power_idx, "stability_idx": stability_idx, "wrist_cock": wrist_cock,
            "head_speed": hs, "stability_val": stability_val, "xf_max": xf_max, "max_wrist": max_wrist,
        },
    }

# ==================================================
# 10 まとめ（07, 08, 09 の結果を動的に統合した最終総括版）
# ==================================================
def build_paid_10(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    各セクションの解析結果を引用し、ユーザーに最適な改善ストーリーを提示する。
    - 07 から：スイング型と最優先課題
    - 08 から：取り組むべきメインドリル
    - 09 から：推奨シャフトとその選定根拠
    """
    sec07 = analysis.get("07") or {}
    meta07 = sec07.get("meta") or {}
    swing_type = meta07.get("swing_type", "バランス型")
    priorities = meta07.get("priorities", [])
    spine_flag = meta07.get("spine_flag", "ok")

    sec08 = analysis.get("08") or {}
    drills = sec08.get("drills", [])
    drill_names = [d["name"] for d in drills if isinstance(d, dict) and d.get("name")]

    sec09 = analysis.get("09") or {}
    table = sec09.get("table", [])
    has_shaft_section = bool(table)

    kp_info = next((item for item in table if item.get("item") == "キックポイント"), {})
    kp_guide = kp_info.get("guide", "中")
    kp_reason = kp_info.get("reason", "")

    summary_text: List[str] = []

    summary_text.append(f"今回の解析結果、あなたのスイングは『{swing_type}』に分類されます。")

    if priorities:
        p_str = "／".join(priorities)
        summary_text.append(f"現在、優先して取り組みたいテーマは『{p_str}』です。")

        if drill_names:
            summary_text.append(
                f"まずは推奨ドリルの中でも「{drill_names[0]}」を中心に取り組むことで、動きの再現性を整えやすくなります。"
            )
            summary_text.append(
                "複数の動きを一度に変えようとするよりも、優先テーマを一つずつ整理していくことが改善につながります。"
            )
    else:
        summary_text.append(
            "全体として大きな破綻はなく、バランスよく振れています。提示されたドリルを活用しながら、さらに再現性を高めていきましょう。"
        )

    # 前傾は補助要因として短めに記載
    if spine_flag == "bad":
        summary_text.append(
            "前傾姿勢の変化にも注意しながら、上体角度の再現性を整えていくと全体の安定につながりやすくなります。"
        )
    elif spine_flag == "warn":
        summary_text.append(
            "前傾姿勢にややばらつきが見られるため、上体角度の再現性も意識するとさらに安定しやすくなります。"
        )
    else:
        summary_text.append(
            "前傾姿勢は概ね安定しており、スイング全体の再現性を支える要素になっています。"
        )

    summary_text.append("")

    if has_shaft_section:
        summary_text.append(
            f"道具の面では、今回のスイング特性に合わせて『{kp_guide}調子』のシャフトを提案しました。"
        )
        if kp_reason:
            summary_text.append(f"【選定根拠】{kp_reason}")

        summary_text.append("")
        summary_text.append(
            "『練習による動きの整理』と『スイング特性に合ったシャフト選び』を組み合わせることで、より安定した結果につながりやすくなります。"
        )
    else:
        summary_text.append(
            "まずは練習によって優先テーマを整理し、動きの再現性を高めていくことが安定した結果につながります。"
        )

    summary_text.append("次回の解析では、今回の優先テーマに対して数値がどのように変化したかを確認していきましょう。")

    summary_text.append("")
    summary_text.append("今後の練習とラウンドが、より良いものになることを願っています。")

    return {
        "title": "10. Summary（まとめ）",
        "text": summary_text,
    }
# ==================================================
# Analysis builder
# ==================================================
def judge_spine_flag(raw: Dict[str, Any]) -> str:
    spines = raw.get("spine_raw") or []

    if not spines:
        return "ok"

    base = float(spines[0])
    deltas = [abs(float(x) - base) for x in spines if x is not None]

    if not deltas:
        return "ok"

    delta_mean = sum(deltas) / len(deltas)

    spine_top = float(raw.get("spine_top") or 0)
    spine_impact = float(raw.get("spine_impact") or 0)

    delta_top = abs(spine_top - base) if spine_top else 0
    delta_impact = abs(spine_impact - base) if spine_impact else 0

    # --- [修正箇所] 評価の最適化 ---
    # 動画の「ほぼ緑（平均値が良い）」という印象と合わせるため、
    # トップ・インパクトの一瞬のズレには緩和係数（0.7など）を掛け、
    # 一瞬のブレだけで極端にBad判定へ引っ張られないように調整します。
    worst = max(delta_mean, delta_top * 0.7, delta_impact * 0.7)

    # --- [修正箇所] クラブ別のしきい値を参照するように変更 ---
    # rawの中に thresholds があればそこから取得、なければ初・中級者デフォルト(5.0)を使用
    thresholds = raw.get("thresholds") or {}
    spine_limit = float(thresholds.get("spine_limit", 5.0))

    # OK基準：設定された spine_limit (5.0〜6.5) を使用
    if worst <= spine_limit:
        return "ok"
    
    # Warn基準：しきい値の約1.5倍〜2倍程度（ここでは1.8倍程度に設定）
    # 例：アイアンなら 9.0度、ドライバーなら 12.0度くらいまでが Warn
    if worst <= spine_limit * 1.8:
        return "warn"
    
    return "bad"

def judge_address_posture(raw: Dict[str, Any]) -> Dict[str, str]:

    angle = raw.get("base_spine_angle")

    try:
        angle = float(angle)
    except Exception:
        return {
            "label": "判定保留",
            "comment": "構えの静止区間が十分に確認できませんでした。"
        }

    if angle <= 0:
        return {
            "label": "判定保留",
            "comment": "構えの静止区間が十分に確認できませんでした。"
        }

    if angle < 18:
        return {
            "label": "やや浅め",
            "comment": "上体が立ちやすく、回転量が不足しやすい姿勢です。"
        }

    if angle > 38:
        return {
            "label": "やや深め",
            "comment": "前傾量は確保されていますが、動作中に維持負荷がかかりやすい姿勢です。"
        }

    return {
        "label": "適正",
        "comment": "回転しやすい前傾角が作れています。"
    }

def judge_spine_maintain_display(raw: Dict[str, Any]) -> Dict[str, str]:
    flag = judge_spine_flag(raw)

    if flag == "ok":
        return {
            "label": "安定",
            "comment": "前傾角の変化は小さく、回転動作の再現性は安定しています。"
        }

    if flag == "warn":
        return {
            "label": "やや不安定",
            "comment": "前傾角の変化がやや見られ、動作の再現性に影響する可能性があります。"
        }

    return {
        "label": "不安定",
        "comment": "前傾角の変化がやや大きく、回転動作の安定性に影響しています。"
    }

# ==================================================
# Helper Functions (比較・分析・スコアリング統合ブロック)
# ==================================================
from typing import Dict, Any, List
import logging
from google.cloud import firestore

def get_past_reports(user_id: str, current_report_id: str, club_type: str, limit: int = 5):
    """
    インデックスエラーを100%回避する安全版。
    """
    reports_ref = db.collection("reports")
    docs = reports_ref.where("user_id", "==", user_id).get()
    
    past_data = []
    for doc in docs:
        if doc.id == current_report_id:
            continue
        r = doc.to_dict()
        if r.get("status") != "DONE":
            continue
        r_club = r.get("raw", {}).get("club_type") or r.get("user_inputs", {}).get("club_type")
        if r_club != club_type:
            continue
        past_data.append(r)
            
    past_data.sort(key=lambda x: x.get("completed_at") or "", reverse=True)
    return past_data[:limit]

def calculate_full_comparison(current_raw: dict, past_reports: list):
    """
    全解析項目の過去平均との差分を計算し、グラフ用の履歴データを作成します。
    """
    if not past_reports:
        return {"past_sessions_count": 0, "deltas": {}, "history": []}

    metrics_keys = ["shoulder", "hip", "wrist", "head", "knee", "x_factor", "spine"]
    deltas = {}
    past_raws = [r.get("raw", {}) for r in past_reports]
    num_past = len(past_raws)
    
    for key in metrics_keys:
        curr_val = current_raw.get(key, {}).get("mean", 0)
        avg_val = sum(r.get(key, {}).get("mean", 0) for r in past_raws) / num_past
        deltas[f"{key}_mean"] = round(curr_val - avg_val, 2)
        
        curr_std = current_raw.get(key, {}).get("std", 0)
        avg_std = sum(r.get(key, {}).get("std", 0) for r in past_raws) / num_past
        deltas[f"{key}_std"] = round(curr_std - avg_std, 2)

    return {
        "past_sessions_count": num_past,
        "deltas": deltas,
        "history": [
            {
                "date": r.get("completed_at"), 
                "score": r.get("total_score") or r.get("raw", {}).get("total_score", 0),
                "raw": r.get("raw")
            } for r in past_reports
        ]
    }

def build_comparison_block(comparison: Dict[str, Any]) -> Dict[str, Any]:
    deltas = comparison.get("deltas", {})
    count = comparison.get("past_sessions_count", 0)
    
    label_map = {
        "shoulder": "肩の回転", "hip": "腰の回転", "wrist": "手首の角度",
        "head": "頭のブレ", "knee": "膝の動き", "x_factor": "捻転差(X-Factor)",
        "spine": "前傾角度(全体)", "spine_top": "トップでの前傾", "spine_impact": "インパクトでの前傾"
    }

    detailed_results = []
    for key, label in label_map.items():
        d_mean = deltas.get(f"{key}_mean") if f"{key}_mean" in deltas else deltas.get(key)
        if d_mean is None: continue

        is_positive_metric = key in ["shoulder", "hip", "x_factor"]
        is_improved = (d_mean > 0) if is_positive_metric else (d_mean < 0)
        
        status_icon = "✅" if is_improved else "⚠️"
        diff_text = f"+{d_mean}" if d_mean > 0 else f"{d_mean}"
        
        if is_positive_metric:
            comment = f"過去{count}回の平均より動きが深くなり、良い傾向です。" if is_improved else f"過去{count}回の平均より動きが浅くなっています。"
        else:
            comment = f"過去{count}回の平均よりブレが少なく、安定しています。" if is_improved else f"過去{count}回の平均よりブレが大きくなっています。"
        
        detailed_results.append({
            "label": label, "diff": diff_text, "status": status_icon,
            "is_improved": is_improved, "comment": comment
        })

    return {
        "title": "全項目・過去比較分析 (Premium)",
        "subtitle": f"過去{count}回の平均データとの全指標比較",
        "detailed_results": detailed_results,
        "deltas": deltas,
        "past_sessions_count": count
    }

def build_paid_07_from_analysis(analysis: Dict[str, Any], raw: Dict[str, Any], comparison: Dict[str, Any] = None) -> Dict[str, Any]:
    c = collect_tag_counter(analysis)
    swing_type = judge_swing_type(c)
    priorities = extract_priorities(c, 2)
    conf = _conf(raw)
    frames = _frames(raw)

    llm_payload = {
        "club_type": raw.get("club_type", "iron"),
        "priority": priorities[0] if priorities else "不明",
        "swing_type": swing_type,
        "raw_metrics": raw,
        "tags": dict(c),

        "hip": raw.get("hip", {}),
        "x_factor": raw.get("x_factor", {}),
        "head": raw.get("head", {}),
        "knee": raw.get("knee", {}),
        "spine": judge_spine_flag(raw),
        
        "comparison_data": {
            "deltas": comparison.get("deltas", {}) if comparison else {},
            "past_count": comparison.get("past_sessions_count", 0) if comparison else 0
        },
    }
    
    lines: List[str] = []
    lines.append(f"今回のスイングは「{swing_type}」です（confidence {conf:.3f} / 区間 {frames} frames）。")
    
    # 比較データがある場合でも、AIへのデータ注入は行わず、回数の表示のみに留めます
    if comparison and comparison.get("past_sessions_count", 0) > 0:
        lines.append(f"※ 過去{comparison['past_sessions_count']}回の平均データと比較した進捗を分析しました。")
    else:
        lines.append("※ 初回の方は、まずは「最優先テーマ」だけを確認してください。")
    
    lines.append("")
    
    if priorities:
        p_str = "／".join(priorities)
        lines.append(f"数値上の最優先テーマは「{p_str}」です。")

        try:
            # AI評価の生成を実行します
            llm_text = generate_llm_comment_07(llm_payload)
            if llm_text:
                lines.append("")
                lines.append(str(llm_text))
        except Exception as e:
            import logging
            logging.exception("LLM summary failed: %s", e)
            lines.append("")
            lines.append("【システムエラー】AI評価の生成に失敗しました。")
            
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
            "spine_flag": judge_spine_flag(raw)
        },
    }

def build_analysis(
    raw: Dict[str, Any], premium: bool, report_id: str, user_inputs: Dict[str, Any],
    comparison: Dict[str, Any] = None, user_profile: Dict[str, Any] = None, user_plan: str = "free"
) -> Dict[str, Any]:
    ui = user_inputs or {}
    club_type = raw.get("club_type") or ui.get("club_type", "unknown")
    analysis: Dict[str, Any] = {}

    is_paid_plan = user_plan in ["single", "monthly"]
    is_monthly = user_plan == "monthly"

    if is_paid_plan:
        try:
            total_score = calculate_swing_score(raw, club_type)
            analysis["00_score"] = build_paid_score_block(total_score)

            if is_monthly and comparison and "deltas" in comparison:
                analysis["00_comparison"] = build_comparison_block(comparison)

        except Exception as e:
            logging.error(f"Score calculation failed: {e}")
            analysis["00_score"] = build_paid_score_block(70)

    analysis["01"] = build_section_01(raw, club_type)
    spine_flag = judge_spine_flag(raw)

    if user_plan == "free":
        analysis["07"] = build_free_07(raw)
        return analysis

    analysis["02"] = build_paid_02_shoulder(raw, seed=report_id)
    analysis["03"] = build_paid_03_hip(raw, seed=report_id)
    analysis["04"] = build_paid_04_wrist(raw, seed=report_id)
    analysis["05"] = build_paid_05_head(raw, seed=report_id)
    analysis["06"] = build_paid_06_knee(raw, seed=report_id)

    # 07評価（LLM）
    if is_monthly:
        analysis["07"] = build_paid_07_from_analysis(
            analysis,
            raw,
            comparison=comparison
        )
    else:
        # single は今回のみの詳細分析。過去比較は出さない
        analysis["07"] = build_paid_07_from_analysis(
            analysis,
            raw,
            comparison=None
        )

    # --- 07前傾補足ロジック ---
    if spine_flag == "ok":
        analysis["07"].setdefault("text", []).append("【前傾維持】前傾角の変化は小さく、回転動作の再現性は安定しています。")
    elif spine_flag == "warn":
        analysis["07"].setdefault("text", []).append("【前傾維持】スイング中に前傾角の変化がやや見られ、動作の再現性に影響する可能性があります。")
    elif spine_flag == "bad":
        analysis["07"].setdefault("text", []).append("【前傾維持】スイング中の前傾角の変化がやや大きく、回転動作の安定性に影響しています。")

    if spine_flag == "bad":
        analysis["07"].setdefault("tags", [])
        if "前傾維持不安定" not in analysis["07"]["tags"]:
            analysis["07"]["tags"].append("前傾維持不安定")
    elif spine_flag == "warn":
        analysis["07"].setdefault("tags", [])
        if "前傾維持やや不安定" not in analysis["07"]["tags"]:
            analysis["07"]["tags"].append("前傾維持やや不安定")

    # 月額プランの場合は過去データを、単発プランの場合は None を渡します
    analysis["08"] = build_paid_08(analysis, raw, comparison=comparison if is_monthly else None)

    if club_type == "driver" and (ui.get("head_speed") is not None or ui.get("miss_tendency") or ui.get("gender")):
        analysis["09"] = build_paid_09(raw, ui, analysis)

    analysis["10"] = build_paid_10(analysis)

    # --- [既存の10前傾補足ロジックを完全復元] ---
    head_mean = float(raw.get("head", {}).get("mean", 0.0))
    knee_mean = float(raw.get("knee", {}).get("mean", 0.0))

    if spine_flag == "ok":
        analysis["10"].setdefault("text", []).insert(0, "前傾姿勢は全体として比較的安定しており、スイングの再現性を支えています。")
    elif spine_flag == "warn":
        analysis["10"].setdefault("text", []).insert(0, "前傾姿勢にはやや変化が見られ、局面によって再現性に影響している可能性があります。")
    elif spine_flag == "bad":
        if head_mean <= 6.0 and knee_mean <= 9.0:
            analysis["10"].setdefault("text", []).insert(0, "前傾姿勢にはやや変化が見られますが、全体として大きくバランスを崩している状態ではありません。")
        else:
            analysis["10"].setdefault("text", []).insert(0, "前傾姿勢の変化がやや大きく、スイング全体の再現性に影響している可能性があります。")
    # ------------------------------------------

    return analysis
        
# ==================================================
# Routes
# ==================================================
@app.route("/report/<report_id>", methods=["GET"])
def report_page(report_id: str):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return "Report not found", 404

    report = doc.to_dict() or {}

    # report.html が templates/ にある前提
    return render_template(
        "report.html",
        report_id=report_id,
        report=report,
        status=report.get("status", "PROCESSING"),
        premium=bool(report.get("is_premium", False)),
        analysis=report.get("analysis"),
        raw=report.get("raw"),
    )


# 末尾スラッシュでも落ちないように保険（LINE/ブラウザが勝手に付ける事故対策）
@app.route("/report/<report_id>/", methods=["GET"])
def report_page_slash(report_id: str):
    return report_page(report_id)

@app.route("/api/report_data/<report_id>", methods=["GET"])
def api_report_data(report_id: str):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"ok": False, "error": "not_found"}), 404

    r = doc.to_dict() or {}

    # report.html 側は COMPLETED のときだけ表示しているので、
    # Firestore の DONE を COMPLETED に寄せて返す（HTMLを直さなくて済む）
    status = (r.get("status") or "PROCESSING")
    st_upper = str(status).upper()
    status_out = "COMPLETED" if st_upper == "DONE" else status

    return jsonify({
        "ok": True,
        "report_id": report_id,
        "status": status_out,
        "is_premium": bool(r.get("is_premium", False)),
        "analysis": r.get("analysis") or {},
        "comparison": r.get("comparison"), # ★ここが追加されているか
        "address_posture": r.get("address_posture"),
        "spine_maintain": r.get("spine_maintain"),
        "overlay_video_url": r.get("overlay_video_url"),
        "overlay_video_download_url": r.get("overlay_video_download_url"),
    })

def upload_video_to_gcs(local_path: str, report_id: str) -> dict:
    bucket_name = "gate20260201"
    object_name = f"overlays/{report_id}.mp4"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    blob.upload_from_filename(local_path, content_type="video/mp4")

    import google.auth
    from google.auth.transport.requests import Request

    credentials, _ = google.auth.default()
    auth_req = Request()
    credentials.refresh(auth_req)

    play_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=24),
        method="GET",
        response_type="video/mp4",
        service_account_email=getattr(credentials, "service_account_email", None),
        access_token=credentials.token,
    )

    download_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=24),
        method="GET",
        response_type="video/mp4",
        response_disposition='attachment; filename="overlay.mp4"',
        service_account_email=getattr(credentials, "service_account_email", None),
        access_token=credentials.token,
    )

    return {
        "play_url": play_url,
        "download_url": download_url,
    }
    
@app.route("/task-handler", methods=["POST"])
def task_handler():
    try:
        data = request.get_json(silent=True) or {}
        report_id = data.get("report_id")
        user_id = data.get("user_id")
        message_id = data.get("message_id")

        if not report_id or not user_id or not message_id:
            return jsonify({"ok": False, "error": "missing fields"}), 400

        report_ref = db.collection("reports").document(report_id)
        snap = report_ref.get()
        if not snap.exists:
            return jsonify({"ok": False, "error": "report not found"}), 404

        report = snap.to_dict() or {}
        premium = bool(report.get("is_premium", False))
        user_inputs = report.get("user_inputs") or {}
        
        # --- [追加] ユーザー情報の取得（プラン、目標スコアなど） ---
        user_snap = db.collection("users").document(user_id).get()
        user_data = user_snap.to_dict() or {}
        user_plan = user_data.get("plan", "free") # free, single, monthly
        user_profile = user_data.get("profile", {}) # {current_avg_score: 100, target_score: 90} 等
        # --------------------------------------------------------

              # 動画DL → 解析（＋overlayアップロード）
        overlay_url = None
        overlay_download_url = None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, f"{report_id}.mp4")
            content = line_bot_api.get_message_content(message_id)

            with open(video_path, "wb") as f:
                for chunk in content.iter_content():
                    f.write(chunk)

            overlay_out = os.path.join(tmpdir, f"{report_id}_overlay.mp4")
            raw = analyze_swing_with_mediapipe(
                video_path,
                overlay_out_path=overlay_out,
                user_id=user_id
            )

            logging.warning(f"[DEBUG] raw_type={type(raw)}")
            logging.warning(f"[DEBUG] raw_keys={(list(raw.keys()) if isinstance(raw, dict) else None)}")
            logging.warning(f"[DEBUG] raw_overlay_path={(raw.get('overlay_path') if isinstance(raw, dict) else None)}")


            # --- overlay動画URLを作る ---
            try:
                logging.warning(f"[DEBUG] overlay_out={overlay_out}")
                logging.warning(f"[DEBUG] overlay_out_exists={os.path.exists(overlay_out)}")

                if os.path.exists(overlay_out):
                    overlay_urls = upload_video_to_gcs(overlay_out, report_id)
                    overlay_url = overlay_urls.get("play_url")
                    overlay_download_url = overlay_urls.get("download_url")
                else:
                    logging.warning("[DEBUG] overlay file not found, upload skipped")

            except Exception:
                logging.exception("[WARN] overlay upload failed")
                
        # ← with を抜けた後
        logging.warning(f"[DEBUG] overlay_url={overlay_url}")
        
        # --- ここまで ---
        # --- [追加] 比較分析ロジックの実行 ---
        comparison_data = None
        club_type = raw.get("club_type") or user_inputs.get("club_type", "unknown")
        
        if user_plan == "monthly":
            # 過去5回に限定して取得（要件定義に基づく）
            past_reports = get_past_reports(user_id, report_id, club_type, limit=5)
            if past_reports:
                comparison_data = calculate_full_comparison(raw, past_reports)
        # ------------------------------------

        analysis = build_analysis(
            raw=raw, 
            premium=premium, 
            report_id=report_id, 
            user_inputs=user_inputs,
            comparison=comparison_data,
            user_profile=user_profile,
            user_plan=user_plan
        )
        # -------------------------------------------------------------
        address_posture = judge_address_posture(raw)
        spine_maintain = judge_spine_maintain_display(raw)
        
        report_ref.set({
            "status": "DONE",
            "raw": raw,
            "analysis": analysis,
            "comparison": comparison_data, # 追加
            "user_plan_at_analysis": user_plan, # 追加（後からプラン変更された時のため）
            "address_posture": address_posture,
            "spine_maintain": spine_maintain,
            "overlay_video_url": overlay_url,
            "overlay_video_download_url": overlay_download_url,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }, merge=True)
        # -------------------------------------------------------------

        from linebot.models import FlexSendMessage
        from urllib.parse import quote

        report_url = f"https://gate-kagayaki-562867875402.asia-northeast2.run.app/report/{report_id}"
        share_text = f"GATEでスイングを解析しました！\n{report_url}"
        share_uri = "https://line.me/R/msg/text/?" + quote(share_text, safe="")

        flex_contents = {
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "🎉 スイング計測が完了しました！",
                        "weight": "bold",
                        "size": "lg",
                        "color": "#1DB446"
                    },
                    {
                        "type": "text",
                        "text": "AIがスイングを精密に解析しました。下記のボタンから結果を確認・共有できます。",
                        "size": "sm",
                        "color": "#666666",
                        "wrap": True,
                        "margin": "md"
                    }
                ]
            },
            "footer": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "contents": [
                    {
                        "type": "button",
                        "style": "primary",
                        "color": "#00b900",
                        "action": {
                            "type": "uri",
                            "label": "📊 診断レポートを見る",
                            "uri": report_url
                        }
                    },
                    {
                        "type": "button",
                        "style": "secondary",
                        "action": {
                            "type": "uri",
                            "label": "🎬 レポートをシェアする",
                            "uri": share_uri
                        }
                    }
                ]
            }
        }

        try:
            line_bot_api.push_message(
                user_id,
                FlexSendMessage(
                    alt_text="スイング診断完了のお知らせ",
                    contents=flex_contents
                )
            )
        except Exception as e:
            logging.exception("LINE push failed: %s", e)
        
        return jsonify({"ok": True}), 200

    except Exception:
        print("[ERROR] task-handler:", traceback.format_exc())
        try:
            data = request.get_json(silent=True) or {}
            rid = data.get("report_id")
            if rid:
                firestore_safe_update(rid, {"status": "TASK_FAILED", "error": traceback.format_exc()})
        except Exception:
            pass
        return jsonify({"ok": False, "error": "internal"}), 500

   

# ==================================================
# Stripe Checkout 作成
# ==================================================    
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
        session_kwargs = {
            "mode": checkout_mode,
            "payment_method_types": ["card"],
            "client_reference_id": line_user_id, # LINE ID
            "metadata": {
                "plan": plan,             # "single", "ticket", "monthly"
                "line_user_id": line_user_id
            },
            "success_url": success_url,
            "cancel_url": cancel_url,
        }

        if plan == "monthly":
            session_kwargs["line_items"] = [{"price": "price_1TR1AQCZFahUhJa5RvkSJtZD", "quantity": 1}]
            session_kwargs["subscription_data"] = {"trial_period_days": 14}
        else:
            session_kwargs["line_items"] = [{"price": price_id, "quantity": 1}]

        session = stripe.checkout.Session.create(**session_kwargs)
        return jsonify({"checkout_url": session.url}), 200

    except Exception as e:
        print(f"[ERROR] Stripe Session Create Failed: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# server.py 上部（1回だけ）
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
db = firestore.Client()

@app.route("/stripe/checkout/monthly", methods=["GET"])
def stripe_checkout_monthly_get():
    line_user_id = request.args.get("client_reference_id")
    if not line_user_id:
        return "Missing client_reference_id", 400

    if not stripe.api_key:
        return "STRIPE_SECRET_KEY is not set", 500

    success_url = os.environ.get("STRIPE_SUCCESS_URL", SERVICE_HOST_URL)
    cancel_url = os.environ.get("STRIPE_CANCEL_URL", SERVICE_HOST_URL)

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": "price_1TR1AQCZFahUhJa5RvkSJtZD", "quantity": 1}],
            client_reference_id=line_user_id,
            subscription_data={"trial_period_days": 14},
            metadata={
                "plan": "monthly",
                "line_user_id": line_user_id
            },
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return redirect(session.url)
    except Exception as e:
        import traceback
        print(f"[ERROR] Stripe Session Create Failed: {traceback.format_exc()}")
        return str(e), 500

@app.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    import os, traceback, logging
    from flask import request
    import stripe
    from google.cloud import firestore

    endpoint_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
    if not endpoint_secret or not stripe.api_key:
        return "Missing stripe env", 500

    db = firestore.Client()
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")

    # 1) 署名検証
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except stripe.error.SignatureVerificationError as e:
        print(f"⚠️ Stripe署名検証に失敗しました: {e}", flush=True)
        return "Invalid signature", 400
    except Exception as e:
        print(f"⚠️ Stripe webhook error: {e}", flush=True)
        return "Error", 400

    event_type = event.type

    # =========================================================
    # A) 購入完了（単発/回数券）
    # =========================================================
    if event_type == "checkout.session.completed":
        session = event["data"]["object"]
        event_id = getattr(event, "id", None)
        session_id = getattr(session, "id", None)
        line_user_id = getattr(session, "client_reference_id", None)

        if not line_user_id:
            print("❌ client_reference_id missing", flush=True)
            return "OK", 200

        try:
            li = stripe.checkout.Session.list_line_items(session_id, limit=1)
            first = li["data"][0] if li and getattr(li, "data", None) else None
            price_id = getattr(getattr(first, "price", None), "id", None) if first else None

            add_tickets = 1
            ticket_price_id = (os.environ.get("STRIPE_PRICE_TICKET", "") or "").strip()
            if ticket_price_id and price_id == ticket_price_id:
                add_tickets = 5


            user_ref = db.collection("users").document(line_user_id)

            # 冪等（Stripe再送でも二重加算しない）
            before = user_ref.get().to_dict() or {}
            if before.get("last_stripe_event_id") == event_id:
                print("✅ duplicate event ignored", flush=True)
                return "OK", 200

            # Firestore更新（ここで増える）
            user_ref.set({
                "plan": "ticket" if add_tickets > 1 else "single",
                "status": "paid",
                "ticket_remaining": firestore.Increment(add_tickets),
                "last_payment_date": firestore.SERVER_TIMESTAMP,
                "last_stripe_event_id": event_id,
                "updated_at": firestore.SERVER_TIMESTAMP,
            }, merge=True)

            # ===== 購入完了メッセージ（新規追加・既存文言は触らない）=====
            after = user_ref.get().to_dict() or {}
            plan = after.get("plan")
            tickets = int(after.get("ticket_remaining", 0))

            if plan == "monthly":
                from datetime import datetime, timedelta, timezone
                jst = timezone(timedelta(hours=9))
                next_date_str = (datetime.now(jst) + timedelta(days=14)).strftime("%Y年%m月%d日")
                message = (
                    "GATEの月額プランへご登録いただき、ありがとうございます！⛳️✨\n\n"
                    f"次回お支払い日は {next_date_str} です。それまでに解約すれば料金は発生しません。\n\n"
                    "ご契約期間中は、AIスイング解析が無制限でご利用いただけます。\n\n"
                    "さっそく、リッチメニューの「分析スタート」からあなたのスイング動画を送ってみましょう🏌️‍♂️"
                )
            else:
                message = (
                    "GATEのチケットをご購入いただき、ありがとうございます！⛳️✨\n\n"
                    f"🎫 残りチケット：{tickets}回\n\n"
                    "さっそくAI解析を始めましょう！\n"
                    "リッチメニューの「分析スタート」から、あなたのスイング動画を送ってみましょう🏌️‍♂️"
                )

            safe_line_push(line_user_id, message, force=True)

            print(f"✅ Firestore updated user={line_user_id} add={add_tickets}", flush=True)
            return "OK", 200

        except Exception:
            print("❌ post-payment handler failed:", traceback.format_exc(), flush=True)
            return "Internal Error", 500

    # =========================================================
    # B) 解約（サブスク削除）
    # =========================================================
    elif event_type == "customer.subscription.deleted":
        try:
            subscription = event["data"]["object"]
            customer_id = getattr(subscription, "customer", None)

            users_ref = db.collection("users")
            docs = users_ref.where("stripe_customer_id", "==", customer_id).limit(1).get()

            for doc in docs:
                doc.reference.update({
                    "status": "free",
                    "updated_at": firestore.SERVER_TIMESTAMP
                })

            print(f"✅ Subscription cancelled for customer: {customer_id}", flush=True)
            return "OK", 200

        except Exception:
            print("❌ Deletion Error:", traceback.format_exc(), flush=True)
            return "Internal Error", 500

    # その他イベントは無視
    return "OK", 200


from datetime import datetime, timezone
import logging
import traceback
from google.cloud import firestore

@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event: MessageEvent):
    user_id = event.source.user_id
    msg = event.message
    report_id = f"{user_id}_{msg.id}"

    db = firestore.Client()
    users_ref = db.collection("users")
    user_ref = users_ref.document(user_id)

    # ===== users 取得（ここで1回だけ）=====
    user_doc = user_ref.get()
    user_data = user_doc.to_dict() if user_doc.exists else {}

    plan = (user_data.get("plan") or "free").lower()
    tickets = int(user_data.get("ticket_remaining", 0))

    # freeなら ticket_remaining は 0 に正規化（不整合対策）
    if plan == "free" and tickets != 0:
        try:
            user_ref.set({"ticket_remaining": 0}, merge=True)
        except Exception:
            logging.exception("[WARN] failed to normalize ticket_remaining for free user")

        tickets = 0

    # ===== 無料（月1回）チェック（必須：plan==free のときだけ）=====
    if plan == "free":
        now_yyyy_mm = datetime.now(timezone.utc).strftime("%Y-%m")

        @firestore.transactional
        def reserve_free_monthly(txn: firestore.Transaction) -> bool:
            snap = user_ref.get(transaction=txn)
            data = snap.to_dict() if snap.exists else {}
            used = data.get("free_used_month")
            if used == now_yyyy_mm:
                return False
            txn.set(user_ref, {"free_used_month": now_yyyy_mm}, merge=True)
            return True

        try:
            txn = db.transaction()
            ok = reserve_free_monthly(txn)
        except Exception:
            logging.exception("[ERROR] free monthly transaction failed")
            ok = False

        if not ok:
            safe_line_reply(
                event.reply_token,
                "今月の無料解析（1回）はすでにご利用済みです。\n\n"
                "引き続き解析をご希望の場合は、単発プランまたは回数券をご利用ください。",
                user_id=user_id
            )
            return

    logging.warning(
        "[DEBUG] handle_video HIT user_id=%s message_id=%s plan=%s free_used_month=%s tickets=%s",
        user_id, msg.id, plan, user_data.get("free_used_month"), tickets
    )

    # ===== prefill → user_inputs =====
    prefill = user_data.get("prefill") or {}
    user_inputs = {
        "head_speed": prefill.get("head_speed"),
        "miss_tendency": prefill.get("miss_tendency"),
        "gender": prefill.get("gender"),
    }

    user_inputs = {k: v for k, v in user_inputs.items() if v is not None}

    # ===== 有料判定（ここは既存ロジックを尊重）=====
    is_subscription = is_premium_user(user_id)
    force_paid_report = is_subscription or tickets > 0

    firestore_safe_set(report_id, {
        "user_id": user_id,
        "status": "PROCESSING",
        "is_premium": force_paid_report,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_inputs": user_inputs,
    })

    reply_text = make_initial_reply(report_id)
    safe_line_reply(event.reply_token, reply_text, user_id=user_id)

    try:
        task_name = create_cloud_task(report_id, user_id, msg.id)
        firestore_safe_update(report_id, {"task_name": task_name})

        # ===== チケット消費（Tasks作成成功後のみ）=====
        if (not is_subscription) and tickets > 0:

            @firestore.transactional
            def consume_and_downgrade(txn: firestore.Transaction):
                snap = user_ref.get(transaction=txn)
                u = snap.to_dict() if snap.exists else {}

                plan_now = (u.get("plan") or "free").lower()
                remaining = int(u.get("ticket_remaining", 0))

                if plan_now not in ("single", "ticket") or remaining <= 0:
                    return

                new_remaining = remaining - 1

                updates = {
                    "ticket_remaining": new_remaining,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                }

                # single / ticket → 残0で free
                if new_remaining <= 0:
                    updates["plan"] = "free"

                txn.update(user_ref, updates)

            txn = db.transaction()
            consume_and_downgrade(txn)

    except Exception:
        logging.exception(
            "[ERROR] handle_video failed after report created user_id=%s report_id=%s",
            user_id,
            report_id,
        )
        firestore_safe_update(
            report_id,
            {"status": "TASK_FAILED", "error": "handle_video exception"},
        )
        safe_line_push(
            user_id,
            "受付後の処理でエラーが発生しました。進行状況URLをご確認ください。",
            force=True,
        )
        return
    
        
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()


    # 1. まずメッセージの内容を取得（空白を削除して判定を正確にする）
    text = event.message.text.strip()

    # ここに追加
    db = firestore.Client()

    if "料金プラン" in text:
        host = (SERVICE_HOST_URL or "").strip().rstrip("/")
        if not host:
            from flask import request
            host = request.host_url.rstrip("/")
        elif not host.startswith(("https://", "http://")):
            host = "https://" + host
            
        monthly_checkout_url = f"{host}/stripe/checkout/monthly?client_reference_id={user_id}"

        from datetime import datetime, timedelta, timezone
        jst = timezone(timedelta(hours=9))
        next_date_str = (datetime.now(jst) + timedelta(days=14)).strftime("%Y年%m月%d日")

        plan_text = (
            "GATE公式LINEへようこそ！⛳️\n\n"
            "正確なAI解析結果をお届けするため、画面上部に「追加」ボタンが表示されている方は、まず登録をお願いいたします。\n\n"
            "決済完了後は、このトーク画面にスイング動画を送るだけでAI解析がスタートします。\n"
            "--------------------\n\n"
            "【単発プラン】300円/1回\n"
            "単発プランで試す → \n"
            f"https://buy.stripe.com/00w6oI6Qb8TogSn6uz18c0a?client_reference_id={user_id}\n\n"
            "【回数券プラン】1,200円/5回\n"
            "回数券を購入する → \n"
            f"https://buy.stripe.com/4gM6oI3DZd9EeKfcSX18c09?client_reference_id={user_id}\n\n"
            "【月額プラン】2,000円/月\n"
            "月額プランを申し込む → \n"
            f"{monthly_checkout_url}\n\n"
            f"※初回14日間無料！次回お支払い日は {next_date_str} です。それまでに解約すれば料金は発生しません。\n"
            "--------------------\n"
            "※操作方法などは、このままトークでお気軽にご質問ください。"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=plan_text))
        return
   
    # 2. 解約・キャンセル判定（最優先）
    if text in ["解約", "キャンセル", "サブスク解除"]:
        # Firestoreからユーザー情報を取得
        user_doc = db.collection("users").document(user_id).get()
    
        stripe_id = None
        if user_doc.exists:
            stripe_id = user_doc.to_dict().get("stripe_customer_id")

        if stripe_id:
            # 外部で定義した get_cancel_portal_url を呼び出し
            url = get_cancel_portal_url(stripe_id)
            if url:
                reply = (
                    "解約・プラン管理のお手続きですね。\n"
                    "以下の専用ページからお手続きいただけます。\n\n"
                    f"{url}\n\n"
                    "※有効期限があるため、お早めにアクセスしてください。"
                )
            else:
                reply = "URLの発行に失敗しました。お手数ですが事務局までご連絡ください。"
        else:
            reply = "決済情報が見つかりませんでした。お手数ですが事務局までご連絡ください。"

        # LINEへ返信して処理を終了
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
        return  # ここで処理を完全に終了させる
        
    # 二重処理防止（最優先）
    event_id = f"{event.source.user_id}:{event.reply_token}"
    ref = db.collection("processed_events").document(event_id)
    if ref.get().exists:
        return
    ref.set({"ts": datetime.now(timezone.utc).isoformat()})

    
    # 3. お問い合わせ判定（切り分け：サーバー返信を止める）
    if text == "お問い合わせ":
        print(f"[CUT] inquiry server reply suppressed msgid={event.message.id}", flush=True)
        return



    # ===== 正規化（全角スペース & 全角数字）=====
    raw_text = event.message.text or ""
    text = raw_text.replace("\u3000", " ").strip()
    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    logging.warning("[DEBUG] raw=%r normalized=%r user_id=%s", raw_text, text, user_id)

    # ===== 1) 分析スタート → クラブ選択 (Quick Reply) =====
    if text == "分析スタート":
        is_premium = is_premium_user(user_id)
        # 最初のステップを 'club_type' に設定
        users_ref.document(user_id).set({
            "prefill_step": "club_type",
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        # 冒頭文とクラブ選択のクイックリプライ
        msg_text = (
            "ご利用ありがとうございます。\n\n"
            "無料簡易解析をご希望の方はそのまま動画を送ってください。\n\n"
            "有料プラン・月額プランの方は今回分析するクラブを下のボタンから選んでください。\n"
        )

        # クイックリプライボタンの作成
        quick_reply = QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="ドライバー", text="ドライバー")),
            QuickReplyButton(action=MessageAction(label="ウッド・UT", text="ウッド・UT")),
            QuickReplyButton(action=MessageAction(label="アイアン", text="アイアン")),
            QuickReplyButton(action=MessageAction(label="スキップ", text="スキップ")),
        ])

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=msg_text, quick_reply=quick_reply)
        )
        return

    # ===== 2) user取得 → step =====
    user_doc = users_ref.document(user_id).get()
    user_data = user_doc.to_dict() or {}
    step = user_data.get("prefill_step")
    logging.warning("[DEBUG] prefill_step=%r", step)

    # 【既存ロジック維持】性別判定を if step: の外に配置。
    if text in ["男性", "女性"] or step == "gender":
        users_ref.document(user_id).set({
            "prefill_step": None,
            "prefill": {"gender": text},
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="ありがとうございます！性別を保存しました。このまま動画を送ってください。")
        )
        return

    # ===== 3) stepが立っているなら最優先で保存 =====
    if step:
        # 任意：途中で抜けたい人はスキップ扱いでリセット
        if text == "スキップ":
            users_ref.document(user_id).set({
                "prefill_step": None,
                "updated_at": firestore.SERVER_TIMESTAMP,
            }, merge=True)
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="OK！入力は中断しました。このまま動画を送ってください。")
            )
            return

        # --- 新設：クラブ種別の保存 ---
        if step == "club_type":
            # 入力されたテキストから内部用の種別を決定
            club_map = {"ドライバー": "driver", "ウッド・UT": "wood_ut", "アイアン": "iron"}
            selected_club = club_map.get(text, "iron") # 該当なしは暫定アイアン

            users_ref.document(user_id).set({
                "prefill_step": "head_speed",
                "prefill": {"club_type": selected_club},
                "updated_at": firestore.SERVER_TIMESTAMP,
            }, merge=True)

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=(
                        f"「{text}」で解析を承ります。\n\n"
                        "クラブ提案をご希望の方は必要事項を順番にお伺いしていきます。クラブ提案が不要な方はそのまま動画を送ってください。\n\n"
                        "まず、ヘッドスピードを数字だけ送ってください（例：43）。"
                    )
                )
            )
            return

        # --- 既存：ヘッドスピードの保存 ---
        if step == "head_speed":
            if not text.isdigit():
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="ヘッドスピードは数字だけで送ってください（例：42）。")
                )
                return

            users_ref.document(user_id).set({
                "prefill_step": "miss_tendency",
                "prefill": {"head_speed": int(text)},
                "updated_at": firestore.SERVER_TIMESTAMP,
            }, merge=True)

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=(
                        "ありがとうございます。\n\n"
                        "続けて、普段のゴルフでの悩みや主なミスの傾向を自由に入力して送ってください。\n"
                        "（例：右へのプッシュアウトが多い、チーピンに悩んでいる、ダフリやすい など）"
                    )
                )
            )
            return

        # --- 既存：ミスの傾向の保存 ---
        if step == "miss_tendency":
            users_ref.document(user_id).set({
                "prefill_step": None,  # ここでリセットされる
                "prefill": {"miss_tendency": text},
                "updated_at": firestore.SERVER_TIMESTAMP,
            }, merge=True)

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=(
                        "ありがとうございます。ミスの傾向を保存しました。\n\n"
                        "性別は任意です。\n"
                        "入力する場合は「男性」または「女性」と送ってください。\n\n"
                        "入力しない場合は、このまま動画を送信してください。"
                    )
                )
            )
            return
            
        # 想定外step保険
        users_ref.document(user_id).set({
            "prefill_step": None,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="入力状態をリセットしました。もう一度「分析スタート」からお願いします。")
        )
        return

    # ===== 4) stepがない場合：09希望/性別ボタンなど =====
    if text == "09希望":
        users_ref.document(user_id).set({
            "prefill_step": "club_type", # 修正：ここもクラブ選択から始める
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="まず、今回分析するクラブをボタンから選んでください。",
                           quick_reply=QuickReply(items=[
                               QuickReplyButton(action=MessageAction(label="ドライバー", text="ドライバー")),
                               QuickReplyButton(action=MessageAction(label="ウッド・UT", text="ウッド・UT")),
                               QuickReplyButton(action=MessageAction(label="アイアン", text="アイアン"))
                           ]))
        )
        return

    if text == "性別":
        users_ref.document(user_id).set({
            "prefill_step": "gender",
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="性別を送ってください（例：男性/女性）。スキップなら「スキップ」")
        )
        return

  
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
