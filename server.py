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

from flask import Flask, request, jsonify, abort, render_template, render_template_string

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore
from google.cloud import tasks_v2
from google.api_core.exceptions import NotFound, PermissionDenied

from google.cloud import storage   # ★追加（動画アップロード用）
from datetime import timedelta     # ★追加（URL期限用）
import google.auth
from google.auth.transport.requests import Request

import stripe
# --- Stripe設定 ---
# 本番環境では環境変数から取得することを推奨します
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

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
def analyze_swing_with_mediapipe(video_path, overlay_out_path=None):
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
    # ここから下の「while cap.isOpened():」などは、以前のインデントのまま動くはずです

     

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

                    if delta_spine <= 3:
                        color = (0, 255, 0)
                    elif delta_spine <= 6:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)

                draw_overlay_skeleton(out, lm, mp_pose, color)
                draw_spine_line(out, lm, mp_pose, color)
                
                if spine_angle > 0:
                    draw_gaze_line(out, lm, mp_pose, spine_angle)
                
                writer.write(out)
    def analyze_swing_with_mediapipe(video_path, overlay_out_path=None, user_id=None):
        if user_id is None:
            raise ValueError("user_id が渡されていません")   
            
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
    result = {
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

    # ★ overlay動画パスを返す
    if overlay_out_path:
        result["overlay_path"] = overlay_out_path

    return result

  
# ==================================================
# Section 01: 修正版（3D・％単位対応）
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
                "value": f'max {raw["shoulder"]["max"]:.1f} / mean {raw["shoulder"]["mean"]:.1f} / σ {raw["shoulder"]["std"]:.1f}',
                "description": "3D空間での上半身の回旋量です。",
                "guide": "maxで80°〜115°",
            },
            {
                "name": "腰回転（°）",
                "value": f'max {raw["hip"]["max"]:.1f} / mean {raw["hip"]["mean"]:.1f} / σ {raw["hip"]["std"]:.1f}',
                "description": "3D空間での下半身の回旋量です。",
                "guide": "maxで30°〜60°",
            },
            {
                "name": "手首コック（°）",
                "value": f'max {raw["wrist"]["max"]:.1f} / mean {raw["wrist"]["mean"]:.1f} / σ {raw["wrist"]["std"]:.1f}',
                "description": "手首のタメの角度（3D）です。",
                "guide": "meanで40°〜80°",
            },
            {
                "name": "頭部ブレ（%）",
                "value": f'max {raw["head"]["max"]:.1f} / mean {raw["head"]["mean"]:.1f} / σ {raw["head"]["std"]:.1f}',
                "description": "アドレス時からの頭部の移動量です（画面幅比）。",
                "guide": "meanで6.5%以下",
            },
            {
                "name": "膝ブレ（%）",
                "value": f'max {raw["knee"]["max"]:.1f} / mean {raw["knee"]["mean"]:.1f} / σ {raw["knee"]["std"]:.1f}',
                "description": "アドレス時からの膝の移動量です（画面幅比）。",
                "guide": "meanで10.0%以下",
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


def build_paid_02_shoulder(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_shoulder(raw)
    sh = raw["shoulder"]
    xf = raw["x_factor"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    # 良い点（やや緩和）
    if sh["std"] <= 12:
        good.append("肩の回し幅は比較的揃っており、上半身の再現性は概ね確保されています。")
    if 80 <= sh["max"] <= 115:
        good.append("肩の最大回旋量は基準レンジに収まっており、無理のない捻転ができています。")
    if xf["max"] >= 30:
        good.append("肩と腰の差（捻転差）は確保できており、出力の準備が整っています。")

    # バッファ：回転量が多い場合
    if sh["max"] > 115:
        good.append("深い肩の回転を可能にする柔軟性があり、大きな飛距離を生む潜在能力があります。")

    # バッファ：数値は外れていても安定している場合
    if sh["std"] <= 8 and not (80 <= sh["max"] <= 115):
        good.append("角度自体は調整の余地がありますが、同じ深さまで回せる安定感は大きな武器です。")

    if not good:
        good = ["基本的な上半身の柔軟性は備わっており、スイングの土台はできています。"]

    # 改善点（やや緩和）
    if sh["max"] < 80:
        bad.append(f"最大肩回転は {sh['max']:.1f}° で、やや捻転が浅い傾向があります。")
    if sh["max"] > 115:
        bad.append(f"肩回転が {sh['max']:.1f}° に達しており、ややオーバースイング傾向が見られます。")
    if xf["max"] < 30:
        bad.append(f"最大捻転差は {xf['max']:.1f}° で、力を溜める余地がまだあります。")
    if xf["max"] > 70:
        bad.append(f"捻転差が {xf['max']:.1f}° と大きく、肩と腰の連動にばらつきが出やすくなっています。")
    if sh["std"] > 18:
        bad.append(f"肩回転のばらつき（σ {sh['std']:.1f}°）がやや大きく、トップ位置の再現性に影響しています。")

    if not bad:
        bad = ["大きな改善点は特にありません。"]

    # プロ目線（やや柔らかく）
    pro_lines: List[str] = []
    pro_lines.append("上半身は回り幅そのものだけでなく、回した量を安定して再現できているかも評価軸です。")

    if sh["std"] <= 12:
        pro_lines.append("本動画では肩の回旋は比較的安定して再現できています。")
    else:
        pro_lines.append("本動画では肩の回旋幅にややばらつきがあり、トップの再現性に影響しています。")

    if xf["max"] < 30:
        pro_lines.append("捻転差はやや小さめで、切り返しで力を溜める余地があります。")
    elif xf["max"] > 70:
        pro_lines.append("捻転差は大きめで、肩と腰の連動を整えるとさらに安定しやすくなります。")
    else:
        pro_lines.append("捻転差は確保されており、切り返しに必要な準備はできています。")

    pro_lines.append("このスイングでは、主因は肩と腰の役割分担です。")

    pro_comment = " ".join(pro_lines[:3])

    bench = [
        _bench_line("肩回転(°)", "°", "max", _range_ideal(80, 115, "°"), current=float(sh["max"])),
        _bench_line("肩回転の安定(°)", "°", "σ", _le_ideal(12.0, "°"), current=float(sh["std"])),
        _bench_line("捻転差(°)", "°", "max", _range_ideal(30, 70, "°"), current=float(xf["max"])),
    ]

    return {
        "title": "02. Shoulder Rotation（肩回転）",
        "value": _value_line(sh["max"], sh["mean"], sh["std"], conf),
        "tags": j["tags"],
        "bench": bench,
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro_comment,
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

def build_paid_03_hip(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_hip(raw)
    hip = raw["hip"]
    xf = raw["x_factor"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    # 良い点（やや緩和）
    if hip["std"] <= 6:
        good.append("下半身の動きが安定しており、ミート率を高める基礎ができています。")
    elif hip["std"] <= 12:
        good.append("腰の回し幅は比較的揃っており、下半身の再現性は概ね確保されています。")
    if 30 <= hip["max"] <= 60:
        good.append("腰の回旋量は基準レンジに収まっており、安定した土台として機能しています。")

    # バッファ：安定性
    if hip["std"] <= 6:
        good.append("下半身の動きが安定しており、ミート率を高める基礎ができています。")

    # バッファ：腰が控えめでも捻転差が作れている場合
    if hip["max"] < 30 and xf["max"] >= 35:
        good.append("腰の回転は控えめですが、肩との捻転差は確保できており、一定の出力準備はできています。")

    if not good:
        good = ["基本的な下半身の可動域は確保されており、スイングの土台はできています。"]

    # 改善点（やや緩和）
    if hip["max"] > 60:
        bad.append(f"最大腰回転は {hip['max']:.1f}° で、やや回りすぎの傾向が見られます。")
    if hip["max"] < 30:
        bad.append(f"最大腰回転は {hip['max']:.1f}° で、下半身主導を高める余地があります。")
    if xf["max"] < 30:
        bad.append(f"最大捻転差は {xf['max']:.1f}° で、上半身との連動をさらに高める余地があります。")
    if hip["std"] > 18:
        bad.append(f"腰の回転幅のばらつき（σ {hip['std']:.1f}°）がやや大きく、インパクトの再現性に影響しています。")

    if not bad:
        bad = ["大きな改善点は特にありません。"]

    # プロ目線（やや柔らかく）
    pro_lines: List[str] = []
    pro_lines.append("腰は「回す量」だけでなく、「肩との順序」と「回し幅の揃い方」で質が決まります。")

    if hip["max"] > 60:
        pro_lines.append("本動画では腰の回転量がやや大きく、軸の安定性に影響しやすい状態です。")
    elif hip["max"] < 30:
        pro_lines.append("本動画では下半身の回旋量はやや控えめで、下半身主導を高める余地があります。")
    else:
        pro_lines.append("本動画では腰の回旋量は適正範囲で、安定した軸回転ができています。")

    if hip["std"] > 15:
        pro_lines.append("腰の回転角度にややばらつきがあり、下半身主導の再現性に影響しています。")
    else:
        pro_lines.append("下半身の回転は比較的安定しており、スイングの再現性を支える土台となっています。")

    pro_lines.append("このスイングでは、主因は下半身主導のタイミングです。")

    pro_comment = " ".join(pro_lines[:3])

    bench = [
        _bench_line("腰回転(°)", "°", "max", _range_ideal(30, 60, "°"), current=float(hip["max"])),
        _bench_line("腰回転の安定(°)", "°", "σ", _le_ideal(12.0, "°"), current=float(hip["std"])),
        _bench_line("捻転差(°)", "°", "max", _ge_ideal(30.0, "°"), current=float(xf["max"])),
    ]

    return {
        "title": "03. Hip Rotation（腰回転）",
        "value": _value_line(hip["max"], hip["mean"], hip["std"], conf),
        "tags": j["tags"],
        "bench": bench,
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro_comment,
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


def build_paid_04_wrist(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    w_raw = raw["wrist"]

    # 解析側で反転済み
    w_mean = float(w_raw["mean"])
    w_max  = float(w_raw["max"])
    w_std  = float(w_raw["std"])

    j = judge_wrist(raw)
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    # 良い点（やや緩和）
    if w_std <= 10:
        good.append("手首の角度変化は比較的一定しており、インパクトでのフェース管理は概ね安定しています。")
    if 40 <= w_mean <= 80:
        good.append("タメは基準レンジに収まっており、ヘッドを加速させる準備ができています。")
    if w_max >= 75 and w_std <= 15:
        good.append("トップで十分なコックが作れており、その角度も比較的安定して再現できています。")
    elif w_max >= 75:
        good.append("トップでは十分なコックが入っており、ヘッドスピードにつながる土台があります。")

    if not good:
        good = ["基本的な手首の可動域は確保されており、スイングの土台はできています。"]

    # 改善点（やや緩和）
    if w_mean < 40:
        bad.append(f"平均コック角 {w_mean:.1f}° はやや浅く、リリースが早くなる傾向があります。")
    if w_mean > 80:
        bad.append(f"平均コック角 {w_mean:.1f}° はやや大きく、リリースのタイミング管理がややシビアになりやすい状態です。")
    if w_std > 18:
        bad.append(f"手首の挙動（σ {w_std:.1f}）にばらつきがあり、打点の再現性に影響しています。")
    if w_max < 35:
        bad.append("バックスイングでのコック量がやや不足しており、力を溜める余地があります。")

    if not bad:
        bad = ["現在、手首の使い方において大きな修正ポイントは見当たりません。"]

    # プロ目線（やや柔らかく）
    pro_lines: List[str] = []

    if w_mean < 40:
        pro_lines.append(f"本動画では手首の角度が {w_mean:.1f}° とやや浅く、ヘッドを運ぶ動きが出やすい傾向です。")
        pro_lines.append("タメを保つ時間が短くなりやすいため、インパクトで合わせる動きにつながることがあります。")
    elif w_mean > 80:
        pro_lines.append(f"平均 {w_mean:.1f}° とタメは深めですが、その分リリースのタイミング管理が重要になります。")
        pro_lines.append("手元の操作が増えると、左右のミスにつながりやすくなります。")
    else:
        pro_lines.append(f"手首のコック角（{w_mean:.1f}°）は基準レンジに近く、効率的なパワー伝達ができています。")

    if w_std > 15:
        pro_lines.append("手首の動きにややばらつきがあり、フェース向きの再現性に影響しています。")
    else:
        pro_lines.append("手首の挙動は比較的安定しており、シャフトのしなりを再現しやすい状態です。")

    pro_comment = " ".join(pro_lines)

    bench = [
        _bench_line("手首コック(°)", "°", "mean", _range_ideal(40, 80, "°"), current=float(w_mean)),
        _bench_line("手首コックの上限(°)", "°", "max", _ge_ideal(75.0, "°"), current=float(w_max)),
        _bench_line("手首の再現性(°)", "°", "σ", _le_ideal(15.0, "°"), current=float(w_std)),
    ]

    return {
        "title": "04. Wrist Cock（手首コック）",
        "value": f"Max Cock {w_max:.1f}° / Mean {w_mean:.1f}° (σ {w_std:.1f})",
        "tags": j["tags"],
        "bench": bench,
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro_comment,
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

def build_paid_05_head(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_head(raw)
    h = raw["head"]
    k = raw["knee"]
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    spine_flag = judge_spine_flag(raw)

    # 良い点
    if h["mean"] <= 4.5:
        good.append("頭部の位置は全体を通して安定しており、スイング軸の再現性は良好です。")
    elif h["mean"] <= 6.0:
        good.append("頭部の移動は一定範囲に収まっており、軸の安定性は概ね保たれています。")

    if h["mean"] <= 5.0 and k["mean"] <= 6.5:
        good.append("頭部と下半身の上下動が比較的整っており、ミート率を支える土台があります。")

    if not good:
        good = ["頭部位置は大きく崩れておらず、基本的な軸の土台はあります。"]

    # 改善点
    if h["mean"] > 6.0:
        bad.append(f"頭部の移動量は {h['mean']:.1f}% とやや大きく、インパクト時の軸の再現性に影響しやすい状態です。")

    if h["std"] > 3.5:
        bad.append(f"頭部位置のばらつき（σ {h['std']:.1f}%）がやや大きく、場面ごとの再現性にムラがあります。")

    if k["mean"] > 10.0:
        bad.append(f"膝ブレは mean {k['mean']:.1f}% と大きめで、頭部の安定性にも影響している可能性があります。")

    if not bad:
        bad = ["大きな改善点は特にありません。"]

    # プロ目線
    pro_lines: List[str] = []
    pro_lines.append("頭部は完全に止めることよりも、動いても同じ位置関係に戻れるかが重要な評価ポイントです。")

    if h["mean"] > 6.0:
        pro_lines.append("本動画では頭部の移動がやや大きく、軸の再現性に影響しやすい状態です。")
    else:
        pro_lines.append("本動画では頭部の位置は比較的安定しており、軸の再現性は概ね保たれています。")

    if h["std"] > 3.5:
        pro_lines.append("場面によって頭の位置にばらつきがあり、再現性にややムラがあります。")
    else:
        pro_lines.append("頭の位置は全体として揃っており、再現性は概ね保たれています。")

    if h["mean"] > 6.0:
        if spine_flag == "bad":
            pro_lines.append("前傾姿勢の変化も重なり、頭部の安定性に影響している可能性があります。")
        elif spine_flag == "warn":
            pro_lines.append("前傾維持のばらつきが、頭部の安定性に一部影響している可能性があります。")

    pro_comment = " ".join(pro_lines[:3])

    bench = [
        _bench_line("頭部ブレ(%)", "%", "mean", _le_ideal(6.0, "%"), current=float(h["mean"])),
        _bench_line("頭部ブレの再現性(%)", "%", "σ", _le_ideal(2.5, "%"), current=float(h["std"])),
        _bench_line("頭部ブレの安定目安(%)", "%", "mean", _le_ideal(4.5, "%"), current=float(h["mean"])),
    ]

    return {
        "title": "05. Head Stability（頭部）",
        "value": f'max {h["max"]:.1f} / mean {h["mean"]:.1f} / σ {h["std"]:.1f} (%) （conf {conf:.3f}）',
        "tags": j["tags"],
        "bench": bench,
        "good": good[:3],
        "bad": bad[:3],
        "pro_comment": pro_comment,
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


def build_paid_06_knee(raw: Dict[str, Any], seed: str) -> Dict[str, Any]:
    j = judge_knee(raw)
    k = raw["knee"]
    h = raw["head"]
    conf = _conf(raw)
    spine_flag = judge_spine_flag(raw)

    good: List[str] = []
    bad: List[str] = []

    # 良い点
    if k["mean"] <= 5.5:
        good.append("膝の左右ブレは小さく抑えられており、下半身の土台は概ね安定しています。")
    elif k["mean"] <= 8.5:
        good.append("下半身の動きは許容範囲に収まっており、土台としての安定感は概ね保たれています。")

    if k["mean"] <= 6.5 and h["mean"] <= 5.5:
        good.append("下半身と上半身の軸が比較的連動しており、全体の安定感につながっています。")

    if not good:
        good = ["下半身の動きは大きく崩れてはいませんが、さらに安定性を高める余地があります。"]

    # 改善点
    if k["mean"] > 9.0:
        bad.append(f"膝の移動量は {k['mean']:.1f}% とやや大きく、土台の再現性に影響しやすい状態です。")

    if k["std"] > 3.5:
        bad.append(f"膝位置のばらつき（σ {k['std']:.1f}%）がやや大きく、場面ごとの安定感にムラがあります。")

    if h["mean"] > 6.5 and k["mean"] > 9.0:
        bad.append("上半身の揺れも重なり、下半身の安定性がやや活かしにくい状態です。")

    if not bad:
        bad = ["大きな改善点は特にありません。"]

    # プロ目線
    pro_lines: List[str] = []
    pro_lines.append("下半身は踏めているかだけでなく、回転中も土台が横に流れすぎないかが重要な評価ポイントです。")

    if k["mean"] > 9.0:
        pro_lines.append("本動画では下半身の横方向の動きがやや大きく、土台の再現性に影響しやすい状態です。")
    else:
        pro_lines.append("本動画では下半身の動きは全体として比較的抑えられており、土台の安定性は概ね保たれています。")

    if k["std"] > 3.5:
        pro_lines.append("場面によって膝の位置にばらつきがあり、土台の再現性にややムラがあります。")
    else:
        pro_lines.append("膝の位置は全体として揃っており、土台の再現性は概ね保たれています。")

    if k["mean"] > 9.0:
        if spine_flag == "bad":
            pro_lines.append("前傾姿勢の変化も重なり、下半身の安定性に影響している可能性があります。")
        elif spine_flag == "warn":
            pro_lines.append("前傾維持のばらつきが、下半身の安定性に一部影響している可能性があります。")

    pro_comment = " ".join(pro_lines[:3])

    bench = [
        _bench_line("膝ブレ(%)", "%", "mean", _le_ideal(9.0, "%"), current=float(k["mean"])),
        _bench_line("膝ブレの再現性(%)", "%", "σ", _le_ideal(2.5, "%"), current=float(k["std"])),
        _bench_line("膝ブレの安定目安(%)", "%", "mean", _le_ideal(5.5, "%"), current=float(k["mean"])),
    ]

    return {
        "title": "06. Knee Stability（膝）",
        "value": f'max {k["max"]:.1f} / mean {k["mean"]:.1f} / σ {k["std"]:.1f} (%) （conf {conf:.3f}）',
        "tags": j["tags"],
        "bench": bench,
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
            "軸が揺れると、打点やフェース向きの再現性に影響が出やすくなります。",
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

    sh = raw.get("shoulder", {})
    h = raw.get("head", {})
    k = raw.get("knee", {})
    xf = raw.get("x_factor", {})
    conf = _conf(raw)
    frames = _frames(raw)
    spine_flag = judge_spine_flag(raw)

    lines: List[str] = []
    lines.append(f"今回のスイングは「{swing_type}」です（confidence {conf:.3f} / 区間 {frames} frames）。")
    lines.append("※ 初回の方は、まずは「最優先テーマ」だけを確認してください。")
    lines.append("")

    h_mean = h.get("mean", 0)
    k_mean = k.get("mean", 0)

    # 軸の安定性（やや緩和）
    if h_mean > 6.5 or k_mean > 10.0 or spine_flag == "bad":
        if spine_flag == "bad":
            lines.append(
                f"【軸の安定性】頭部（{h_mean:.1f}%）や膝（{k_mean:.1f}%）の左右動に加え、前傾維持の崩れも大きく見られます。"
                " 回転量よりも先に、まずは『土台と上体の安定性』を高めることが、打点の再現性を整える近道です。"
            )
        else:
            lines.append(
                f"【軸の安定性】頭部（{h_mean:.1f}%）や膝（{k_mean:.1f}%）の左右動がやや大きく、"
                " スイング全体の安定性に影響しています。まずは『土台の安定』を整えることが重要です。"
            )
    elif h_mean > 5.0 or k_mean > 8.5 or spine_flag == "warn":
        lines.append(
            f"【軸の安定性】頭部（{h_mean:.1f}%）や膝（{k_mean:.1f}%）の動き、または前傾維持にややばらつきが見られます。"
            " 大きな崩れではありませんが、土台と上体の再現性を高める余地があります。"
        )
    else:
        lines.append(
            "【軸の安定性】頭部・下半身ともにブレは比較的抑えられており、前傾維持も概ね安定しています。"
            " 軸回転の土台はおおむね整っています。"
        )

    xf_max = xf.get("max", 0)

    # パワー効率（やや緩和）
    if xf_max < 30:
        lines.append(
            f"【パワー効率】捻転差（max {xf_max:.1f}°）はやや小さめで、"
            " 切り返しで力を溜める余地があります。"
        )
    elif xf_max > 70:
        lines.append(
            f"【パワー効率】捻転差（max {xf_max:.1f}°）は大きめで、"
            " 肩と腰の連動を整えるとさらに安定しやすくなります。"
        )
    else:
        lines.append(
            f"【パワー効率】捻転差（max {xf_max:.1f}°）は十分に確保されており、"
            " 切り返しに必要な準備はできています。"
        )

    sh_std = sh.get("std", 0)

    # 再現性（やや緩和）
    if sh_std > 18.0:
        lines.append(
            f"【再現性】肩回転の深さにばらつき（σ {sh_std:.1f}°）が見られます。"
            " トップ位置の再現性に影響しやすい状態です。"
        )
    elif sh_std > 12.0:
        lines.append(
            f"【再現性】肩回転の深さにややばらつき（σ {sh_std:.1f}°）があり、"
            " ミート率の安定性を高める余地があります。"
        )

    lines.append("")

    if priorities:
        p_str = "／".join(priorities)
        lines.append(f"数値上の最優先テーマは「{p_str}」です。")
    else:
        lines.append("数値上の優先テーマはありません。")

    if "前傾維持不安定" in priorities:
        lines.append("特に前傾維持の崩れが大きいため、切り返しからインパクトまで上体角度を保つ意識が重要です。")
    elif "前傾維持にばらつき" in priorities:
        lines.append("特に前傾維持にややばらつきがあるため、場面ごとの上体角度の再現性を高めたい状態です。")

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

    xf_mean = float(xf.get("mean", 0.0))

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

    # 捻転差
    if xf_mean < 30:
        tags.append("捻転差不足")
    elif xf_mean > 70:
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
        lines.append(f"本動画では捻転差が mean {xf_mean:.1f}° でやや小さく、切り返しで力を溜める余地があります。")

    if "捻転差過多" in priorities:
        lines.append(f"本動画では捻転差が mean {xf_mean:.1f}° で大きめで、肩と腰の連動を整える余地があります。")

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

    if xf_mean >= 30:
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
    # 02〜06の各セクションからタグを収集
    for k in ["02", "03", "04", "05", "06"]:
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


def build_paid_08(analysis: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:

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

    selected_drills = select_drills_with_priority(
        all_tags,
        priorities,
        3
    )

    for d in selected_drills:

        extra_notes = []

        if sh_std > 18:

            extra_notes.append(
                f"● 【再現性補足】肩回転のばらつき（σ {sh_std:.1f}）がやや大きいため、"
                "回数よりも『同じトップ位置をゆっくり再現すること』を優先してください。"
            )

        if spine_flag == "bad" and d["id"] in {

            "spine_posture_keep",
            "hip_depth_wall"

        }:

            extra_notes.append(
                "● 【前傾維持補足】切り返しからインパクトまで胸の向きと上体角度を揃える意識を優先してください。"
            )

        elif spine_flag == "warn" and d["id"] == "spine_posture_keep":

            extra_notes.append(
                "● 【前傾維持補足】動作スピードを少し落とし、上体角度が大きく変わらない範囲で行ってください。"
            )

        if extra_notes:

            d["how"] += "\n\n" + "\n".join(extra_notes)

    return {

        "title": "08. Training Drills（練習ドリル）",

        "drills": [

            {

                "name": d["name"],

                "purpose": d["purpose"],

                "how": d["how"]

            }

            for d in selected_drills

        ]

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

    a = _norm_inverse(head, 3.0, 8.0)
    b = _norm_inverse(knee, 4.0, 10.0)
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
    import logging
    from typing import List, Dict, Any

    logging.warning("[DEBUG] build_paid_09 START (Using aggregate raw data)")

    # --- 1. 基礎データの取得と正規化（完全維持） ---
    hs = _to_float_or_none(user_inputs.get("head_speed"))
    miss = _norm_miss(user_inputs.get("miss_tendency"))
    gender = _norm_gender(user_inputs.get("gender"))

    # 外部関数（KeyErrorガード）: 返り型に合わせて int fallback
    try:
        power_idx = int(calc_power_idx(raw))
    except Exception:
        logging.warning("[DEBUG] calc_power_idx failed; fallback")
        power_idx = 10

    try:
        stability_idx = int(calc_stability_idx(raw))
    except Exception:
        logging.warning("[DEBUG] calc_stability_idx failed; fallback")
        stability_idx = 10

    # --- 2. 実測値（単数形キー）の抽出（完全維持） ---
    def _f(path: List[str], default: float) -> float:
        """raw のネスト辞書から float を安全に取り出す"""
        cur: Any = raw
        try:
            for k in path:
                if not isinstance(cur, dict):
                    return default
                cur = cur.get(k)
            return float(cur)
        except Exception:
            return default

    # head/knee は analyze 側で %相当になっている想定
    h_mean = _f(["head", "mean"], 10.0)
    k_mean = _f(["knee", "mean"], 10.0)
    stability_val = (h_mean + k_mean) / 2.0  # %相当

    avg_xfactor = _f(["x_factor", "mean"], 0.0)

    # 手首の実測値（mean基準へ完全同期）
    wrist_cock = _f(["wrist", "mean"], 0.0) 
    max_wrist = _f(["wrist", "max"], wrist_cock)
    wrist_std = _f(["wrist", "std"], 0.0)

    # --- 3. 判定バンド（元のロジックを1文字も削らず復旧） ---
    def _band_stability(stability_val: float) -> str:
        if stability_val < 5.0: return "stable"
        if stability_val < 7.0: return "normal"
        return "unstable"

    def _band_xfactor(avg_xfactor: float) -> str:
        if avg_xfactor < 40.0: return "low"
        if avg_xfactor < 45.0: return "mid"
        return "high"

    def _band_tame(max_wrist: float, mean_wrist: float, std_wrist: float) -> str:
        if max_wrist < 45.0:     
            return "shallow"
        if max_wrist < 75.0:     
            return "normal"
        if mean_wrist < 45.0 or std_wrist >= 15.0:  
            return "unstable_deep"
        return "deep"

    # バンド割り当て
    stab_band = _band_stability(stability_val)
    xf_band = _band_xfactor(avg_xfactor)
    tame_band = _band_tame(max_wrist, wrist_cock, wrist_std)

    # 【2軸分析用レベル判定：mean基準 45°-75°】
    if hs is not None:
        hs_level = "low" if hs < 38 else ("mid" if hs <= 45 else "high")
    else:
        hs_level = "low" if power_idx < 12 else ("mid" if power_idx <= 18 else "high")
    
    # 手首コック(mean)による分類：45°未満を「浅め」、45-75°を「標準」、75°超を「深め」と判定
    cock_level = "shallow" if wrist_cock < 45.0 else ("deep" if wrist_cock > 75.0 else "normal")
    cock_label = "浅め" if cock_level == "shallow" else ("深め" if cock_level == "deep" else "標準")

    # --- 4. rows の作成 ---
    rows: List[Dict[str, str]] = []

    # 項目: 診断サマリ
    rows.append({
        "item": "診断サマリ",
        "guide": "今回の分析根拠",
        "reason": "\n".join([
            f"● 軸ブレ：{stability_val:.1f}%（{stab_band}）",
            f"● 捻転差：{avg_xfactor:.1f}°（{xf_band}）",
            f"● タメ平均：{wrist_cock:.1f}°（目安：45-75°に対して {cock_label}）",
        ])
    })

    # --- 項目: 重量（HS × 性別 × 安定性 × タメの2軸反映） ---
    if hs is not None:
        if gender == "female":
            if hs < 35: weight = "30〜40g"
            elif hs < 40: weight = "40〜50g"
            elif hs < 45: weight = "50g前後"
            else: weight = "60g前後"
            reason = f"● 2軸評価：HS {hs:.1f}m/s × タメ平均{wrist_cock:.1f}° に対する適正重量を選定\n● 女性の身体特性を考慮し、振り抜きやすさを最優先"
        else:
            if hs < 35: weight = "40〜50g"
            elif hs < 40: weight = "50g前後"
            elif hs < 45: weight = "50〜60g"
            else: weight = "60〜70g"
            reason = f"● 2軸評価：HS {hs:.1f}m/s × タメ平均{wrist_cock:.1f}° の負荷に耐えうる基準重量"

        # 【追加補正】
        if (hs >= 40 and stability_val > 7.0) or (cock_level == "deep" and stability_val > 5.0):
            if "60g" not in weight:
                weight = "60g前後"
                reason += f"\n● 【補正】タメの深さと軸ブレ実測（{stability_val:.1f}%）を考慮し、重量を上げて挙動を安定化"
    else:
        band = "low" if power_idx < 12 else ("mid" if power_idx <= 18 else "high")
        weight = {"low": "40〜50g", "mid": "50〜60g", "high": "60〜70g"}[band]
        reason = f"● 2軸評価：パワー指数（{power_idx}）に基づく推奨重量（タメ{cock_label}を考慮）"

    rows.append({"item": "重量", "guide": weight, "reason": reason})

    # --- 項目: フレックス（HS × 捻転差 × タメの2軸反映） ---
    if hs is not None:
        flex_map = [(33, "L〜A"), (38, "A〜R"), (42, "R〜SR"), (46, "SR〜S"), (50, "S〜X")]
        flex = next((f for h, f in flex_map if hs < h), "X")
        
        reason = f"● 2軸評価：HS {hs:.1f}m/s × タメ平均{wrist_cock:.1f}° に対する適正剛性"
        
        if avg_xfactor > 45.0 or cock_level == "deep":
            if flex in ["L〜A", "A〜R", "R〜SR"]:
                flex = "SR〜S"
            else:
                flex = "一ランク硬め"
            reason += f"\n● 【補正】強い捻転差（{avg_xfactor:.1f}°）と深いタメによるシャフトへの高負荷を考慮"
    else:
        flex = {"low": "A〜R", "mid": "R〜SR", "high": "SR〜S"}[hs_level]
        reason = f"● 2軸評価：パワー指数に対する適正剛性（タメ{cock_label}を考慮）"

    rows.append({"item": "フレックス", "guide": flex, "reason": reason})

    # --- 項目: キックポイント（ミス傾向 × タメ角：逆転ロジックを指示通り詳細化） ---
    if miss == "right":
        kp, base_reason = "先〜中", "● 右ミスに対し、つかまりを助ける先調子系を基準"
    elif miss == "left":
        kp, base_reason = "中〜元", "● 左ミスに対し、先端の動きを抑えた元調子系を基準"
    else:
        kp, base_reason = "中", "● ニュートラルな挙動の中調子を基準"

    reason_lines = [
        base_reason,
        f"● 実測タメ：最大 {max_wrist:.1f}° / 平均 {wrist_cock:.1f}°"
    ]

    # 【重要：逆転ロジック】右ミス ×（タメ浅い ＝ 元へ逆転）
    if miss == "right" and (cock_level == "shallow" or tame_band == "unstable_deep" or wrist_std >= 15.0):
        kp = "元"
        reason_lines += [
            "● 【判定】自力でのタメが浅い（または不安定な）ため、元調子を推奨",
            "● シャフトのしなりで『タメの間』を意図的に作り、右ミスを抑制する"
        ]
    elif miss == "right" and stability_val > 7.0:
        kp = "中"
        reason_lines += [f"● 【判定】軸ブレ {stability_val:.1f}% が大きいため、中調子で安定を優先"]

    rows.append({"item": "キックポイント", "guide": kp, "reason": "\n".join(reason_lines)})

    # --- 項目: トルク（安定性 × ミス補正） ---
    if stability_val >= 9.0:
        tq, base_reason = "3.0〜4.0", f"● 軸ブレ実測（{stability_val:.1f}%）が大きいため低トルクで抑制"
    elif stability_val >= 5.0:
        tq, base_reason = "3.5〜5.0", f"● 軸ブレ実測（{stability_val:.1f}%）に基づき標準帯を選択"
    else:
        tq, base_reason = "4.5〜6.0", f"● 軸ブレ実測（{stability_val:.1f}%）が小さく再現性を重視"

    if miss == "right":
        tq = "4.5〜5.5" if stability_val >= 5.0 else "5.5以上"
        reason = base_reason + "\n● 右ミス補正：トルクを増やしてフェースターンをサポート"
    elif miss == "left":
        tq = "2.5〜3.5"
        reason = base_reason + "\n● 左ミス補正：トルクを絞り、つかまり過ぎを抑制"
    else:
        reason = base_reason

    rows.append({"item": "トルク", "guide": tq, "reason": reason})

    # --- 8. 【項目: 総評】最適シャフトスペック ---
    matrix_desc = {
        ("low", "shallow"): "自力でのタメが浅めな分をシャフト全体のしなり戻りで補い、飛距離効率を高めやすいセッティングです。",
        ("low", "normal"):  "振り抜きやすさを重視し、スイング中のリズムと打点の安定を両立しやすいセッティングです。",
        ("low", "deep"):    "深めのタメによるエネルギーを活かし、インパクトで効率よくボールに伝えやすいセッティングです。",
        ("mid", "shallow"): "切り返しでの打ち急ぎをシャフトの粘りで抑え、インパクトの厚みと安定性を高めやすいセッティングです。",
        ("mid", "normal"):  "スイングのクセを抑えながら、操作性と安定性のバランスを取りやすい実用的なセッティングです。",
        ("mid", "deep"):    "深めのタメを受け止め、インパクト効率と飛距離性能を両立しやすいセッティングです。",
        ("high", "shallow"):"しっかり振ってもヘッド挙動が暴れにくく、左方向へのミスを抑えながら振り抜きやすいセッティングです。",
        ("high", "normal"): "パワーを効率よくボールに伝えやすく、弾道コントロールと安定性を両立しやすいセッティングです。",
        ("high", "deep"):   "強いタメを受け止めやすく、インパクト効率と方向安定性を高めやすいセッティングです。",
    }
    
    # 逆転判定（元調子）との矛盾解消用説明文
    if kp == "元" and cock_level == "shallow":
        final_desc = "手元側のしなりにより『タメの間』を自動生成し、物理的に飛距離ロスと右ミスを防ぐセッティングです。"
    else:
        final_desc = matrix_desc.get((hs_level, cock_level), "解析数値に基づき、個別フィッティングでの最終調整を推奨します。")

    rows.append({
        "item": "総評",
        "guide": "最適シャフトスペック",
        "reason": f"● {final_desc}\n● 推奨詳細：【 重量{weight} / {flex} / {kp}調子 / トルク{tq} 】"
    })

    return {
        "title": "09. Shaft Fitting Guide（推奨）",
        "table": rows,
        "note": "※本結果は解析数値に基づく指標です。購入時は試打での最終確認を推奨します。",
        "meta": {
            "power_idx": power_idx,
            "stability_idx": stability_idx,
            "wrist_cock": wrist_cock,
            "head_speed": hs,
            "stability_val": stability_val,
            "avg_xfactor": avg_xfactor,
            "max_wrist": max_wrist,
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

    worst = max(delta_mean, delta_top, delta_impact)

    if worst <= 5.5:
        return "ok"
    if worst <= 10.0:
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
    
def build_analysis(raw: Dict[str, Any], premium: bool, report_id: str, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {"01": build_section_01(raw)}

    spine_flag = judge_spine_flag(raw)

    if not premium:
        analysis["07"] = build_free_07(raw)
        return analysis

    analysis["02"] = build_paid_02_shoulder(raw, seed=report_id)
    analysis["03"] = build_paid_03_hip(raw, seed=report_id)
    analysis["04"] = build_paid_04_wrist(raw, seed=report_id)
    analysis["05"] = build_paid_05_head(raw, seed=report_id)
    analysis["06"] = build_paid_06_knee(raw, seed=report_id)

    # 05 / 06 への前傾補足は削除
    # 理由:
    # - build_paid_05_head / build_paid_06_knee の内部ですでに spine_flag を見ている
    # - ここで追加すると前傾が二重反映になり、主因に見えやすくなるため

    analysis["07"] = build_paid_07_from_analysis(analysis, raw)

    # 07 Summary では前傾を1回だけ補足
    if spine_flag == "ok":
        analysis["07"].setdefault("text", []).append(
            "【前傾維持】前傾角の変化は小さく、回転動作の再現性は安定しています。"
        )

    elif spine_flag == "warn":
        analysis["07"].setdefault("text", []).append(
            "【前傾維持】スイング中に前傾角の変化がやや見られ、動作の再現性に影響する可能性があります。"
        )

    elif spine_flag == "bad":
        analysis["07"].setdefault("text", []).append(
            "【前傾維持】スイング中の前傾角の変化がやや大きく、回転動作の安定性に影響しています。"
        )

    # 必要ならタグとして持たせる（ドリル選定用）
    if spine_flag == "bad":
        analysis["07"].setdefault("tags", [])
        if "前傾維持不安定" not in analysis["07"]["tags"]:
            analysis["07"]["tags"].append("前傾維持不安定")
    elif spine_flag == "warn":
        analysis["07"].setdefault("tags", [])
        if "前傾維持やや不安定" not in analysis["07"]["tags"]:
            analysis["07"]["tags"].append("前傾維持やや不安定")

    analysis["08"] = build_paid_08(analysis, raw)

    # 08 は無条件の前傾ドリル insert をやめる
    # build_paid_08 内の通常ロジックに任せる
    # どうしても補強するなら、head / knee も崩れている時だけ追加
    if spine_flag == "bad":
        head_mean = float(raw.get("head", {}).get("mean", 0.0))
        knee_mean = float(raw.get("knee", {}).get("mean", 0.0))

        if head_mean > 6.0 or knee_mean > 9.0:
            analysis["08"].setdefault("drills", [])

            existing_names = {d.get("name") for d in analysis["08"]["drills"] if isinstance(d, dict)}
            if "前傾維持ドリル" not in existing_names:
                analysis["08"]["drills"].append({
                    "name": "前傾維持ドリル",
                    "purpose": "アドレスで作った姿勢を保ちながら回転する感覚を身につけ、フォーム全体の再現性を高める",
                    "how": "アドレスで作った前傾を保ちながら、肩と腰をゆっくり回すハーフスイングを10回×2セット行う"
                })

    # 09は入力がある場合のみ出力
    ui = user_inputs or {}
    if ui.get("head_speed") is not None or ui.get("miss_tendency") or ui.get("gender"):
        analysis["09"] = build_paid_09(raw, ui)

    analysis["10"] = build_paid_10(analysis)

    # 10 Summary は前傾を先頭に入れるが、表現は弱める
    head_mean = float(raw.get("head", {}).get("mean", 0.0))
    knee_mean = float(raw.get("knee", {}).get("mean", 0.0))

    if spine_flag == "ok":
        analysis["10"].setdefault("text", []).insert(0,
            "前傾姿勢は全体として比較的安定しており、スイングの再現性を支えています。"
        )
    elif spine_flag == "warn":
        analysis["10"].setdefault("text", []).insert(0,
            "前傾姿勢にはやや変化が見られ、局面によって再現性に影響している可能性があります。"
        )
    elif spine_flag == "bad":
        # head / knee がそこまで悪くない時は mild 表現
        if head_mean <= 6.0 and knee_mean <= 9.0:
            analysis["10"].setdefault("text", []).insert(0,
                "前傾姿勢にはやや変化が見られますが、全体として大きくバランスを崩している状態ではありません。"
            )
        else:
            analysis["10"].setdefault("text", []).insert(0,
                "前傾姿勢の変化がやや大きく、スイング全体の再現性に影響している可能性があります。"
            )

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
                overlay_out_path=overlay_out
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

        analysis = build_analysis(raw=raw, premium=premium, report_id=report_id, user_inputs=user_inputs)
        address_posture = judge_address_posture(raw)
        spine_maintain = judge_spine_maintain_display(raw)
        
       
        report_ref.set({
            "status": "DONE",
            "raw": raw,
            "analysis": analysis,
            "address_posture": address_posture,
            "spine_maintain": spine_maintain,
            "overlay_video_url": overlay_url,
            "overlay_video_download_url": overlay_download_url,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }, merge=True)

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
                            "label": "🎬 動画をシェアする",
                            "uri": share_uri
                        }
                    }
                ]
            }
        }

        line_bot_api.push_message(
            user_id,
            FlexSendMessage(
                alt_text="スイング診断完了のお知らせ",
                contents=flex_contents
            )
        )
        

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


# server.py 上部（1回だけ）
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
db = firestore.Client()

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

    event_type = event.get("type")

    # =========================================================
    # A) 購入完了（単発/回数券）
    # =========================================================
    if event_type == "checkout.session.completed":
        session = event["data"]["object"]
        event_id = event.get("id")
        session_id = session.get("id")
        line_user_id = session.get("client_reference_id")

        if not line_user_id:
            print("❌ client_reference_id missing", flush=True)
            return "OK", 200

        try:
            li = stripe.checkout.Session.list_line_items(session_id, limit=1)
            first = li["data"][0] if li and li.get("data") else None
            price_id = first.get("price", {}).get("id") if first else None

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
                message = (
                    "月額プランのご契約、誠にありがとうございます。\n\n"
                    "ご契約期間中は、無制限でご利用いただけます。\n\n"
                    "リッチメニューの「分析スタート」からスイング動画を送信してください。"
                )
            else:
                message = (
                    "チケットのご購入、誠にありがとうございます。\n\n"
                    f"残りチケット：{tickets}回\n\n"
                    "リッチメニューの「分析スタート」からスイング動画を送信してください。"
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
            customer_id = subscription.get("customer")

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
        plan_text = (
            "GATE公式LINEへようこそ！⛳️\n\n"
            "正確なAI解析結果をお届けするため、画面上部に「追加」ボタンが表示されている方は、まず登録をお願いいたします。\n\n"
            "決済完了後は、このトーク画面にスイング動画を送るだけでAI解析がスタートします。\n"
            "--------------------\n\n"
            "【単発プラン】500円/1回\n"
            "単発プランで試す → \n"
            f"https://buy.stripe.com/00w28sdezc5A8lR2ej18c00?client_reference_id={user_id}\n\n"
            "【回数券プラン】1,980円/5回\n"
            "回数券を購入する → \n"
            f"https://buy.stripe.com/bJeaEY1vR9Xs7hN4mr18c07?client_reference_id={user_id}\n\n"
            "【月額プラン】4,980円/月\n"
            "月額プランを申し込む → \n"
            f"https://buy.stripe.com/3cIfZi2zVd9E1XtdX118c05?client_reference_id={user_id}\n\n"
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

    # ===== 1) 分析スタート → Quick Reply =====
    if text == "分析スタート":
        is_premium = is_premium_user(user_id)
        users_ref.document(user_id).set({
            "prefill_step": "head_speed",
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        # 冒頭文を整理し、各セクションの間に空行（\n\n）を入れて視認性を向上
        msg_text = (
            "ご利用ありがとうございます。\n\n"
            "無料版の方は、このまま動画を送ってください。\n\n"
            "有料版の方で、より正確なフィッティング分析レポート（09）をご希望の方は、分かる範囲で入力をお願いします。\n\n"
            "---------------------\n"
            "【必須】ヘッドスピード／主なミスの傾向（1つ）\n"
            "【任意】性別\n\n"
            "このあと順番にご案内します。\n"
            "まずはヘッドスピードを数字だけで送ってください（例：43）。\n\n"
            "※フィッティング分析レポートを希望されない場合は、そのまま動画を送信してください。\n"
            "※途中で入力をやめたい場合は「スキップ」と送ってください。"
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=msg_text)
        )
        return

    # ===== 2) user取得 → step =====
    user_doc = users_ref.document(user_id).get()
    user_data = user_doc.to_dict() or {}
    step = user_data.get("prefill_step")
    logging.warning("[DEBUG] prefill_step=%r", step)

    # 【修正箇所】性別判定を if step: の外に配置。
    # ここに置くことで、stepがNoneになっても「男性/女性」という文字を最優先で捕まえます。
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
                        "続けて、主なミスの傾向を1つだけ送ってください。\n"
                        "（例：スライス／フック／トップ／ダフリ）"
                    )
                )
            )
            return

        if step == "miss_tendency":
            users_ref.document(user_id).set({
                "prefill_step": None,  # ここでリセットされるため、性別判定は上にないと動かない
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
            "prefill_step": "head_speed",
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="まずヘッドスピードを数字だけで送ってください（例：42）。")
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
