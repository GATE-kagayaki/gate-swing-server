import os
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

import stripe

app = Flask(__name__, template_folder="templates")
app.config["JSON_AS_ASCII"] = False

line_bot_api = LineBotApi(os.environ.get('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.environ.get('LINE_CHANNEL_SECRET'))

def get_stripe_secrets():
    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    endpoint_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    return stripe.api_key, endpoint_secret


@app.route("/webhook", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    try:
        # ここで処理を実行します
        handler.handle(body, signature)
    except Exception as e:
        # エラーが起きたら、ログに詳しく書き出すようにします！
        print(f"!!! Webhook Error !!!: {e}")
        logging.error(traceback.format_exc())
        return 'Internal Error', 500 # 400から500に変えて、サーバー側のミスだと明示します
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # LINEからメッセージが来たら、下の handle_text_message を実行する
    handle_text_message(event)

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
    import math
    import os  # 環境変数設定用に追加
    from typing import List, Dict, Any

    # Cloud Run 等のGPU非搭載環境でEGLエラー(0x3008)が出るのを防ぐため、CPUを強制指定します。
    os.environ['MP_DEVICE'] = 'cpu'

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

    # model_complexity=1 はCPU環境で速度と精度のバランスが最も良い設定です。
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        base_nose = None
        base_lknee = None
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ここでGPUを探しに行ってエラーが出ていましたが、CPU指定により回避されます。
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            valid_frames += 1

            def xyz(i):
                return (lm[i].x, lm[i].y, lm[i].z)

                           
            LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LH = mp_pose.PoseLandmark.LEFT_HIP.value
            RH = mp_pose.PoseLandmark.RIGHT_HIP.value
            LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
            LW = mp_pose.PoseLandmark.LEFT_WRIST.value
            LI = mp_pose.PoseLandmark.LEFT_INDEX.value
            NO = mp_pose.PoseLandmark.NOSE.value
            LK = mp_pose.PoseLandmark.LEFT_KNEE.value

              
            curr_nose = xyz(NO)
            curr_lknee = xyz(LK)

            if base_nose is None:
                base_nose = curr_nose
                base_lknee = curr_lknee

            sh = angle_3d(xyz(LS), xyz(RS), xyz(RH))
            hip = angle_3d(xyz(LH), xyz(RH), xyz(LK))
            wr = 180.0 - angle_3d(xyz(LE), xyz(LW), xyz(LI))
            # 頭部・膝：アドレス基準の「3次元移動距離」を計算
            # 3次元距離公式: $$d = \sqrt{(x-x_0)^2 + (y-y_0)^2 + (z-z_0)^2}$$
            def dist_3d(p, base):
                return math.sqrt(sum((a - b)**2 for a, b in zip(p, base)))

            hd = dist_3d(curr_nose, base_nose) * 100  # 画面幅に対する%に変換
            kn = dist_3d(curr_lknee, base_lknee) * 100

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

    def _safe_mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def _safe_std(xs):
        if not xs: return 0.0
        m = _safe_mean(xs)
        return math.sqrt(sum((x - m)**2 for x in xs) / len(xs))

    def pack(xs: List[float], nd: int = 2) -> Dict[str, float]:
        if not xs:
            return {"max": 0.0, "min": 0.0, "mean": 0.0, "std": 0.0}
        return {
            "max": round(float(max(xs)), nd),
            "min": round(float(min(xs)), nd), # 最小値を追加
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

    # 良い点（最低1行） --- ポジティブバッファ拡充 ---
    if sh["std"] <= 10:
        good.append("肩の回し幅は揃っており、上半身の再現性は確保されています。")
    if 85 <= sh["mean"] <= 105:
        good.append("肩の回旋量は基準レンジに収まっており、効率的な捻転ができています。")
    if xf["mean"] >= 35:
        good.append("肩と腰の差（捻転差）は確保できており、出力の準備が整っています。")
    
    # バッファ：回転量が多い場合
    if sh["mean"] > 105:
        good.append("深い肩の回転を可能にする柔軟性があり、大きな飛距離を生む潜在能力があります。")
    # バッファ：数値は外れていても安定している場合
    if sh["std"] <= 7 and not (85 <= sh["mean"] <= 105):
        good.append("角度自体は調整の余地がありますが、常に同じ深さまで回せる安定感は大きな武器です。")

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

    # 良い点（最低1行） --- ポジティブバッファ拡充 ---
    if hip["std"] <= 10:
        good.append("腰の回し幅は揃っており、下半身の再現性は確保されています。")
    if 36 <= hip["mean"] <= 50:
        good.append("腰の回旋量は基準レンジに収まっており、土台として機能しています。")
    
    # バッファ：安定性
    if hip["std"] <= 5:
        good.append("下半身の動きが非常に安定しており、ミート率を高める基礎ができています。")
    # バッファ：捻転の深さ
    if hip["mean"] < 36 and xf["mean"] >= 40:
        good.append("腰の回転は控えめですが、その分肩との捻転差をしっかり作れています。")

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
    # --- 【重要】180度からの引き算で「コック角」に変換 ---
    # raw["wrist"]["mean"] が 159.0 の場合、w_mean は 21.0 になります
    w_mean = 180.0 - float(raw["wrist"]["mean"])
    xf_mean = float(raw["x_factor"]["mean"])

    main = "mid"
    if w_mean < 70:      # 70度より曲がっていない（タメが浅い）
        main = "low"
    elif w_mean > 90:    # 90度より深く曲がっている（タメが深い）
        main = "high"

    rel = "mid"
    if xf_mean < 35:
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
    # 数値変換：180 - 内角 = コック角
    w_mean = 180.0 - float(w_raw["mean"])
    w_max  = 180.0 - float(w_raw["min"]) if "min" in w_raw else (180.0 - w_mean)
    w_std  = float(w_raw["std"])
    
    j = judge_wrist(raw)
    conf = _conf(raw)

    good: List[str] = []
    bad: List[str] = []

    # --- 良い点（プロの視点） ---
    if w_std <= 8:
        good.append("手首の角度変化が非常に一定しており、インパクトでのフェース管理能力が極めて高いです。")
    if 70 <= w_mean <= 90:
        good.append("理想的なタメ（L字）が形成されており、効率的にヘッドを加速させる準備ができています。")
    if w_max > 100:
        good.append("トップでの深いコックを許容する柔軟性があり、爆発的な飛距離を生み出す潜在能力があります。")
    
    if not good: good = ["基本的な手首の可動域は確保されており、スイングの土台はできています。"]

    # --- 改善点（プロの指摘） ---
    if w_mean < 70:
        bad.append(f"平均コック角 {w_mean:.1f}° は浅く、アーリーリリースの傾向があります。")
    if w_std > 15:
        bad.append(f"手首の挙動（σ {w_std:.1f}）が不安定で、インパクトの打点がバラつきやすい状態です。")
    if w_max < 60:
        bad.append("バックスイングでのコックが完了する前に切り返しており、パワーロスが生じています。")

    if not bad: bad = ["現在、手首の使い方において大きな修正ポイントは見当たりません。"]

    # --- プロ目線の詳細な言語化（ここを大幅に強化） ---
    pro_lines: List[str] = []
    
    # 状態別の深い解説
    if w_mean < 70:
        pro_lines.append(f"本動画では手首の角度が {w_mean:.1f}° と浅いため、ヘッドを“運ぶ”動きが強く、飛距離がロスしやすい傾向です。")
        pro_lines.append("本来あるべき『タメ』が解けるのが早いため、インパクトで合わせる動きが必要になっています。")
    elif w_mean > 100:
        pro_lines.append(f"最大 {w_max:.1f}° という非常に深いタメを作れていますが、その分、リリースのタイミングがシビアです。")
        pro_lines.append("手元の操作に頼りすぎると、急激なフックやプッシュアウトの原因となります。")
    else:
        pro_lines.append(f"手首のコック角（{w_mean:.1f}°）はプロの基準値に近く、効率的なパワー伝達が行われています。")

    # 安定性に関する洞察
    if w_std > 12:
        pro_lines.append("特に気になるのは再現性です。手首の動きが一定でないため、フェース向きの管理が困難になっています。")
    else:
        pro_lines.append("手首の挙動が安定しているため、シャフトのしなりを一定に使いこなせる状態です。")

    pro_comment = " ".join(pro_lines)

    return {
        "title": "04. Wrist Cock（手首コック）",
        "value": f"Max Cock {w_max:.1f}° / Mean {w_mean:.1f}° (σ {w_std:.1f})",
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
        tags.append("膝ブレ大")
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

    # 良い点（最低1行） --- ポジティブバッファ拡充 ---
    if h["std"] <= 0.03:
        good.append("頭の位置は非常に揃っており、スイング軸の再現性は極めて高いです。")
    if h["mean"] <= 0.10:
        good.append("頭の左右ブレは最小限に抑えられており、理想的な軸の安定感があります。")
    
    # バッファ：許容範囲内の動き
    if 0.10 < h["mean"] <= 0.15:
        good.append("多少の左右移動はありますが、許容範囲内でダイナミックな動きができています。")
    # バッファ：下半身との連動
    if h["mean"] <= 0.12 and k["mean"] <= 0.15:
        good.append("上下の軸が連動して安定しており、ミート率を支える良い土台があります。")

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

    # 良い点（最低1行） --- ポジティブバッファ拡充 ---
    if k["std"] <= 0.04:
        good.append("膝の位置は揃っており、下半身の再現性がインパクトの安定感を生んでいます。")
    if k["mean"] <= 0.12:
        good.append("膝の左右ブレが抑えられており、エネルギーを逃がさない強い土台があります。")
    
    # バッファ：粘りのある下半身
    if 0.12 < k["mean"] <= 0.18:
        good.append("下半身に粘りがあり、スイング中のパワーをしっかり受け止めています。")
    # バッファ：再現性重視
    if k["std"] <= 0.05 and k["mean"] > 0.20:
        good.append("ブレ自体はありますが、毎回同じ場所で踏み込めている点は安定への足がかりになります。")

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
# 08 ドリル（07の優先順位連動 ＋ バリエーション拡充版）
# ==================================================
DRILL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "x_factor_turn",
        "name": "捻転差づくりドリル（肩先行ターン）",
        "category": "体幹",
        "tags": ["捻転差不足"],
        "purpose": "● 肩と腰の回転差（X-Factor）を最大化し、飛距離の源泉となる溜めを作る\n● 切り返しで上半身が突っ込む動きを抑制し、インサイドからの軌道を安定させる",
        "how": "① 腰の位置を固定したまま、肩を90度以上深く回す意識でトップを作る\n② 切り返しの一歩目で左膝をわずかに外へ開き、上半身の始動を一瞬遅らせる\n③ 10回×2セット、筋肉が引き伸ばされる感覚を確認しながらスローモーションで行う",
    },
    {
        "id": "shoulder_control",
        "name": "肩回転コントロールドリル",
        "category": "上半身",
        "tags": ["肩回転過多"],
        "purpose": "● 肩の過剰な回転によるアウトサイドイン軌道を修正し、スイングプレーンを安定させる\n● 回転の「量」ではなく、軸を動かさない「質」を重視し、ミート率を向上させる",
        "how": "① 前傾角度を維持したまま、肩が地面に対し斜め45度のプレーン上を動くよう回転する\n② 肩が浮いたり沈んだりしないよう、鏡の前で軸の傾きを確認しながら行う\n③ 10球×2セット、同じ高さのライナーが打てるまで繰り返す",
    },
    {
        "id": "hip_drive",
        "name": "腰主導ターンドリル",
        "category": "下半身",
        "tags": ["腰回転不足"],
        "purpose": "● 下半身主導の切り返し（ヒップドライブ）を習得し、手打ちを根本から解消する\n● 地面反力を使い、全身の連動性によってヘッドスピードを物理的に底上げする",
        "how": "① トップで静止し、腕の力を使わず「左腰のポケット」を後ろに引く動きから始動する\n② 上体はトップの形を維持し、腰が先に45度回る時間差（タメ）を掴む\n③ 連続素振り15回、足裏で地面を踏みしめる強さを意識する",
    },
    {
        "id": "late_hit",
        "name": "レイトヒットドリル",
        "category": "手首",
        "tags": ["コック不足"],
        "purpose": "● 手首のコックを直前まで維持（タメ）し、インパクトでの加速効率を最大化する\n● アーリーリリースを撲滅し、ダウンブローでボールを捉える厚い当たりを習得する",
        "how": "① トップで1秒静止し、手首の角度を変えずにグリップエンドがボールを指すように下ろす\n② 右腰の高さまで手が降りてきたところで、一気に体の正面でリリースする\n③ 連続素振り10回、重めのクラブやウェッジで行うとより効果的",
    },
    {
        "id": "release_control",
        "name": "リリース抑制ドリル（LtoL）",
        "category": "手首",
        "tags": ["コック過多"],
        "purpose": "● 手首の過剰な介入を抑え、フェース管理を体幹主導に戻すことで方向性を安定させる\n● 急激なフックや引っ掛けを防止し、ライン出しのような正確なショットを習得する",
        "how": "① 腰から腰の振り幅で、腕とクラブが「L」の字を保ったまま体全体のターンで振る\n② インパクト以降も手首をこねず、フェース面が常に自分の方を向いているか確認する\n③ 20回、方向性のばらつきがなくなるまで低く短い球を打つ",
    },
    {
        "id": "head_still",
        "name": "頭固定ドリル（壁チェック）",
        "category": "安定性",
        "tags": ["頭部ブレ大"],
        "purpose": "● スイング軸（首の付け根）の左右ブレを解消し、正確な打点と高いミート率を実現する\n● 視界を一定に保つことで距離感を掴みやすくし、トップやシャンクを防止する",
        "how": "① 壁に頭が軽く触れる位置で構える（または鏡に目印をつける）\n② フィニッシュまで、その位置から頭の幅半分もズレないよう独楽のように回転する\n③ 素振り10回、自分の軸がどこにあるか感覚を研ぎ澄ます",
    },
    {
        "id": "knee_stable",
        "name": "膝ブレ抑制ドリル",
        "category": "下半身",
        "tags": ["膝ブレ大"],
        "purpose": "● 膝の横流れ（スウェー）を抑制し、パワーを逃がさない強固な下半身の壁を構築する\n● 土台を安定させることで、上半身の捻じれを最大限に引き出し回転スピードを上げる",
        "how": "① 両膝の間隔をアドレス時の幅で完全に固定する\n② 体重移動を横ではなく「縦（踏み込み）」に意識し、右膝の向きを正面に保つ\n③ 10回×2セット、太ももの内側に張りが感じられるまで集中して行う",
    },
    {
        "id": "sync_turn",
        "name": "全身同調ターンドリル（クロスアーム）",
        "category": "体幹",
        "tags": ["捻転差不足"],
        "purpose": "● 腕と胴体の一体感を高め、手だけではなく体全体が連動した「ボディターン」を習得する\n● 部位ごとのタイミングのズレを解消し、ショット全体の再現性を向上させる",
        "how": "① 腕を胸の前でクロスさせ、手ではなく「胸の面」を回してバックスイングする\n② 胸と腰がバラバラにならず、かつ適度な時間差を保って同調して回る感覚を掴む\n③ 左右に大きく10回、背骨を中心とした軸回転を深く行う",
    },
    {
        "id": "step_transition",
        "name": "足踏みステップドリル",
        "category": "下半身",
        "tags": ["腰回転不足", "下半身不安定", "捻転差不足"],
        "purpose": "● 下半身主導の切り返しタイミングと、全身のダイナミックな連動性を習得する\n● 重心移動をスムーズに行い、フィニッシュまで一気に振り抜く推進力を養う",
        "how": "① 足を閉じて構え、バックスイングの頂点に達する瞬間に左足を踏み出す\n② 左足が着地した反動を利用して、ダウンスイングを爆発的に始動させる\n③ 止まらずに一気に振り抜く動作を15回連続で行い、リズム感を体得する",
    },
    {
        "id": "tempo_rhythm",
        "name": "テンポ一定ドリル（メトロノーム）",
        "category": "再現性",
        "tags": ["ばらつき大"],
        "purpose": "● スイングのリズムを一定にし、各部位が連動するタイミングのズレ（ばらつき）を解消する\n● プレッシャーのかかる場面でも崩れない、自分だけの安定したテンポを構築する",
        "how": "① 一定のリズム（イチ、ニ、サン）を口に出しながら、フィニッシュまで澱みなく振る\n② メトロノームを使用し、同じテンポで何度も素振りを行う\n③ 連続素振り20回、無意識でも同じ速さで振れるまで神経系を繋ぐ",
    },
    {
        "id": "towel_release",
        "name": "タオルスイング（リリース管理）",
        "category": "手首",
        "tags": ["コック過多", "リリースのばらつき大"],
        "purpose": "● 手首の早解けを防ぎ、遠心力が最大化されるポイント（左足前）でのリリースを養う\n● 体幹の回転とリリースのタイミングを一致させ、分厚いインパクトを実現する",
        "how": "① タオルの先端を結び、ダウンスイングで結び目が背中に当たるのを待ってから振る\n② インパクト以降（左足前）で「シュッ」と音が鳴るように加速ポイントを意識する\n③ 10回×3セット、音が鳴る位置が安定するまで集中して行う",
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
    # 07の解析結果から優先課題を取得
    sec07 = analysis.get("07") or {}
    meta07 = sec07.get("meta") or {}
    priorities = meta07.get("priorities", [])
    
    # すべての検知タグを収集
    all_tags = collect_all_tags(analysis)
    
    # 【数値による動的タグ付与】ばらつきが大きい場合、再現性ドリルを候補に入れる
    sh_std = raw.get("shoulder", {}).get("std", 0)
    if sh_std > 15:
        all_tags.append("ばらつき大")
        all_tags.append("肩回転のばらつき大")

    # 優先順位を考慮してドリルを選定
    selected_drills = select_drills_with_priority(all_tags, priorities, 3)
    
    # 【AI数値アドバイス】ばらつき（σ）が大きいユーザーへの動的注釈
    # build_paid_08 関数内の for d in selected_drills: ループ内
    for d in selected_drills:
        if sh_std > 15:
            # プロらしい詳細な指導文に差し替え
            d["how"] += f"\n\n● 【プロの特別指導】現在、動作に $\sigma$ {sh_std:.1f} という大きなばらつきが検出されています。回数よりも『ゆっくりとした正確な動き』による神経系への定着を最優先してください。"

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
    import logging
    logging.warning("[DEBUG] build_paid_09 user_inputs=%r", user_inputs)

    # --- 数値の取得と変換 ---
    power_idx = calc_power_idx(raw)
    stability_idx = calc_stability_idx(raw)
    hs = _to_float_or_none(user_inputs.get("head_speed"))
    miss = _norm_miss(user_inputs.get("miss_tendency"))
    gender = _norm_gender(user_inputs.get("gender"))

    # 【重要】手首の数値をコック角（180 - 内角）に変換
    wrist_cock = 180.0 - float(raw["wrist"]["mean"])
    sh_mean = float(raw["shoulder"]["mean"])

    rows: List[Dict[str, str]] = []

    # --- 1. 重量（HS × 安定性解析） ---
    if hs is not None:
        if hs < 35: weight = "40〜50g"
        elif hs < 40: weight = "50g前後"
        elif hs < 45: weight = "50〜60g"
        else: weight = "60〜70g"
        reason = f"● ヘッドスピード {hs:.1f}m/s の基準重量\n"
        if hs >= 40 and stability_idx < 45:
            weight = "60g前後"
            reason += f"● 安定性指数（{stability_idx}）が低いため、重量を増やして軌道を物理的に安定化"
    else:
        band = infer_hs_band(power_idx)
        weight = {"low": "40〜50g", "mid": "50〜60g", "high": "60〜70g"}[band]
        reason = f"● パワー指数（{power_idx}）に基づく推奨重量"

    rows.append({"item": "重量", "guide": weight, "reason": reason})

    # --- 2. フレックス（HS × 捻転パワー） ---
    if hs is not None:
        flex_map = [(33, "L〜A"), (38, "A〜R"), (42, "R〜SR"), (46, "SR〜S"), (50, "S〜X")]
        flex = next((f for h, f in flex_map if hs < h), "X")
        reason = f"● HS {hs:.1f}m/s に対してしなり戻りが適正な硬さ\n"
        if power_idx > 75:
            flex = "一ランク硬め"
            reason += f"● パワー指数（{power_idx}）が高く、シャフトへの負荷が強いため"
    else:
        flex = {"low": "A〜R", "mid": "R〜SR", "high": "SR〜S"}[infer_hs_band(power_idx)]
        reason = f"● パワー指数（{power_idx}）に対する適正剛性"

    rows.append({"item": "フレックス", "guide": flex, "reason": reason})

    # --- 3. キックポイント（ミス傾向 × 手首タメ解析：逆転ロジック） ---
    if miss == "right":
        kp, base_reason = "先〜中", "● 右ミスに対し、つかまりを助ける先調子系を基準"
    elif miss == "left":
        kp, base_reason = "中〜元", "● 左ミスに対し、先端の動きを抑えた元調子系を基準"
    else:
        kp, base_reason = "中", "● ニュートラルな挙動の中調子を基準"

    # build_paid_09 内のキックポイント判定箇所
    # 右ミス ＋ タメが浅い（30度未満 ＝ 内角150度以上）場合に逆転発想を発動
    if miss == "right" and wrist_cock < 30:
        kp = "元"
        reason = (f"● 右ミス傾向かつ、手首のタメ（{wrist_cock:.1f}°）が非常に浅い状態です\n"
                  "● アーリーリリースによる振り遅れがスライスの主因と判定\n"
                  "● あえて手元がしなる『元調子』を使用し、強制的にタメを維持させインパクトの厚みを作ります")
    elif miss == "right" and stability_idx < 35:
        kp = "中"
        reason = f"● 右ミスがあるが、安定性指数（{stability_idx}）が低く軸が不安定\n● 先端を走らせるより、挙動が安定する『中調子』で打点を整えるのが優先"
    else:
        reason = base_reason

    rows.append({"item": "キックポイント", "guide": kp, "reason": reason})

    # --- 4. トルク（安定性解析 × ミス補正：矛盾解消版） ---
    if stability_idx <= 40:
        tq, base_reason = "3.0〜4.0", f"● 安定性指数（{stability_idx}）が低いため、低トルクでねじれを抑制"
    elif stability_idx <= 70:
        tq, base_reason = "3.5〜5.0", f"● 安定性指数（{stability_idx}）に基づき、標準帯のねじれ量でバランスを確保"
    else:
        tq, base_reason = "4.5〜6.0", f"● 安定性指数（{stability_idx}）が高く、高トルクでも再現性が維持可能"

    # ミス傾向による微調整
    if miss == "right":
        # 右ミスにはトルクを「増やす（大きい数値にする）」ことでつかまりを良くする
        tq = "4.5〜5.5" if stability_idx <= 70 else "5.5以上"
        reason = base_reason + "\n● 右ミス補正：トルクを増やしてフェースの返り（つかまり）をサポート"
    elif miss == "left":
        # 左ミスにはトルクを「減らす」ことでつかまりを抑える
        tq = "2.5〜3.5"
        reason = base_reason + "\n● 左ミス補正：トルクを絞り、つかまり過ぎ（引っ掛け）を抑制"
    else:
        reason = base_reason

    rows.append({"item": "トルク", "guide": tq, "reason": reason})

    return {
        "title": "09. Shaft Fitting Guide（推奨）",
        "table": rows,
        "note": "※本結果は解析数値に基づく指標です。購入時は試打での最終確認を推奨します。",
        "meta": {
            "power_idx": power_idx,
            "stability_idx": stability_idx,
            "wrist_cock": wrist_cock,
            "head_speed": hs
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
    # --- 07. 総合評価（型と優先課題）の抽出 ---
    # analysis辞書内の 07 セクション、またはそれに相当するキーから取得
    sec07 = analysis.get("07") or {}
    meta07 = sec07.get("meta") or {}
    swing_type = meta07.get("swing_type", "バランス型")
    priorities = meta07.get("priorities", [])

    # --- 08. 練習ドリルの抽出 ---
    sec08 = analysis.get("08") or {}
    drills = sec08.get("drills", [])
    drill_names = [d["name"] for d in drills]

    # --- 09. フィッティングの抽出 ---
    sec09 = analysis.get("09") or {}
    table = sec09.get("table", [])
    meta09 = sec09.get("meta") or {}
    
    # キックポイントの推奨情報と選定理由(AI逆転判定の根拠)を取得
    kp_info = next((item for item in table if item["item"] == "キックポイント"), {})
    kp_guide = kp_info.get("guide", "中")
    kp_reason = kp_info.get("reason", "")

    # --- 文章の組み立て（ストーリー構築） ---
    summary_text = []

    # 1. スイング型の総評
    summary_text.append(f"今回の解析結果、あなたのスイングは『{swing_type}』に分類されます。")
    
    # 2. 優先課題とアクションプランの連動
    if priorities:
        p_str = "／".join(priorities)
        summary_text.append(f"現在、スコアアップのために最も優先すべきテーマは『{p_str}』の改善です。")
        
        if drill_names:
            summary_text.append(f"この課題を克服するために、まずは推奨ドリル筆頭の「{drill_names[0]}」に集中して取り組んでください。")
            summary_text.append("複数の動きを同時に直すよりも、この一点を整えることで他の数値も連鎖的に向上します。")
    else:
        summary_text.append("全体的に大きな破綻はなく、非常にバランスの良いスイングです。提示されたドリルでさらなる再現性の向上を目指しましょう。")

    summary_text.append("")  # 視認性のための改行

    # 3. フィッティングとスイングの相関（09の逆転ロジックを尊重）
    summary_text.append(f"道具の面では、AIの解析数値に基づき『{kp_guide}調子』のシャフトを提案しました。")
    if kp_reason:
        # 09で生成された「理由」には、スライス傾向と解析数値の矛盾などが含まれているため、そのまま引用
        summary_text.append(f"【選定根拠】{kp_reason}")

    summary_text.append("")  # 視認性のための改行

    # 4. 結びの言葉（動的メッセージ）
    summary_text.append("『練習による動作の最適化』と『シャフトによる挙動の補正』。")
    summary_text.append("この両輪を回すことが、目標達成への最短距離となります。")
    summary_text.append("次回の解析で、各数値がどのように進化しているかを楽しみにしています！")

    summary_text.append("")  # 視認性のための改行

    # 5. 共通メッセージ（全ての利用者共通）
    summary_text.append("あなたのゴルフライフが、より充実したものになることを願っています。")

    return {
        "title": "10. Summary（まとめ）",
        "text": summary_text,
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

    # 07は「解析結果のまとめ(analysis)」と「生データ(raw)」の両方を使用
    analysis["07"] = build_paid_07_from_analysis(analysis, raw)

    # ✅ 修正箇所1：build_paid_08 は (analysis, raw) の2つが必要です
    analysis["08"] = build_paid_08(analysis, raw)

    # ✅ 09は常にセクションを出す（入力があれば本体、無ければ案内のみ）
    ui = user_inputs or {}
    if ui.get("head_speed") is not None or ui.get("miss_tendency") or ui.get("gender"):
        analysis["09"] = build_paid_09(raw, ui)
    else:
        analysis["09"] = build_paid_09_placeholder()

    # ✅ 修正箇所2：build_paid_10 は 01〜09の結果をまとめるため (analysis) を渡します
    # ※ raw を渡すと、まとめロジック内でデータが参照できずエラーになります
    analysis["10"] = build_paid_10(analysis)

    return analysis

# ==================================================
# Routes
# ==================================================
# ===== Firestore 接続確認用（一時テスト）=====
@app.route("/debug/create_user", methods=["GET"])
def debug_create_user():
    test_user_id = "U_DEBUG_CREATE_001"

    db.collection("users").document(test_user_id).set({
        "plan": "free",
        "ticket_remaining": 0,
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }, merge=True)

    return "users created", 200

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
        "status": status_out,                 # PROCESSING / COMPLETED
        "is_premium": bool(r.get("is_premium", False)),
        "analysis": r.get("analysis") or {},
    })
    
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

        # 動画DL → 解析
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, f"{report_id}.mp4")
            content = line_bot_api.get_message_content(message_id)
            with open(video_path, "wb") as f:
                for chunk in content.iter_content():
                    f.write(chunk)

            raw = analyze_swing_with_mediapipe(video_path)

        analysis = build_analysis(raw=raw, premium=premium, report_id=report_id, user_inputs=user_inputs)

        report_ref.set({
            "status": "DONE",
            "raw": raw,
            "analysis": analysis,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }, merge=True)

        # 完了通知
        safe_line_push(user_id, make_done_push(report_id), force=True)

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
    import os, traceback
    from flask import request
    import stripe
    from google.cloud import firestore

    endpoint_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "") # ★追加
    db = firestore.Client() # ★追加
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")
    print(f"[BOOT] webhook_secret_prefix={endpoint_secret[:10]} len={len(endpoint_secret)}", flush=True)


    # 1) 署名検証（ここが通らないとFirestoreは絶対更新されない）
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except stripe.error.SignatureVerificationError as e:
        print(f"⚠️ Stripe署名検証に失敗しました: {e}")
        return "Invalid signature", 400
    except Exception as e:
        print(f"⚠️ Stripe webhook error: {e}")
        return "Error", 400

    # 2) 必要なイベント以外は即OK
    if event.get("type") != "checkout.session.completed":
        return "OK", 200

    session = event["data"]["object"]
    event_id = event.get("id")
    session_id = session.get("id")
    line_user_id = session.get("client_reference_id")

    print(f"[STRIPE] livemode={event.get('livemode')} event_id={event_id} session_id={session_id} client_reference_id={line_user_id}")

    if not line_user_id:
        print("❌ client_reference_id missing")
        return "OK", 200

    try:
        # 3) price_idを確実に取る（expandより堅い）
        li = stripe.checkout.Session.list_line_items(session_id, limit=1)
        first = li["data"][0] if li and li.get("data") else None
        price_id = first.get("price", {}).get("id") if first else None
        print(f"[STRIPE] price_id={price_id}", flush=True)

        # 4) 付与数（回数券だけ +5 / それ以外 +1）
        add_tickets = 1
        if price_id == "price_1SrGGcK85rGl4ns4FpiYMXtt":
            add_tickets = 5

        user_ref = db.collection("users").document(line_user_id)

        # 5) 冪等（Stripe再送でも二重加算しない）
        before = user_ref.get().to_dict() or {}
        print(f"[BEFORE] ticket_remaining={before.get('ticket_remaining')} last_stripe_event_id={before.get('last_stripe_event_id')}")

        if before.get("last_stripe_event_id") == event_id:
            print("✅ duplicate event ignored")
            return "OK", 200

        # 6) Firestore更新（ここで必ず増える）
        user_ref.set({
            "plan": "ticket" if add_tickets > 1 else "single",
            "ticket_remaining": firestore.Increment(add_tickets),
            "last_payment_date": firestore.SERVER_TIMESTAMP,
            "last_stripe_event_id": event_id,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        after = user_ref.get().to_dict() or {}
        print(f"[AFTER] ticket_remaining={after.get('ticket_remaining')} plan={after.get('plan')}")
        print(f"✅ Firestore updated user={line_user_id} add={add_tickets}", flush=True)


    except Exception:
        print("❌ post-payment handler failed:", traceback.format_exc(), flush=True)
        return "OK", 500

    return "OK", 200




@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event: MessageEvent):
    user_id = event.source.user_id
    msg = event.message
    report_id = f"{user_id}_{msg.id}"

    import logging
    logging.warning(
        "[DEBUG] handle_video HIT user_id=%s message_id=%s",
        user_id,
        msg.id
    )

    # ===== users 取得 =====
    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()
    user_data = user_doc.to_dict() if user_doc.exists else {}
    tickets = user_data.get('ticket_remaining', 0)

    # ===== prefill → user_inputs に確定コピー =====
    prefill = user_data.get("prefill") or {}
    user_inputs = {
        "head_speed": prefill.get("head_speed"),
        "miss_tendency": prefill.get("miss_tendency"),
        "gender": prefill.get("gender"),
    }
    user_inputs = {k: v for k, v in user_inputs.items() if v is not None}

    logging.warning("[DEBUG] user_inputs=%r", user_inputs)

    # ===== 有料判定 =====
    force_paid_report = is_premium_user(user_id) or tickets > 0

    # ===== report 作成 =====
    firestore_safe_set(report_id, {
        "user_id": user_id,
        "status": "PROCESSING",
        "is_premium": force_paid_report,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_inputs": user_inputs,  # ★ ここが唯一の入力ソース
    })

    try:
        # ===== 解析タスク作成 =====
        task_name = create_cloud_task(report_id, user_id, msg.id)
        firestore_safe_update(report_id, {"task_name": task_name})

        # ===== チケット / 無料回数消費 =====
        if not is_premium_user(user_id) and tickets > 0:
            user_ref.update({'ticket_remaining': firestore.Increment(-1)})

        if not force_paid_report:
            increment_free_usage(user_id)

        # ===== 初期返信（URLのみ）=====
        reply_text = make_initial_reply(report_id)
        safe_line_reply(event.reply_token, reply_text, user_id=user_id)

    except Exception:
        logging.exception("[ERROR] handle_video failed")
        firestore_safe_update(report_id, {
            "status": "TASK_FAILED",
            "error": traceback.format_exc()
        })
        safe_line_reply(
            event.reply_token,
            "動画は受け取りましたが、解析の予約に失敗しました。時間を置いて再度お試しください。",
            user_id=user_id
        )



@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id

    import logging
    from google.cloud import firestore

    # ===== 正規化（全角スペース & 全角数字）=====
    raw_text = event.message.text or ""
    text = raw_text.replace("\u3000", " ").strip()
    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    logging.warning("[DEBUG] raw=%r normalized=%r user_id=%s", raw_text, text, user_id)

    # ===== 1) 分析スタート → Quick Reply =====
    if text == "分析スタート":
        users_ref.document(user_id).set({
            "prefill_step": "head_speed",
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        msg_text = (
            "ご利用ありがとうございます。\n\n"
            "より正確なフィッティング分析レポート（09）をご希望の方は、分かる範囲で入力をお願いします。\n\n"
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

   # ===== 5) 最後に既存分岐（料金プランなど） =====
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
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
