import os
import re
import io
import json
import time
import math
import shutil
import tempfile
import traceback
from typing import Any, Dict, Optional, Tuple

import ffmpeg
import numpy as np
import cv2
import mediapipe as mp

from flask import Flask, request, abort, jsonify

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, VideoMessage, TextSendMessage

from google.cloud import tasks_v2
from google.api_core.exceptions import NotFound, PermissionDenied
from google.cloud import firestore as gcfirestore

import firebase_admin
from firebase_admin import credentials, firestore as fbfirestore, initialize_app

from google import genai
from google.genai import errors as genai_errors


# ==================================================
# ENV
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")

TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")  # Osaka default
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")

# Premium behavior
FORCE_PREMIUM = os.environ.get("FORCE_PREMIUM", "true").lower() in ("1", "true", "yes", "on")

# Gemini model preferences (fallback list)
GEMINI_MODEL_ENV = os.environ.get("GEMINI_MODEL", "").strip()

# Analysis tuning
MAX_SECONDS_FOR_ANALYSIS = int(os.environ.get("MAX_SECONDS_FOR_ANALYSIS", "30"))
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "2"))  # analyze every N frames


# ==================================================
# App init
# ==================================================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


# LINE
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None


# Firestore (Firebase Admin recommended on Cloud Run)
db = None
try:
    if not firebase_admin._apps:
        cred = credentials.ApplicationDefault()
        initialize_app(cred, {"projectId": GCP_PROJECT_ID or None})
    db = fbfirestore.client()
except Exception as e:
    print("[Firestore] init failed:", e)
    db = None


# Cloud Tasks
tasks_client = None
queue_path = None
try:
    if GCP_PROJECT_ID:
        tasks_client = tasks_v2.CloudTasksClient()
        queue_path = tasks_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)
except Exception as e:
    print("[CloudTasks] init failed:", e)
    tasks_client = None
    queue_path = None


# ==================================================
# Utilities
# ==================================================
def now_ts() -> float:
    return time.time()


def safe_print_exc(prefix: str = "") -> None:
    print(prefix)
    print(traceback.format_exc())


def firestore_safe_set(report_id: str, data: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("reports").document(report_id).set(data, merge=True)
    except Exception:
        safe_print_exc("[Firestore] set failed")


def firestore_safe_update(report_id: str, patch: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("reports").document(report_id).update(patch)
    except Exception:
        safe_print_exc("[Firestore] update failed")


def firestore_get(doc_path: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    if not db:
        return None
    try:
        col, doc_id = doc_path
        doc = db.collection(col).document(doc_id).get()
        if doc.exists:
            return doc.to_dict() or {}
        return None
    except Exception:
        safe_print_exc("[Firestore] get failed")
        return None


def safe_line_reply(reply_token: str, text: str) -> None:
    if not line_bot_api:
        return
    try:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text))
    except LineBotApiError:
        safe_print_exc("[LINE] reply failed")


def safe_line_push(user_id: str, text: str) -> None:
    if not line_bot_api:
        return
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text=text))
    except LineBotApiError:
        safe_print_exc("[LINE] push failed")


def make_initial_reply(report_id: str, plan_label: str) -> str:
    report_url = f"{SERVICE_HOST_URL}/report/{report_id}" if SERVICE_HOST_URL else f"/report/{report_id}"
    return (
        "✅ 動画を受信しました。解析を開始します！\n"
        f"（モード：{plan_label}）\n\n"
        "AIによるスイング診断には数分かかります。\n"
        "【処理状況確認URL】\n"
        f"{report_url}\n\n"
        "【料金プラン】\n"
        "・都度契約：500円／1回\n"
        "・回数券　：1,980円／5回券\n"
        "・月額契約：4,980円／月"
    )


def make_done_push(report_id: str, is_premium: bool) -> str:
    report_url = f"{SERVICE_HOST_URL}/report/{report_id}" if SERVICE_HOST_URL else f"/report/{report_id}"
    if is_premium:
        return (
            "🎉 AIスイング診断が完了しました！\n\n"
            "【診断レポートURL】\n"
            f"{report_url}\n\n"
            "詳細なレポートはURLからご確認ください。次の練習にお役立てください！"
        )
    return (
        "✅ 無料版AIスイング診断が完了しました。\n\n"
        "【簡易レポートURL】\n"
        f"{report_url}\n\n"
        "骨格データ（01）と総合診断（07）をご確認いただけます。"
    )


# ==================================================
# User input capture (optional)
#   Users may send text like:
#     HS:45 ミス:スライス 性別:男 番手:DR
#   Store as "pending_profile" for next video.
# ==================================================
PROFILE_REGEX = re.compile(
    r"""
    (?:
        (?:HS|ヘッドスピード)\s*[:：]\s*(?P<hs>\d+(?:\.\d+)?) |
        (?:ミス|miss)\s*[:：]\s*(?P<miss>[^ \n\r\t]+) |
        (?:性別|gender)\s*[:：]\s*(?P<gender>男|女|男性|女性|m|f|M|F) |
        (?:番手|club)\s*[:：]\s*(?P<club>DR|D|ドライバー|FW|UT|IRON|アイアン|WEDGE|ウェッジ|P|SW|AW)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_profile_text(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in PROFILE_REGEX.finditer(text or ""):
        if m.group("hs"):
            try:
                out["head_speed"] = float(m.group("hs"))
            except Exception:
                pass
        if m.group("miss"):
            out["miss_tendency"] = m.group("miss").strip()
        if m.group("gender"):
            g = m.group("gender").strip().lower()
            if g in ("男", "男性", "m"):
                out["gender"] = "male"
            elif g in ("女", "女性", "f"):
                out["gender"] = "female"
        if m.group("club"):
            c = m.group("club").strip().upper()
            if c in ("D", "DR", "ドライバー"):
                out["club"] = "DR"
            elif c in ("FW",):
                out["club"] = "FW"
            elif c in ("UT",):
                out["club"] = "UT"
            elif c in ("IRON", "アイアン"):
                out["club"] = "IRON"
            elif c in ("WEDGE", "ウェッジ", "SW", "AW", "P"):
                out["club"] = "WEDGE"
    return out


def set_pending_profile(user_id: str, profile: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("users").document(user_id).set(
            {"pending_profile": profile, "pending_profile_updated_at": fbfirestore.SERVER_TIMESTAMP},
            merge=True,
        )
    except Exception:
        safe_print_exc("[Firestore] set pending_profile failed")


def pop_pending_profile(user_id: str) -> Dict[str, Any]:
    if not db:
        return {}
    try:
        ref = db.collection("users").document(user_id)
        doc = ref.get()
        if not doc.exists:
            return {}
        data = doc.to_dict() or {}
        prof = data.get("pending_profile") or {}
        # clear after use
        ref.set({"pending_profile": fbfirestore.DELETE_FIELD}, merge=True)
        return prof if isinstance(prof, dict) else {}
    except Exception:
        safe_print_exc("[Firestore] pop pending_profile failed")
        return {}


# ==================================================
# Cloud Tasks enqueue with OIDC (required for Cloud Run auth)
# ==================================================
def create_cloud_task(report_id: str, user_id: str, message_id: str) -> str:
    if not tasks_client or not queue_path:
        raise RuntimeError("Cloud Tasks client is not initialized.")
    if not SERVICE_HOST_URL:
        raise RuntimeError("SERVICE_HOST_URL is missing.")
    if not TASK_SA_EMAIL:
        raise RuntimeError("TASK_SA_EMAIL is missing.")

    payload = json.dumps({"report_id": report_id, "user_id": user_id, "message_id": message_id}).encode("utf-8")

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{SERVICE_HOST_URL}/worker/process_video",
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
# Video handling
# ==================================================
def download_line_video_to_file(message_id: str, dst_path: str) -> None:
    if not line_bot_api:
        raise RuntimeError("LINE bot not initialized.")
    content = line_bot_api.get_message_content(message_id)
    with open(dst_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)


def transcode_video(input_path: str, output_path: str) -> None:
    """
    Normalize to mp4/h264/aac, resize, limit duration if needed.
    """
    # Use ffmpeg-python
    (
        ffmpeg
        .input(input_path)
        .output(
            output_path,
            vcodec="libx264",
            acodec="aac",
            movflags="+faststart",
            vf="scale='min(720,iw)':-2",
            preset="veryfast",
            crf=28,
            **{"t": MAX_SECONDS_FOR_ANALYSIS},
        )
        .overwrite_output()
        .run(quiet=True)
    )


# ==================================================
# MediaPipe analysis (practical, robust)
#   We compute simple proxy metrics:
#   - frame_count: total frames read
#   - max_head_drift_x: max normalized horizontal drift of nose vs starting
#   - max_knee_sway_x: max normalized horizontal drift of knee midpoint vs starting
#   - max_wrist_cock: angle at lead wrist (shoulder-elbow-wrist) proxy
#   - max_shoulder_rotation: shoulder line angle change (2D) vs address
#   - min_hip_rotation: hip line angle change (2D) vs address
#
# NOTE:
#   This is not "perfect biomechanics", but stable and consistent for service v1.
# ==================================================
def angle_deg(p1, p2, p3) -> float:
    a = np.array([p1[0], p1[1]])
    b = np.array([p2[0], p2[1]])
    c = np.array([p3[0], p3[1]])
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cos = float(np.dot(v1, v2) / denom)
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def safe_get_landmark_xy(lms, idx: int) -> Optional[Tuple[float, float]]:
    try:
        lm = lms[idx]
        return (float(lm.x), float(lm.y))
    except Exception:
        return None


def line_angle_deg(p_left: Tuple[float, float], p_right: Tuple[float, float]) -> float:
    dx = p_right[0] - p_left[0]
    dy = p_right[1] - p_left[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def analyze_swing(video_path: str) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        return {"error": "video file not found"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "failed to open video"}

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_count = 0
    stride_count = 0

    # Baselines
    start_nose_x = None
    start_knee_mid_x = None
    start_shoulder_angle = None
    start_hip_angle = None

    max_head_drift_x = 0.0
    max_knee_sway_x = 0.0
    max_wrist_cock = 0.0
    max_shoulder_rotation = -999.0
    min_hip_rotation = 999.0

    # mediapipe indices
    NOSE = 0
    L_SHOULDER = 11
    R_SHOULDER = 12
    L_HIP = 23
    R_HIP = 24
    L_ELBOW = 13
    R_ELBOW = 14
    L_WRIST = 15
    R_WRIST = 16
    L_KNEE = 25
    R_KNEE = 26

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            stride_count += 1
            if stride_count % max(1, FRAME_STRIDE) != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            lms = res.pose_landmarks.landmark

            nose = safe_get_landmark_xy(lms, NOSE)
            ls = safe_get_landmark_xy(lms, L_SHOULDER)
            rs = safe_get_landmark_xy(lms, R_SHOULDER)
            lh = safe_get_landmark_xy(lms, L_HIP)
            rh = safe_get_landmark_xy(lms, R_HIP)
            lk = safe_get_landmark_xy(lms, L_KNEE)
            rk = safe_get_landmark_xy(lms, R_KNEE)

            le = safe_get_landmark_xy(lms, L_ELBOW)
            lw = safe_get_landmark_xy(lms, L_WRIST)
            re_ = safe_get_landmark_xy(lms, R_ELBOW)
            rw = safe_get_landmark_xy(lms, R_WRIST)

            if nose and start_nose_x is None:
                start_nose_x = nose[0]
            if lk and rk and start_knee_mid_x is None:
                start_knee_mid_x = (lk[0] + rk[0]) / 2.0

            # drift
            if nose and start_nose_x is not None:
                max_head_drift_x = max(max_head_drift_x, abs(nose[0] - start_nose_x))
            if lk and rk and start_knee_mid_x is not None:
                knee_mid_x = (lk[0] + rk[0]) / 2.0
                max_knee_sway_x = max(max_knee_sway_x, abs(knee_mid_x - start_knee_mid_x))

            # shoulder / hip angles
            if ls and rs:
                ang = line_angle_deg(ls, rs)
                if start_shoulder_angle is None:
                    start_shoulder_angle = ang
                rot = ang - start_shoulder_angle
                max_shoulder_rotation = max(max_shoulder_rotation, rot)
            if lh and rh:
                ang = line_angle_deg(lh, rh)
                if start_hip_angle is None:
                    start_hip_angle = ang
                rot = ang - start_hip_angle
                min_hip_rotation = min(min_hip_rotation, rot)

            # wrist cock proxy (use lead arm: left for right-handed majority; still works as proxy)
            if ls and le and lw:
                wc = angle_deg(ls, le, lw)  # shoulder-elbow-wrist
                max_wrist_cock = max(max_wrist_cock, wc)

        # sanity defaults
        if max_shoulder_rotation == -999.0:
            max_shoulder_rotation = 0.0
        if min_hip_rotation == 999.0:
            min_hip_rotation = 0.0

        return {
            "frame_count": int(frame_count),
            "max_shoulder_rotation": float(round(max_shoulder_rotation, 1)),
            "min_hip_rotation": float(round(min_hip_rotation, 1)),
            "max_wrist_cock": float(round(max_wrist_cock, 1)),
            "max_head_drift_x": float(round(max_head_drift_x, 4)),
            "max_knee_sway_x": float(round(max_knee_sway_x, 4)),
        }

    except Exception:
        safe_print_exc("[MediaPipe] analysis failed")
        return {"error": "mediapipe analysis failed"}

    finally:
        cap.release()
        try:
            pose.close()
        except Exception:
            pass


# ==================================================
# Gemini helpers: structured JSON output
#   AI writes only the content parts, we keep layout fixed.
# ==================================================
def choose_gemini_models() -> Tuple[str, ...]:
    if GEMINI_MODEL_ENV:
        return (GEMINI_MODEL_ENV,)
    return (
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash",
        "models/gemini-2.0-flash",
    )


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Gemini may wrap in ```json. Extract first {...} block safely.
    """
    if not text:
        return None
    t = text.strip()
    # remove fences
    t = re.sub(r"```(?:json)?", "", t, flags=re.IGNORECASE).replace("```", "")
    # find first JSON object
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def run_gemini_sections(raw_data: Dict[str, Any], is_premium: bool) -> Dict[str, Any]:
    """
    Returns dict with:
      sec02_bullets, sec02_pro,
      sec03_bullets, sec03_pro,
      sec04_bullets, sec04_pro,
      sec05_bullets, sec05_pro,
      sec06_bullets, sec06_pro,
      sec07_good, sec07_improve,
      sec10_text
    """
    # fallback templates in case Gemini fails
    fallback = {
        "sec02_bullets": [
            "頭の左右移動が小さく、スイング中の軸が比較的保たれていると評価できます。",
            "上体が突っ込みにくく、インパクトの再現性を作りやすい土台があります。",
            "この安定性を活かすことで、回転量の改善がそのまま成果につながりやすい状態です。",
        ],
        "sec02_pro": "プロ目線では「まず直す点」ではなく「活かす土台」と判断します。ここが安定していると、他の改善が速く形になります。",
        "sec03_bullets": [
            "上半身の捻転量が不足しており、バックスイングでエネルギーを溜めきれていないと評価できます。",
            "体幹より腕主導になりやすく、飛距離と再現性の両面でロスが出やすい状態です。",
            "切り返し以降で手先の補正が増え、タイミングが日によって変わりやすくなります。",
        ],
        "sec03_pro": "肩回旋が増えるだけでスイング効率が一段上がるタイプです。安定性があるので、回せるようになると伸び方が大きいと見ます。",
        "sec04_bullets": [
            "下半身の回旋が上半身と噛み合いにくく、捻転差が作りにくい状態と評価できます。",
            "腰が先に動きやすいと、クラブの下りる位置が不安定になりやすくなります。",
            "上半身の回転が改善すると、腰の動きも整理されやすい傾向です。",
        ],
        "sec04_pro": "腰単体を直すより、肩回旋と連動の作り直しが優先です。順番を間違えないことが重要です。",
        "sec05_bullets": [
            "手首の動きが大きくなりやすく、リリースのタイミングがぶれやすい状態と評価できます。",
            "体幹の回転不足を手首で補うと、ミスの再現性が上がりにくくなります。",
            "回転量が整うほど、手首の動きは自然に適正化しやすいタイプです。",
        ],
        "sec05_pro": "手首を抑え込むより「体で振れる条件」を作る方が改善が速いと判断します。",
        "sec06_bullets": [
            "下半身の左右ブレが小さく、土台が安定していると評価できます。",
            "切り返しでバランスを崩しにくく、再現性を積み上げやすい状態です。",
            "上半身の改善が進むと、安定性がそのままショットの安定につながりやすいです。",
        ],
        "sec06_pro": "下半身が安定している人は、上半身の回転改善の成果が出やすいです。伸び代が大きい部類です。",
        "sec07_good": [
            "頭と下半身のブレが少なく、スイングの土台が安定しています。",
            "安定性があるため、改善を入れたときに結果へ反映されやすい状態です。",
        ],
        "sec07_improve": [
            "上半身の捻転量が不足しており、体幹のパワー伝達が弱くなっています。",
            "その影響で手首の動きが大きくなり、タイミングのズレが生じやすい状態です。",
        ],
        "sec10_text": (
            "今回のスイングは、頭と下半身の安定性という強い土台を持っています。これは再現性を高めるうえで大きな武器です。\n\n"
            "一方で、肩の回旋量が不足していることで、体幹を使ったパワー生成が十分に行われておらず、飛距離と安定性の両面でロスが生じています。"
            "その不足分を手首の動きで補う形になりやすく、日によってタイミングが変わりやすい傾向も見られます。\n\n"
            "まずは上半身の回転量を増やし、回転主導でクラブが動く条件を作ることが最優先です。"
            "土台が安定しているため、改善が進むほど成果が出やすいタイプです。"
            "定期的に計測し、数値の変化と感覚をセットで確認しながら進めていきましょう。\n\n"
            "お客様のゴルフライフが充実したものになることを切に願っています。"
        ),
    }

    if not GEMINI_API_KEY:
        return fallback

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Ask for strict JSON
    prompt = (
        "あなたは世界トップクラスのゴルフスイングコーチです。"
        "以下の骨格計測データに基づき、各セクションの『評価文』を作成してください。"
        "重要: 一般論ではなく『今回の数値からこのスイングをどう評価するか』だけを書いてください。"
        "出力は必ず JSON のみ。日本語。トーンは普通。箇条書きは短文。\n\n"
        "【出力JSONスキーマ】\n"
        "{\n"
        '  "sec02_bullets": ["...","...","..."],\n'
        '  "sec02_pro": "...",\n'
        '  "sec03_bullets": ["...","...","..."],\n'
        '  "sec03_pro": "...",\n'
        '  "sec04_bullets": ["...","...","..."],\n'
        '  "sec04_pro": "...",\n'
        '  "sec05_bullets": ["...","...","..."],\n'
        '  "sec05_pro": "...",\n'
        '  "sec06_bullets": ["...","...","..."],\n'
        '  "sec06_pro": "...",\n'
        '  "sec07_good": ["...","..."],\n'
        '  "sec07_improve": ["...","..."],\n'
        '  "sec10_text": "...."\n'
        "}\n\n"
        "【制約】\n"
        "- sec02〜06: bulletsは各3個。proは2〜3文。\n"
        "- sec07_good / sec07_improve は各2個。\n"
        "- sec10_text は4〜8段落相当で、最後は『お客様のゴルフライフが充実したものになることを切に願っています。』で締める。\n\n"
        f"【骨格計測データ】\n{json.dumps(raw_data, ensure_ascii=False, indent=2)}\n"
    )

    last_err = None
    for model in choose_gemini_models():
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            text = getattr(resp, "text", "") or ""
            data = extract_json_object(text)
            if not isinstance(data, dict):
                raise RuntimeError("Gemini output is not JSON")
            # minimal validation / fill missing with fallback
            for k, v in fallback.items():
                if k not in data or not data[k]:
                    data[k] = v
            return data
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            last_err = e
            print("[Gemini] model failed:", model, str(e))
            continue
        except Exception as e:
            last_err = e
            print("[Gemini] unexpected:", model, str(e))
            continue

    print("[Gemini] fallback due to error:", last_err)
    return fallback


# ==================================================
# 09 fitting rules (driver only, premium only)
#   Based on:
#     - head_speed (optional)
#     - raw_data tendencies
# ==================================================
def fit_weight(head_speed: Optional[float]) -> str:
    if head_speed is None:
        return "50g台（目安）"
    hs = head_speed
    if hs < 32:
        return "40g台〜50g台前半"
    if hs < 38:
        return "50g台前半〜中盤"
    if hs < 45:
        return "50g台後半〜60g台"
    return "60g台〜70g台"


def fit_flex(head_speed: Optional[float]) -> str:
    if head_speed is None:
        return "R〜SR（目安）"
    hs = head_speed
    if hs < 32:
        return "L / A"
    if hs < 38:
        return "R"
    if hs < 45:
        return "SR"
    return "S〜X"


def fit_torque(head_speed: Optional[float]) -> str:
    if head_speed is None:
        return "3.8〜4.8（目安）"
    hs = head_speed
    if hs < 32:
        return "5.0〜6.5"
    if hs < 38:
        return "4.5〜5.5"
    if hs < 45:
        return "3.8〜4.8"
    return "3.0〜4.2"


def fit_kick(raw_data: Dict[str, Any]) -> str:
    # simple tendency-based choice:
    shoulder = float(raw_data.get("max_shoulder_rotation", 0.0) or 0.0)
    wrist = float(raw_data.get("max_wrist_cock", 0.0) or 0.0)
    # If shoulder turn low, help launch/feel head: mid or mid-high
    if shoulder < 20:
        return "中調子"
    # If wrist motion huge (tendency to timing issue), stabilize: middle or butt
    if wrist > 140:
        return "元調子"
    return "中調子"


def fitting_table(raw_data: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, str]:
    hs = profile.get("head_speed")
    try:
        hs = float(hs) if hs is not None else None
    except Exception:
        hs = None

    return {
        "シャフト重量": fit_weight(hs),
        "フレックス": fit_flex(hs),
        "キックポイント": fit_kick(raw_data),
        "トルク": fit_torque(hs),
    }


# ==================================================
# Report assembly (STRUCTURE FIXED)
#   Free: 01 + 07 only
#   Premium: 01-10 full (02-06 bullets + pro, 08/09 tables)
# ==================================================
IDEALS = {
    "frame_count": "60フレーム以上",
    "max_shoulder_rotation": "約80°〜100°",
    "min_hip_rotation": "約35°〜45°",
    "max_wrist_cock": "約90°〜120°",
    "max_head_drift_x": "0.05以下（小さいほど安定）",
    "max_knee_sway_x": "0.05以下（小さいほど安定）",
}

MEANINGS = {
    "frame_count": "スイング全体を通した分析の粒度です。十分なフレーム数があるほど傾向が安定して見えます。",
    "max_shoulder_rotation": "上半身の捻転量の目安です。この数値が大きいほど体幹を使った効率的なスイングになりやすいとされます。",
    "min_hip_rotation": "腰の回旋量の目安です。上半身との捻転差（Xファクター）を作る重要要素です。",
    "max_wrist_cock": "手首のコック量の目安です。適正範囲で保てるとヘッドスピード向上に繋がりやすいです。",
    "max_head_drift_x": "頭の左右ブレの目安です。小さいほど軸が安定し、再現性の高いインパクトに繋がりやすいです。",
    "max_knee_sway_x": "膝（下半身）の左右ブレの目安です。小さいほど土台が安定し、ショットが安定しやすいです。",
}


def fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        # keep as-is; caller may append unit
        return str(v)
    return str(v)


def build_markdown_report(
    raw: Dict[str, Any],
    sections: Dict[str, Any],
    is_premium: bool,
    profile: Dict[str, Any],
) -> str:
    # 01 table (always)
    lines = []
    lines.append("## 01. 骨格計測データ（AIが測った数値）\n")

    # Table + short explanation under each (as requested)
    lines.append("| 計測項目 | 測定値 | 理想の目安 |")
    lines.append("|---|---:|---|")

    lines.append(f"| 解析フレーム数 | {fmt(raw.get('frame_count'))} | {IDEALS['frame_count']} |")
    lines.append(f"| 最大肩回転 | {fmt(raw.get('max_shoulder_rotation'))}° | {IDEALS['max_shoulder_rotation']} |")
    lines.append(f"| 最小腰回転 | {fmt(raw.get('min_hip_rotation'))}° | {IDEALS['min_hip_rotation']} |")
    lines.append(f"| 最大コック角 | {fmt(raw.get('max_wrist_cock'))}° | {IDEALS['max_wrist_cock']} |")
    lines.append(f"| 最大頭ブレ（Sway） | {fmt(raw.get('max_head_drift_x'))} | {IDEALS['max_head_drift_x']} |")
    lines.append(f"| 最大膝ブレ（Sway） | {fmt(raw.get('max_knee_sway_x'))} | {IDEALS['max_knee_sway_x']} |")

    lines.append("\n### 各数値の見方（簡単な説明）\n")
    lines.append(f"**解析フレーム数**：{MEANINGS['frame_count']}")
    lines.append(f"\n**最大肩回転**：{MEANINGS['max_shoulder_rotation']}")
    lines.append(f"\n**最小腰回転**：{MEANINGS['min_hip_rotation']}")
    lines.append(f"\n**最大コック角**：{MEANINGS['max_wrist_cock']}")
    lines.append(f"\n**最大頭ブレ（Sway）**：{MEANINGS['max_head_drift_x']}")
    lines.append(f"\n**最大膝ブレ（Sway）**：{MEANINGS['max_knee_sway_x']}\n")

    # 02-06 (premium only)
    if is_premium:
        def sec(title: str, measure_label: str, measure_value: str, bullets_key: str, pro_key: str):
            lines.append(f"\n## {title}\n")
            lines.append(f"**測定値：{measure_label} {measure_value}**\n")
            for b in sections.get(bullets_key, []):
                lines.append(f"- {b}")
            lines.append("\n**プロ評価**")
            lines.append(f"{sections.get(pro_key, '')}\n")

        sec("02. 頭の安定性（軸のブレ）", "最大頭ブレ（Sway）", fmt(raw.get("max_head_drift_x")), "sec02_bullets", "sec02_pro")
        sec("03. 肩の回旋（上半身のねじり）", "最大肩回転", f"{fmt(raw.get('max_shoulder_rotation'))}°", "sec03_bullets", "sec03_pro")
        sec("04. 腰の回旋（下半身の動き）", "最小腰回転", f"{fmt(raw.get('min_hip_rotation'))}°", "sec04_bullets", "sec04_pro")
        sec("05. 手首のメカニクス（コック角）", "最大コック角", f"{fmt(raw.get('max_wrist_cock'))}°", "sec05_bullets", "sec05_pro")
        sec("06. 下半身の安定性（膝のブレ）", "最大膝ブレ（Sway）", fmt(raw.get("max_knee_sway_x")), "sec06_bullets", "sec06_pro")

    # 07 (always; requested: two items; bullets)
    lines.append("\n## 07. 総合診断\n")
    lines.append("### 安定している点")
    for b in sections.get("sec07_good", []):
        lines.append(f"- {b}")
    lines.append("\n### 改善が期待される点")
    for b in sections.get("sec07_improve", []):
        lines.append(f"- {b}")

    # 08 (premium only) - table with richer steps
    if is_premium:
        lines.append("\n## 08. 改善戦略とドリル\n")
        lines.append("| ドリル名 | 目的 | やり方 |")
        lines.append("|---|---|---|")
        lines.append("| クロスアームターン | 肩回旋量の向上 | ① 両腕を胸の前で軽くクロスする<br>② 下半身を固定したまま、胸をバックスイング方向へ回す<br>③ 肩ではなく「胸が回る感覚」を意識して左右交互に行う |")
        lines.append("| ウォールターン | 軸を保った回転習得 | ① お尻を壁に軽く触れさせてアドレス姿勢を作る<br>② お尻の位置を保ったまま上半身を回す<br>③ 壁から離れずに回れるかを確認する |")
        lines.append("| L to L スイング | 手首依存の軽減 | ① クラブを腰から腰までの振り幅で構える<br>② 体の回転でクラブを動かす意識で振る<br>③ 手首で操作せず、振り幅とリズムを一定に保つ |")

    # 09 (premium only) - driver only + notice line
    if is_premium:
        lines.append("\n## 09. スイング傾向補正型フィッティング（ドライバーのみ）\n")
        if (profile.get("club") and str(profile.get("club")).upper() != "DR"):
            lines.append("※本セクションはドライバー専用のため、番手がドライバー以外の場合は参考情報としてご確認ください。\n")

        ft = fitting_table(raw, profile)
        lines.append("| 項目 | 推奨 | 理由 |")
        lines.append("|---|---|---|")
        lines.append(f"| シャフト重量 | {ft['シャフト重量']} | スイングの安定性を損なわず、振り遅れ・手先補正を増やしにくい帯域を優先します。 |")
        lines.append(f"| フレックス | {ft['フレックス']} | 切り返しからインパクトまでのタイミングを整え、再現性を優先します。 |")
        lines.append(f"| キックポイント | {ft['キックポイント']} | 現状の回転量と手首の使い方の傾向を踏まえ、打ち出しと操作性のバランスを取ります。 |")
        lines.append(f"| トルク | {ft['トルク']} | ヘッドスピード帯を考慮し、戻り過ぎ・遅れ過ぎを避ける範囲を推奨します。 |")
        lines.append("\n※本診断は骨格分析に基づく傾向提案です。")
        lines.append("リシャフトについては、お客様ご自身で実際に試打した上でご検討ください。\n")

    # 10 (premium only) – requested volume and closing sentence
    if is_premium:
        lines.append("\n## 10. まとめ（次のステップ）\n")
        lines.append(sections.get("sec10_text", "").strip())

    return "\n".join(lines).strip()


# ==================================================
# Routes
# ==================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "service": "gate-swing-server",
            "queue_location": TASK_QUEUE_LOCATION,
            "queue_name": TASK_QUEUE_NAME,
            "force_premium": FORCE_PREMIUM,
            "has_firestore": bool(db),
            "has_line": bool(line_bot_api and handler),
            "has_tasks": bool(tasks_client and queue_path),
        }
    )


@app.route("/webhook", methods=["POST"])
def webhook():
    if not handler:
        abort(500)
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception:
        safe_print_exc("[Webhook] handler error")
        abort(500)

    return "OK"


@handler.add(MessageEvent, message=TextMessage)  # type: ignore[misc]
def handle_text_message(event: MessageEvent):
    """
    Optional: accept user inputs for 09 fitting:
      HS:45 ミス:スライス 性別:男 番手:DR
    Store for next video.
    """
    user_id = event.source.user_id
    text = getattr(event.message, "text", "") or ""
    prof = parse_profile_text(text)

    if prof:
        set_pending_profile(user_id, prof)
        safe_line_reply(
            event.reply_token,
            "✅ 受け取りました。\n"
            "次に送る動画の診断で、フィッティング診断（有料版）に参考情報として反映します。\n"
            "（例：HS:45 ミス:スライス 性別:男 番手:DR）"
        )
    else:
        safe_line_reply(
            event.reply_token,
            "テキストを受け取りました。\n"
            "フィッティング診断用に入力する場合は、例の形式で送ってください。\n"
            "例：HS:45 ミス:スライス 性別:男 番手:DR"
        )


@handler.add(MessageEvent, message=VideoMessage)  # type: ignore[misc]
def handle_video_message(event: MessageEvent):
    user_id = event.source.user_id
    message_id = event.message.id
    report_id = f"{user_id}_{message_id}"

    # Determine plan (dev stage: force premium)
    is_premium = True if FORCE_PREMIUM else False
    plan_label = "全機能プレビュー" if is_premium else "無料版"

    # Pull pending profile (optional)
    pending_profile = pop_pending_profile(user_id)

    # 1) Firestore initial
    firestore_safe_set(
        report_id,
        {
            "user_id": user_id,
            "message_id": message_id,
            "status": "PROCESSING",
            "created_at": fbfirestore.SERVER_TIMESTAMP if db else None,
            "is_premium": is_premium,
            "plan_type": "preview" if is_premium else "free",
            "summary": "動画解析を開始しました。",
            "profile": pending_profile,
        },
    )

    # 2) enqueue
    try:
        task_name = create_cloud_task(report_id=report_id, user_id=user_id, message_id=message_id)
        firestore_safe_update(report_id, {"task_name": task_name})
    except NotFound:
        firestore_safe_update(
            report_id,
            {"status": "TASK_QUEUE_NOT_FOUND", "summary": f"Queue not found: {TASK_QUEUE_NAME} @ {TASK_QUEUE_LOCATION}"},
        )
        safe_line_reply(event.reply_token, "【システムエラー】タスクキューが見つかりません。管理者にご連絡ください。")
        return
    except PermissionDenied:
        firestore_safe_update(
            report_id,
            {"status": "TASK_PERMISSION_DENIED", "summary": "Cloud Tasks permission denied"},
        )
        safe_line_reply(event.reply_token, "【システムエラー】タスク権限が不足しています。管理者にご連絡ください。")
        return
    except Exception as e:
        firestore_safe_update(
            report_id,
            {"status": "TASK_CREATE_FAILED", "summary": f"Task create failed: {str(e)[:200]}"},
        )
        safe_line_reply(event.reply_token, "【システムエラー】動画解析ジョブの登録に失敗しました。")
        return

    # 3) first reply (your preferred polite message)
    safe_line_reply(event.reply_token, make_initial_reply(report_id, plan_label=plan_label))


@app.route("/worker/process_video", methods=["POST"])
def process_video_worker():
    started = now_ts()
    payload = request.get_json(silent=True) or {}

    report_id = payload.get("report_id")
    user_id = payload.get("user_id")
    message_id = payload.get("message_id")

    if not report_id or not user_id or not message_id:
        return jsonify({"status": "error", "message": "missing report_id/user_id/message_id"}), 400

    firestore_safe_update(report_id, {"status": "IN_PROGRESS", "summary": "動画解析を実行中です..."})

    temp_dir = None
    try:
        # plan
        report_doc = firestore_get(("reports", report_id)) or {}
        is_premium = bool(report_doc.get("is_premium", True))
        profile = report_doc.get("profile") or {}
        if not isinstance(profile, dict):
            profile = {}

        # temp files
        temp_dir = tempfile.mkdtemp()
        original_path = os.path.join(temp_dir, "original")
        input_path = os.path.join(temp_dir, "input.mp4")
        normalized_path = os.path.join(temp_dir, "normalized.mp4")

        # 1) download from LINE
        download_line_video_to_file(message_id, original_path)

        # 2) normalize container to mp4
        # Sometimes LINE content has no extension; attempt transcode anyway
        try:
            transcode_video(original_path, input_path)
        except Exception:
            # If fail, try assuming it's already mp4
            shutil.copyfile(original_path, input_path)

        # 3) re-transcode to stable mp4 for analysis
        transcode_video(input_path, normalized_path)

        # 4) mediapipe analyze
        raw_data = analyze_swing(normalized_path)
        if raw_data.get("error"):
            raise RuntimeError(raw_data["error"])

        # 5) Build sections by Gemini (premium only needs 02-06 & 10; free uses 07 too)
        sections = run_gemini_sections(raw_data, is_premium=is_premium)

        # 6) Assemble final markdown with fixed structure
        report_md = build_markdown_report(raw_data, sections, is_premium=is_premium, profile=profile)

        # Save
        firestore_safe_update(
            report_id,
            {
                "status": "COMPLETED",
                "summary": "AIによる診断レポートが生成されました。",
                "raw_data": raw_data,
                "ai_report": report_md,
                "elapsed_sec": round(now_ts() - started, 2),
                "completed_at": fbfirestore.SERVER_TIMESTAMP if db else None,
            },
        )

        # push done
        safe_line_push(user_id, make_done_push(report_id, is_premium=is_premium))
        return jsonify({"status": "success", "report_id": report_id}), 200

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        safe_print_exc("[Worker] failed")
        firestore_safe_update(
            report_id,
            {
                "status": "ANALYSIS_FAILED",
                "summary": f"動画解析処理中にエラーが発生しました。{err[:200]}",
                "elapsed_sec": round(now_ts() - started, 2),
            },
        )
        safe_line_push(user_id, "【解析エラー】動画解析が失敗しました。別角度や明るい場所で撮影してみてください。")
        # Return 200 to prevent infinite retries
        return jsonify({"status": "error", "message": "analysis failed"}), 200

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.route("/api/report_data/<report_id>", methods=["GET"])
def api_report_data(report_id: str):
    if not db:
        return jsonify({"error": "Firestore is not initialized"}), 500
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    data = doc.to_dict() or {}
    return jsonify(
        {
            "status": data.get("status", "UNKNOWN"),
            "summary": data.get("summary", ""),
            "is_premium": data.get("is_premium", True),
            "plan_type": data.get("plan_type", ""),
            "mediapipe_data": data.get("raw_data", {}),
            "ai_report_text": data.get("ai_report", ""),
            "profile": data.get("profile", {}),
        }
    )


# ==================================================
# Report View (Professional HTML, no green-heavy)
# - Safe from f-string brace bugs: return raw triple-quoted string with placeholder replacement
# - Markdown renderer handles headings, bold, lists, tables, <br>
# ==================================================
@app.route("/report/<report_id>", methods=["GET"])
def report_view(report_id: str):
    html = r"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GATE AIスイングドクター 診断レポート</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @media print { .no-print { display: none !important; } body { background:#fff !important; } }
    :root{
      --ink:#0f172a;        /* slate-900 */
      --muted:#475569;      /* slate-600 */
      --line:#e2e8f0;       /* slate-200 */
      --panel:#ffffff;
      --bg:#f1f5f9;         /* slate-100 */
      --accent:#1f2937;     /* gray-800 */
      --soft:#f8fafc;       /* slate-50 */
    }
    body{ background:var(--bg); color:var(--ink); }
    .card{ background:var(--panel); border:1px solid var(--line); border-radius:16px; }
    .h2{ font-size:1.35rem; font-weight:800; letter-spacing:.02em; margin-top:2.2rem; padding-bottom:.6rem; border-bottom:2px solid var(--ink); }
    .h3{ font-size:1.05rem; font-weight:800; margin-top:1.2rem; }
    .muted{ color:var(--muted); }
    .pill{ border:1px solid var(--line); background:var(--soft); border-radius:999px; padding:.25rem .6rem; font-size:.78rem; }
    .metric{
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:14px;
      padding:12px;
      text-align:center;
    }
    .metric .k{ font-size:.78rem; color:var(--muted); }
    .metric .v{ font-size:1.5rem; font-weight:900; color:var(--ink); margin-top:2px; }
    .metric .s{ font-size:.75rem; color:var(--muted); margin-top:4px; }
    table{ width:100%; border-collapse:collapse; margin-top:12px; }
    th,td{ border:1px solid #cbd5e1; padding:10px; vertical-align:top; font-size:.95rem; }
    th{ background:#e2e8f0; font-weight:800; text-align:left; }
    .probox{
      border:1px solid var(--line);
      background:var(--soft);
      border-radius:14px;
      padding:12px;
      margin-top:10px;
    }
    .probox .tag{ font-weight:900; color:var(--accent); margin-bottom:6px; }
    .bullets{ margin-top:10px; }
    .bullets li{
      list-style:none;
      margin:8px 0;
      padding:10px 12px;
      border:1px solid var(--line);
      background:var(--panel);
      border-radius:12px;
      line-height:1.5;
    }
    .para{ line-height:1.75; margin-top:10px; }
    .loading{ padding:40px 0; text-align:center; color:var(--muted); }
  </style>
</head>
<body class="font-sans">
  <div class="max-w-4xl mx-auto p-4 md:p-8">

    <div class="card shadow-sm p-5">
      <div class="text-center">
        <div class="text-2xl font-extrabold tracking-wide">GATE AIスイングドクター</div>
        <div class="mt-2 flex flex-wrap items-center justify-center gap-2 text-sm muted">
          <span class="pill">診断レポート</span>
          <span class="pill">ID: <span id="rid"></span></span>
          <span class="pill">Status: <span id="status"></span></span>
        </div>
      </div>
      <div class="no-print flex justify-end mt-4">
        <button onclick="window.print()" class="px-4 py-2 rounded-lg bg-slate-900 text-white font-semibold hover:bg-slate-800">
          PDFとして保存 / 印刷
        </button>
      </div>
    </div>

    <div id="loading" class="loading">レポートを読み込み中…</div>

    <div id="main" class="hidden">
      <div class="card shadow-sm p-5 mt-6">
        <div class="h2">01. 骨格計測データ（AIが測った数値）</div>
        <div id="metrics" class="grid grid-cols-2 md:grid-cols-3 gap-3 mt-4"></div>
        <div class="mt-5">
          <div class="h3">各数値の見方（簡単な説明）</div>
          <div id="metric_desc" class="para muted"></div>
        </div>
      </div>

      <div class="card shadow-sm p-5 mt-6">
        <div class="h2">AIスイング診断レポート</div>
        <div id="report" class="mt-3"></div>
      </div>
    </div>

  </div>

<script>
  const reportId = "__REPORT_ID__";
  document.getElementById("rid").innerText = reportId;

  function esc(s){
    return String(s ?? "")
      .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }

  function clean(md){
    let t = String(md || "");
    t = t.replace(/```[\\s\\S]*?```/g, ""); // remove fenced blocks
    t = t.replace(/```/g, "");
    return t.trim();
  }

  function renderMetric(title, value, unit, ideal){
    const v = (value === undefined || value === null || value === "") ? "N/A" : String(value);
    return `
      <div class="metric">
        <div class="k">${esc(title)}</div>
        <div class="v">${esc(v)}${esc(unit||"")}</div>
        <div class="s">理想目安: ${esc(ideal||"-")}</div>
      </div>
    `;
  }

  // Very small Markdown renderer:
  // - h2 (##)
  // - h3 (###)
  // - bold (**)
  // - bullet list (- )
  // - tables (|...|)
  // - <br> kept as-is
  function mdToHtml(md){
    let t = clean(md);

    // Convert tables first: detect consecutive lines starting with |
    t = t.replace(/(^\\|.*\\|\\s*$\\n(?:^\\|.*\\|\\s*$\\n?)+)/gm, (block)=>{
      const lines = block.trim().split(/\\n/).filter(Boolean);
      if(lines.length < 2) return block;

      // split cells
      const rows = lines.map(l => l.trim().replace(/^\\|/,"").replace(/\\|$/,"").split("|").map(c=>c.trim()));
      // remove separator row like |---|---|
      const filtered = [];
      for(let i=0;i<rows.length;i++){
        const r = rows[i];
        const isSep = r.every(c => /^:?-{3,}:?$/.test(c));
        if(!isSep) filtered.push(r);
      }
      if(filtered.length < 1) return block;

      const head = filtered[0];
      const body = filtered.slice(1);

      let html = "<table><thead><tr>";
      head.forEach(c=> html += "<th>"+esc(c)+"</th>");
      html += "</tr></thead><tbody>";
      body.forEach(r=>{
        html += "<tr>";
        r.forEach(c=> html += "<td>"+c.replace(/<br>/g,"<br>")+"</td>");
        html += "</tr>";
      });
      html += "</tbody></table>";
      return html;
    });

    // Headings
    t = t.replace(/^##\\s+(.*)$/gm, '<div class="h2">$1</div>');
    t = t.replace(/^###\\s+(.*)$/gm, '<div class="h3">$1</div>');

    // Bold
    t = t.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');

    // Lists
    t = t.replace(/(^-\\s+.*(?:\\n-\\s+.*)*)/gm, (block)=>{
      const items = block.split(/\\n/).map(l=>l.replace(/^-\\s+/,"").trim()).filter(Boolean);
      if(!items.length) return block;
      return '<ul class="bullets">' + items.map(it=>'<li>'+it+'</li>').join('') + '</ul>';
    });

    // Paragraph breaks
    t = t.replace(/\\n\\n+/g, "</div><div class='para'>");
    t = "<div class='para'>" + t.replace(/\\n/g,"<br>") + "</div>";

    // Pro evaluation emphasis: wrap lines starting with **プロ評価** already present in markdown, so leave it.
    return t;
  }

  fetch("/api/report_data/" + reportId)
    .then(r=>r.json())
    .then(d=>{
      document.getElementById("loading").classList.add("hidden");
      document.getElementById("main").classList.remove("hidden");
      document.getElementById("status").innerText = d.status || "UNKNOWN";

      const m = d.mediapipe_data || {};
      const metrics = document.getElementById("metrics");

      metrics.innerHTML =
        renderMetric("解析フレーム数", m.frame_count, "", "60フレーム以上") +
        renderMetric("最大肩回転", m.max_shoulder_rotation, "°", "約80°〜100°") +
        renderMetric("最小腰回転", m.min_hip_rotation, "°", "約35°〜45°") +
        renderMetric("最大コック角", m.max_wrist_cock, "°", "約90°〜120°") +
        renderMetric("最大頭ブレ（Sway）", m.max_head_drift_x, "", "0.05以下") +
        renderMetric("最大膝ブレ（Sway）", m.max_knee_sway_x, "", "0.05以下");

      // metric descriptions (pulled from report 01 section, already contains explanations)
      // We'll show a compact fixed text here for readability; the detailed is in markdown too.
      const desc = `
        <div><strong>解析フレーム数</strong>：分析の粒度。十分なフレーム数があるほど傾向が安定します。</div>
        <div><strong>最大肩回転</strong>：上半身の捻転量。大きいほど体幹主導になりやすい目安です。</div>
        <div><strong>最小腰回転</strong>：腰の回旋量。捻転差の作りやすさに関係します。</div>
        <div><strong>最大コック角</strong>：手首のコック量。適正範囲が再現性に繋がります。</div>
        <div><strong>最大頭ブレ</strong>：頭の左右ブレ。小さいほど軸が安定しやすいです。</div>
        <div><strong>最大膝ブレ</strong>：下半身の左右ブレ。小さいほど土台が安定しやすいです。</div>
      `;
      document.getElementById("metric_desc").innerHTML = desc;

      const reportMd = d.ai_report_text || "";
      document.getElementById("report").innerHTML = mdToHtml(reportMd);
    })
    .catch(()=>{
      document.getElementById("loading").innerText = "読み込みに失敗しました。";
    });
</script>

</body>
</html>
"""
    return html.replace("__REPORT_ID__", report_id), 200


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
