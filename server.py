import os
import json
import math
import time
import tempfile
import shutil
import traceback
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, abort, jsonify

# LINE
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

# GCP
from google.cloud import firestore, tasks_v2
from google.api_core.exceptions import NotFound, PermissionDenied

# Gemini (google-genai)
from google import genai
from google.genai import errors as genai_errors

# Video / CV
import ffmpeg
import cv2
import mediapipe as mp
import numpy as np


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
TASK_HANDLER_PATH = os.environ.get("TASK_HANDLER_PATH", "/worker/process_video")

# Premium/Free control (dev)
FORCE_PREMIUM = os.environ.get("FORCE_PREMIUM", "true").lower() in ("1", "true", "yes", "on")

# Gemini model candidates override (comma-separated)
# e.g. GEMINI_MODELS="gemini-1.5-pro,gemini-1.5-flash"
GEMINI_MODELS = os.environ.get("GEMINI_MODELS", "").strip()

# Analysis knobs
MIN_FRAMES_REQUIRED = int(os.environ.get("MIN_FRAMES_REQUIRED", "20"))
MAX_VIDEO_SECONDS = int(os.environ.get("MAX_VIDEO_SECONDS", "20"))  # safety cap


# ==================================================
# App init
# ==================================================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

line_bot_api: Optional[LineBotApi] = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
handler: Optional[WebhookHandler] = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None

db: Optional[firestore.Client] = firestore.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else None

tasks_client: Optional[tasks_v2.CloudTasksClient] = tasks_v2.CloudTasksClient() if GCP_PROJECT_ID else None
queue_path: Optional[str] = None
if tasks_client and GCP_PROJECT_ID:
    queue_path = tasks_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)


# ==================================================
# Helpers: LINE safe send
# ==================================================
def safe_line_reply(reply_token: str, text: str) -> None:
    if not line_bot_api:
        return
    try:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text))
    except LineBotApiError:
        print("[LINE] reply failed")
        print(traceback.format_exc())


def safe_line_push(user_id: str, text: str) -> None:
    if not line_bot_api:
        return
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text=text))
    except LineBotApiError:
        print("[LINE] push failed")
        print(traceback.format_exc())


# ==================================================
# Helpers: Firestore safe ops
# ==================================================
def fs_set(report_id: str, data: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("reports").document(report_id).set(data, merge=True)
    except Exception:
        print("[Firestore] set failed:", report_id)
        print(traceback.format_exc())


def fs_update(report_id: str, patch: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("reports").document(report_id).update(patch)
    except Exception:
        print("[Firestore] update failed:", report_id)
        print(traceback.format_exc())


# ==================================================
# Pricing message (you asked to keep the “first” polite version)
# ==================================================
def make_initial_reply(report_id: str, plan_label: str) -> str:
    report_url = f"{SERVICE_HOST_URL}/report/{report_id}"
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
    report_url = f"{SERVICE_HOST_URL}/report/{report_id}"
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
        "骨格データと総合コメントをご確認いただけます。"
    )


# ==================================================
# Cloud Tasks
# ==================================================
def create_cloud_task(report_id: str, user_id: str, message_id: str) -> str:
    if not tasks_client or not queue_path:
        raise RuntimeError("Cloud Tasks client is not initialized. Check GCP_PROJECT_ID.")
    if not SERVICE_HOST_URL:
        raise RuntimeError("SERVICE_HOST_URL is missing.")
    if not TASK_SA_EMAIL:
        raise RuntimeError("TASK_SA_EMAIL is missing (service account for OIDC).")

    payload = json.dumps(
        {"report_id": report_id, "user_id": user_id, "message_id": message_id}
    ).encode("utf-8")

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{SERVICE_HOST_URL}{TASK_HANDLER_PATH}",
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
# Video utilities
# ==================================================
def _save_line_video_to_file(message_id: str, out_path: str) -> None:
    """
    Download video content from LINE and save to file.
    """
    if not line_bot_api:
        raise RuntimeError("LINE bot API not initialized.")
    content = line_bot_api.get_message_content(message_id)
    with open(out_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)


def _probe_duration_seconds(path: str) -> float:
    try:
        probe = ffmpeg.probe(path)
        fmt = probe.get("format", {})
        dur = float(fmt.get("duration", "0") or 0)
        return dur
    except Exception:
        return 0.0


def _transcode_to_mp4(in_path: str, out_path: str) -> None:
    """
    Convert to H.264/AAC mp4 for stable decoding in OpenCV.
    """
    (
        ffmpeg
        .input(in_path)
        .output(
            out_path,
            vcodec="libx264",
            acodec="aac",
            preset="veryfast",
            movflags="faststart",
            pix_fmt="yuv420p",
            r=30,
        )
        .overwrite_output()
        .run(quiet=True)
    )


# ==================================================
# Pose / metrics
# ==================================================
mp_pose = mp.solutions.pose


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    angle ABC (at point b)
    """
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return 0.0
    cosv = float(np.dot(ba, bc) / (nba * nbc))
    cosv = max(-1.0, min(1.0, cosv))
    return float(np.degrees(np.arccos(cosv)))


def analyze_swing(video_path: str) -> Dict[str, Any]:
    """
    MediaPipe Pose based analysis.
    Returns the 6 metrics you use:
      - frame_count
      - max_shoulder_rotation
      - min_hip_rotation
      - max_wrist_cock
      - max_head_drift_x
      - max_knee_sway_x
    """
    if not os.path.exists(video_path):
        return {"error": "video_not_found"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "video_open_failed"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(min(MAX_VIDEO_SECONDS * fps, 900))  # hard cap

    # Baselines
    frame_count = 0

    nose_x0: Optional[float] = None
    knee_x0: Optional[float] = None

    max_head_drift_x = 0.0
    max_knee_sway_x = 0.0

    # Rotation proxies (image-plane)
    # shoulder line angle and hip line angle in degrees
    max_shoulder_turn = -999.0
    min_hip_turn = 999.0

    max_wrist_cock = 0.0

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while frame_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            h, w = frame.shape[:2]
            if w == 0 or h == 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark

            def pt(idx: int) -> np.ndarray:
                return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

            # Key points
            NOSE = 0
            L_SH = 11
            R_SH = 12
            L_HIP = 23
            R_HIP = 24
            L_ELB = 13
            R_ELB = 14
            L_WRI = 15
            R_WRI = 16
            L_IDX = 19
            R_IDX = 20
            L_KNE = 25
            R_KNE = 26

            nose = pt(NOSE)
            lsh, rsh = pt(L_SH), pt(R_SH)
            lhip, rhip = pt(L_HIP), pt(R_HIP)
            lelb, relb = pt(L_ELB), pt(R_ELB)
            lwri, rwri = pt(L_WRI), pt(R_WRI)
            lidx, ridx = pt(L_IDX), pt(R_IDX)
            lkne, rkne = pt(L_KNE), pt(R_KNE)

            # Head drift (normalized by width)
            if nose_x0 is None:
                nose_x0 = float(nose[0])
            head_drift = abs(float(nose[0]) - nose_x0) / float(w)
            if head_drift > max_head_drift_x:
                max_head_drift_x = head_drift

            # Knee sway (use average knee x, normalized)
            knee_mid_x = float((lkne[0] + rkne[0]) * 0.5)
            if knee_x0 is None:
                knee_x0 = knee_mid_x
            knee_sway = abs(knee_mid_x - knee_x0) / float(w)
            if knee_sway > max_knee_sway_x:
                max_knee_sway_x = knee_sway

            # Shoulder & hip line angles (image plane)
            # angle = atan2(dy, dx) in degrees
            sh_dx = float(lsh[0] - rsh[0])
            sh_dy = float(lsh[1] - rsh[1])
            hip_dx = float(lhip[0] - rhip[0])
            hip_dy = float(lhip[1] - rhip[1])

            sh_angle = math.degrees(math.atan2(sh_dy, sh_dx))
            hip_angle = math.degrees(math.atan2(hip_dy, hip_dx))

            # "rotation" proxy: difference between shoulder and hip line angles
            # keeps sign so your UI can show negative/positive consistently
            turn = sh_angle - hip_angle

            if turn > max_shoulder_turn:
                max_shoulder_turn = turn
            if hip_angle < min_hip_turn:
                min_hip_turn = hip_angle

            # Wrist cock: use angle at wrist (elbow-wrist-index)
            l_cock = _angle_deg(lelb, lwri, lidx)
            r_cock = _angle_deg(relb, rwri, ridx)
            cock = max(l_cock, r_cock)
            if cock > max_wrist_cock:
                max_wrist_cock = cock

    finally:
        cap.release()
        pose.close()

    if frame_count < MIN_FRAMES_REQUIRED:
        return {"error": "too_short", "frame_count": frame_count}

    # Normalize outputs to your key names
    # Note: values are proxies; your existing training/UX can refine later.
    return {
        "frame_count": int(frame_count),
        "max_shoulder_rotation": float(round(max_shoulder_turn, 1)) if max_shoulder_turn > -900 else 0.0,
        "min_hip_rotation": float(round(min_hip_turn, 1)) if min_hip_turn < 900 else 0.0,
        "max_wrist_cock": float(round(max_wrist_cock, 1)),
        "max_head_drift_x": float(round(max_head_drift_x, 4)),
        "max_knee_sway_x": float(round(max_knee_sway_x, 4)),
    }


# ==================================================
# Report building
#   - 01 is generated deterministically (no Gemini)
#   - 02-10 is generated by Gemini using your final rules
# ==================================================
IDEAL_01 = {
    "frame_count": "（目安：60以上）",
    "max_shoulder_rotation": "約80°〜100°",
    "min_hip_rotation": "約40°〜50°",
    "max_wrist_cock": "約90°〜120°",
    "max_head_drift_x": "小さいほど良い（目安：0.03以下）",
    "max_knee_sway_x": "小さいほど良い（目安：0.04以下）",
}

DESC_01 = {
    "frame_count": "動画が何枚の静止画に分割され、分析されたかを示すコマ数です。",
    "max_shoulder_rotation": "肩の回転量を示します。体の捻転を使ったスイングほど大きくなりやすい指標です。",
    "min_hip_rotation": "腰の回転量を示します。適度に抑えられると上半身との捻転差を作りやすくなります。",
    "max_wrist_cock": "手首のコック量を示します。適正範囲に収まるとタイミングが安定しやすくなります。",
    "max_head_drift_x": "スイング中の頭の左右移動量を示します。小さいほど軸が安定している状態です。",
    "max_knee_sway_x": "スイング中の下半身（膝付近）の左右ブレを示します。小さいほど安定しやすい指標です。",
}


def build_markdown_section_01(raw: Dict[str, Any]) -> str:
    def v(key: str, unit: str = "") -> str:
        val = raw.get(key, "N/A")
        if isinstance(val, (int, float)):
            if unit:
                return f"{val}{unit}"
            return f"{val}"
        return str(val)

    md = []
    md.append("## 01. 骨格計測データ（AIが測った数値）\n")

    items = [
        ("解析フレーム数", "frame_count", "", IDEAL_01["frame_count"], DESC_01["frame_count"]),
        ("最大肩回転", "max_shoulder_rotation", "°", IDEAL_01["max_shoulder_rotation"], DESC_01["max_shoulder_rotation"]),
        ("最小腰回転", "min_hip_rotation", "°", IDEAL_01["min_hip_rotation"], DESC_01["min_hip_rotation"]),
        ("最大コック角", "max_wrist_cock", "°", IDEAL_01["max_wrist_cock"], DESC_01["max_wrist_cock"]),
        ("最大頭ブレ（Sway）", "max_head_drift_x", "", IDEAL_01["max_head_drift_x"], DESC_01["max_head_drift_x"]),
        ("最大膝ブレ（Sway）", "max_knee_sway_x", "", IDEAL_01["max_knee_sway_x"], DESC_01["max_knee_sway_x"]),
    ]

    for title, key, unit, ideal, desc in items:
        md.append(f"**{title}**  ")
        md.append(f"測定値：**{v(key, unit)}**  ")
        md.append(f"説明：{desc}  ")
        md.append(f"理想の目安：{ideal}\n")

    return "\n".join(md).strip() + "\n"


def choose_gemini_models() -> Tuple[str, ...]:
    if GEMINI_MODELS:
        models = [m.strip() for m in GEMINI_MODELS.split(",") if m.strip()]
        if models:
            return tuple(models)
    # safe defaults (try both bare and "models/" prefix)
    return (
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash",
        "models/gemini-2.0-flash",
    )


def strip_code_fences(text: str) -> str:
    """
    Remove accidental ``` blocks while keeping inner content.
    """
    if not text:
        return ""
    lines = text.splitlines()
    out: List[str] = []
    in_fence = False
    for ln in lines:
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            continue
        out.append(ln)
    return "\n".join(out).strip()


def build_gemini_prompt_02_10(raw_data: Dict[str, Any], head_speed: Optional[str] = None) -> str:
    hs = head_speed.strip() if isinstance(head_speed, str) and head_speed.strip() else "未入力"

    return (
        "以下の骨格計測データに基づき、指定された構成・ルールを厳密に守って、日本語のスイング診断レポートを作成してください。\n"
        "\n"
        "【重要ルール】\n"
        "・余計な前置きや自己紹介文は一切書かないでください\n"
        "・指定された見出し構成を変更しないでください\n"
        "・文量とトーンは全体で統一してください\n"
        "・商品名・メーカー名は絶対に書かないでください\n"
        "・普通のトーンで記載してください\n"
        "・Markdownで出力してください\n"
        "\n"
        "────────────────────\n"
        "■ レポート構成（厳守）\n"
        "────────────────────\n"
        "\n"
        "※『01』はシステム側で生成します。あなたは **02〜10のみ** 出力してください。\n"
        "\n"
        "## 02. 頭の安定性（軸のブレ）\n"
        "以下の構成で記載してください。\n"
        "・**① 最大頭ブレ（Sway）**（太字）\n"
        "・**測定値：○○**（太字）\n"
        "・解説（箇条書き・2〜3項目）\n"
        "・👉 プロ評価では〜（1文・他セクションと文量統一）\n"
        "※理想値の記載は不要\n"
        "\n"
        "## 03. 肩の回旋（上半身のねじり）\n"
        "## 04. 腰の回旋（下半身の動き）\n"
        "## 05. 手首のメカニクス（クラブを操る技術）\n"
        "## 06. 下半身の安定性（軸のブレ）\n"
        "02と完全に同じ構成・文量・トーンで記載\n"
        "\n"
        "## 07. 総合診断\n"
        "以下の2項目のみで構成。\n"
        "### 安定している点\n"
        "・箇条書き（2〜3項目）\n"
        "### 改善が期待される点\n"
        "・箇条書き（2〜3項目）\n"
        "\n"
        "## 08. 改善戦略とドリル（今日からできる練習法）\n"
        "・最大3つまで\n"
        "・表形式\n"
        "・列：ドリル名／目的／簡易的なやり方（①②③程度）\n"
        "※ポイント解説や理論説明は不要\n"
        "\n"
        "## 09. フィッティング診断（ドライバー）\n"
        "以下の条件を厳守。\n"
        "・表形式のみ\n"
        "・プロ評価は一切入れない\n"
        "・商品名／メーカー名は書かない\n"
        "・推奨とその理由のみ\n"
        "・ドライバー限定であることを明記\n"
        "\n"
        "対象項目（範囲厳守）：\n"
        "・シャフト重量（40g台〜70g台）\n"
        "・フレックス（L/A/R/SR/S/X）\n"
        "・キックポイント（先調子／中調子／元調子）\n"
        "・トルク（3.0〜6.5）\n"
        "\n"
        f"申告ヘッドスピード：{hs}\n"
        "・申告値がある場合は必ず考慮\n"
        "・未入力の場合は骨格分析のみで判断\n"
        "\n"
        "最後に注意書きを必ず2行入れる：\n"
        "「※本フィッティング診断はドライバーを対象としています。」\n"
        "「※リシャフトについては、お客様ご自身で試打した上でご検討ください。」\n"
        "\n"
        "## 10. まとめ（次のステップ）\n"
        "・現状の内容を普通の文量でまとめる\n"
        "・最後は必ず次の一文で締める：\n"
        "「お客様のゴルフライフが充実したものになることを切に願っています。」\n"
        "\n"
        "────────────────────\n"
        "【骨格計測データ】\n"
        f"{json.dumps(raw_data, ensure_ascii=False, indent=2)}\n"
    )


def run_gemini_02_10(raw_data: Dict[str, Any], head_speed: Optional[str] = None) -> Tuple[str, str]:
    if not GEMINI_API_KEY:
        return ("", "AI診断が実行できませんでした（APIキー未設定）")

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = build_gemini_prompt_02_10(raw_data, head_speed=head_speed)

    last_err: Optional[Exception] = None
    for model in choose_gemini_models():
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            text = getattr(resp, "text", "") or ""
            text = strip_code_fences(text)
            if not text.strip():
                raise RuntimeError("Empty Gemini response")
            return text.strip(), f"AIレポート生成完了（model={model}）"
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            last_err = e
            print("[Gemini] model failed:", model, str(e))
            continue
        except Exception as e:
            last_err = e
            print("[Gemini] unexpected error:", model, str(e))
            continue

    msg = "AI診断レポートの生成に失敗しました。利用可能なモデル名をご確認ください。"
    if last_err:
        msg += f" / last={type(last_err).__name__}: {str(last_err)[:200]}"
    return ("", msg)


def assemble_full_report_markdown(raw_data: Dict[str, Any], head_speed: Optional[str] = None) -> Tuple[str, str]:
    sec01 = build_markdown_section_01(raw_data)
    sec02_10, summary = run_gemini_02_10(raw_data, head_speed=head_speed)
    if not sec02_10:
        # still return 01 so user sees something
        fallback = (
            "## 02. 頭の安定性（軸のブレ）\n"
            "**① 最大頭ブレ（Sway）**  \n"
            f"**測定値：{raw_data.get('max_head_drift_x', 'N/A')}**  \n"
            "- レポート生成に失敗しました。  \n"
            "- モデル設定やAPIキーをご確認ください。  \n"
            "👉 プロ評価では「安定性評価はデータが揃い次第更新可能」と判断されます。\n"
            "\n"
            "## 10. まとめ（次のステップ）\n"
            "レポート生成に失敗しました。設定をご確認のうえ再度お試しください。  \n"
            "お客様のゴルフライフが充実したものになることを切に願っています。\n"
        )
        return (sec01 + "\n\n" + fallback, summary)

    return (sec01 + "\n\n" + sec02_10.strip() + "\n", summary)


# ==================================================
# Web report HTML (single string, no f-strings)
#   - fetch /api/report_data/<report_id>
#   - render:
#       01 cards + 01 detail (static)
#       rest sections from Markdown
# ==================================================
REPORT_HTML = r"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GATE AIスイングドクター 診断レポート</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @media print { .no-print { display:none !important; } body{ background:#fff !important; } }
    .report h2{
      font-size:1.9rem; font-weight:900; color:#111827;
      border-bottom:4px solid #10b981; padding-bottom:.45rem;
      margin-top:2.2rem; margin-bottom:1.2rem;
      letter-spacing:.02em;
    }
    .report h3{
      font-size:1.25rem; font-weight:800; color:#374151;
      border-left:6px solid #6ee7b7; padding-left:.8rem;
      margin-top:1.4rem; margin-bottom:.8rem;
    }
    .report p{ margin:0 0 .9rem 0; line-height:1.7; color:#374151; }
    .report ul{ list-style:none; padding:0; margin:.8rem 0; }
    .report li{
      padding:.9rem 1rem; margin:.55rem 0;
      background:#ecfdf5; border-left:6px solid #10b981;
      border-radius:.75rem; font-weight:650; color:#065f46;
      box-shadow:0 1px 2px rgba(0,0,0,.05);
    }
    .report table{
      width:100%; border-collapse:collapse; margin:1rem 0;
      font-size:.95rem;
    }
    .report th, .report td{
      border:1px solid #d1d5db; padding:.75rem; vertical-align:top;
    }
    .report th{ background:#f3f4f6; font-weight:800; color:#111827; }
    .metric-card{
      background:#f9fafb; border:1px solid #e5e7eb; border-radius:1rem;
      padding:1rem; text-align:center;
      box-shadow:0 1px 3px rgba(0,0,0,.06);
    }
    .metric-k{ font-size:.75rem; color:#6b7280; margin-bottom:.15rem; }
    .metric-v{ font-size:1.6rem; font-weight:900; color:#111827; }
    .chip{
      display:inline-block; padding:.25rem .6rem; border-radius:999px;
      background:#d1fae5; color:#065f46; font-weight:800; font-size:.75rem;
    }
    .muted{ color:#6b7280; }
    .divider{ border-top:1px solid #e5e7eb; margin:1.25rem 0; }
    .note{
      background:#f0fdf4; border:1px solid #bbf7d0; border-radius:1rem;
      padding:1rem; color:#065f46;
    }
    .small{ font-size:.9rem; }
    .mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
  </style>
</head>

<body class="bg-gray-100 font-sans">
  <div id="loading" class="fixed inset-0 bg-white/80 flex flex-col items-center justify-center z-50">
    <div class="animate-spin rounded-full h-14 w-14 border-t-4 border-b-4 border-emerald-500"></div>
    <div class="mt-4 text-lg font-bold text-gray-700">AIレポートを読み込み中...</div>
  </div>

  <div class="max-w-4xl mx-auto p-4 md:p-8">
    <div class="bg-white rounded-2xl shadow p-5 md:p-7 border border-gray-100">
      <div class="flex items-start justify-between gap-4">
        <div>
          <div class="text-2xl md:text-3xl font-black text-emerald-600">GATE AIスイングドクター</div>
          <div class="mt-2 text-gray-600 font-semibold">診断レポート</div>
          <div class="mt-2 text-sm text-gray-500">レポートID: <span id="rid" class="mono"></span></div>
          <div class="mt-1 text-sm text-gray-500">ステータス: <span id="status" class="chip">---</span></div>
        </div>

        <div class="no-print text-right">
          <button onclick="window.print()" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl shadow font-bold">
            📄 PDFとして保存 / 印刷
          </button>
          <div class="mt-2 text-xs text-gray-500">スマホは共有→印刷でも保存できます</div>
        </div>
      </div>

      <div class="divider"></div>

      <div id="summaryBox" class="note hidden">
        <div class="font-extrabold">総合コメント</div>
        <div id="summaryText" class="mt-2 small"></div>
      </div>

      <div id="pendingBox" class="note hidden">
        <div class="font-extrabold">処理状況</div>
        <div class="mt-2 small">
          まだ解析が完了していません。しばらくしてから再読み込みしてください。
        </div>
      </div>

    </div>

    <div class="mt-6 bg-white rounded-2xl shadow p-5 md:p-7 border border-gray-100">
      <div class="flex items-center justify-between gap-3">
        <div class="text-xl font-black text-gray-900">01. 骨格計測データ（AIが測った数値）</div>
        <div class="text-sm text-gray-500">測定値は動画条件により変動します</div>
      </div>

      <div id="metrics" class="mt-4 grid grid-cols-2 md:grid-cols-3 gap-3"></div>

      <div class="mt-5">
        <div class="text-base font-extrabold text-gray-900">各項目の説明と理想の目安</div>
        <div id="metricDetail" class="mt-3 report small"></div>
      </div>
    </div>

    <div class="mt-6 bg-white rounded-2xl shadow p-5 md:p-7 border border-gray-100">
      <div class="text-xl font-black text-gray-900">AIスイング診断レポート</div>
      <div id="report" class="mt-4 report"></div>
    </div>

    <div class="mt-8 text-center text-sm text-gray-500 no-print">
      <div>© GATE AI Swing Doctor</div>
    </div>
  </div>

<script>
  const reportId = location.pathname.split("/").pop();
  document.getElementById("rid").innerText = reportId;

  function esc(s){
    return String(s ?? "")
      .replaceAll("&","&amp;")
      .replaceAll("<","&lt;")
      .replaceAll(">","&gt;");
  }

  function metricCard(k, v){
    return `
      <div class="metric-card">
        <div class="metric-k">${esc(k)}</div>
        <div class="metric-v">${esc(v)}</div>
      </div>
    `;
  }

  function toFixedMaybe(x, digits){
    if (x === null || x === undefined) return "N/A";
    if (typeof x === "number") return x.toFixed(digits);
    return String(x);
  }

  // Minimal Markdown renderer: headings, bold, lists, tables
  function mdToHtml(md){
    let t = String(md || "").trim();

    // normalize line endings
    t = t.replaceAll("\r\n", "\n");

    // Tables (GitHub style)
    // We'll do a simple parser that converts consecutive | lines into a table.
    const lines = t.split("\n");
    let out = [];
    let i = 0;

    function isTableLine(line){
      return line.trim().startsWith("|") && line.includes("|");
    }

    while (i < lines.length){
      const line = lines[i];

      if (isTableLine(line)){
        // collect table block
        let block = [];
        while (i < lines.length && isTableLine(lines[i])){
          block.push(lines[i]);
          i++;
        }

        // remove alignment row if present (---)
        // split rows
        const rows = block.map(r => r.trim()).filter(r => r.length > 0);
        if (rows.length >= 2 && rows[1].replaceAll("|","").trim().match(/^:?-+:?(\s*:?-+:?)*$/)){
          // keep header + body
        }

        // parse header
        const header = rows[0].split("|").slice(1,-1).map(c => c.trim());
        let startBody = 1;
        if (rows.length >= 2 && rows[1].replaceAll("|","").trim().match(/^:?-+:?(\s*:?-+:?)*$/)){
          startBody = 2;
        }
        const body = rows.slice(startBody).map(r => r.split("|").slice(1,-1).map(c => c.trim()));

        let html = "<table><thead><tr>";
        header.forEach(h => html += "<th>"+esc(h)+"</th>");
        html += "</tr></thead><tbody>";
        body.forEach(row => {
          html += "<tr>";
          row.forEach(cell => html += "<td>"+esc(cell)+"</td>");
          html += "</tr>";
        });
        html += "</tbody></table>";
        out.push(html);
        continue;
      }

      out.push(line);
      i++;
    }

    t = out.join("\n");

    // Headings
    t = t.replace(/^##\s+(.*)$/gm, "<h2>$1</h2>");
    t = t.replace(/^###\s+(.*)$/gm, "<h3>$1</h3>");

    // Bold
    t = t.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Lists: blocks of "- " or "* "
    t = t.replace(/(?:^|\n)(?:\s*[-*]\s+.*(?:\n|$))+?/g, (block) => {
      const items = block.trim().split("\n")
        .map(l => l.replace(/^\s*[-*]\s+/, "").trim())
        .filter(Boolean);
      return "\n<ul>\n" + items.map(it => "<li>"+esc(it)+"</li>").join("\n") + "\n</ul>\n";
    });

    // Paragraphs / line breaks
    // Convert double newlines to paragraphs
    t = t.split("\n\n").map(chunk => {
      chunk = chunk.trim();
      if (!chunk) return "";
      // If starts with block tags, keep
      if (chunk.startsWith("<h2>") || chunk.startsWith("<h3>") || chunk.startsWith("<ul>") || chunk.startsWith("<table>")){
        return chunk;
      }
      return "<p>"+chunk.replaceAll("\n","<br>")+"</p>";
    }).join("\n");

    return t;
  }

  // Build 01 details (static text; measured value inserted)
  function build01Detail(m){
    function row(title, val, desc, ideal){
      return `
        <p><strong>${esc(title)}</strong><br>
        測定値：<strong>${esc(val)}</strong><br>
        説明：${esc(desc)}<br>
        理想の目安：${esc(ideal)}</p>
      `;
    }

    const frameCount = (m.frame_count ?? "N/A");
    const sh = (m.max_shoulder_rotation ?? "N/A") + (m.max_shoulder_rotation === null || m.max_shoulder_rotation === undefined ? "" : "°");
    const hip = (m.min_hip_rotation ?? "N/A") + (m.min_hip_rotation === null || m.min_hip_rotation === undefined ? "" : "°");
    const cock = (m.max_wrist_cock ?? "N/A") + (m.max_wrist_cock === null || m.max_wrist_cock === undefined ? "" : "°");
    const head = (m.max_head_drift_x ?? "N/A");
    const knee = (m.max_knee_sway_x ?? "N/A");

    return (
      row("解析フレーム数", String(frameCount), "動画が何枚の静止画に分割され、分析されたかを示すコマ数です。", "（目安：60以上）") +
      row("最大肩回転", String(sh), "肩の回転量を示します。体の捻転を使ったスイングほど大きくなりやすい指標です。", "約80°〜100°") +
      row("最小腰回転", String(hip), "腰の回転量を示します。適度に抑えられると上半身との捻転差を作りやすくなります。", "約40°〜50°") +
      row("最大コック角", String(cock), "手首のコック量を示します。適正範囲に収まるとタイミングが安定しやすくなります。", "約90°〜120°") +
      row("最大頭ブレ（Sway）", String(head), "スイング中の頭の左右移動量を示します。小さいほど軸が安定している状態です。", "小さいほど良い（目安：0.03以下）") +
      row("最大膝ブレ（Sway）", String(knee), "スイング中の下半身（膝付近）の左右ブレを示します。小さいほど安定しやすい指標です。", "小さいほど良い（目安：0.04以下）")
    );
  }

  fetch("/api/report_data/" + reportId)
    .then(r => r.json())
    .then(d => {
      document.getElementById("loading").classList.add("hidden");

      const st = d.status || "UNKNOWN";
      const statusEl = document.getElementById("status");
      statusEl.innerText = st;

      if (st !== "COMPLETED"){
        document.getElementById("pendingBox").classList.remove("hidden");
      }

      if (d.summary){
        document.getElementById("summaryBox").classList.remove("hidden");
        document.getElementById("summaryText").innerText = d.summary;
      }

      const m = d.mediapipe_data || {};

      // cards
      const metrics = document.getElementById("metrics");
      metrics.innerHTML =
        metricCard("解析フレーム数", String(m.frame_count ?? "N/A")) +
        metricCard("最大肩回転", String(m.max_shoulder_rotation ?? "N/A") + (m.max_shoulder_rotation === null || m.max_shoulder_rotation === undefined ? "" : "°")) +
        metricCard("最小腰回転", String(m.min_hip_rotation ?? "N/A") + (m.min_hip_rotation === null || m.min_hip_rotation === undefined ? "" : "°")) +
        metricCard("最大コック角", String(m.max_wrist_cock ?? "N/A") + (m.max_wrist_cock === null || m.max_wrist_cock === undefined ? "" : "°")) +
        metricCard("最大頭ブレ(Sway)", String(m.max_head_drift_x ?? "N/A")) +
        metricCard("最大膝ブレ(Sway)", String(m.max_knee_sway_x ?? "N/A"));

      document.getElementById("metricDetail").innerHTML = build01Detail(m);

      // markdown report
      const report = document.getElementById("report");
      const md = d.ai_report_text || "(まだレポートが生成されていません)";
      report.innerHTML = mdToHtml(md);
    })
    .catch(() => {
      document.getElementById("loading").classList.add("hidden");
      alert("読み込みに失敗しました。しばらくしてから再読み込みしてください。");
    });
</script>
</body>
</html>
"""


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
            "has_gemini_key": bool(GEMINI_API_KEY),
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
        print("[Webhook] handler error")
        print(traceback.format_exc())
        abort(500)

    return "OK"


@handler.add(MessageEvent, message=VideoMessage)  # type: ignore[misc]
def handle_video_message(event: MessageEvent):
    user_id = event.source.user_id
    message_id = event.message.id
    report_id = f"{user_id}_{message_id}"

    # For now: always premium in dev (your request)
    is_premium = True if FORCE_PREMIUM else False
    plan_type = "full_preview" if is_premium else "free"

    # 1) Save initial status
    fs_set(
        report_id,
        {
            "user_id": user_id,
            "message_id": message_id,
            "status": "PROCESSING",
            "plan_type": plan_type,
            "is_premium": is_premium,
            "summary": "動画解析を開始しました。",
            "created_at": firestore.SERVER_TIMESTAMP if db else None,
        },
    )

    # 2) Enqueue task
    try:
        task_name = create_cloud_task(report_id=report_id, user_id=user_id, message_id=message_id)
        fs_update(report_id, {"task_name": task_name})
    except NotFound:
        fs_update(
            report_id,
            {"status": "TASK_QUEUE_NOT_FOUND", "summary": f"Queue not found: {TASK_QUEUE_NAME} @ {TASK_QUEUE_LOCATION}"},
        )
        safe_line_reply(event.reply_token, "【システムエラー】タスクキューが見つかりません。管理者にご連絡ください。")
        return
    except PermissionDenied:
        fs_update(report_id, {"status": "TASK_PERMISSION_DENIED", "summary": "Cloud Tasks permission denied"})
        safe_line_reply(event.reply_token, "【システムエラー】タスク権限が不足しています。管理者にご連絡ください。")
        return
    except Exception as e:
        fs_update(report_id, {"status": "TASK_CREATE_FAILED", "summary": f"Task create failed: {str(e)[:200]}"})
        safe_line_reply(event.reply_token, "【システムエラー】動画解析ジョブの登録に失敗しました。")
        return

    # 3) Reply (your preferred “first polite message”)
    safe_line_reply(
        event.reply_token,
        make_initial_reply(report_id, plan_label="全機能プレビュー" if is_premium else "無料版")
    )


@app.route(TASK_HANDLER_PATH, methods=["POST"])
def process_video_worker():
    """
    Cloud Tasks worker
    - download video from LINE
    - transcode via ffmpeg
    - mediapipe analyze
    - gemini generate 02-10
    - assemble full markdown (01 fixed + 02-10)
    - save firestore
    - push LINE done message
    """
    started = time.time()
    payload = request.get_json(silent=True) or {}

    report_id = payload.get("report_id")
    user_id = payload.get("user_id")
    message_id = payload.get("message_id")

    if not report_id or not user_id or not message_id:
        return jsonify({"status": "error", "message": "missing report_id/user_id/message_id"}), 400

    fs_update(report_id, {"status": "IN_PROGRESS", "summary": "動画解析を実行中です..."})

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        original_path = os.path.join(temp_dir, "original.bin")
        mp4_path = os.path.join(temp_dir, "video.mp4")

        # 1) Download from LINE
        _save_line_video_to_file(message_id, original_path)

        # 2) Duration safety cap
        dur = _probe_duration_seconds(original_path)
        if dur and dur > MAX_VIDEO_SECONDS:
            raise RuntimeError(f"video_too_long: {dur:.1f}s")

        # 3) Transcode to mp4
        _transcode_to_mp4(original_path, mp4_path)

        # 4) Analyze with MediaPipe
        raw = analyze_swing(mp4_path)
        if raw.get("error"):
            raise RuntimeError(f"analysis_failed: {raw.get('error')}")

        # 5) Build report markdown
        # Optional fields (later): head_speed, miss tendencies etc.
        head_speed = None
        full_md, summary = assemble_full_report_markdown(raw_data=raw, head_speed=head_speed)

        # 6) Save
        fs_update(
            report_id,
            {
                "status": "COMPLETED",
                "summary": summary,
                "raw_data": raw,
                "ai_report": full_md,
                "completed_at": firestore.SERVER_TIMESTAMP if db else None,
                "elapsed_sec": round(time.time() - started, 2),
            },
        )

        # 7) Push done
        safe_line_push(user_id, make_done_push(report_id, is_premium=True if FORCE_PREMIUM else False))

        return jsonify({"status": "success", "report_id": report_id}), 200

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        print("[Worker] failed:", err)
        print(traceback.format_exc())

        fs_update(
            report_id,
            {
                "status": "ANALYSIS_FAILED",
                "summary": f"動画解析処理中にエラーが発生しました。{err[:200]}",
                "elapsed_sec": round(time.time() - started, 2),
            },
        )
        safe_line_push(user_id, "【解析エラー】動画の変換・解析に失敗しました。別角度や明るい場所で撮影してみてください。")
        # return 200 to stop Cloud Tasks infinite retries for user-facing errors
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
        }
    )


@app.route("/report/<report_id>", methods=["GET"])
def report_view(report_id: str):
    # The HTML reads report_id from URL path in JS, so no interpolation required.
    return REPORT_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    # For local testing. On Cloud Run, gunicorn will serve `app`.
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)



