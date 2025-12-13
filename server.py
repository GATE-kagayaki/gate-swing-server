import os
import json
import time
import math
import shutil
import tempfile
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import ffmpeg
import cv2
import mediapipe as mp

from flask import Flask, request, abort, jsonify

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore, tasks_v2
from google.api_core.exceptions import NotFound, PermissionDenied
from google import genai
from google.genai import errors as genai_errors


# ==================================================
# ENV
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")
TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")
TASK_HANDLER_PATH = os.environ.get("TASK_HANDLER_PATH", "/worker/process_video")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "").strip()

ESTIMATED_SECONDS = int(os.environ.get("ESTIMATED_SECONDS", "180"))

# 開発中は「常に有料版」を強制（あなたの要望）
FORCE_PREMIUM_ALWAYS = os.environ.get("FORCE_PREMIUM_ALWAYS", "true").lower() in ("1", "true", "yes", "on")

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
# Helpers: Firestore
# ==================================================
def firestore_safe_set(report_id: str, data: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("reports").document(report_id).set(data, merge=True)
    except Exception:
        print("[Firestore] set failed:", report_id)
        print(traceback.format_exc())


def firestore_safe_update(report_id: str, patch: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("reports").document(report_id).update(patch)
    except Exception:
        print("[Firestore] update failed:", report_id)
        print(traceback.format_exc())


def firestore_get(report_id: str) -> Optional[Dict[str, Any]]:
    if not db:
        return None
    try:
        doc = db.collection("reports").document(report_id).get()
        if not doc.exists:
            return None
        return doc.to_dict() or {}
    except Exception:
        print("[Firestore] get failed:", report_id)
        print(traceback.format_exc())
        return None


# ==================================================
# LINE messages
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


def make_initial_reply(report_id: str, mode_label: str) -> str:
    report_url = f"{SERVICE_HOST_URL}/report/{report_id}"
    return (
        "✅ 動画を受信しました。解析を開始します！\n"
        f"（モード：{mode_label}）\n\n"
        f"AIによるスイング診断には最大{max(1, ESTIMATED_SECONDS // 60)}分ほどかかります。\n"
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
        "骨格データ（01）と総合診断（07）をご確認いただけます。"
    )


# ==================================================
# Cloud Tasks
# ==================================================
def create_cloud_task(report_id: str, user_id: str, message_id: str) -> str:
    if not tasks_client or not queue_path:
        raise RuntimeError("Cloud Tasks client is not initialized.")
    if not SERVICE_HOST_URL:
        raise RuntimeError("SERVICE_HOST_URL is missing.")
    if not TASK_SA_EMAIL:
        raise RuntimeError("TASK_SA_EMAIL is missing.")

    payload = json.dumps(
        {"report_id": report_id, "user_id": user_id, "message_id": message_id},
        ensure_ascii=False,
    ).encode("utf-8")

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{SERVICE_HOST_URL}{TASK_HANDLER_PATH}",
            "headers": {"Content-Type": "application/json"},
            "body": payload,
            # Cloud Run 認証ON前提：OIDC必須
            "oidc_token": {
                "service_account_email": TASK_SA_EMAIL,
                "audience": SERVICE_HOST_URL,
            },
        }
    }
    resp = tasks_client.create_task(parent=queue_path, task=task)
    return resp.name


# ==================================================
# Video download & preprocess
# ==================================================
def download_line_video_to_file(message_id: str, out_path: str) -> None:
    if not line_bot_api:
        raise RuntimeError("LINE API is not configured.")
    content = line_bot_api.get_message_content(message_id)
    with open(out_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)


def transcode_to_mp4(in_path: str, out_path: str) -> None:
    """
    短尺 / 可変fps / コーデック差を吸収するため、H.264/AAC + yuv420p + faststart を強制
    """
    try:
        (
            ffmpeg
            .input(in_path)
            .output(
                out_path,
                vcodec="libx264",
                acodec="aac",
                pix_fmt="yuv420p",
                movflags="+faststart",
                preset="veryfast",
                crf=28,
                r=30,
                vf="scale='min(1280,iw)':-2",
                **{"max_muxing_queue_size": 1024},
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        err = (e.stderr or b"").decode("utf-8", errors="ignore")[:3000]
        raise RuntimeError(f"動画の変換に失敗しました（ffmpeg）: {err}")


# ==================================================
# Mediapipe analysis
# ==================================================
def _angle(p1, p2, p3) -> float:
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    p3 = np.array(p3, dtype=np.float32)
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def analyze_swing(video_path: str) -> Dict[str, Any]:
    """
    ✅ 180°/異常値を避けるための完成版
    - 極値取りではなく「トップ基準」
    - 物理的に成立する範囲へクリップ
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("動画を読み込めませんでした。ファイル形式をご確認ください。")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []

    def rot_deg(lx, ly, rx, ry):
        return math.degrees(math.atan2(ry - ly, rx - lx))  # -180..180

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark

            L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            L_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
            R_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
            NOSE = mp_pose.PoseLandmark.NOSE.value
            L_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
            R_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
            R_ELB = mp_pose.PoseLandmark.RIGHT_ELBOW.value
            R_WRI = mp_pose.PoseLandmark.RIGHT_WRIST.value
            R_IND = mp_pose.PoseLandmark.RIGHT_INDEX.value

            shoulder = rot_deg(lm[L_SH].x, lm[L_SH].y, lm[R_SH].x, lm[R_SH].y)
            hip = rot_deg(lm[L_HIP].x, lm[L_HIP].y, lm[R_HIP].x, lm[R_HIP].y)

            wrist = _angle(
                (lm[R_ELB].x, lm[R_ELB].y),
                (lm[R_WRI].x, lm[R_WRI].y),
                (lm[R_IND].x, lm[R_IND].y),
            )

            head_x = lm[NOSE].x
            knee_x = (lm[L_KNEE].x + lm[R_KNEE].x) / 2.0

            frames.append({
                "shoulder": shoulder,
                "hip": hip,
                "wrist": wrist,
                "head_x": head_x,
                "knee_x": knee_x,
            })

    finally:
        cap.release()
        pose.close()

    if len(frames) < 15:
        raise RuntimeError("解析に必要なフレーム数が不足しています。もう少し長めの動画でお試しください。")

    # トップ = 肩回転（絶対値）が最大のフレーム
    top = max(frames, key=lambda f: abs(f["shoulder"]))

    head_base = frames[0]["head_x"]
    knee_base = frames[0]["knee_x"]

    def clip(v, lo, hi):
        return max(lo, min(hi, v))

    def clean(v, nd=4):
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return round(float(v), nd)

    max_shoulder = clean(clip(abs(top["shoulder"]), 0, 120), 1)
    hip_rot = clean(clip(abs(top["hip"]), 0, 90), 1)
    wrist_cock = clean(clip(top["wrist"], 0, 120), 1)
    head_drift = clean(abs(top["head_x"] - head_base), 4)
    knee_sway = clean(abs(top["knee_x"] - knee_base), 4)

    return {
        "frame_count": int(len(frames)),
        "max_shoulder_rotation": max_shoulder,
        "min_hip_rotation": hip_rot,
        "max_wrist_cock": wrist_cock,
        "max_head_drift_x": head_drift,
        "max_knee_sway_x": knee_sway,
    }


# ==================================================
# Data split (章崩壊を物理的に防ぐ)
# ==================================================
def split_section_data(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        "section_01": {
            "frame_count": raw.get("frame_count"),
            "max_shoulder_rotation": raw.get("max_shoulder_rotation"),
            "min_hip_rotation": raw.get("min_hip_rotation"),
            "max_wrist_cock": raw.get("max_wrist_cock"),
            "max_head_drift_x": raw.get("max_head_drift_x"),
            "max_knee_sway_x": raw.get("max_knee_sway_x"),
        },
        "section_02": {"max_head_drift_x": raw.get("max_head_drift_x")},
        "section_03": {"max_shoulder_rotation": raw.get("max_shoulder_rotation")},
        "section_04": {"min_hip_rotation": raw.get("min_hip_rotation")},
        "section_05": {"max_wrist_cock": raw.get("max_wrist_cock")},
        "section_06": {"max_knee_sway_x": raw.get("max_knee_sway_x")},
    }


# ==================================================
# Gemini
# ==================================================
def _choose_models() -> Tuple[str, ...]:
    if GEMINI_MODEL:
        return (GEMINI_MODEL,)
    # 通りやすい順（環境差吸収）
    return (
        "gemini-2.0-flash",
        "models/gemini-2.0-flash",
        "gemini-1.5-pro",
        "models/gemini-1.5-pro",
        "gemini-1.5-flash",
        "models/gemini-1.5-flash",
    )


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    # 念のため：コードブロックを排除（Geminiが ``` を混ぜた時の保険）
    t = t.replace("```json", "").replace("```JSON", "").replace("```", "")
    return t.strip()


def build_prompt_full(raw_data: Dict[str, Any], declared: Optional[Dict[str, Any]] = None) -> str:
    declared = declared or {}
    sections = split_section_data(raw_data)

    # ※あなたの確定仕様を厳守
    return f"""
あなたはプロゴルファーを指導するゴルフコーチ兼フィッターです。
以下の「章ごとの使用可能データ（数値）」のみに基づき、指定された構成・ルールを厳守して日本語の診断レポートを作成してください。

【絶対ルール（破ると不合格）】
・章番号、章タイトルは必ず指定どおりに出力
・各章は「その章に提示されたデータのみ」を使う（他章の数値に触れない）
・「説明」と「評価（プロ評価）」を混同しない
・推測で数値を補完しない（未計測の話は禁止）
・商品名、メーカー名は一切出さない
・全体トーンは「ハイ（専門的だが読みやすい）」
・Markdownのみ使用（```json 等のコードブロックは禁止）
・出力は必ず「## 01...」から開始し、最後は「10.」で締める（前置き禁止）

【01の理想の目安（一般的な参考）】
- 解析フレーム数：60フレーム以上
- 最大肩回転：約80°〜100°
- 最小腰回転：約35°〜45°（目安）
- 最大コック角：約90°〜120°
- 最大頭ブレ（Sway）：0.05以下（小さいほど安定）
- 最大膝ブレ（Sway）：0.05以下（小さいほど安定）

【申告情報（任意）】未入力なら骨格分析のみで判断。
{json.dumps(declared, ensure_ascii=False, indent=2)}

────────────────
## 01. 骨格計測データ（AIが測った数値）
必ず「表形式」。列は「計測項目｜測定値｜理想の目安」。
対象6項目：解析フレーム数／最大肩回転／最小腰回転／最大コック角／最大頭ブレ（Sway）／最大膝ブレ（Sway）
※この章では「評価」「プロ評価」「改善提案」を書かない。
※表の直後に「### 各数値の見方（簡単な説明）」を必ず付け、6項目それぞれを **太字の見出し** にして1〜2文で説明を書く。
※ここでも「プロ評価」は書かない。

使用可能データ：
{json.dumps(sections["section_01"], ensure_ascii=False, indent=2)}

────────────────
## 02. 頭の安定性（軸のブレ）
対象数値：最大頭ブレ（Sway）のみ
構成：
・**測定値：xxxx**
・箇条書きの解説（最大3つ、少し詳しめ）
・プロ評価（1段落）
※肩・腰・手首・膝の話題は出さない（絶対）

使用可能データ：
{json.dumps(sections["section_02"], ensure_ascii=False, indent=2)}

────────────────
## 03. 肩の回旋（上半身のねじり）
対象数値：最大肩回転のみ
・**測定値：xxxx**
・箇条書き（最大3つ、少し詳しめ）
・プロ評価（1段落）

使用可能データ：
{json.dumps(sections["section_03"], ensure_ascii=False, indent=2)}

────────────────
## 04. 腰の回旋（下半身の動き）
対象数値：最小腰回転のみ
・**測定値：xxxx**
・箇条書き（最大3つ、少し詳しめ）
・プロ評価（1段落）

使用可能データ：
{json.dumps(sections["section_04"], ensure_ascii=False, indent=2)}

────────────────
## 05. 手首のメカニクス（コック角）
対象数値：最大コック角のみ
・**測定値：xxxx**
・箇条書き（最大3つ、少し詳しめ）
・プロ評価（1段落）

使用可能データ：
{json.dumps(sections["section_05"], ensure_ascii=False, indent=2)}

────────────────
## 06. 下半身の安定性（膝のブレ）
対象数値：最大膝ブレ（Sway）のみ
・**測定値：xxxx**
・箇条書き（最大3つ、少し詳しめ）
・プロ評価（1段落）

使用可能データ：
{json.dumps(sections["section_06"], ensure_ascii=False, indent=2)}

────────────────
## 07. 総合診断
以下の2項目のみ。各項目は箇条書き。
・安定している点
・改善が期待される点

────────────────
## 08. 改善戦略とドリル
最大3つ。必ず「表形式」。列：ドリル名｜目的｜やり方
やり方は必ず「①②③」の3ステップで、初心者でも実行できる程度に“少し詳しめ”。

────────────────
## 09. スイング傾向補正型フィッティング（ドライバーのみ）
必ず「表形式」。列：項目｜推奨｜理由
商品名禁止。対象項目：
①シャフト重量（40g台〜70g台）
②フレックス（L/A/R/SR/S/X）
③キックポイント（先・中・元）
④トルク（3.0〜6.5）
ヘッドスピードが申告されていれば考慮。未入力なら骨格分析のみで判断。
表の直後に必ず次の注意書きをそのまま入れる：
「本診断は骨格分析に基づく傾向提案です。
リシャフトについては、お客様ご自身で実際に試打した上でご検討ください。」

────────────────
## 10. まとめ（次のステップ）
“現状より一段ボリューム多め”で。
最後は必ず次の締め文で終える：
「お客様のゴルフライフが、より充実したものになることを切に願っています。」
""".strip()


def build_prompt_free(raw_data: Dict[str, Any], declared: Optional[Dict[str, Any]] = None) -> str:
    """
    無料版：01 と 07 のみ（あなたの確定仕様）
    """
    declared = declared or {}
    sections = split_section_data(raw_data)
    return f"""
あなたはゴルフコーチです。以下の数値だけに基づき、指定形式で「無料版レポート」を日本語で作成してください。

【ルール】
・出力は必ず「## 01...」から開始（前置き禁止）
・章は 01 と 07 のみ（02〜06,08〜10は書かない）
・Markdownのみ。コードブロック禁止（```禁止）
・01は表（計測項目｜測定値｜理想の目安）＋「### 各数値の見方（簡単な説明）」を付ける（6項目）
・07は「安定している点」「改善が期待される点」の2項目だけ、箇条書きで

【01の理想の目安】
- 解析フレーム数：60フレーム以上
- 最大肩回転：約80°〜100°
- 最小腰回転：約35°〜45°（目安）
- 最大コック角：約90°〜120°
- 最大頭ブレ（Sway）：0.05以下（小さいほど安定）
- 最大膝ブレ（Sway）：0.05以下（小さいほど安定）

使用可能データ：
{json.dumps(sections["section_01"], ensure_ascii=False, indent=2)}
""".strip()


def call_gemini(prompt: str) -> Tuple[str, str]:
    if not GEMINI_API_KEY:
        return "## AI診断エラー\nGEMINI_API_KEY が未設定です。", "AI診断が実行できませんでした。"

    client = genai.Client(api_key=GEMINI_API_KEY)

    last_err: Optional[Exception] = None
    for model in _choose_models():
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            text = _strip_code_fences(getattr(resp, "text", "") or "")
            if not text:
                raise RuntimeError(f"Empty response from model: {model}")

            # 保険：先頭が 01 から始まらない出力は軽く整形（前置き混入対策）
            idx = text.find("## 01.")
            if idx > 0:
                text = text[idx:].strip()

            return text, f"AIレポート生成完了（model: {model}）"
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            last_err = e
            print("[Gemini] model failed:", model, str(e))
            continue
        except Exception as e:
            last_err = e
            print("[Gemini] unexpected error:", model, str(e))
            continue

    msg = "AI診断レポートの生成に失敗しました。利用可能モデルをご確認ください。"
    if last_err:
        msg += f"\n（最後のエラー）{type(last_err).__name__}: {str(last_err)[:300]}"
    return "## AI診断エラー\n" + msg, "AI診断が実行できませんでした。"


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
            "service_host_url": SERVICE_HOST_URL,
            "force_premium_always": FORCE_PREMIUM_ALWAYS,
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

    # 開発中：常に有料版にする
    is_premium = True if FORCE_PREMIUM_ALWAYS else False

    firestore_safe_set(
        report_id,
        {
            "user_id": user_id,
            "message_id": message_id,
            "status": "PROCESSING",
            "plan_type": "premium" if is_premium else "free",
            "is_premium": is_premium,
            "summary": "動画解析を開始しました。",
            "created_at": firestore.SERVER_TIMESTAMP if db else None,
        },
    )

    try:
        task_name = create_cloud_task(report_id, user_id, message_id)
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

    safe_line_reply(event.reply_token, make_initial_reply(report_id, mode_label="全機能プレビュー" if is_premium else "無料版"))


@app.route("/worker/process_video", methods=["POST"])
def process_video_worker():
    started = time.time()
    payload = request.get_json(silent=True) or {}
    report_id = payload.get("report_id")
    user_id = payload.get("user_id")
    message_id = payload.get("message_id")

    if not report_id or not user_id or not message_id:
        return jsonify({"status": "error", "message": "missing report_id/user_id/message_id"}), 400

    meta = firestore_get(report_id) or {}
    is_premium = bool(meta.get("is_premium", True))

    firestore_safe_update(report_id, {"status": "IN_PROGRESS", "summary": "動画解析を実行中です..."})

    temp_dir = tempfile.mkdtemp(prefix="gate_swing_")
    raw_video = os.path.join(temp_dir, "raw_video.bin")
    mp4_video = os.path.join(temp_dir, "input.mp4")

    try:
        download_line_video_to_file(message_id, raw_video)
        transcode_to_mp4(raw_video, mp4_video)
        raw_data = analyze_swing(mp4_video)

        declared = meta.get("declared") if isinstance(meta.get("declared"), dict) else {}

        # レポート生成（無料/有料）
        prompt = build_prompt_full(raw_data, declared=declared) if is_premium else build_prompt_free(raw_data, declared=declared)
        ai_report_md, summary_text = call_gemini(prompt)

        firestore_safe_update(
            report_id,
            {
                "status": "COMPLETED",
                "summary": summary_text,
                "raw_data": raw_data,
                "ai_report": ai_report_md,
                "elapsed_sec": round(time.time() - started, 2),
                "completed_at": firestore.SERVER_TIMESTAMP if db else None,
            },
        )

        safe_line_push(user_id, make_done_push(report_id, is_premium=is_premium))
        return jsonify({"status": "success", "report_id": report_id}), 200

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        print("[Worker] failed:", err)
        print(traceback.format_exc())

        firestore_safe_update(
            report_id,
            {
                "status": "ANALYSIS_FAILED",
                "summary": f"動画解析処理中にエラーが発生しました。{err[:200]}",
                "elapsed_sec": round(time.time() - started, 2),
            },
        )
        safe_line_push(user_id, "【解析エラー】動画の変換または解析に失敗しました。別角度や明るい場所で撮影してみてください。")
        # Cloud Tasks は 200 を返すと無限リトライにならない
        return jsonify({"status": "error", "message": "analysis failed"}), 200

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route("/api/report_data/<report_id>", methods=["GET"])
def api_report_data(report_id: str):
    if not db:
        return jsonify({"error": "Firestore is not initialized"}), 500

    data = firestore_get(report_id)
    if not data:
        return jsonify({"error": "not found"}), 404

    return jsonify(
        {
            "status": data.get("status", "UNKNOWN"),
            "summary": data.get("summary", ""),
            "is_premium": bool(data.get("is_premium", True)),
            "plan_type": data.get("plan_type", "premium"),
            "mediapipe_data": data.get("raw_data", {}),
            "ai_report_text": data.get("ai_report", ""),
        }
    )


# ==================================================
# Web Report Viewer (Markdown表を“確実に”HTML化)
# ==================================================
@app.route("/report/<report_id>", methods=["GET"])
def report_view(report_id: str):
    # f-string を使わない（JSの {} / ${} 事故防止）
    return r"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GATE AIスイングドクター 診断レポート</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @media print { .no-print { display:none !important; } body{ background:#fff !important; } }
    .md h2 { font-size: 1.5rem; font-weight: 900; margin: 1.8rem 0 0.8rem; border-bottom: 2px solid #e5e7eb; padding-bottom: .35rem; }
    .md h3 { font-size: 1.15rem; font-weight: 800; margin: 1.2rem 0 0.6rem; }
    .md p  { margin: 0 0 0.9rem 0; line-height: 1.75; color:#111827; }
    .md ul { margin: .6rem 0 1rem 1.2rem; list-style: disc; }
    .md li { margin: .35rem 0; line-height:1.7; }
    .md table { width:100%; border-collapse: collapse; margin: 1rem 0; }
    .md th, .md td { border:1px solid #e5e7eb; padding:.65rem .6rem; vertical-align: top; }
    .md th { background:#f9fafb; font-weight: 900; }
    .card { background:#fff; border:1px solid #e5e7eb; border-radius: 0.9rem; }
    .k { font-size:.75rem; color:#6b7280; }
    .v { font-size:1.35rem; font-weight:900; color:#111827; }
    .pill { display:inline-block; padding:.2rem .6rem; border-radius:9999px; font-size:.75rem; background:#f3f4f6; color:#111827; }
  </style>
</head>
<body class="bg-gray-100 font-sans">
  <div class="max-w-4xl mx-auto p-4 md:p-8">
    <div class="card shadow-sm p-4 md:p-5 mb-4">
      <div class="text-2xl md:text-3xl font-black text-center text-gray-900">GATE AIスイングドクター</div>
      <div class="text-sm text-gray-500 text-center mt-1">診断レポートID: <span id="rid"></span></div>
      <div class="text-sm text-gray-500 text-center mt-1">ステータス: <span class="pill" id="status"></span></div>
      <div class="no-print flex justify-end mt-4">
        <button onclick="window.print()" class="px-4 py-2 bg-gray-900 text-white rounded-lg shadow hover:bg-black">📄 PDFとして保存 / 印刷</button>
      </div>
    </div>

    <div id="loading" class="card shadow-sm p-6 text-center text-gray-600">読み込み中...</div>

    <div id="main" class="hidden">
      <div class="card shadow-sm p-5 mb-6">
        <div class="text-xl font-extrabold mb-3 text-gray-900">01. 骨格計測データ（AIが測った数値）</div>
        <div id="metrics" class="grid grid-cols-2 md:grid-cols-3 gap-3"></div>
      </div>

      <div class="card shadow-sm p-5 md:p-6">
        <div class="text-xl font-extrabold mb-3 text-gray-900">AIスイング診断レポート</div>
        <div id="report" class="md"></div>
      </div>
    </div>
  </div>

<script>
  const reportId = location.pathname.split("/").pop();
  document.getElementById("rid").innerText = reportId;

  function esc(s){
    return String(s ?? "")
      .replace(/&/g,"&amp;").replace(/</g,"&lt;")
      .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
  }

  // “確実に”Markdown表をHTMLにする（ブロック単位で処理）
  function renderTables(md){
    const lines = String(md || "").split("\\n");
    const out = [];
    let i = 0;

    function isTableLine(line){
      const t = line.trim();
      return t.startsWith("|") && t.endsWith("|");
    }
    function isSepLine(line){
      const t = line.trim();
      return /^\\|\\s*[:-]-[-|\\s:]*\\|\\s*$/.test(t);
    }

    while(i < lines.length){
      if (isTableLine(lines[i]) && i+1 < lines.length && isSepLine(lines[i+1])){
        const header = lines[i].trim();
        i += 2;
        const rows = [];
        while(i < lines.length && isTableLine(lines[i])){
          rows.push(lines[i].trim());
          i++;
        }

        const headCells = header.split("|").slice(1,-1).map(x=>x.trim());
        const bodyRows = rows.map(r => r.split("|").slice(1,-1).map(x=>x.trim()));

        let html = "<table><thead><tr>";
        html += headCells.map(c=>`<th>${esc(c)}</th>`).join("");
        html += "</tr></thead><tbody>";
        html += bodyRows.map(r=>"<tr>"+r.map(c=>`<td>${esc(c).replace(/&lt;br&gt;/g,"<br>")}</td>`).join("")+"</tr>").join("");
        html += "</tbody></table>";
        out.push(html);
        continue;
      }
      out.push(esc(lines[i]));
      i++;
    }
    return out.join("\\n");
  }

  function mdToHtml(md){
    let t = String(md || "").trim();

    // 先に表をHTML化（残りはエスケープ済み + 表HTMLが混在）
    t = renderTables(t);

    // 太字
    t = t.replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>");

    // 見出し
    t = t.replace(/^##\\s+(.*)$/gm, "<h2>$1</h2>");
    t = t.replace(/^###\\s+(.*)$/gm, "<h3>$1</h3>");

    // 箇条書き（- / * / ・）
    t = t.replace(/^(?:\\s*(?:[-*]|・)\\s+.*(?:\\n|$))+?/gm, (block) => {
      const items = block.trim().split(/\\n/)
        .map(line => line.replace(/^\\s*(?:[-*]|・)\\s+/, "").trim())
        .filter(Boolean);
      return "<ul>" + items.map(it => "<li>"+it+"</li>").join("") + "</ul>";
    });

    // 段落化：HTML要素（table/h2/h3/ul）はそのまま
    const parts = t.split(/\\n\\n+/).map(p => p.trim()).filter(Boolean);
    const out = parts.map(p => {
      if (p.startsWith("<h2>") || p.startsWith("<h3>") || p.startsWith("<table") || p.startsWith("<ul>")) return p;
      return "<p>"+p.replace(/\\n/g,"<br>")+"</p>";
    }).join("\\n");
    return out;
  }

  function metricCard(title, value, unit){
    return `
      <div class="card p-4">
        <div class="k">${esc(title)}</div>
        <div class="v">${esc(value)}${esc(unit||"")}</div>
      </div>
    `;
  }

  fetch("/api/report_data/" + reportId)
    .then(r => r.json())
    .then(d => {
      document.getElementById("loading").classList.add("hidden");
      document.getElementById("main").classList.remove("hidden");
      document.getElementById("status").innerText = d.status || "UNKNOWN";

      const m = d.mediapipe_data || {};
      const metrics = document.getElementById("metrics");
      metrics.innerHTML =
        metricCard("解析フレーム数", m.frame_count ?? "N/A", "") +
        metricCard("最大肩回転", m.max_shoulder_rotation ?? "N/A", "°") +
        metricCard("最小腰回転", m.min_hip_rotation ?? "N/A", "°") +
        metricCard("最大コック角", m.max_wrist_cock ?? "N/A", "°") +
        metricCard("最大頭ブレ（Sway）", m.max_head_drift_x ?? "N/A", "") +
        metricCard("最大膝ブレ（Sway）", m.max_knee_sway_x ?? "N/A", "");

      const report = document.getElementById("report");
      const md = (d.ai_report_text || "").trim();
      report.innerHTML = md ? mdToHtml(md) : "<p>まだレポートが生成されていません。</p>";
    })
    .catch(() => {
      document.getElementById("loading").innerText = "読み込みに失敗しました。";
    });
</script>
</body>
</html>
""", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
