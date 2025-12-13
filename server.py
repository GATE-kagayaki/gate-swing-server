import os
import json
import time
import traceback
from typing import Any, Dict, Optional, Tuple

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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")
TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")
FORCE_PREMIUM = os.environ.get("FORCE_PREMIUM", "true").lower() in ("1", "true")

# ==================================================
# App init
# ==================================================
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None
db = firestore.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else None
tasks_client = tasks_v2.CloudTasksClient() if GCP_PROJECT_ID else None

queue_path = (
    tasks_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)
    if tasks_client and GCP_PROJECT_ID
    else None
)

# ==================================================
# Utils
# ==================================================
def now_ts() -> float:
    return time.time()


def firestore_safe_set(report_id: str, data: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("reports").document(report_id).set(data, merge=True)
    except Exception:
        print(traceback.format_exc())


def firestore_safe_update(report_id: str, patch: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("reports").document(report_id).update(patch)
    except Exception:
        print(traceback.format_exc())


# ==================================================
# Swing Analysis (Dummy)
# ==================================================
def analyze_swing_stub() -> Dict[str, Any]:
    return {
        "frame_count": 72,
        "max_shoulder_rotation": 38.4,
        "min_hip_rotation": 22.1,
        "max_wrist_cock": 95.6,
        "max_head_drift_x": 0.018,
        "max_knee_sway_x": 0.031,
    }


def calc_overall_score(d: Dict[str, Any]) -> int:
    score = 100
    if abs(d.get("max_head_drift_x", 0)) > 0.03:
        score -= 10
    if abs(d.get("max_knee_sway_x", 0)) > 0.04:
        score -= 10
    if d.get("max_wrist_cock", 0) < 80:
        score -= 10
    return max(60, score)


def enrich_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(raw)
    raw["overall_score"] = calc_overall_score(raw)
    raw["metric_notes"] = {
        "max_shoulder_rotation": {
            "label": "肩の最大回旋角",
            "pro_range": "45°〜60°",
        },
        "min_hip_rotation": {
            "label": "腰の最小回旋角",
            "pro_range": "20°〜35°",
        },
        "max_wrist_cock": {
            "label": "最大コック角",
            "pro_range": "90°〜120°",
        },
        "max_head_drift_x": {
            "label": "頭の左右ブレ",
            "pro_range": "±0.02以内",
        },
        "max_knee_sway_x": {
            "label": "膝の左右ブレ",
            "pro_range": "±0.03以内",
        },
    }
    return raw


# ==================================================
# Gemini
# ==================================================
def choose_gemini_model():
    return (
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "models/gemini-1.5-pro",
    )


def run_gemini_full_report(raw_data: Dict[str, Any]) -> Tuple[str, str]:
    if not GEMINI_API_KEY:
        return "## AI診断エラー\nAPIキー未設定", "AI診断失敗"

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
あなたは世界トップクラスのゴルフスイングコーチであり、AIドクターです。
以下の骨格計測データをもとに、日本語で診断レポートを作成してください。

専門用語は使用して構いませんが、直後に必ず平易な補足を入れてください。
前向きで冷静なプロインストラクターの文体で書いてください。

---

## 01. 総合スコア
- 100点満点中【{raw_data.get("overall_score")}点】として評価
- 理由を2〜3行で説明

## 02. データ評価基準（プロとの違い）
## 03. 肩の回旋（上半身のねじり）
## 04. 腰の回旋（下半身の動き）
## 05. 手首のメカニクス（クラブを操る技術）
## 06. 下半身の安定性（軸のブレ）

## 07. 総合診断（一番の課題はここ！）
- 冒頭でポジティブな一文
- 最優先課題は1つだけ

## 08. 改善戦略とドリル
- 最大3つ
- **ドリル名**：目的（1行）

## 10. まとめ（次のステップ）

---

【骨格計測データ】
{json.dumps(raw_data, ensure_ascii=False, indent=2)}
"""

    for model in choose_gemini_model():
        try:
            r = client.models.generate_content(model=model, contents=prompt)
            if r.text:
                return r.text, "AI診断完了"
        except Exception:
            continue

    return "## AI診断エラー\n生成失敗", "AI診断失敗"


# ==================================================
# Worker
# ==================================================
@app.route("/worker/process_video", methods=["POST"])
def process_video_worker():
    started = now_ts()
    payload = request.get_json() or {}
    report_id = payload.get("report_id")
    user_id = payload.get("user_id")

    if not report_id or not user_id:
        return jsonify({"error": "bad request"}), 400

    firestore_safe_update(report_id, {"status": "IN_PROGRESS"})

    try:
        raw = analyze_swing_stub()
        raw = enrich_metrics(raw)

        report_md, summary = run_gemini_full_report(raw)

        firestore_safe_update(
            report_id,
            {
                "status": "COMPLETED",
                "raw_data": raw,
                "ai_report": report_md,
                "summary": summary,
                "elapsed_sec": round(now_ts() - started, 2),
            },
        )

        if line_bot_api:
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text=f"🎉 AIスイング診断が完了しました\n{SERVICE_HOST_URL}/report/{report_id}")
            )

        return jsonify({"ok": True})

    except Exception as e:
        firestore_safe_update(report_id, {"status": "FAILED", "error": str(e)})
        return jsonify({"ok": False}), 200


# ==================================================
# Webhook
# ==================================================
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
    return "OK"


@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event: MessageEvent):
    user_id = event.source.user_id
    report_id = f"{user_id}_{event.message.id}"

    firestore_safe_set(
        report_id,
        {
            "user_id": user_id,
            "status": "PROCESSING",
            "created_at": firestore.SERVER_TIMESTAMP if db else None,
        },
    )

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{SERVICE_HOST_URL}/worker/process_video",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"report_id": report_id, "user_id": user_id}).encode(),
            "oidc_token": {
                "service_account_email": TASK_SA_EMAIL,
                "audience": SERVICE_HOST_URL,
            },
        }
    }

    tasks_client.create_task(parent=queue_path, task=task)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="✅ 動画を受信しました。AI解析を開始します。"),
    )


# ==================================================
# API
# ==================================================
@app.route("/api/report_data/<report_id>")
def api_report(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    return jsonify(doc.to_dict())


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))

