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

# Gemini
GEMINI_MODEL_ENV = os.environ.get("GEMINI_MODEL", "").strip()
FORCE_PREMIUM = os.environ.get("FORCE_PREMIUM", "true").lower() in ("1", "true", "yes", "on")


# ==================================================
# App init
# ==================================================
app = Flask(__name__)

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    # ここで落とすとデプロイ後のログがわかりにくいので、起動は継続しつつ /health で検知できるようにします
    pass

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None

db = firestore.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else None

tasks_client = tasks_v2.CloudTasksClient() if GCP_PROJECT_ID else None
queue_path = None
if tasks_client and GCP_PROJECT_ID:
    queue_path = tasks_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)


# ==================================================
# Helpers
# ==================================================
def now_ts() -> float:
    return time.time()


def firestore_safe_update(report_id: str, patch: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("reports").document(report_id).update(patch)
    except Exception:
        # Firestore update failure should not crash worker
        print("[Firestore] update failed:", report_id)
        print(traceback.format_exc())


def firestore_safe_set(report_id: str, data: Dict[str, Any]) -> None:
    if not db:
        return
    try:
        db.collection("reports").document(report_id).set(data, merge=True)
    except Exception:
        print("[Firestore] set failed:", report_id)
        print(traceback.format_exc())


def make_initial_reply(report_id: str, plan_label: str = "全機能プレビュー") -> str:
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


def make_done_push(report_id: str, is_premium: bool = True) -> str:
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


def choose_gemini_model() -> Tuple[str, ...]:
    """
    v1beta環境で「モデル名の表記ゆれ」や「利用可能モデル差」があるため、
    複数候補を順に試す。
    """
    if GEMINI_MODEL_ENV:
        # ユーザー指定が最優先
        return (GEMINI_MODEL_ENV,)

    # できるだけ「通りやすい」順に並べる（環境差を吸収）
    return (
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash",
        "models/gemini-2.0-flash",
    )


def run_gemini_full_report(raw_data: Dict[str, Any], is_premium: bool = True) -> Tuple[str, str]:
    """
    有料版フルレポート（あなたが貼ってくれた構成に寄せる）
    """
    if not GEMINI_API_KEY:
        return (
            "## AI診断エラー\nAI診断レポートの生成に必要なAPIキーが設定されていません。",
            "AI診断が実行できませんでした。",
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        "あなたは世界トップクラスのゴルフスイングコーチであり、AIドクターです。\n"
        "提供されたスイングの骨格データ（数値）に基づき、以下の構造で詳細な日本語の診断レポートを作成してください。\n"
        "専門用語（捻転、アーリーリリース等）は使用しつつ、直後に平易な言葉で補足して読みやすくしてください。\n"
        "数値に異常値が混じる場合は『計測エラーの可能性』に触れた上で、他の指標も使って診断を進めてください。\n\n"
        "【レポートの構造】\n"
        "## 02. データ評価基準（プロとの違い）\n"
        "## 03. 肩の回旋（上半身のねじり）\n"
        "## 04. 腰の回旋（下半身の動き）\n"
        "## 05. 手首のメカニクス（クラブを操る技術）\n"
        "## 06. 下半身の安定性（軸のブレ）\n"
        "## 07. 総合診断（一番の課題はここ！）\n"
        " - 07の冒頭で、まずお客様のポテンシャルを褒めるポジティブな一文を入れてください。\n"
        "## 08. 改善戦略とドリル（今日からできる練習法）\n"
        " - ここには『最重要課題に絞ったドリルを最大3つ』、Markdown箇条書きで『ドリル名と目的』のみを簡潔に書いてください。手順は書かないでください。\n"
        "## 10. まとめ（次のステップ）\n\n"
        "【骨格計測データ】\n"
        f"{json.dumps(raw_data, ensure_ascii=False, indent=2)}\n"
    )

    last_err: Optional[Exception] = None
    for model in choose_gemini_model():
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = getattr(resp, "text", "") or ""
            if not text.strip():
                raise RuntimeError(f"Empty response from model: {model}")
            return text, f"AIによる診断レポートが生成されました。（model: {model}）"
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            # 404/403/429 などを吸収して次候補へ
            last_err = e
            print("[Gemini] model failed:", model, str(e))
            continue
        except Exception as e:
            last_err = e
            print("[Gemini] unexpected error:", model, str(e))
            continue

    msg = "AI診断レポートの生成に失敗しました。利用可能なモデル名をご確認ください。"
    if last_err:
        msg += f"\n\n（最後のエラー）{type(last_err).__name__}: {str(last_err)[:300]}"
    return "## AI診断エラー\n" + msg, "AI診断が実行できませんでした。"


def analyze_swing_stub() -> Dict[str, Any]:
    """
    MediaPipe解析の代替（現状は確実に動くダミー）。
    既存のMediaPipe解析に置き換える場合は、この関数を差し替えるだけでOK。
    """
    return {
        "frame_count": 73,
        "max_shoulder_rotation": -23.8,
        "min_hip_rotation": -179.9,
        "max_wrist_cock": 179.6,
        "max_head_drift_x": 0.0264,
        "max_knee_sway_x": 0.0375,
    }


def create_cloud_task(report_id: str, user_id: str) -> str:
    if not tasks_client or not queue_path:
        raise RuntimeError("Cloud Tasks client is not initialized.")
    if not SERVICE_HOST_URL:
        raise RuntimeError("SERVICE_HOST_URL is missing.")
    if not TASK_SA_EMAIL:
        raise RuntimeError("TASK_SA_EMAIL is missing.")

    payload = json.dumps({"report_id": report_id, "user_id": user_id}).encode("utf-8")

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{SERVICE_HOST_URL}/worker/process_video",
            "headers": {"Content-Type": "application/json"},
            "body": payload,
            # OIDC 必須（Cloud Run 認証ON前提）
            "oidc_token": {
                "service_account_email": TASK_SA_EMAIL,
                # audience は「サービスURL（ホスト）」で通るのが一般的
                "audience": SERVICE_HOST_URL,
            },
        }
    }
    resp = tasks_client.create_task(parent=queue_path, task=task)
    return resp.name


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
        }
    )


@app.route("/webhook", methods=["POST"])
def webhook():
    # LINE署名検証
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

    # 1) Firestore: 初期保存
    firestore_safe_set(
        report_id,
        {
            "user_id": user_id,
            "status": "PROCESSING",
            "created_at": firestore.SERVER_TIMESTAMP if db else None,
            "is_premium": True if FORCE_PREMIUM else False,
            "plan_type": "free_preview" if FORCE_PREMIUM else "free",
            "summary": "動画解析を開始しました。",
        },
    )

    # 2) Cloud Tasks enqueue
    try:
        task_name = create_cloud_task(report_id=report_id, user_id=user_id)
        firestore_safe_update(report_id, {"task_name": task_name})
    except NotFound:
        # キューが無い
        firestore_safe_update(
            report_id,
            {
                "status": "TASK_QUEUE_NOT_FOUND",
                "summary": f"Cloud Tasks queue not found: {TASK_QUEUE_NAME} @ {TASK_QUEUE_LOCATION}",
            },
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

    # 3) 返信（あなたの「最初の丁寧な文面」）
    safe_line_reply(event.reply_token, make_initial_reply(report_id, plan_label="全機能プレビュー" if FORCE_PREMIUM else "無料版"))


@app.route("/worker/process_video", methods=["POST"])
def process_video_worker():
    """
    Cloud Tasks から呼ばれる Worker
    """
    started = now_ts()
    payload = request.get_json(silent=True) or {}
    report_id = payload.get("report_id")
    user_id = payload.get("user_id")

    if not report_id or not user_id:
        return jsonify({"status": "error", "message": "missing report_id or user_id"}), 400

    # IN_PROGRESS
    firestore_safe_update(report_id, {"status": "IN_PROGRESS", "summary": "動画解析を実行中です..."})

    try:
        # 解析（いまは安定ダミー。MediaPipe実装に置き換えてOK）
        raw_data = analyze_swing_stub()

        # 有料版フルレポートを常に生成（開発フェーズ最優先）
        is_premium = True if FORCE_PREMIUM else False

        ai_report_md, summary_text = run_gemini_full_report(raw_data, is_premium=is_premium)

        # COMPLETED 保存
        firestore_safe_update(
            report_id,
            {
                "status": "COMPLETED",
                "summary": summary_text,
                "raw_data": raw_data,
                "ai_report": ai_report_md,
                "is_premium": is_premium,
                "elapsed_sec": round(now_ts() - started, 2),
                "completed_at": firestore.SERVER_TIMESTAMP if db else None,
            },
        )

        # LINE push（完了）
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
                "elapsed_sec": round(now_ts() - started, 2),
            },
        )
        safe_line_push(user_id, "【解析エラー】動画解析が失敗しました。別角度や明るい場所で撮影してみてください。")
        # Cloud Tasks は200を返すとリトライしない（無限リトライを避ける）
        return jsonify({"status": "error", "message": "analysis failed"}), 200


@app.route("/api/report_data/<report_id>", methods=["GET"])
def api_report_data(report_id: str):
    if not db:
        return jsonify({"error": "Firestore is not initialized"}), 500

    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404

    data = doc.to_dict() or {}

    # Web側レンダリング用に整形
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
    # f-string禁止（JSの {} / ${} 事故防止のため完全固定HTML）
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GATE AIスイングドクター 診断レポート</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @media print { .no-print { display: none !important; } }
    .report-content h2 {
      font-size: 1.6rem;
      font-weight: 800;
      border-bottom: 3px solid #10b981;
      padding-bottom: 0.4rem;
      margin-top: 2rem;
      margin-bottom: 1rem;
    }
    .report-content h3 {
      font-size: 1.2rem;
      font-weight: 700;
      border-left: 5px solid #6ee7b7;
      padding-left: 0.8rem;
      margin-top: 1.4rem;
      margin-bottom: 0.8rem;
    }
    .card {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 0.75rem;
      padding: 1rem;
      text-align: center;
    }
    .card .k { font-size: 0.75rem; color: #6b7280; margin-bottom: 0.2rem; }
    .card .v { font-size: 1.5rem; font-weight: 800; color: #111827; }
    .md p { margin: 0 0 0.9rem 0; line-height: 1.65; color: #374151; }
    .md ul { margin: 0.8rem 0; padding: 0; list-style: none; }
    .md li { padding: 0.8rem; margin-bottom: 0.5rem; background: #ecfdf5; border-left: 5px solid #10b981; border-radius: 0.6rem; font-weight: 600; color: #065f46; }
  </style>
</head>
<body class="bg-gray-100 font-sans">
  <div class="max-w-4xl mx-auto p-4 md:p-8">
    <div class="bg-white rounded-lg shadow p-4 mb-4">
      <div class="text-2xl font-extrabold text-center text-emerald-600">GATE AIスイングドクター</div>
      <div class="text-sm text-gray-500 text-center mt-1">診断レポートID: <span id="rid"></span></div>
      <div class="text-sm text-gray-500 text-center mt-1">ステータス: <span id="status"></span></div>
      <div class="no-print text-right mt-3">
        <button onclick="window.print()" class="px-4 py-2 bg-emerald-600 text-white rounded-lg shadow hover:bg-emerald-700">📄 PDFとして保存 / 印刷</button>
      </div>
    </div>

    <div id="loading" class="bg-white rounded-lg shadow p-6 text-center text-gray-600">読み込み中...</div>

    <div id="main" class="hidden">
      <div class="bg-white rounded-lg shadow p-6 mb-6">
        <div class="text-xl font-bold mb-4">01. 骨格計測データ（AIが測った数値）</div>
        <div id="metrics" class="grid grid-cols-2 md:grid-cols-3 gap-3"></div>
      </div>

      <div class="bg-white rounded-lg shadow p-6 report-content md">
        <div class="text-xl font-bold mb-4">AIスイング診断レポート</div>
        <div id="report"></div>
      </div>
    </div>
  </div>

<script>
  const reportId = location.pathname.split("/").pop();
  document.getElementById("rid").innerText = reportId;

  function esc(s){ return String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

  function mdToHtml(md){
    let t = String(md || "");
    // 見出し
    t = t.replace(/^##\\s+(.*)$/gm, '<h2>$1</h2>');
    t = t.replace(/^###\\s+(.*)$/gm, '<h3>$1</h3>');
    // 箇条書き（- / *）
    t = t.replace(/^(?:\\s*[-*]\\s+.*(?:\\n|$))+?/gm, (block) => {
      const items = block.trim().split(/\\n/).map(line => line.replace(/^\\s*[-*]\\s+/, '').trim()).filter(Boolean);
      return "<ul>" + items.map(it => "<li>"+esc(it)+"</li>").join("") + "</ul>";
    });
    // 太字
    t = t.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
    // 改行→段落っぽく
    t = t.replace(/\\n\\n+/g, "</p><p>");
    t = "<p>" + t.replace(/\\n/g, "<br>") + "</p>";
    return t;
  }

  function card(title, value, unit){
    return `
      <div class="card">
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
        card("解析フレーム数", m.frame_count ?? "N/A", "") +
        card("最大肩回転", m.max_shoulder_rotation ?? "N/A", "°") +
        card("最小腰回転", m.min_hip_rotation ?? "N/A", "°") +
        card("最大コック角", m.max_wrist_cock ?? "N/A", "°") +
        card("最大頭ブレ(Sway)", m.max_head_drift_x ?? "N/A", "") +
        card("最大膝ブレ(Sway)", m.max_knee_sway_x ?? "N/A", "");

      const report = document.getElementById("report");
      const md = d.ai_report_text || ("(まだレポートが生成されていません)\\n\\nステータス: " + (d.status || "UNKNOWN"));
      report.innerHTML = mdToHtml(md);
    })
    .catch(() => {
      document.getElementById("loading").innerText = "読み込みに失敗しました。";
    });
</script>
</body>
</html>
"""


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    # Cloud Run は gunicorn 起動なので通常ここは使われないが、ローカル動作用に残す
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

