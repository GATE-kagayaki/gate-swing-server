import os
import json
import time
import traceback
from typing import Dict, Any

from flask import Flask, request, abort, jsonify, render_template_string

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore, tasks_v2
from google import genai

# ==================================================
# ENV
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")

FORCE_PREMIUM = True  # テスト中は常に有料版

# ==================================================
# INIT
# ==================================================
app = Flask(__name__)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None

db = firestore.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else None
tasks_client = tasks_v2.CloudTasksClient() if GCP_PROJECT_ID else None

queue_path = None
if tasks_client and GCP_PROJECT_ID:
    queue_path = tasks_client.queue_path(
        GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME
    )

genai_client = None
if GEMINI_API_KEY:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ==================================================
# UTIL
# ==================================================
def extract_json_object(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(s)
    except Exception:
        # JSONの一部抽出を試みる
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1:
            # 失敗時は空のデータを返すのではなくエラー用構造を返す
            return {"section07": {"title": "解析エラー", "stable_points": ["AI応答の解析に失敗しました"], "improvement_points": []}}
        return json.loads(s[start:end + 1])


def send_line_reply(token: str, text: str):
    if line_bot_api:
        try:
            line_bot_api.reply_message(token, TextSendMessage(text=text))
        except Exception:
            print(traceback.format_exc())


def send_line_push(user_id: str, text: str):
    if line_bot_api:
        try:
            line_bot_api.push_message(user_id, TextSendMessage(text=text))
        except Exception:
            print(traceback.format_exc())


# ==================================================
# ANALYSIS (MediaPipe Stub)
# ==================================================
def analyze_swing() -> Dict[str, Any]:
    return {
        "frame_count": 73,
        "max_shoulder_rotation": -23.8,
        "min_hip_rotation": -179.9,
        "max_wrist_cock": 179.6,
        "max_head_sway": 0.0264,
        "max_knee_sway": 0.0375
    }


# ==================================================
# GEMINI
# ==================================================
def generate_report_json(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    if not genai_client:
        return {}

    prompt = f"""
あなたはゴルフスイング解析AIです。
以下のJSONスキーマと完全一致するJSONのみを出力してください。
文章や説明は一切不要です。

【骨格分析データ】
{json.dumps(raw_data, ensure_ascii=False)}

【出力JSONスキーマ】
{{
  "section02": {{ "title": "データ評価基準", "analysis": ["..."] }},
  "section03": {{ "title": "肩の回旋", "analysis": ["..."] }},
  "section04": {{ "title": "腰の回旋", "analysis": ["..."] }},
  "section05": {{ "title": "手首のメカニクス", "analysis": ["..."] }},
  "section06": {{ "title": "下半身の安定性", "analysis": ["..."] }},
  "section07": {{
    "title": "総合診断",
    "stable_points": ["..."],
    "improvement_points": ["..."]
  }},
  "section08": {{
    "title": "改善戦略とドリル",
    "drills": [
      {{ "name": "...", "howto": ["..."] }}
    ]
  }},
  "section09": {{
    "title": "フィッティング提案",
    "table": [
      {{ "item": "...", "recommendation": "...", "reason": "..." }}
    ],
    "disclaimer": "..."
  }},
  "section10": {{
    "title": "まとめ",
    "text": "..."
  }}
}}
"""
    try:
        # 安定板モデルを指定
        res = genai_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return extract_json_object(res.text)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {}


# ==================================================
# ROUTES
# ==================================================
@app.route("/webhook", methods=["POST"])
def webhook():
    if not handler:
        return "Config Error", 500
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event):
    user_id = event.source.user_id
    message_id = event.message.id
    report_id = f"{user_id}_{message_id}"

    if db:
        db.collection("reports").document(report_id).set({
            "status": "PROCESSING",
            "created_at": firestore.SERVER_TIMESTAMP
        })

    # Cloud Tasks 登録
    if tasks_client and queue_path:
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{SERVICE_HOST_URL}/worker",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "report_id": report_id,
                    "user_id": user_id
                }).encode(),
                "oidc_token": {
                    "service_account_email": TASK_SA_EMAIL,
                    "audience": SERVICE_HOST_URL
                }
            }
        }
        try:
            tasks_client.create_task(parent=queue_path, task=task)
        except Exception as e:
            print(f"Task creation failed: {e}")

    send_line_reply(
        event.reply_token,
        "✅ 動画を受信しました。解析を開始します。\n完了後に通知します。"
    )


@app.route("/worker", methods=["POST"])
def worker():
    payload = request.json
    report_id = payload.get("report_id")
    user_id = payload.get("user_id")

    raw = analyze_swing()
    report_json = generate_report_json(raw)

    if db:
        db.collection("reports").document(report_id).update({
            "status": "COMPLETED",
            "raw_data": raw,
            "report": report_json
        })

    send_line_push(
        user_id,
        f"🎉 AIスイング診断が完了しました！\n{SERVICE_HOST_URL}/report/{report_id}"
    )

    return jsonify({"ok": True})


@app.route("/api/report_data/<report_id>")
def api_report(report_id):
    if not db:
        return jsonify({"error": "DB error"}), 500
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    return jsonify(doc.to_dict())


# ==================================================
# Web レポート (HTML埋め込み版)
# ==================================================
@app.route("/report/<report_id>")
def report_view(report_id):
    # ここにHTMLを埋め込みます
    html = """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GATE AIスイングドクター</title>
  <script src="[https://cdn.tailwindcss.com](https://cdn.tailwindcss.com)"></script>
  <style>
    @media print { .no-print { display: none !important; } }
    body { background-color: #f3f4f6; color: #1f2937; }
    .card { background: white; border-radius: 0.75rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    h2 { font-size: 1.5rem; font-weight: 800; color: #064e3b; border-bottom: 3px solid #10b981; padding-bottom: 0.5rem; margin-bottom: 1rem; margin-top: 0.5rem; }
    h3 { font-size: 1.1rem; font-weight: 700; color: #374151; border-left: 5px solid #34d399; padding-left: 0.75rem; margin-top: 1.25rem; margin-bottom: 0.75rem; }
    ul { list-style: none; padding: 0; }
    ul li { background-color: #ecfdf5; border-left: 4px solid #10b981; padding: 0.75rem; margin-bottom: 0.5rem; border-radius: 0.375rem; font-weight: 500; color: #065f46; }
    .metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; }
    .metric-box { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 0.75rem; text-align: center; }
    .metric-label { font-size: 0.7rem; color: #6b7280; font-weight: 700; margin-bottom: 0.25rem; }
    .metric-val { font-size: 1.25rem; font-weight: 900; color: #111827; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 1rem; }
    th { background: #f3f4f6; color: #374151; font-weight: 700; padding: 0.75rem; border: 1px solid #d1d5db; }
    td { padding: 0.75rem; border: 1px solid #d1d5db; vertical-align: top; }
  </style>
</head>
<body>
  <div class="max-w-3xl mx-auto p-4 md:p-8">
    <div class="card">
        <h1 class="text-2xl font-black text-center text-emerald-600 mb-2">GATE AIスイングドクター</h1>
        <div class="text-center text-sm text-gray-500">ID: <span id="rid"></span></div>
    </div>
    
    <div id="loading" class="text-center py-10 text-gray-500">レポート読み込み中...</div>

    <div id="main-content" class="hidden">
        <!-- 骨格データ -->
        <div class="card">
            <h2>01. 骨格計測データ</h2>
            <div id="metrics" class="metric-grid"></div>
        </div>
        <!-- AIレポート -->
        <div id="ai-sections"></div>
    </div>
  </div>

<script>
  const reportId = "__REPORT_ID__";
  document.getElementById("rid").innerText = reportId;

  function esc(s){ return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
  function renderList(items) {
      if(!items || items.length === 0) return "";
      return `<ul>${items.map(i => `<li>${esc(i)}</li>`).join('')}</ul>`;
  }
  function renderTable(tableData) {
      if(!tableData || tableData.length === 0) return "";
      let html = '<table><thead><tr><th>項目</th><th>推奨</th><th>理由</th></tr></thead><tbody>';
      tableData.forEach(row => {
          html += `<tr><td class="font-bold">${esc(row.item)}</td><td>${esc(row.recommendation)}</td><td>${esc(row.reason)}</td></tr>`;
      });
      html += '</tbody></table>';
      return html;
  }
  function renderDrills(drills) {
      if(!drills || drills.length === 0) return "";
      let html = '<div class="space-y-4">';
      drills.forEach(d => {
          html += `<div class="bg-white border border-gray-200 rounded-lg p-4 shadow-sm"><div class="font-bold text-lg text-emerald-700 mb-1">⛳ ${esc(d.name)}</div><div class="text-sm text-gray-700 bg-gray-50 p-2 rounded">${renderList(d.howto)}</div></div>`;
      });
      html += '</div>';
      return html;
  }

  fetch("/api/report_data/" + reportId)
    .then(r => r.json())
    .then(data => {
      document.getElementById("loading").classList.add("hidden");
      document.getElementById("main-content").classList.remove("hidden");

      // 1. Metrics
      const m = data.raw_data || {};
      const metricsDiv = document.getElementById("metrics");
      metricsDiv.innerHTML = `
        <div class="metric-box"><div class="metric-label">肩回転</div><div class="metric-val">${esc(m.max_shoulder_rotation)}°</div></div>
        <div class="metric-box"><div class="metric-label">腰回転</div><div class="metric-val">${esc(m.min_hip_rotation)}°</div></div>
        <div class="metric-box"><div class="metric-label">コック</div><div class="metric-val">${esc(m.max_wrist_cock)}°</div></div>
        <div class="metric-box"><div class="metric-label">頭ブレ</div><div class="metric-val">${esc(m.max_head_sway)}</div></div>
        <div class="metric-box"><div class="metric-label">膝ブレ</div><div class="metric-val">${esc(m.max_knee_sway)}</div></div>
        <div class="metric-box"><div class="metric-label">フレーム</div><div class="metric-val">${esc(m.frame_count)}</div></div>
      `;

      // 2. AI Sections
      const json = data.report || {};
      const container = document.getElementById("ai-sections");
      const keys = ["section02", "section03", "section04", "section05", "section06", "section07", "section08", "section09", "section10"];
      
      keys.forEach(key => {
          const sec = json[key];
          if(!sec) return;
          const div = document.createElement("div");
          div.className = "card";
          let html = "";
          if(sec.title) html += `<h2>${key.replace("section", "")}. ${esc(sec.title)}</h2>`;
          if(sec.text) html += `<p>${esc(sec.text)}</p>`;
          if(sec.analysis) html += renderList(sec.analysis);
          if(sec.stable_points) html += `<h3>安定している点</h3>` + renderList(sec.stable_points);
          if(sec.improvement_points) html += `<h3>改善が期待される点</h3>` + renderList(sec.improvement_points);
          if(sec.drills) html += renderDrills(sec.drills);
          if(sec.table) html += renderTable(sec.table);
          if(sec.disclaimer) html += `<div class="text-xs text-gray-400 mt-2 text-right">${esc(sec.disclaimer)}</div>`;
          div.innerHTML = html;
          container.appendChild(div);
      });
    })
    .catch(e => {
        document.getElementById("loading").innerText = "読み込みエラー";
        console.error(e);
    });
</script>
</body>
</html>
    """
    return html.replace("__REPORT_ID__", report_id)


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

