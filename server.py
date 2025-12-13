import os
import json
import time
import math
import shutil
import tempfile
import traceback
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import ffmpeg
import cv2
import mediapipe as mp

from flask import Flask, request, abort, jsonify, render_template_string

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore, tasks_v2
from google.api_core.exceptions import NotFound, PermissionDenied
from google import genai
from google.genai import errors as genai_errors

# 日付処理用
from datetime import datetime, timezone, timedelta

# ==================================================
# ENV
# ==================================================
# エラー回避のため .get() を使用し、デフォルト値を設定
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")

# Cloud Tasks (asia-northeast2 固定)
TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")

# 開発用: 常に有料版として振る舞うか
FORCE_PREMIUM = True

# ==================================================
# App init
# ==================================================
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# クライアント初期化
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
def firestore_set(report_id: str, data: Dict[str, Any]):
    if db:
        try:
            db.collection("reports").document(report_id).set(data, merge=True)
        except Exception:
            print(traceback.format_exc())

def firestore_update(report_id: str, data: Dict[str, Any]):
    if db:
        try:
            db.collection("reports").document(report_id).update(data)
        except Exception:
            print(traceback.format_exc())

def safe_reply(token: str, text: str):
    if line_bot_api:
        try:
            line_bot_api.reply_message(token, TextSendMessage(text=text))
        except Exception as e:
            print(f"LINE Reply Error: {e}")

def safe_push(user_id: str, text: str):
    if line_bot_api:
        try:
            line_bot_api.push_message(user_id, TextSendMessage(text=text))
        except Exception as e:
            print(f"LINE Push Error: {e}")

# ==================================================
# Video Processing
# ==================================================
def download_line_video_to_file(message_id: str, out_path: str) -> None:
    if not line_bot_api:
        raise RuntimeError("LINE API not configured")
    content = line_bot_api.get_message_content(message_id)
    with open(out_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

def transcode_to_mp4(in_path: str, out_path: str) -> None:
    try:
        ffmpeg.input(in_path).output(
            out_path, vcodec="libx264", acodec="aac", preset="veryfast", crf=28, vf="scale='min(640,iw)':-2"
        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print(f"FFmpeg Error: {e.stderr.decode() if e.stderr else str(e)}")
        raise

# ==================================================
# MediaPipe Analysis (Real Logic)
# ==================================================
def _rot_deg(lx, ly, rx, ry):
    return math.degrees(math.atan2(ry - ly, rx - lx))

def _angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 0
    cos = np.dot(v1, v2) / norm
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def analyze_swing(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_count = 0
    max_shoulder_rot = -180
    min_hip_rot = 180
    max_wrist_cock = 0
    head_start_x = None
    max_head_drift = 0
    knee_start_x = None
    max_knee_sway = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # RGB変換して解析
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            
            # ランドマーク取得
            L_SH = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            R_SH = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            L_HIP = lm[mp_pose.PoseLandmark.LEFT_HIP]
            R_HIP = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            R_ELB = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            R_WRI = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            R_IND = lm[mp_pose.PoseLandmark.RIGHT_INDEX]
            NOSE = lm[mp_pose.PoseLandmark.NOSE]
            L_KNEE = lm[mp_pose.PoseLandmark.LEFT_KNEE]
            R_KNEE = lm[mp_pose.PoseLandmark.RIGHT_KNEE]

            # 1. 回転角
            sh_rot = _rot_deg(L_SH.x, L_SH.y, R_SH.x, R_SH.y)
            max_shoulder_rot = max(max_shoulder_rot, sh_rot)
            
            hip_rot = _rot_deg(L_HIP.x, L_HIP.y, R_HIP.x, R_HIP.y)
            min_hip_rot = min(min_hip_rot, hip_rot)

            # 2. コック角 (右腕)
            cock = _angle((R_ELB.x, R_ELB.y), (R_WRI.x, R_WRI.y), (R_IND.x, R_IND.y))
            max_wrist_cock = max(max_wrist_cock, cock)

            # 3. ブレ (Sway)
            if head_start_x is None: head_start_x = NOSE.x
            max_head_drift = max(max_head_drift, abs(NOSE.x - head_start_x))

            k_center = (L_KNEE.x + R_KNEE.x) / 2
            if knee_start_x is None: knee_start_x = k_center
            max_knee_sway = max(max_knee_sway, abs(k_center - knee_start_x))

    cap.release()
    
    return {
        "frame_count": frame_count,
        "max_shoulder_rotation": round(max_shoulder_rot, 1),
        "min_hip_rotation": round(min_hip_rot, 1),
        "max_wrist_cock": round(max_wrist_cock, 1),
        "max_head_drift_x": round(max_head_drift, 4),
        "max_knee_sway_x": round(max_knee_sway, 4)
    }

# ==================================================
# Gemini Logic (Markdown出力)
# ==================================================
def run_gemini_full_report(raw: Dict[str, Any]) -> str:
    if not GEMINI_API_KEY:
        return "## エラー\nAPIキーが設定されていません。"

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        # JSONではなくMarkdownでの出力を指示（HTML側の表示ロジックと一致させるため）
        prompt = f"""
あなたはゴルフスイング解析AIです。以下の骨格データをもとに、Markdown形式で診断レポートを作成してください。

【構成ルール】
1. ## 02. データ評価基準
2. ## 03. 肩の回旋
3. ## 04. 腰の回旋
4. ## 05. 手首のメカニクス
5. ## 06. 下半身の安定性
6. ## 07. 総合診断
7. ## 08. 改善戦略とドリル
   - 箇条書きで3つまで
8. ## 09. フィッティング提案
   - 表形式で出力（フレックス、重量、調子、トルク）
9. ## 10. まとめ

※ 01. 骨格計測データはシステム側で表示するため、AI出力には含めないでください。

【データ】
{json.dumps(raw, ensure_ascii=False, indent=2)}
"""
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "## エラー\nAI診断中にエラーが発生しました。"

# ==================================================
# Webhook
# ==================================================
@app.route("/webhook", methods=["POST"])
def webhook():
    if not handler: return "Config Error", 500
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

    # 初期保存
    firestore_set(report_id, {
        "status": "PROCESSING",
        "user_id": user_id,
        "is_premium": True, # 仮
        "created_at": firestore.SERVER_TIMESTAMP if db else None
    })

    # Cloud Tasks 登録
    if tasks_client and queue_path and SERVICE_HOST_URL and TASK_SA_EMAIL:
        try:
            payload = json.dumps({"report_id": report_id, "user_id": user_id, "message_id": message_id}).encode()
            task = {
                "http_request": {
                    "http_method": tasks_v2.HttpMethod.POST,
                    "url": f"{SERVICE_HOST_URL}/worker/process_video",
                    "headers": {"Content-Type": "application/json"},
                    "body": payload,
                    "oidc_token": {"service_account_email": TASK_SA_EMAIL}
                }
            }
            tasks_client.create_task(parent=queue_path, task=task)
        except Exception as e:
            print(f"Task Create Error: {e}")
            safe_reply(event.reply_token, "システムエラー: 解析の開始に失敗しました。")
            return
    
    report_url = f"{SERVICE_HOST_URL}/report/{report_id}"
    safe_reply(event.reply_token, f"✅ 動画を受信しました。解析を開始します！\n\n【処理状況確認URL】\n{report_url}")

# ==================================================
# Worker
# ==================================================
@app.route("/worker/process_video", methods=["POST"])
def process_video_worker():
    payload = request.json
    report_id = payload.get("report_id")
    user_id = payload.get("user_id")
    message_id = payload.get("message_id")

    if not report_id: return jsonify({"error": "no report_id"}), 400

    firestore_update(report_id, {"status": "IN_PROGRESS", "summary": "動画解析中..."})

    temp_dir = tempfile.mkdtemp()
    try:
        raw_path = os.path.join(temp_dir, "raw")
        mp4_path = os.path.join(temp_dir, "input.mp4")
        
        download_line_video_to_file(message_id, raw_path)
        transcode_to_mp4(raw_path, mp4_path)
        
        # 解析実行
        raw = analyze_swing(mp4_path)
        # 診断生成 (Markdownテキストとして取得)
        report_text = run_gemini_full_report(raw)

        firestore_update(report_id, {
            "status": "COMPLETED",
            "raw_data": raw,
            "report": report_text # テキストとして保存
        })

        if user_id:
            safe_push(user_id, f"🎉 AIスイング診断が完了しました！\n\n【診断レポートURL】\n{SERVICE_HOST_URL}/report/{report_id}")
        
    except Exception as e:
        print(f"Worker Error: {e}")
        firestore_update(report_id, {"status": "FAILED", "summary": str(e)})
        safe_push(user_id, "解析中にエラーが発生しました。")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return jsonify({"ok": True})

# ==================================================
# API & Report View
# ==================================================
@app.route("/api/report_data/<rid>")
def api_report(rid):
    if not db: return jsonify({"error": "DB error"}), 500
    doc = db.collection("reports").document(rid).get()
    if not doc.exists: return jsonify({"error": "not found"}), 404
    return jsonify(doc.to_dict())

@app.route("/report/<rid>")
def report_view(rid):
    # HTML: Markdownを解析してきれいに表示するロジックを搭載
    html = """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GATE AIスイングドクター</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @media print { .no-print { display: none !important; } }
    body { background-color: #f3f4f6; color: #1f2937; }
    .card { background: white; border-radius: 0.75rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    h2 { font-size: 1.5rem; font-weight: 800; color: #064e3b; border-bottom: 3px solid #10b981; padding-bottom: 0.5rem; margin-bottom: 1rem; margin-top: 1rem; }
    h3 { font-size: 1.1rem; font-weight: 700; color: #374151; border-left: 5px solid #34d399; padding-left: 0.75rem; margin-top: 1.25rem; margin-bottom: 0.75rem; }
    ul { list-style: none; padding: 0; }
    ul li { background-color: #ecfdf5; border-left: 4px solid #10b981; padding: 0.75rem; margin-bottom: 0.5rem; border-radius: 0.375rem; font-weight: 500; color: #065f46; }
    
    /* マークダウンテーブルのスタイル */
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 1rem; }
    th { background: #f3f4f6; color: #374151; font-weight: 700; padding: 0.75rem; border: 1px solid #d1d5db; }
    td { padding: 0.75rem; border: 1px solid #d1d5db; vertical-align: top; }

    .metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; }
    .metric-box { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 0.75rem; text-align: center; }
    .metric-label { font-size: 0.7rem; color: #6b7280; font-weight: 700; margin-bottom: 0.25rem; }
    .metric-val { font-size: 1.25rem; font-weight: 900; color: #111827; }
  </style>
</head>
<body>
  <div class="max-w-3xl mx-auto p-4 md:p-8">
    <div class="card">
        <h1 class="text-2xl font-black text-center text-emerald-600 mb-2">GATE AIスイングドクター</h1>
        <div class="text-center text-sm text-gray-500">ID: <span id="rid"></span></div>
    </div>
    <div id="loading" class="text-center py-20 text-gray-500">レポート生成中...</div>

    <div id="main-content" class="hidden">
        <!-- 骨格データ -->
        <div class="card">
            <h2>01. 骨格計測データ</h2>
            <div id="metrics" class="metric-grid"></div>
        </div>
        <!-- AIレポート -->
        <div id="ai-sections" class="card"></div>
    </div>
  </div>

<script>
  const reportId = "__REPORT_ID__";
  document.getElementById("rid").innerText = reportId;

  function esc(s){ return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

  // Markdown表をHTMLテーブルに変換
  function renderTable(md) {
     const rows = md.trim().split("\\n");
     let html = "<table>";
     let isHeader = true;
     rows.forEach((row, i) => {
        if(row.includes("---")) return; // 区切り線スキップ
        const cols = row.split("|").filter(c => c.trim() !== "");
        if(cols.length === 0) return;
        
        html += "<tr>";
        cols.forEach(col => {
            html += isHeader ? `<th>${esc(col.trim())}</th>` : `<td>${esc(col.trim())}</td>`;
        });
        html += "</tr>";
        if(isHeader) isHeader = false;
     });
     return html + "</table>";
  }

  // MarkdownテキストをHTMLに変換（見出し、リスト、表）
  function mdToHtml(md) {
      let html = "";
      const lines = md.split("\\n");
      let inList = false;
      let inTable = false;
      let tableBuffer = "";

      lines.forEach(line => {
          line = line.trim();
          
          // 表処理
          if(line.startsWith("|")) {
              tableBuffer += line + "\\n";
              inTable = true;
              return;
          } else if(inTable) {
              html += renderTable(tableBuffer);
              inTable = false;
              tableBuffer = "";
          }

          // 見出し
          if(line.startsWith("## ")) {
              html += `<h2>${esc(line.substring(3))}</h2>`;
          } else if(line.startsWith("### ")) {
              html += `<h3>${esc(line.substring(4))}</h3>`;
          } 
          // リスト
          else if(line.startsWith("- ") || line.startsWith("* ")) {
              if(!inList) { html += "<ul>"; inList = true; }
              html += `<li>${esc(line.substring(2))}</li>`;
          } else {
              if(inList) { html += "</ul>"; inList = false; }
              if(line.length > 0) html += `<p class="mb-4">${esc(line)}</p>`;
          }
      });
      if(inList) html += "</ul>";
      if(inTable) html += renderTable(tableBuffer);
      
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
        <div class="metric-box"><div class="metric-label">頭ブレ</div><div class="metric-val">${esc(m.max_head_drift_x)}</div></div>
        <div class="metric-box"><div class="metric-label">膝ブレ</div><div class="metric-val">${esc(m.max_knee_sway_x)}</div></div>
        <div class="metric-box"><div class="metric-label">フレーム</div><div class="metric-val">${esc(m.frame_count)}</div></div>
      `;

      // 2. AIレポート
      const reportText = data.report || "レポート生成待ち...";
      
      document.getElementById("ai-sections").innerHTML = mdToHtml(reportText);
    })
    .catch(e => {
        document.getElementById("loading").innerText = "読み込みエラー";
        console.error(e);
    });
</script>
</body>
</html>
"""
    return html.replace("__REPORT_ID__", rid)

# ==================================================
# Flask実行
# ==================================================
@app.route("/NotificationContent.js")
def dummy_notification_js():
    return "", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
