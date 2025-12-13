import os
import json
import time
import math
import shutil
import tempfile
import traceback
from typing import Any, Dict

import numpy as np
import ffmpeg
import cv2
import mediapipe as mp

from flask import Flask, request, abort, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage, FlexSendMessage
from google.cloud import firestore, tasks_v2
from google import genai

# ==================================================
# ENV & INIT
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")
TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None
db = firestore.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else None
tasks_client = tasks_v2.CloudTasksClient() if GCP_PROJECT_ID else None
queue_path = tasks_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME) if tasks_client else None

# ==================================================
# Helpers
# ==================================================
def firestore_update(report_id: str, data: Dict[str, Any], merge=True):
    if db:
        try:
            db.collection("reports").document(report_id).set(data, merge=merge)
        except Exception:
            print(traceback.format_exc())

def safe_push(user_id: str, text: str):
    if line_bot_api:
        try:
            line_bot_api.push_message(user_id, TextSendMessage(text=text))
        except Exception as e:
            print(f"Push Error: {e}")

def create_flex_report(report_id, raw_data):
    """分析完了時のリッチなメッセージを作成"""
    shoulder = raw_data.get("max_shoulder_rotation", 0)
    hip = raw_data.get("min_hip_rotation", 0)
    cock = raw_data.get("max_wrist_cock", 0)
    
    # 簡易スコア算出 (演出用)
    twist = shoulder - hip
    score = min(100, max(40, int(twist * 1.2))) if shoulder > 0 else 0
    report_url = f"{SERVICE_HOST_URL}/report/{report_id}"

    return FlexSendMessage(
        alt_text="スイング診断レポートが届きました",
        contents={
            "type": "bubble", "size": "mega",
            "header": {
                "type": "box", "layout": "vertical", "backgroundColor": "#065f46", "paddingAll": "20px",
                "contents": [
                    {"type": "text", "text": "AI SWING ANALYSIS", "color": "#ffffff", "weight": "bold", "size": "xxs", "align": "center", "letterSpacing": "1px"},
                    {"type": "text", "text": str(score), "color": "#ffffff", "weight": "bold", "size": "5xl", "align": "center", "margin": "md"},
                    {"type": "text", "text": "TOTAL SCORE", "color": "#a7f3d0", "size": "xxs", "align": "center"}
                ]
            },
            "body": {
                "type": "box", "layout": "vertical",
                "contents": [
                    {"type": "text", "text": "計測データ要約", "weight": "bold", "size": "sm", "color": "#888888", "margin": "md"},
                    {"type": "separator", "margin": "sm"},
                    {"type": "box", "layout": "horizontal", "margin": "md", "contents": [
                        {"type": "text", "text": "肩の回転", "size": "sm", "color": "#555555", "flex": 1},
                        {"type": "text", "text": f"{shoulder}°", "size": "sm", "color": "#111111", "weight": "bold", "align": "end"}
                    ]},
                    {"type": "box", "layout": "horizontal", "margin": "sm", "contents": [
                        {"type": "text", "text": "腰の回転", "size": "sm", "color": "#555555", "flex": 1},
                        {"type": "text", "text": f"{hip}°", "size": "sm", "color": "#111111", "weight": "bold", "align": "end"}
                    ]},
                ]
            },
            "footer": {
                "type": "box", "layout": "vertical", "paddingAll": "20px",
                "contents": [{"type": "button", "style": "primary", "color": "#059669", "action": {"type": "uri", "label": "詳細レポートを見る", "uri": report_url}}]
            }
        }
    )

# ==================================================
# MediaPipe Analysis
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # RGB変換
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            # 座標取得
            L_SH, R_SH = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            L_HIP, R_HIP = lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.RIGHT_HIP]
            R_ELB, R_WRI, R_IND = lm[mp_pose.PoseLandmark.RIGHT_ELBOW], lm[mp_pose.PoseLandmark.RIGHT_WRIST], lm[mp_pose.PoseLandmark.RIGHT_INDEX]
            NOSE = lm[mp_pose.PoseLandmark.NOSE]
            L_KNEE, R_KNEE = lm[mp_pose.PoseLandmark.LEFT_KNEE], lm[mp_pose.PoseLandmark.RIGHT_KNEE]

            # 計測ロジック
            sh_rot = _rot_deg(L_SH.x, L_SH.y, R_SH.x, R_SH.y)
            max_shoulder_rot = max(max_shoulder_rot, sh_rot)
            
            hip_rot = _rot_deg(L_HIP.x, L_HIP.y, R_HIP.x, R_HIP.y)
            min_hip_rot = min(min_hip_rot, hip_rot)
            
            cock = _angle((R_ELB.x, R_ELB.y), (R_WRI.x, R_WRI.y), (R_IND.x, R_IND.y))
            max_wrist_cock = max(max_wrist_cock, cock)

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
# Gemini Logic
# ==================================================
def run_gemini_full_report(raw: Dict[str, Any]) -> str:
    if not GEMINI_API_KEY: return "## エラー\nAPIキー未設定"
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""
あなたはゴルフスイング解析のプロです。以下のデータからMarkdown形式で診断レポートを作成してください。
読み手がポジティブになれるよう、最後に必ず「改善ドリル」を提案してください。
【データ】{json.dumps(raw)}
"""
        response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "## エラー\nAI診断に失敗しました。"

# ==================================================
# Routes: Webhook & Worker
# ==================================================
@app.route("/webhook", methods=["POST"])
def webhook():
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

    # Firestore初期化
    firestore_update(report_id, {
        "status": "PROCESSING", "user_id": user_id, "created_at": firestore.SERVER_TIMESTAMP
    })

    # Cloud Tasksへ投げる
    if tasks_client and queue_path:
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
        
        # 受信応答
        url = f"{SERVICE_HOST_URL}/report/{report_id}"
        if line_bot_api:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"🎥 動画を受け取りました。\nAIが詳細分析を開始します... (約1-2分)\n\n進捗はこちら:\n{url}"))

@app.route("/worker/process_video", methods=["POST"])
def process_video_worker():
    payload = request.json
    report_id = payload.get("report_id")
    user_id = payload.get("user_id")
    message_id = payload.get("message_id")

    firestore_update(report_id, {"status": "IN_PROGRESS"})
    
    temp_dir = tempfile.mkdtemp()
    try:
        # 動画ダウンロード & 変換
        content = line_bot_api.get_message_content(message_id)
        raw_path = os.path.join(temp_dir, "raw")
        mp4_path = os.path.join(temp_dir, "input.mp4")
        with open(raw_path, "wb") as f:
            for chunk in content.iter_content(): f.write(chunk)
        
        # ffmpeg変換
        ffmpeg.input(raw_path).output(mp4_path, vcodec="libx264", acodec="aac", preset="veryfast", crf=28).run(quiet=True, overwrite_output=True)

        # 解析 & AI生成
        raw = analyze_swing(mp4_path)
        report_text = run_gemini_full_report(raw)

        # 保存
        firestore_update(report_id, {
            "status": "COMPLETED", "raw_data": raw, "report": report_text
        })

        # ★Flex MessageでPush通知
        if user_id:
            try:
                flex_msg = create_flex_report(report_id, raw)
                line_bot_api.push_message(user_id, flex_msg)
            except Exception as e:
                print(f"Flex Error: {e}")
                safe_push(user_id, f"解析完了: {SERVICE_HOST_URL}/report/{report_id}")

    except Exception as e:
        print(f"Worker Error: {e}")
        firestore_update(report_id, {"status": "FAILED"})
        if user_id: safe_push(user_id, "解析中にエラーが発生しました。")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return jsonify({"ok": True})

# ==================================================
# Routes: Web Report (Chart.js + Tailwind)
# ==================================================
@app.route("/api/report_data/<rid>")
def api_report(rid):
    if not db: return jsonify({"error": "DB error"}), 500
    doc = db.collection("reports").document(rid).get()
    return jsonify(doc.to_dict()) if doc.exists else (jsonify({"error": "not found"}), 404)

@app.route("/report/<rid>")
def report_view(rid):
    html = """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>スイング診断レポート</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700;900&display=swap');
    body { font-family: 'Noto Sans JP', sans-serif; background-color: #f0fdf4; color: #1f2937; }
    .glass { background: rgba(255,255,255,0.95); border-radius: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    /* Markdown Style */
    #ai-content h2 { font-size: 1.25rem; font-weight: 800; color: #065f46; border-bottom: 2px solid #34d399; margin-top: 1.5rem; margin-bottom: 0.5rem; }
    #ai-content ul { list-style: disc; padding-left: 1.25rem; margin-bottom: 1rem; }
    #ai-content p { margin-bottom: 0.75rem; }
  </style>
</head>
<body class="pb-10">
  <div class="bg-gradient-to-r from-emerald-800 to-emerald-600 text-white p-6 shadow-lg text-center">
    <h1 class="text-xl font-bold tracking-widest uppercase opacity-80">AI Swing Doctor</h1>
    <div class="mt-2 text-5xl font-black" id="score">--</div>
    <div class="text-xs font-medium text-emerald-100">TOTAL SCORE</div>
  </div>
  <div class="max-w-md mx-auto px-4 -mt-6">
    <div class="glass p-4 mb-4">
      <div class="relative h-64 w-full"><canvas id="radarChart"></canvas></div>
    </div>
    <div class="glass p-6 min-h-[200px]">
       <div id="loader" class="text-center py-10 text-gray-400 animate-pulse">Analyzing...</div>
       <div id="ai-content" class="hidden text-sm text-gray-700 leading-relaxed"></div>
    </div>
  </div>
<script>
  const rid = "__RID__";
  fetch(`/api/report_data/${rid}`).then(r=>r.json()).then(d=>{
    const r = d.raw_data || {};
    // Score
    const twist = (r.max_shoulder_rotation||0) - (r.min_hip_rotation||0);
    const score = Math.min(100, Math.max(40, Math.floor(twist * 1.2)));
    document.getElementById('score').innerText = score;
    // Chart
    new Chart(document.getElementById('radarChart'), {
      type: 'radar',
      data: {
        labels: ['捻転差','コック','安定(頭)','安定(膝)','インパクト'],
        datasets: [{
          label: 'あなた', data: [Math.min(100, twist), Math.min(100, (r.max_wrist_cock||0)*0.8), Math.max(0,100-(r.max_head_drift_x||0)*200), Math.max(0,100-(r.max_knee_sway_x||0)*200), score],
          backgroundColor: 'rgba(5,150,105,0.2)', borderColor: '#059669', borderWidth: 2, pointBackgroundColor: '#059669'
        }, {
            label: 'プロ平均', data: [90, 85, 90, 90, 95], borderColor: '#ccc', borderDash: [5,5], borderWidth: 1, pointRadius: 0
        }]
      },
      options: { scales: { r: { min: 0, max: 100, ticks: { display: false } } } }
    });
    // AI Text
    document.getElementById('loader').classList.add('hidden');
    document.getElementById('ai-content').classList.remove('hidden');
    document.getElementById('ai-content').innerHTML = marked.parse(d.report || "No Report");
  });
</script>
</body>
</html>
    """
    return html.replace("__RID__", rid)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
