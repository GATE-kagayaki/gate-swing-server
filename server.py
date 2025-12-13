import os
import json
import time
import math
import shutil
import traceback
import tempfile
import numpy as np # 数値計算用
from typing import Any, Dict

from flask import Flask, request, abort, jsonify

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent, VideoMessage, FileMessage, TextSendMessage
)

from google.cloud import firestore, tasks_v2

# ==================================================
# ENV & CONFIG
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")

TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")
TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")
TASK_QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")

# ==================================================
# APP INIT
# ==================================================
app = Flask(__name__)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

db = firestore.Client(project=GCP_PROJECT_ID)
tasks_client = tasks_v2.CloudTasksClient()
queue_path = tasks_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)

# ==================================================
# [LOGIC] SWING ANALYSIS (PLAN B)
# ==================================================
def get_horizontal_angle(p1, p2):
    """2点を結ぶ線と水平線の角度を計算"""
    vec = np.array(p1) - np.array(p2)
    return math.degrees(math.atan2(vec[1], vec[0]))

def analyze_swing(video_path: str) -> Dict[str, Any]:
    """
    MediaPipeを使ってスイング動画を解析し、
    トップ位置での捻転差(X-Factor)などを計算して返す
    """
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    frames_data = []

    # 1. 全フレームの座標抽出
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 処理高速化・安定化のためリサイズ
        image = cv2.resize(frame, (640, 360))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 必要な部位のみ抽出 (x, y)
            frames_data.append({
                "nose": (lm[0].x, lm[0].y),
                "l_shoulder": (lm[11].x, lm[11].y),
                "r_shoulder": (lm[12].x, lm[12].y),
                "l_elbow": (lm[13].x, lm[13].y),
                "l_wrist": (lm[15].x, lm[15].y),
                "l_hip": (lm[23].x, lm[23].y),
                "r_hip": (lm[24].x, lm[24].y),
                "l_knee": (lm[25].x, lm[25].y),
                "l_ankle": (lm[27].x, lm[27].y),
            })
    cap.release()

    if not frames_data:
        return {} # 解析失敗

    # 2. フェーズ特定 (トップ・アドレス・インパクト)
    # 手首(Left Wrist)の高さ(y)で判定。yは画面下が1.0なので、最小値が一番高い位置
    wrist_ys = [f["l_wrist"][1] for f in frames_data]
    
    # 【トップ】手首が一番高い位置
    top_idx = np.argmin(wrist_ys)
    
    # 【アドレス】トップより前で、手首が低く安定している場所（簡易的にトップの1秒前付近）
    search_start = max(0, top_idx - 50)
    address_slice = wrist_ys[search_start:top_idx]
    if len(address_slice) > 0:
        address_idx = search_start + np.argmax(address_slice)
    else:
        address_idx = 0

    # 【インパクト】トップの後、手首が最下点にくる場所
    search_end = min(len(frames_data), top_idx + 40)
    impact_slice = wrist_ys[top_idx:search_end]
    if len(impact_slice) > 0:
        impact_idx = top_idx + np.argmax(impact_slice)
    else:
        impact_idx = top_idx + 10

    # 3. 数値計算
    def calc_metrics(idx):
        d = frames_data[idx]
        
        # 肩の回転角 (水平線との角度)
        shoulder_rot = get_horizontal_angle(d["l_shoulder"], d["r_shoulder"])
        
        # 腰の回転角
        hip_rot = get_horizontal_angle(d["l_hip"], d["r_hip"])
        
        # 前傾角度 (Spine Angle): 股関節中点と首を結ぶ線 vs 垂直線
        mid_hip = ((d["l_hip"][0]+d["r_hip"][0])/2, (d["l_hip"][1]+d["r_hip"][1])/2)
        mid_sh = ((d["l_shoulder"][0]+d["r_shoulder"][0])/2, (d["l_shoulder"][1]+d["r_shoulder"][1])/2)
        spine_vec = np.array(mid_sh) - np.array(mid_hip)
        spine_angle = math.degrees(math.atan2(spine_vec[0], -spine_vec[1]))
        
        return {
            "shoulder_rot": shoulder_rot,
            "hip_rot": hip_rot,
            "spine_angle": spine_angle,
            "head_x": d["nose"][0]
        }

    addr = calc_metrics(address_idx)
    top = calc_metrics(top_idx)
    imp = calc_metrics(impact_idx)

    # 最終的な指標
    # Xファクター: トップでの (肩回転 - 腰回転) の差
    x_factor = abs(top["shoulder_rot"] - top["hip_rot"])
    
    # スウェー: アドレスとトップの頭の位置の差 (画面幅に対する%)
    sway = (top["head_x"] - addr["head_x"]) * 100
    
    # 前傾キープ: アドレスとインパクトの前傾角度の差
    spine_diff = abs(addr["spine_angle"] - imp["spine_angle"])

    return {
        "x_factor": round(x_factor, 1),
        "shoulder_rotation": round(abs(top["shoulder_rot"]), 1),
        "hip_rotation": round(abs(top["hip_rot"]), 1),
        "sway": round(sway, 2),
        "spine_maintain": round(spine_diff, 1),
        "phases": {
            "address_frame": int(address_idx),
            "top_frame": int(top_idx),
            "impact_frame": int(impact_idx)
        }
    }

# ==================================================
# [DESIGN] HTML TEMPLATE (PLAN B)
# ==================================================
REPORT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>詳細スイング診断</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;700&display=swap" rel="stylesheet">
<style>
    body { font-family: 'Noto Serif JP', serif; background-color: #f3f4f6; color: #1f2937; }
    .a4-sheet {
        background: white; width: 100%; max-width: 210mm; min-height: 297mm;
        margin: 20px auto; padding: 40px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    @media (max-width: 640px) { .a4-sheet { margin: 0; padding: 20px; min-height: 100vh; } }
    
    .metric-box { border-bottom: 1px solid #e5e7eb; padding: 16px 0; display: flex; justify-content: space-between; align-items: center; }
    .metric-label { font-weight: bold; color: #4b5563; }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #059669; }
    .sub-text { font-size: 0.8rem; color: #9ca3af; }
    
    .status-badge { padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: bold; }
    .status-processing { background: #fef3c7; color: #d97706; }
    .status-error { background: #fee2e2; color: #b91c1c; }
</style>
</head>
<body>

<div class="a4-sheet">
    <div class="text-center border-b-2 border-emerald-800 pb-6 mb-8">
        <h1 class="text-3xl font-bold text-emerald-900">SWING ANALYSIS</h1>
        <p class="text-gray-500 mt-2">GATE AI Golf Doctor</p>
        <p class="text-xs text-gray-300 mt-1">ID: <span id="reportIdDisplay"></span></p>
    </div>

    <div id="loading" class="text-center py-20">
        <div class="animate-spin h-8 w-8 border-4 border-emerald-500 rounded-full border-t-transparent mx-auto"></div>
        <p class="mt-4 text-gray-400">解析データを取得中...</p>
    </div>
    
    <div id="error" class="hidden text-center py-10 bg-red-50 text-red-700 rounded mb-4">
        <p class="font-bold">データが見つかりません</p>
        <p class="text-sm">URLを確認するか、再度動画を送信してください。</p>
    </div>

    <div id="content" class="hidden">
        <div class="mb-10">
            <h2 class="text-xl font-bold text-emerald-800 mb-4 flex items-center">
                <span class="bg-emerald-100 text-emerald-800 px-2 py-1 rounded text-sm mr-2">Power</span>
                捻転とパワー (トップ位置)
            </h2>
            <div class="metric-box">
                <div>
                    <div class="metric-label">Xファクター (捻転差)</div>
                    <div class="sub-text">トップでの肩と腰の回転差</div>
                </div>
                <div class="text-right">
                    <span id="val_xfactor" class="metric-value">-</span><span class="text-sm">deg</span>
                </div>
            </div>
            <div class="metric-box">
                <div>
                    <div class="metric-label">肩の回転量</div>
                </div>
                <div class="text-right">
                    <span id="val_shoulder" class="metric-value">-</span><span class="text-sm">deg</span>
                </div>
            </div>
            <div class="metric-box">
                <div>
                    <div class="metric-label">腰の回転量</div>
                </div>
                <div class="text-right">
                    <span id="val_hip" class="metric-value">-</span><span class="text-sm">deg</span>
                </div>
            </div>
        </div>

        <div class="mb-10">
            <h2 class="text-xl font-bold text-emerald-800 mb-4 flex items-center">
                <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm mr-2">Stability</span>
                安定性と軸
            </h2>
            <div class="metric-box">
                <div>
                    <div class="metric-label">スウェー (頭の移動)</div>
                    <div class="sub-text">アドレスからトップまでの頭のズレ</div>
                </div>
                <div class="text-right">
                    <span id="val_sway" class="metric-value">-</span><span class="text-sm">%</span>
                </div>
            </div>
            <div class="metric-box">
                <div>
                    <div class="metric-label">前傾キープ誤差</div>
                    <div class="sub-text">アドレスとインパクトの角度差</div>
                </div>
                <div class="text-right">
                    <span id="val_spine" class="metric-value">-</span><span class="text-sm">deg</span>
                </div>
            </div>
        </div>

        <div class="bg-gray-50 p-6 rounded-lg text-sm text-gray-600 mt-8">
            <h3 class="font-bold mb-2">💡 診断基準</h3>
            <ul class="list-disc pl-5 space-y-1">
                <li><strong>Xファクター:</strong> 45度以上が理想的です。大きいほど飛距離が出ます。</li>
                <li><strong>スウェー:</strong> 5%以内が目安です。動きすぎるとミート率が下がります。</li>
                <li><strong>前傾キープ:</strong> 0に近いほどプロに近いスイングです。</li>
            </ul>
        </div>
    </div>
</div>

<script>
    const reportId = window.location.pathname.split("/").pop();
    document.getElementById("reportIdDisplay").innerText = reportId;

    fetch(`/api/report_data/${reportId}`)
    .then(r => r.json())
    .then(data => {
        document.getElementById("loading").classList.add("hidden");

        if (data.error || data.status === "FAILED") {
            document.getElementById("error").classList.remove("hidden");
            return;
        }
        
        if (data.status === "PROCESSING") {
             const errDiv = document.getElementById("error");
             errDiv.classList.remove("hidden");
             errDiv.className = "text-center py-10 bg-yellow-50 text-yellow-800 rounded mb-4";
             errDiv.innerHTML = "<p class='font-bold'>解析中です</p><p class='text-sm'>1〜2分後に再読み込みしてください。</p>";
             return;
        }

        // Success
        document.getElementById("content").classList.remove("hidden");
        const d = data.mediapipe_data || {};
        
        document.getElementById("val_xfactor").innerText = d.x_factor || "-";
        document.getElementById("val_shoulder").innerText = d.shoulder_rotation || "-";
        document.getElementById("val_hip").innerText = d.hip_rotation || "-";
        document.getElementById("val_sway").innerText = d.sway || "-";
        document.getElementById("val_spine").innerText = d.spine_maintain || "-";
    })
    .catch(e => {
        document.getElementById("loading").classList.add("hidden");
        document.getElementById("error").classList.remove("hidden");
    });
</script>
</body>
</html>
"""

# ==================================================
# HELPERS
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

def safe_line_reply(reply_token: str, text: str) -> None:
    try:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text))
    except LineBotApiError:
        print(traceback.format_exc())

def safe_line_push(user_id: str, text: str) -> None:
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text=text))
    except LineBotApiError:
        print(traceback.format_exc())

def make_initial_reply(report_id: str) -> str:
    return (
        "✅ 動画を受信しました。\n"
        "AIによる詳細解析を開始します。\n\n"
        "トップ位置の特定や捻転差の計算を行います。\n"
        "数分後に完了通知をお送りします。\n\n"
        f"{SERVICE_HOST_URL}/report/{report_id}"
    )

def make_done_push(report_id: str) -> str:
    return (
        "🎉 解析が完了しました！\n\n"
        "Xファクターやスウェー量などの詳細データを確認できます。\n\n"
        f"{SERVICE_HOST_URL}/report/{report_id}"
    )

# ==================================================
# CLOUD TASKS
# ==================================================
def create_cloud_task(report_id: str, user_id: str, message_id: str) -> None:
    payload = json.dumps(
        {"report_id": report_id, "user_id": user_id, "message_id": message_id}
    ).encode("utf-8")

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
    tasks_client.create_task(parent=queue_path, task=task)

# ==================================================
# ROUTES
# ==================================================
@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent)
def handle_any(event: MessageEvent):
    msg = event.message
    user_id = event.source.user_id

    if isinstance(msg, (VideoMessage, FileMessage)):
        report_id = f"{user_id}_{msg.id}"
        firestore_safe_set(
            report_id, 
            {"user_id": user_id, "status": "PROCESSING", "created_at": firestore.SERVER_TIMESTAMP}
        )
        create_cloud_task(report_id, user_id, msg.id)
        safe_line_reply(event.reply_token, make_initial_reply(report_id))
    else:
        safe_line_reply(event.reply_token, "🎥 スイング動画またはファイルを送信してください。")

@app.route("/worker/process_video", methods=["POST"])
def worker():
    payload = request.get_json()
    report_id = payload.get("report_id")
    message_id = payload.get("message_id")

    if not report_id or not message_id:
        return jsonify({"error": "invalid payload"}), 400

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, f"{message_id}.mp4")

    try:
        message_content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for chunk in message_content.iter_content():
                f.write(chunk)

        # ここで関数を直接呼ぶ
        raw_data = analyze_swing(video_path)

        firestore_safe_update(report_id, {
            "status": "COMPLETED",
            "raw_data": raw_data,
            "completed_at": firestore.SERVER_TIMESTAMP,
        })

        doc = db.collection("reports").document(report_id).get()
        if doc.exists:
            user_id = doc.to_dict().get("user_id")
            safe_line_push(user_id, make_done_push(report_id))

    except Exception as e:
        print(f"Error processing video: {e}")
        firestore_safe_update(report_id, {"status": "FAILED", "error": str(e)})
        return jsonify({"status": "failed", "error": str(e)}), 200
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return jsonify({"ok": True})

@app.route("/api/report_data/<report_id>")
def api_report_data(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    d = doc.to_dict()
    return jsonify({
        "status": d.get("status"),
        "mediapipe_data": d.get("raw_data", {}),
    })

@app.route("/report/<report_id>")
def report_view(report_id):
    # 変数内のHTMLをそのまま返す
    return REPORT_HTML_TEMPLATE

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
