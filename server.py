import os
import json
import time
import math
import shutil
import traceback
import tempfile
import numpy as np
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

app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
db = firestore.Client(project=GCP_PROJECT_ID)
tasks_client = tasks_v2.CloudTasksClient()
queue_path = tasks_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)

# ==================================================
# [LOGIC] PRO-LEVEL ANALYSIS ENGINE
# ==================================================
def calculate_angle_3points(a, b, c):
    """3点(a,b,c)のなす角度を計算 (肘の曲がりやコック角など)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_horizontal_angle(p1, p2):
    """2点を結ぶ線と水平線の角度"""
    vec = np.array(p1) - np.array(p2)
    return math.degrees(math.atan2(vec[1], vec[0]))

def generate_pro_comments(metrics):
    """計測値からプロ視点のコメントテキストを生成する"""
    comments = {}
    
    # 02. 頭の安定性 (Sway)
    sway = metrics["sway"]
    if abs(sway) < 5.0:
        comments["head_main"] = "頭部の左右移動量が小さく、回転軸は極めて明確です。\n切り返し局面でも頭の位置が保たれており、体幹主導のスイングに移行できる下地が整っています。"
        comments["head_pro"] = "すでに“壊れにくいスイング構造”を持っていると判断します。"
    elif sway > 0: # 右へスウェー
        comments["head_main"] = "バックスイングで頭が右に流れる傾向があります。\nパワーを溜めようとする意識が強いですが、軸がブレることでミート率が低下するリスクがあります。"
        comments["head_pro"] = "「回転」よりも「横移動」で上げている状態です。"
    else: # 左へリバース
        comments["head_main"] = "トップで頭がターゲット方向に突っ込む「リバースピボット」の傾向が見られます。\n切り返しで詰まりやすくなります。"
        comments["head_pro"] = "軸が左に倒れており、パワーロスが大きいです。"

    # 03. 肩の回旋
    shoulder = metrics["shoulder_rotation"]
    xfactor = metrics["x_factor"]
    if xfactor < 35:
        comments["shoulder_main"] = "肩の回旋量が小さく、捻転差（Xファクター）が十分に形成されていません。\n上半身の可動域というより、回旋の使い方が抑制的になっている可能性が高いです。"
        comments["shoulder_pro"] = "「可動域不足」ではなく“使えていない”タイプに分類されます。"
    elif xfactor > 60:
        comments["shoulder_main"] = "非常に深く肩が入っており、柔軟性はプロ並みです。\nただし、回りすぎによるオーバースイングに注意が必要です。"
        comments["shoulder_pro"] = "柔軟性は武器ですが、戻すタイミングの制御が鍵になります。"
    else:
        comments["shoulder_main"] = "肩の回転量は適正範囲内です。\n無理なく捻転差が作れており、再現性の高いトップが作れています。"
        comments["shoulder_pro"] = "バランスの取れた良い回転量です。"

    # 04. 腰の回旋
    hip = metrics["hip_rotation"]
    if hip > 60:
        comments["hip_main"] = "腰の回転が早く・大きく出やすい傾向です。\n上半身より先に回ることでパワーが分散し、腕の介入を招きやすくなります。"
        comments["hip_pro"] = "切り返しタイミングの調整余地が大きいスイングです。"
    elif hip < 30:
        comments["hip_main"] = "腰の回転が止まり気味で、手打ちになりやすい状態です。\n下半身リードをもっと意識する必要があります。"
        comments["hip_pro"] = "下半身が使えておらず、腕力に頼ったスイングです。"
    else:
        comments["hip_main"] = "腰の回転量は理想的（45度前後）です。\n土台としてしっかり機能しています。"
        comments["hip_pro"] = "プロレベルの安定した下半身使いです。"

    # 05. 手首 (Wrist Cock)
    cock = metrics["wrist_cock"]
    if cock < 90: # 鋭角＝深いコック
        comments["wrist_main"] = "コック角が深く、タメを作ろうとする意識が強いです。\n捻転量不足を手首動作で補おうとする代償動作の可能性もあります。"
        comments["wrist_pro"] = "「再現性を上げる余地が明確」です。"
    else:
        comments["wrist_main"] = "コックが浅く、ノーコックに近いスイングです。\n方向性は安定しますが、飛距離面では損をしている可能性があります。"
        comments["wrist_pro"] = "手首を固めすぎてヘッドが走っていません。"

    # 06. 下半身安定性 (Knee Sway)
    knee_sway = metrics["knee_sway"]
    if abs(knee_sway) < 0.05:
        comments["knee_main"] = "膝の左右ブレが小さく、地面反力を活かしやすい状態です。\nインパクトゾーンで下半身が暴れないのは大きな強みです。"
        comments["knee_pro"] = "これは完全にプロ・競技者側の特徴です。"
    else:
        comments["knee_main"] = "スイング中に膝が大きく動き、土台が不安定です。\n特に膝が割れる動きはパワーロスに直結します。"
        comments["knee_pro"] = "足元のグリップ力が不足しています。"

    # 07. 総合診断
    comments["summary_good"] = "スイング軸と下半身の安定性\n再現性を高めやすい構造"
    comments["summary_bad"] = "上半身の捻転不足によるパワーロス\n手首主導になりやすい動作配分"
    comments["summary_msg"] = "「伸び代が明確で、改善効率が高いタイプ」"

    return comments


def analyze_swing(video_path: str) -> Dict[str, Any]:
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.resize(frame, (640, 360))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            frames_data.append({
                "nose": (lm[0].x, lm[0].y),
                "l_shoulder": (lm[11].x, lm[11].y),
                "r_shoulder": (lm[12].x, lm[12].y),
                "l_elbow": (lm[13].x, lm[13].y),
                "l_wrist": (lm[15].x, lm[15].y),
                "l_hip": (lm[23].x, lm[23].y),
                "r_hip": (lm[24].x, lm[24].y),
                "l_knee": (lm[25].x, lm[25].y),
            })
    cap.release()
    if not frames_data: return {}

    # フェーズ特定
    wrist_ys = [f["l_wrist"][1] for f in frames_data]
    top_idx = np.argmin(wrist_ys)
    
    # アドレス
    search_start = max(0, top_idx - 50)
    address_slice = wrist_ys[search_start:top_idx]
    address_idx = search_start + np.argmax(address_slice) if len(address_slice) > 0 else 0

    # 計測
    d_top = frames_data[top_idx]
    d_addr = frames_data[address_idx]

    # 1. 角度計算
    top_shoulder = abs(get_horizontal_angle(d_top["l_shoulder"], d_top["r_shoulder"]))
    top_hip = abs(get_horizontal_angle(d_top["l_hip"], d_top["r_hip"]))
    x_factor = abs(top_shoulder - top_hip)
    
    # 2. Sway (頭のブレ)
    sway = (d_top["nose"][0] - d_addr["nose"][0]) * 100
    
    # 3. Knee Sway (膝のブレ)
    knee_sway = d_top["l_knee"][0] - d_addr["l_knee"][0]

    # 4. Wrist Cock (簡易: 肩-肘-手首の角度で推定)
    wrist_cock = calculate_angle_3points(d_top["l_shoulder"], d_top["l_elbow"], d_top["l_wrist"])

    metrics = {
        "x_factor": round(x_factor, 1),
        "shoulder_rotation": round(top_shoulder, 1),
        "hip_rotation": round(top_hip, 1),
        "sway": round(sway, 2),
        "knee_sway": round(knee_sway, 4),
        "wrist_cock": round(wrist_cock, 1)
    }

    # コメント生成
    comments = generate_pro_comments(metrics)

    return {
        "metrics": metrics,
        "comments": comments
    }

# ==================================================
# [DESIGN] HTML TEMPLATE (Ver 4.0 Ultimate Report)
# ==================================================
REPORT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AIスイング診断書</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&family=Noto+Serif+JP:wght@600&display=swap" rel="stylesheet">
<style>
    body { font-family: 'Noto Sans JP', sans-serif; background-color: #f0f2f5; color: #333; }
    .paper { background: white; max-width: 800px; margin: 0 auto; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
    @media (min-width: 640px) { .paper { margin: 40px auto; border-radius: 4px; } }
    
    .section-header { border-left: 6px solid #047857; padding-left: 12px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; }
    .section-title { font-size: 1.1rem; font-weight: bold; color: #064e3b; font-family: 'Noto Serif JP', serif; }
    .metric-value { font-weight: bold; font-size: 1.2rem; color: #059669; }
    
    .text-body { font-size: 0.95rem; line-height: 1.8; color: #4b5563; white-space: pre-line; }
    
    .pro-box { background-color: #ecfdf5; border: 1px solid #d1fae5; border-radius: 8px; padding: 16px; margin-top: 16px; position: relative; }
    .pro-label { font-size: 0.8rem; font-weight: bold; color: #059669; margin-bottom: 4px; display: block; }
    .pro-text { font-size: 1rem; font-weight: bold; color: #065f46; font-family: 'Noto Serif JP', serif; }
    
    .table-custom { width: 100%; font-size: 0.9rem; border-collapse: collapse; margin-top: 10px; }
    .table-custom th { background: #047857; color: white; padding: 8px; text-align: left; font-weight: normal; }
    .table-custom td { border-bottom: 1px solid #e5e7eb; padding: 8px; color: #374151; }
</style>
</head>
<body>

<div class="paper">
    <div class="bg-emerald-900 text-white p-8 text-center">
        <h1 class="text-2xl font-serif font-bold tracking-wider mb-2">SWING DIAGNOSIS REPORT</h1>
        <p class="text-emerald-200 text-sm">GATE AI Golf Analysis System</p>
    </div>

    <div id="loading" class="text-center py-20">
        <div class="animate-spin h-8 w-8 border-4 border-emerald-600 rounded-full border-t-transparent mx-auto"></div>
        <p class="mt-4 text-gray-500 text-sm">プロフェッショナル分析を実行中...</p>
    </div>

    <div id="error" class="hidden text-center py-20 bg-red-50 text-red-700">
        <p class="font-bold">データ取得エラー</p>
    </div>

    <div id="content" class="hidden p-6 md:p-10 space-y-10">
        
        <section>
            <div class="section-header">
                <span class="section-title">02. 頭の安定性（軸のブレ）</span>
                <span class="text-sm text-gray-500">Sway: <span id="v_sway" class="metric-value">-</span></span>
            </div>
            <p id="t_head" class="text-body">-</p>
            <div class="pro-box">
                <span class="pro-label">👉 プロ視点では</span>
                <p id="p_head" class="pro-text">-</p>
            </div>
        </section>

        <section>
            <div class="section-header">
                <span class="section-title">03. 肩の回旋（上半身のねじり）</span>
                <span class="text-sm text-gray-500">X-Factor: <span id="v_xfactor" class="metric-value">-</span></span>
            </div>
            <p id="t_shoulder" class="text-body">-</p>
            <div class="pro-box">
                <span class="pro-label">👉 プロ目線では</span>
                <p id="p_shoulder" class="pro-text">-</p>
            </div>
        </section>

        <section>
            <div class="section-header">
                <span class="section-title">04. 腰の回旋（下半身の動き）</span>
                <span class="text-sm text-gray-500">Rotation: <span id="v_hip" class="metric-value">-</span></span>
            </div>
            <p id="t_hip" class="text-body">-</p>
            <div class="pro-box">
                <span class="pro-label">👉 プロ的には</span>
                <p id="p_hip" class="pro-text">-</p>
            </div>
        </section>

        <section>
            <div class="section-header">
                <span class="section-title">05. 手首のメカニクス</span>
                <span class="text-sm text-gray-500">Cock Angle: <span id="v_cock" class="metric-value">-</span></span>
            </div>
            <p id="t_wrist" class="text-body">-</p>
            <div class="pro-box">
                <span class="pro-label">👉 プロ評価では</span>
                <p id="p_wrist" class="pro-text">-</p>
            </div>
        </section>

        <section>
            <div class="section-header">
                <span class="section-title">06. 下半身の安定性</span>
            </div>
            <p id="t_knee" class="text-body">-</p>
            <div class="pro-box">
                <span class="pro-label">👉 これは完全に</span>
                <p id="p_knee" class="pro-text">-</p>
            </div>
        </section>

        <section class="bg-gray-50 p-6 rounded border border-gray-200">
            <h3 class="font-bold text-gray-800 mb-4 border-b pb-2">07. 総合診断</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <h4 class="text-sm font-bold text-blue-600 mb-2">✅ 安定している点</h4>
                    <p id="s_good" class="text-sm text-gray-600 whitespace-pre-line">-</p>
                </div>
                <div>
                    <h4 class="text-sm font-bold text-red-600 mb-2">⚠️ 改善が期待される点</h4>
                    <p id="s_bad" class="text-sm text-gray-600 whitespace-pre-line">-</p>
                </div>
            </div>
            <div class="mt-6 font-serif font-bold text-emerald-800 text-center text-lg">
                👉 <span id="s_msg">-</span>
            </div>
        </section>

        <section>
            <div class="section-header"><span class="section-title">08. 改善戦略とドリル</span></div>
            <table class="table-custom">
                <thead><tr><th>ドリル</th><th>目的</th><th>やり方</th></tr></thead>
                <tbody>
                    <tr><td class="font-bold">セパレーションターン</td><td>上下の捻転差を作る</td><td>下半身を止め、胸だけを回す</td></tr>
                    <tr><td class="font-bold">ハーフトップキープ</td><td>切り返し安定</td><td>トップで一拍止めてから振る</td></tr>
                    <tr><td class="font-bold">体幹主導LtoL</td><td>手首介入抑制</td><td>腕を使わず体の回転で振る</td></tr>
                </tbody>
            </table>
            <div class="mt-4 text-right text-sm text-gray-500 font-bold">👉 プロレベルでは「意識」ではなく「役割分担」を教えます</div>
        </section>

        <section>
            <div class="section-header"><span class="section-title">09. スイング傾向補正型フィッティング</span></div>
            <table class="table-custom">
                <tr><td class="bg-gray-100 font-bold w-1/4">重量</td><td>55〜65g</td><td class="text-xs text-gray-500">下半身安定を活かしつつ操作性確保</td></tr>
                <tr><td class="bg-gray-100 font-bold">フレックス</td><td>SR〜S</td><td class="text-xs text-gray-500">タイミングを合わせやすい</td></tr>
                <tr><td class="bg-gray-100 font-bold">キック</td><td>先中</td><td class="text-xs text-gray-500">打ち出し角と初速を補正</td></tr>
                <tr><td class="bg-gray-100 font-bold">トルク</td><td>3.8〜4.5</td><td class="text-xs text-gray-500">手元の暴れを抑制</td></tr>
            </table>
        </section>

        <div class="bg-emerald-50 p-8 text-center rounded mt-12">
            <h3 class="font-bold text-emerald-800 mb-2">10. まとめ</h3>
            <p class="text-sm text-emerald-700 leading-relaxed">
                このスイングは、「直せばすぐ変わる」タイプです。<br>
                土台はすでに整っています。あとは上半身の役割を正しく使えるかどうか。<br><br>
                お客様のゴルフライフが、<br>より戦略的で、再現性の高いものになることを切に願っています。
            </p>
        </div>

    </div>
</div>

<script>
    const reportId = window.location.pathname.split("/").pop();
    fetch(`/api/report_data/${reportId}`)
    .then(r => r.json())
    .then(data => {
        if(data.status !== "COMPLETED") {
             if(data.status === "PROCESSING") {
                 document.getElementById("loading").innerHTML = "解析中...<br>1〜2分後にリロードしてください";
             } else {
                 document.getElementById("error").classList.remove("hidden");
                 document.getElementById("loading").classList.add("hidden");
             }
             return;
        }

        document.getElementById("loading").classList.add("hidden");
        document.getElementById("content").classList.remove("hidden");
        
        const m = data.mediapipe_data.metrics;
        const c = data.mediapipe_data.comments;

        // 数値埋め込み
        document.getElementById("v_sway").innerText = m.sway + "%";
        document.getElementById("v_xfactor").innerText = m.x_factor + "°";
        document.getElementById("v_hip").innerText = m.hip_rotation + "°";
        document.getElementById("v_cock").innerText = m.wrist_cock + "°";

        // テキスト埋め込み (Head)
        document.getElementById("t_head").innerText = c.head_main;
        document.getElementById("p_head").innerText = c.head_pro;

        // Shoulder
        document.getElementById("t_shoulder").innerText = c.shoulder_main;
        document.getElementById("p_shoulder").innerText = c.shoulder_pro;

        // Hip
        document.getElementById("t_hip").innerText = c.hip_main;
        document.getElementById("p_hip").innerText = c.hip_pro;

        // Wrist
        document.getElementById("t_wrist").innerText = c.wrist_main;
        document.getElementById("p_wrist").innerText = c.wrist_pro;

        // Knee
        document.getElementById("t_knee").innerText = c.knee_main;
        document.getElementById("p_knee").innerText = c.knee_pro;

        // Summary
        document.getElementById("s_good").innerText = c.summary_good;
        document.getElementById("s_bad").innerText = c.summary_bad;
        document.getElementById("s_msg").innerText = c.summary_msg;
    });
</script>
</body>
</html>
"""

# ==================================================
# SERVER HANDLERS
# ==================================================
def firestore_safe_set(report_id, data):
    try: db.collection("reports").document(report_id).set(data, merge=True)
    except: pass

def firestore_safe_update(report_id, patch):
    try: db.collection("reports").document(report_id).update(patch)
    except: pass

def create_cloud_task(report_id, user_id, message_id):
    payload = json.dumps({"report_id": report_id, "user_id": user_id, "message_id": message_id}).encode("utf-8")
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{SERVICE_HOST_URL}/worker/process_video",
            "headers": {"Content-Type": "application/json"},
            "body": payload,
            "oidc_token": {"service_account_email": TASK_SA_EMAIL, "audience": SERVICE_HOST_URL},
        }
    }
    tasks_client.create_task(parent=queue_path, task=task)

@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try: handler.handle(body, signature)
    except InvalidSignatureError: abort(400)
    return "OK"

@handler.add(MessageEvent)
def handle_msg(event: MessageEvent):
    msg = event.message
    if isinstance(msg, (VideoMessage, FileMessage)):
        report_id = f"{event.source.user_id}_{msg.id}"
        firestore_safe_set(report_id, {"user_id": event.source.user_id, "status": "PROCESSING", "created_at": firestore.SERVER_TIMESTAMP})
        create_cloud_task(report_id, event.source.user_id, msg.id)
        try: line_bot_api.reply_message(event.reply_token, TextSendMessage(text="✅ スイング診断を開始します。\nプロフェッショナル分析を実行中です...（約1分）"))
        except: pass
    else:
        try: line_bot_api.reply_message(event.reply_token, TextSendMessage(text="動画を送信してください。"))
        except: pass

@app.route("/worker/process_video", methods=["POST"])
def worker():
    d = request.get_json()
    report_id = d.get("report_id")
    message_id = d.get("message_id")
    if not report_id: return jsonify({"error": "no id"}), 400

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, f"{message_id}.mp4")

    try:
        content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for c in content.iter_content(): f.write(c)

        result = analyze_swing(video_path) # 解析実行

        firestore_safe_update(report_id, {
            "status": "COMPLETED",
            "raw_data": result, # {metrics:..., comments:...}
            "completed_at": firestore.SERVER_TIMESTAMP
        })
        
        doc = db.collection("reports").document(report_id).get()
        if doc.exists:
            uid = doc.to_dict().get("user_id")
            try: line_bot_api.push_message(uid, TextSendMessage(text=f"🏌️‍♂️ 診断レポートが完成しました！\n{SERVICE_HOST_URL}/report/{report_id}"))
            except: pass

    except Exception as e:
        print(traceback.format_exc())
        firestore_safe_update(report_id, {"status": "FAILED", "error": str(e)})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return jsonify({"ok": True})

@app.route("/api/report_data/<report_id>")
def api_data(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists: return jsonify({"error": "not found"}), 404
    return jsonify({"status": doc.to_dict().get("status"), "mediapipe_data": doc.to_dict().get("raw_data")})

@app.route("/report/<report_id>")
def view_report(report_id):
    return REPORT_HTML_TEMPLATE

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
