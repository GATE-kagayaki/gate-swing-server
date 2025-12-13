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
# [LOGIC] DYNAMIC ANALYSIS ENGINE
# ==================================================
def calculate_angle_3points(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_horizontal_angle(p1, p2):
    vec = np.array(p1) - np.array(p2)
    return math.degrees(math.atan2(vec[1], vec[0]))

def generate_dynamic_prescription(metrics):
    """
    数値に基づいて、コメント・ドリル・フィッティング・まとめを
    すべて動的に生成する
    """
    c = {} # comments container
    drills = []
    fitting = {}
    summary_msg = ""
    
    # 指標の取得
    sway = metrics["sway"]
    xfactor = metrics["x_factor"]
    hip_rot = metrics["hip_rotation"]
    cock = metrics["wrist_cock"]
    
    # --- 1. 個別診断コメント生成 ---
    
    # Head Sway
    if abs(sway) > 8.0:
        c["head_main"] = f"バックスイングで頭が{sway:.1f}%も大きく移動しており、軸が安定していません。"
        c["head_pro"] = "「回転」ではなく「横移動」になっており、ミート率低下の主原因です。"
        drills.append({"name": "クローズスタンス打ち", "obj": "スウェー防止", "method": "両足を閉じてスイングし、その場で回る感覚を養う"})
    elif abs(sway) < 3.0:
        c["head_main"] = "頭部の左右移動が非常に小さく、回転軸は極めて明確です。"
        c["head_pro"] = "すでに“壊れにくいスイング構造”を持っていると判断します。"
    else:
        c["head_main"] = "許容範囲内の動きですが、疲労時に軸がブレる可能性があります。"
        c["head_pro"] = "悪くはありませんが、もう少しその場で回る意識があっても良いでしょう。"

    # Shoulder & X-Factor
    if xfactor < 35:
        c["shoulder_main"] = "肩の回転が浅く、腰と一緒に回っているため捻転差が作れていません。"
        c["shoulder_pro"] = "手打ちの原因となります。柔軟性よりも「分離」の意識が必要です。"
        drills.append({"name": "セパレーションターン", "obj": "上下の捻転差を作る", "method": "椅子に座ったまま、胸だけを左右に限界まで回す"})
    elif xfactor > 60:
        c["shoulder_main"] = "素晴らしい柔軟性ですが、オーバースイングのリスクがあります。"
        c["shoulder_pro"] = "回転過多により、戻すタイミングが遅れやすくなっています。"
        drills.append({"name": "ハーフトップキープ", "obj": "トップの収まり", "method": "トップで一拍止めてから打つ"})
    else:
        c["shoulder_main"] = "理想的な捻転差（Xファクター）が形成されています。"
        c["shoulder_pro"] = "効率よくパワーを生み出せる、バランスの良いトップです。"

    # Hip Rotation
    if hip_rot > 60:
        c["hip_main"] = "腰が回りすぎており、ゴムが緩んだ状態になっています。"
        c["hip_pro"] = "腰の回転を「止める」意識が必要です。"
        drills.append({"name": "右足ベタ足スイング", "obj": "腰の開き抑制", "method": "インパクトまで右かかとを上げずに打つ"})
        # Fitting: 回りすぎる人は重・硬・元調子
        fitting = {"weight": "60g台後半〜70g", "flex": "S〜X", "kick": "元調子", "torque": "3.0〜3.5", "reason": "身体の開きを抑え、左へのミスを消す"}
    elif hip_rot < 30:
        c["hip_main"] = "腰の回転が止まっており、腕力に頼ったスイングです。"
        c["hip_pro"] = "下半身リードが不足しています。"
        drills.append({"name": "ステップ打ち", "obj": "体重移動と回転", "method": "足踏みをしながらリズム良く振る"})
        # Fitting: 回らない人は軽・柔・先調子
        fitting = {"weight": "40g〜50g台", "flex": "R〜SR", "kick": "先調子", "torque": "4.5〜5.5", "reason": "シャフトの走りで回転不足を補う"}
    else:
        c["hip_main"] = "腰の回転量は適正（45度前後）で、土台が安定しています。"
        c["hip_pro"] = "プロレベルの下半身使いです。"
        # Fitting: 標準
        if not fitting:
            fitting = {"weight": "50g〜60g", "flex": "SR〜S", "kick": "中調子", "torque": "3.8〜4.5", "reason": "癖のない挙動で安定性を最大化"}

    # Wrist
    if cock < 80:
        c["wrist_main"] = "コックが深すぎて、リリースが難しくなっています。"
        c["wrist_pro"] = "入射角が鋭角になりやすく、ダフリのリスクがあります。"
    elif cock > 120:
        c["wrist_main"] = "ノーコック気味で、タメが作れていません。"
        c["wrist_pro"] = "ヘッドスピードが上がりにくい構造です。"
        drills.append({"name": "LtoLドリル", "obj": "コックの習得", "method": "腰から腰の振り幅で、手首を90度に折る"})
    else:
        c["wrist_main"] = "適度なコック角が維持されています。"
        c["wrist_pro"] = "再現性の高いリストワークです。"

    # Knee (Dummy logic for now based on sway)
    c["knee_main"] = "下半身の粘りについては動画解析の特性上、推定となりますが、"
    if abs(sway) > 5:
        c["knee_main"] += "スウェーに伴い膝が流れている可能性が高いです。"
        c["knee_pro"] = "足元のグリップ力強化が必要です。"
    else:
        c["knee_main"] += "軸が安定しているため、膝の使い方も良好と推測されます。"
        c["knee_pro"] = "地面反力を活かしやすい土台です。"

    # --- 2. 総合診断 & まとめ ---
    if len(drills) > 2:
        c["summary_good"] = "スイングへの意欲とパワーポテンシャル"
        c["summary_bad"] = "各パーツの連動不足とオーバーアクション"
        c["summary_msg"] = "「要素を削ぎ落とし、シンプルにする段階」"
        summary_footer = "現在のスイングは、少し複雑になりすぎています。\n余計な動きを減らすことで、驚くほどミート率が向上するはずです。\nまずは土台となるアドレスと、小さな振り幅から調整しましょう。"
    elif len(drills) == 0:
        c["summary_good"] = "全体のバランスと再現性の高さ"
        c["summary_bad"] = "特になし（微調整レベル）"
        c["summary_msg"] = "「完成度が高く、スコアに直結するスイング」"
        summary_footer = "素晴らしいスイングです。\n大きな改造は必要ありません。\n今のリズムを維持しつつ、ショートゲームやコースマネジメントに注力してください。"
        drills.append({"name": "片手打ち", "obj": "リズム維持", "method": "片手でウェッジを持ち、ゆったり振る"})
    else:
        c["summary_good"] = "軸の意識と基本的なボディターン"
        c["summary_bad"] = "特定の局面での代償動作"
        c["summary_msg"] = "「ワンポイント修正で激変するタイプ」"
        summary_footer = "土台は整っています。\n指摘した1〜2点の課題を修正するだけで、球筋が劇的に変わるでしょう。\nまずは推奨ドリルを2週間続けてみてください。"

    return {
        "comments": c,
        "drills": drills[:3], # 最大3つまで
        "fitting": fitting,
        "summary_footer": summary_footer
    }


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
    search_start = max(0, top_idx - 50)
    address_slice = wrist_ys[search_start:top_idx]
    address_idx = search_start + np.argmax(address_slice) if len(address_slice) > 0 else 0

    d_top = frames_data[top_idx]
    d_addr = frames_data[address_idx]

    # 数値計算
    top_shoulder = abs(get_horizontal_angle(d_top["l_shoulder"], d_top["r_shoulder"]))
    top_hip = abs(get_horizontal_angle(d_top["l_hip"], d_top["r_hip"]))
    x_factor = abs(top_shoulder - top_hip)
    sway = (d_top["nose"][0] - d_addr["nose"][0]) * 100
    knee_sway = d_top["l_knee"][0] - d_addr["l_knee"][0]
    wrist_cock = calculate_angle_3points(d_top["l_shoulder"], d_top["l_elbow"], d_top["l_wrist"])

    metrics = {
        "x_factor": round(x_factor, 1),
        "shoulder_rotation": round(top_shoulder, 1),
        "hip_rotation": round(top_hip, 1),
        "sway": round(sway, 2),
        "knee_sway": round(knee_sway, 4),
        "wrist_cock": round(wrist_cock, 1)
    }

    # 動的コンテンツ生成
    prescription = generate_dynamic_prescription(metrics)

    return {
        "metrics": metrics,
        "comments": prescription["comments"],
        "drills": prescription["drills"],
        "fitting": prescription["fitting"],
        "summary_footer": prescription["summary_footer"]
    }

# ==================================================
# [DESIGN] HTML TEMPLATE (Dynamic Content)
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
    .pro-box { background-color: #ecfdf5; border: 1px solid #d1fae5; border-radius: 8px; padding: 16px; margin-top: 16px; }
    .pro-label { font-size: 0.8rem; font-weight: bold; color: #059669; margin-bottom: 4px; display: block; }
    .pro-text { font-size: 1rem; font-weight: bold; color: #065f46; font-family: 'Noto Serif JP', serif; }
    .table-custom { width: 100%; font-size: 0.9rem; border-collapse: collapse; margin-top: 10px; }
    .table-custom th { background: #047857; color: white; padding: 8px; text-align: left; }
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
        <p class="mt-4 text-gray-500 text-sm">解析中...</p>
    </div>

    <div id="content" class="hidden p-6 md:p-10 space-y-10">
        
        <section>
            <div class="section-header"><span class="section-title">02. 頭の安定性（軸のブレ）</span><span class="text-sm text-gray-500">Sway: <span id="v_sway" class="metric-value">-</span></span></div>
            <p id="t_head">-</p>
            <div class="pro-box"><span class="pro-label">👉 プロ視点では</span><p id="p_head" class="pro-text">-</p></div>
        </section>

        <section>
            <div class="section-header"><span class="section-title">03. 肩の回旋（上半身のねじり）</span><span class="text-sm text-gray-500">X-Factor: <span id="v_xfactor" class="metric-value">-</span></span></div>
            <p id="t_shoulder">-</p>
            <div class="pro-box"><span class="pro-label">👉 プロ目線では</span><p id="p_shoulder" class="pro-text">-</p></div>
        </section>

        <section>
            <div class="section-header"><span class="section-title">04. 腰の回旋（下半身の動き）</span><span class="text-sm text-gray-500">Rotation: <span id="v_hip" class="metric-value">-</span></span></div>
            <p id="t_hip">-</p>
            <div class="pro-box"><span class="pro-label">👉 プロ的には</span><p id="p_hip" class="pro-text">-</p></div>
        </section>

        <section>
            <div class="section-header"><span class="section-title">05. 手首のメカニクス</span><span class="text-sm text-gray-500">Cock: <span id="v_cock" class="metric-value">-</span></span></div>
            <p id="t_wrist">-</p>
            <div class="pro-box"><span class="pro-label">👉 プロ評価では</span><p id="p_wrist" class="pro-text">-</p></div>
        </section>

        <section class="bg-gray-50 p-6 rounded border border-gray-200">
            <h3 class="font-bold text-gray-800 mb-4 border-b pb-2">07. 総合診断</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div><h4 class="text-sm font-bold text-blue-600 mb-2">✅ 安定している点</h4><p id="s_good" class="text-sm">-</p></div>
                <div><h4 class="text-sm font-bold text-red-600 mb-2">⚠️ 改善が期待される点</h4><p id="s_bad" class="text-sm">-</p></div>
            </div>
            <div class="mt-6 font-serif font-bold text-emerald-800 text-center text-lg">👉 <span id="s_msg">-</span></div>
        </section>

        <section>
            <div class="section-header"><span class="section-title">08. 改善戦略とドリル</span></div>
            <table class="table-custom">
                <thead><tr><th>ドリル</th><th>目的</th><th>やり方</th></tr></thead>
                <tbody id="drill_table_body"></tbody>
            </table>
        </section>

        <section>
            <div class="section-header"><span class="section-title">09. スイング傾向補正型フィッティング</span></div>
            <table class="table-custom">
                <tr><td class="bg-gray-100 font-bold w-1/4">重量</td><td id="fit_weight">-</td><td class="text-xs text-gray-500" rowspan="4" id="fit_reason">-</td></tr>
                <tr><td class="bg-gray-100 font-bold">フレックス</td><td id="fit_flex">-</td></tr>
                <tr><td class="bg-gray-100 font-bold">キック</td><td id="fit_kick">-</td></tr>
                <tr><td class="bg-gray-100 font-bold">トルク</td><td id="fit_torque">-</td></tr>
            </table>
        </section>

        <div class="bg-emerald-50 p-8 text-center rounded mt-12">
            <h3 class="font-bold text-emerald-800 mb-2">10. まとめ</h3>
            <p id="footer_msg" class="text-sm text-emerald-700 leading-relaxed whitespace-pre-line">-</p>
        </div>
    </div>
</div>

<script>
    const reportId = window.location.pathname.split("/").pop();
    fetch(`/api/report_data/${reportId}`)
    .then(r => r.json())
    .then(data => {
        if(data.status === "COMPLETED") {
            document.getElementById("loading").classList.add("hidden");
            document.getElementById("content").classList.remove("hidden");
            
            const m = data.mediapipe_data.metrics;
            const c = data.mediapipe_data.comments;
            const drills = data.mediapipe_data.drills;
            const fit = data.mediapipe_data.fitting;
            const footer = data.mediapipe_data.summary_footer;

            // Metrics
            document.getElementById("v_sway").innerText = m.sway + "%";
            document.getElementById("v_xfactor").innerText = m.x_factor + "°";
            document.getElementById("v_hip").innerText = m.hip_rotation + "°";
            document.getElementById("v_cock").innerText = m.wrist_cock + "°";

            // Comments
            document.getElementById("t_head").innerText = c.head_main;
            document.getElementById("p_head").innerText = c.head_pro;
            document.getElementById("t_shoulder").innerText = c.shoulder_main;
            document.getElementById("p_shoulder").innerText = c.shoulder_pro;
            document.getElementById("t_hip").innerText = c.hip_main;
            document.getElementById("p_hip").innerText = c.hip_pro;
            document.getElementById("t_wrist").innerText = c.wrist_main;
            document.getElementById("p_wrist").innerText = c.wrist_pro;
            
            document.getElementById("s_good").innerText = c.summary_good;
            document.getElementById("s_bad").innerText = c.summary_bad;
            document.getElementById("s_msg").innerText = c.summary_msg;

            // Drills (Loop)
            const drillBody = document.getElementById("drill_table_body");
            drills.forEach(d => {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td class="font-bold">${d.name}</td><td>${d.obj}</td><td>${d.method}</td>`;
                drillBody.appendChild(tr);
            });

            // Fitting
            document.getElementById("fit_weight").innerText = fit.weight;
            document.getElementById("fit_flex").innerText = fit.flex;
            document.getElementById("fit_kick").innerText = fit.kick;
            document.getElementById("fit_torque").innerText = fit.torque;
            document.getElementById("fit_reason").innerText = fit.reason;

            // Footer
            document.getElementById("footer_msg").innerText = footer;
        }
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
        try: line_bot_api.reply_message(event.reply_token, TextSendMessage(text="✅ 動画を受信しました。\n個別のスイング傾向に合わせて、ドリルやギア推奨を作成中です..."))
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

        result = analyze_swing(video_path)

        firestore_safe_update(report_id, {
            "status": "COMPLETED",
            "raw_data": result,
            "completed_at": firestore.SERVER_TIMESTAMP
        })
        
        doc = db.collection("reports").document(report_id).get()
        if doc.exists:
            uid = doc.to_dict().get("user_id")
            try: line_bot_api.push_message(uid, TextSendMessage(text=f"🏌️‍♂️ 診断完了！\nあなたのタイプに合わせた練習メニューを作成しました。\n{SERVICE_HOST_URL}/report/{report_id}"))
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
