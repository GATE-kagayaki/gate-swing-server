import os
import json
import math
import shutil
import traceback
import tempfile
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List

# --- Flask のインポート (send_from_directory を修正済み) ---
from flask import Flask, request, jsonify, abort, send_from_directory 

# --- LINE Bot 関連のライブラリ ---
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent, VideoMessage, TextSendMessage
)

# Cloud Functions/Run で MediaPipe を動かすために必要
import cv2
import mediapipe as mp

from google.cloud import firestore
from google.cloud import tasks_v2

# ==================================================
# CONFIGURATION
# ==================================================
app = Flask(__name__)
db = firestore.Client()

# 環境変数の取得と設定
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("GCP_PROJECT_ID"))
QUEUE_NAME = os.getenv("TASK_QUEUE_NAME", "video-analysis-queue")
QUEUE_LOCATION = os.getenv("TASK_QUEUE_LOCATION", "asia-northeast2")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_HANDLER_URL = f"{SERVICE_HOST_URL}/task-handler"
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "") 

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
tasks_client = tasks_v2.CloudTasksClient()
queue_path = tasks_client.queue_path(PROJECT_ID, QUEUE_LOCATION, QUEUE_NAME)

# ==================================================
# [CORE LOGIC] MATH, TEXT & MEDIAPIPE EXTRACTION
# ==================================================

def calculate_angle_3points(a, b, c):
    """3点間の角度計算"""
    if not a or not b or not c: return 0.0
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_horizontal_angle(p1, p2):
    """水平角計算"""
    if not p1 or not p2: return 0.0
    vec = np.array(p1) - np.array(p2)
    return math.degrees(math.atan2(vec[1], vec[0]))

def extract_mediapipe_data(video_path) -> List[Dict[str, float]]:
    """動画ファイルからMediaPipeを使って骨格データを抽出"""
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
                "nose": (lm[0].x, lm[0].y), "l_shoulder": (lm[11].x, lm[11].y), "r_shoulder": (lm[12].x, lm[12].y), 
                "l_elbow": (lm[13].x, lm[13].y), "l_wrist": (lm[15].x, lm[15].y), "l_hip": (lm[23].x, lm[23].y), 
                "r_hip": (lm[24].x, lm[24].y), "l_knee": (lm[25].x, lm[25].y),
            })
    cap.release()
    return frames_data

def process_landmarks(frames_data):
    """時系列の骨格データからスイング指標(metrics)を計算する"""
    if not frames_data: return None
    wrist_ys = [f.get("l_wrist", [0, 1.0])[1] for f in frames_data]
    top_idx = np.argmin(wrist_ys)
    
    search_start = max(0, top_idx - 50)
    address_slice = wrist_ys[search_start:top_idx]
    if len(address_slice) > 0:
        address_idx = search_start + np.argmax(address_slice)
    else:
        address_idx = 0

    d_top = frames_data[top_idx]
    d_addr = frames_data[address_idx]

    top_shoulder = abs(get_horizontal_angle(d_top.get("l_shoulder"), d_top.get("r_shoulder")))
    top_hip = abs(get_horizontal_angle(d_top.get("l_hip"), d_top.get("r_hip")))
    x_factor = abs(top_shoulder - top_hip)
    
    nose_top = d_top.get("nose", [0,0])[0]
    nose_addr = d_addr.get("nose", [0,0])[0]
    sway = (nose_top - nose_addr) * 100
    
    knee_top = d_top.get("l_knee", [0,0])[0]
    knee_addr = d_addr.get("l_knee", [0,0])[0]
    knee_sway = knee_top - knee_addr

    wrist_cock = calculate_angle_3points(d_top.get("l_shoulder"), d_top.get("l_elbow"), d_top.get("l_wrist"))

    return {
        "x_factor": round(x_factor, 1), "shoulder_rotation": round(top_shoulder, 1), "hip_rotation": round(top_hip, 1),
        "sway": round(sway, 2), "knee_sway": round(knee_sway, 4), "wrist_cock": round(wrist_cock, 1), "frame_count": len(frames_data)
    }

def generate_pro_quality_text(metrics):
    """
    短く、鋭く、プロの言い回しを用いた診断テキストを生成する
    """
    c = {} 
    drills = []
    fitting = {}
    
    sway = metrics["sway"]
    xfactor = metrics["x_factor"]
    hip_rot = metrics["hip_rotation"]
    cock = metrics["wrist_cock"]
    
    # --- 02. Head Sway ---
    if abs(sway) > 8.0:
        c["head_main"] = (
            f"最大頭ブレ（Sway）：{sway:.1f}% (要改善)\n\n"
            "頭部が大きく移動しており、回転軸が定まっていません。\n"
            "ミート率が安定しない主原因。"
        )
        c["head_pro"] = "「回転」ではなく「横移動」になっています。"
        drills.append({"name": "クローズスタンス打ち", "obj": "軸の固定感覚", "method": "両足を閉じてスイングし、その場で回る"})
        
    elif abs(sway) < 4.0:
        c["head_main"] = (
            f"最大頭ブレ（Sway）：{sway:.1f}%\n\n"
            "頭部の左右移動量が小さく、回転軸は明確。\n"
            "体幹主導のスイングに移行できる下地が整っている。"
        )
        c["head_pro"] = "すでに“壊れにくいスイング構造”を持っています。"
        
    else:
        c["head_main"] = (
            f"最大頭ブレ（Sway）：{sway:.1f}%\n\n"
            "許容範囲内だが、疲労時に軸がブレるリスクあり。\n"
            "「背骨の角度を変えない」意識が必要です。"
        )
        c["head_pro"] = "悪くはないが、もっと「その場」で回れます。"

    # --- 03. Shoulder & X-Factor ---
    if xfactor < 35:
        c["shoulder_main"] = (
            "肩の回旋量が小さく、捻転差（Xファクター）が不十分。\n"
            "腕力で代償しようとする動きが発生しています。"
        )
        c["shoulder_pro"] = "「可動域不足」ではなく“使えていない”タイプ。"
        drills.append({"name": "椅子座り捻転", "obj": "分離動作の習得", "method": "椅子に座り、胸椎だけを回す"})
        
    elif xfactor > 60:
        c["shoulder_main"] = (
            "プロ並みの柔軟性だが、オーバースイング気味。\n"
            "戻すタイミングが遅れやすく、振り遅れの原因になりかねない。"
        )
        c["shoulder_pro"] = "柔軟性は武器だが、現在は「振り遅れ」のリスク要因です。"
        drills.append({"name": "3秒トップ停止", "obj": "トップの収まり", "method": "トップで3秒静止し、グラつきを確認する"})
        
    else:
        c["shoulder_main"] = (
            "理想的な捻転差が形成され、再現性の高いトップ。\n"
            "下半身との拮抗（引っ張り合い）もバランスが良い。"
        )
        c["shoulder_pro"] = "文句なし。非常に効率の良いエネルギー構造です。"

    # --- 04. Hip Rotation ---
    if hip_rot > 60:
        c["hip_main"] = (
            "腰の回転が早く・大きく出やすい傾向。\n"
            "上半身より先に回ることで、パワーが分散している。"
        )
        c["hip_pro"] = "切り返しタイミングの調整余地が大きいスイング。"
        drills.append({"name": "右足ベタ足打ち", "obj": "腰の開き抑制", "method": "右かかとを上げずにインパクトする"})
        fitting = {"weight": "60g後半〜70g", "flex": "S〜X", "kick": "元調子", "torque": "3.0〜3.5", "reason": "重く硬いシャフトで、身体の開きを抑える"}
        
    elif hip_rot < 30:
        c["hip_main"] = (
            "腰の回転が止まり気味で、手打ちになりやすい状態。\n"
            "下半身リードをもっと意識する必要があります。"
        )
        c["hip_pro"] = "下半身のエンジンを使わず、腕力に頼りすぎです。"
        fitting = {"weight": "40g〜50g前半", "flex": "R〜SR", "kick": "先調子", "torque": "4.5〜5.5", "reason": "シャフトの走りで回転不足を補う"}
        
    else:
        c["hip_main"] = (
            "腰の回転量は理想的（45度前後）で、土台としてしっかり機能している。"
        )
        c["hip_pro"] = "プロレベルの安定した下半身使いです。"
        fitting = {"weight": "50g〜60g", "flex": "SR〜S", "kick": "中調子", "torque": "3.8〜4.5", "reason": "癖のない挙動で安定性を最大化"}

    # --- 05. Wrist ---
    if cock < 80:
        c["wrist_main"] = (
            "コック角が深く、タメを作ろうとする意識が強い。\n"
            "タイミング次第で飛ぶ日と飛ばない日の差が出やすいタイプ。"
        )
        c["wrist_pro"] = "リストに依存しすぎています。"
    elif cock > 120:
        c["wrist_main"] = (
            "ノーコックに近いスイング。\n"
            "方向性は安定するが、ヘッドスピードのポテンシャルを活かせていない。"
        )
        c["wrist_pro"] = "安全策を取りすぎです。もっと飛ばせます。"
        if len(drills) < 3: drills.append({"name": "連続素振り", "obj": "手首の柔軟性", "method": "止まらずに連続で振り、遠心力を感じる"})
    else:
        c["wrist_main"] = (
            "適度なコック角が維持され、クラブの重みをうまく扱えている。"
        )
        c["wrist_pro"] = "シンプルで再現性の高い手首使いです。"

    # --- 06. Knee (Logic) ---
    if abs(sway) > 5:
        c["knee_main"] = "スウェーにつられて、膝も一緒に流れている。\n地面反力が逃げてしまっている状態。"
        c["knee_pro"] = "足元のグリップ力が足りていません。"
    else:
        c["knee_main"] = "膝のブレが少なく、地面をしっかり捉えられている。\nインパクトゾーンで下半身が暴れないのは大きな強み。"
        c["knee_pro"] = "粘りのある、良い下半身です。"

    # --- Summary ---
    if len(drills) >= 2:
        c["summary_good"] = "スイング軸と下半身の安定性\n再現性を高めやすい構造"
        c["summary_bad"] = "各パーツの連動不足\n特定の局面での代償動作"
        c["summary_msg"] = "「要素を削ぎ落とし、シンプルにする段階」"
        summary_footer = ("このスイングは、「直せばすぐ変わる」タイプです。\n土台は整っています。あとは上半身の役割を正しく使えるかどうか。\n\nお客様のゴルフライフが、\nより戦略的で、再現性の高いものになることを切に願っています。")
    else:
        c["summary_good"] = "全体のバランスと再現性の高さ\n強固なスイング軸"
        c["summary_bad"] = "特になし（微調整レベル）"
        c["summary_msg"] = "「完成度が高く、スコアに直結するスイング」"
        summary_footer = ("素晴らしいスイングです。\n大きな改造は必要ありません。\n今のリズムを維持しつつ、ショートゲームやマネジメントに磨きをかけてください。")

    return c, drills[:3], fitting, summary_footer


# ==================================================
# CLOUD TASKS UTILITY
# ==================================================

def create_cloud_task(report_id, user_id, message_id):
    """LINEの動画IDとユーザーIDをCloud Tasksに渡す"""
    payload = json.dumps({
        "report_id": report_id, 
        "user_id": user_id, 
        "message_id": message_id
    }).encode("utf-8")
    
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": TASK_HANDLER_URL,
            "headers": {"Content-Type": "application/json"},
            "body": payload,
            "oidc_token": {"service_account_email": TASK_SA_EMAIL, "audience": SERVICE_HOST_URL},
        }
    }
    tasks_client.create_task(parent=queue_path, task=task)


# ==================================================
# API ROUTES (LINE Webhook & Worker)
# ==================================================

@app.route("/webhook", methods=["POST"])
def webhook():
    """LINEからのイベントを受け取るエンドポイント"""
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try: 
        handler.handle(body, signature)
    except InvalidSignatureError: 
        abort(400)
    except LineBotApiError as e:
        print(f"LINE API Error: {e}")
    return "OK"

@handler.add(MessageEvent)
def handle_msg(event: MessageEvent):
    """メッセージイベントハンドラー（動画受信時の処理）"""
    msg = event.message
    
    if isinstance(msg, VideoMessage):
        report_id = f"{event.source.user_id}_{msg.id}"
        
        # 1. Firestoreに初期状態を保存
        db.collection("reports").document(report_id).set({
            "user_id": event.source.user_id,
            "status": "PROCESSING",
            "created_at": datetime.now(timezone.utc).isoformat()
        }, merge=True)
        
        # 2. Cloud Tasksに処理を依頼
        try:
            create_cloud_task(report_id, event.source.user_id, msg.id)
            
            # 3. ユーザーに即座に返信（解析中通知）
            line_bot_api.reply_message(
                event.reply_token, 
                TextSendMessage(text=f"✅ 動画を受信しました。\nスイング解析中です。しばらくお待ちください。\n\n※所要時間：約1〜2分")
            )
        except Exception as e:
            # タスク作成失敗時の処理
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="システムエラーが発生しました。時間を置いて再度お試しください。"))
            print(f"Failed to create task: {e}")
            
    else:
        # 動画以外のメッセージへの一般的な応答
        try:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="スイング動画を送信してください。"))
        except:
            pass


@app.route("/task-handler", methods=["POST"])
def handle_task():
    """
    [Worker] Cloud Tasks から呼び出され、動画をダウンロードし解析を実行する
    """
    d = request.get_json(silent=True)
    report_id = d.get("report_id")
    message_id = d.get("message_id")
    user_id = d.get("user_id")

    if not report_id or not message_id or not user_id: return "Invalid payload", 400

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, f"{message_id}.mp4")
    doc_ref = db.collection("reports").document(report_id)
    
    try:
        # 1. LINEから動画ファイルをダウンロード
        content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for chunk in content.iter_content(): f.write(chunk)

        # 2. MediaPipe解析（重い処理）を実行
        frames_data = extract_mediapipe_data(video_path)

        if not frames_data:
            doc_ref.update({"status": "FAILED", "error": "No valid pose detected"})
            line_bot_api.push_message(user_id, TextSendMessage(text="⚠️ 解析失敗: スイングが検出できませんでした。全身が映るよう撮影し直してください。"))
            return "No frames", 200

        # 3. 指標計算と診断テキスト生成
        metrics = process_landmarks(frames_data)
        comments, drills, fitting, summary_text = generate_pro_quality_text(metrics)

        # 4. JSON構造に整形
        result_json = {
            "01": { "frame_count": metrics["frame_count"], "shoulder_rotation": metrics["shoulder_rotation"], "hip_rotation": metrics["hip_rotation"], "cock_angle": metrics["wrist_cock"], "head_sway": metrics["sway"], "knee_sway": metrics["knee_sway"] },
            "02": { "title": "頭の安定性（軸のブレ）", "sway": metrics["sway"], "comment": comments["head_main"], "pro_view": comments["head_pro"] },
            "03": { "title": "肩の回旋（上半身のねじり）", "comment": comments["shoulder_main"], "pro_view": comments["shoulder_pro"] },
            "04": { "title": "腰の回旋（下半身の動き）", "comment": comments["hip_main"], "pro_view": comments["hip_pro"] },
            "05": { "title": "手首のメカニクス（クラブ操作）", "cock": metrics["wrist_cock"], "comment": comments["wrist_main"], "pro_view": comments["wrist_pro"] },
            "06": { "title": "下半身の安定性", "comment": comments["knee_main"], "pro_view": comments["knee_pro"] },
            "07": { "title": "総合診断", "strong": comments["summary_good"], "weak": comments["summary_bad"], "note": comments["summary_msg"] },
            "08": { "drills": drills },
            "09": { "fitting": fitting },
            "10": { "summary": summary_text }
        }

        # 5. Firestore更新とユーザーへの通知
        doc_ref.update({
            "status": "COMPLETED",
            "report_data": result_json,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "raw_data": frames_data 
        })
        
        report_url = f"{SERVICE_HOST_URL}/report/{report_id}"
        line_bot_api.push_message(user_id, TextSendMessage(text=f"✅ 診断が完了しました。\n以下のURLから診断内容をご確認ください。\n{report_url}"))
        
    except Exception as e:
        print(f"Task Failed (Report ID: {report_id}): {traceback.format_exc()}")
        doc_ref.update({"status": "FAILED", "error": f"Task error: {str(e)}"})
        line_bot_api.push_message(user_id, TextSendMessage(text="システムエラーが発生し、解析を完了できませんでした。"))
        return "Internal Error", 500
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        
    return jsonify({"ok": True}), 200

# --- レポート表示API ---
@app.route("/report/<report_id>")
def serve_report(report_id):
    """HTMLレポートを返す (templates/report.html が必要)"""
    return send_from_directory("templates", "report.html") # 修正済みの関数

@app.route("/api/report_data/<report_id>")
def api_report_data(report_id):
    """レポートデータをJSONで返す (フロントエンド用)"""
    try:
        doc = db.collection("reports").document(report_id).get()
        if not doc.exists: return jsonify({"error": "not found"}), 404
        d = doc.to_dict()
        return jsonify({
            "status": d.get("status"),
            "report_data": d.get("report_data"),
            "created_at": d.get("created_at")
        })
    except Exception:
        return jsonify({"error": "internal error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
