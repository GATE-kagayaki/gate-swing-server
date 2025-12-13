import os
import json
import math
import traceback
import uuid
import numpy as np
from datetime import datetime, timezone

from flask import Flask, request, jsonify, send_from_directory
from google.cloud import firestore
from google.cloud import tasks_v2

# ==================================================
# CONFIGURATION
# ==================================================
app = Flask(__name__)
db = firestore.Client()

# 環境変数の取得（設定がない場合のデフォルト値も考慮）
QUEUE_NAME = os.getenv("TASK_QUEUE_NAME", "video-analysis-queue")
QUEUE_LOCATION = os.getenv("TASK_QUEUE_LOCATION", "asia-northeast2")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("GCP_PROJECT_ID"))
# Cloud Runの自分自身のURLを指定（タスクが叩く先）
TASK_HANDLER_URL = os.getenv("TASK_HANDLER_URL", f"{os.getenv('SERVICE_HOST_URL', '')}/task-handler")

# ==================================================
# [CORE LOGIC] MATH & TEXT GENERATION
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

def process_landmarks(frames_data):
    """
    時系列の骨格データ(frames_data)から、
    トップ・アドレスを特定し、スイング指標(metrics)を計算する
    """
    if not frames_data:
        return None

    # 1. フェーズ特定 (手首の高さでトップを判定)
    # yは画面下がプラスなので、最小値が一番高い位置
    wrist_ys = [f.get("l_wrist", [0, 1.0])[1] for f in frames_data]
    top_idx = np.argmin(wrist_ys)
    
    # アドレス（トップの少し前）
    search_start = max(0, top_idx - 50)
    address_slice = wrist_ys[search_start:top_idx]
    if len(address_slice) > 0:
        address_idx = search_start + np.argmax(address_slice)
    else:
        address_idx = 0

    d_top = frames_data[top_idx]
    d_addr = frames_data[address_idx]

    # 2. 各種数値計算
    top_shoulder = abs(get_horizontal_angle(d_top.get("l_shoulder"), d_top.get("r_shoulder")))
    top_hip = abs(get_horizontal_angle(d_top.get("l_hip"), d_top.get("r_hip")))
    x_factor = abs(top_shoulder - top_hip)
    
    # Sway (鼻のX座標の変化率)
    nose_top = d_top.get("nose", [0,0])[0]
    nose_addr = d_addr.get("nose", [0,0])[0]
    sway = (nose_top - nose_addr) * 100
    
    # Knee Sway (左膝のX座標変化)
    knee_top = d_top.get("l_knee", [0,0])[0]
    knee_addr = d_addr.get("l_knee", [0,0])[0]
    knee_sway = knee_top - knee_addr

    # Wrist Cock (肩-肘-手首)
    wrist_cock = calculate_angle_3points(d_top.get("l_shoulder"), d_top.get("l_elbow"), d_top.get("l_wrist"))

    return {
        "x_factor": round(x_factor, 1),
        "shoulder_rotation": round(top_shoulder, 1),
        "hip_rotation": round(top_hip, 1),
        "sway": round(sway, 2),
        "knee_sway": round(knee_sway, 4),
        "wrist_cock": round(wrist_cock, 1),
        "frame_count": len(frames_data)
    }

def generate_pro_report(metrics):
    """
    計算されたmetricsを元に、プロ品質の診断コメントとドリルを生成する
    """
    c = {} 
    drills = []
    fitting = {}
    summary_data = {}
    
    sway = metrics["sway"]
    xfactor = metrics["x_factor"]
    hip_rot = metrics["hip_rotation"]
    cock = metrics["wrist_cock"]
    
    # --- 02. Head Sway ---
    if abs(sway) > 8.0:
        c["head"] = {
            "comment": f"バックスイングで頭が{sway:.1f}%動いています。パワーを溜めるつもりが「横移動」になっており、軸が崩壊しています。",
            "pro_view": "「回転」ではなく「スライド」で上げている状態です。"
        }
        drills.append({"name": "クローズスタンス打ち", "purpose": "その場回転の習得", "how": "両足を揃えてスイングし、軸ブレを物理的に防ぐ"})
    elif abs(sway) < 4.0:
        c["head"] = {
            "comment": f"頭のズレは{sway:.1f}%で非常に優秀です。回転軸が明確で、再現性の高いインパクトが見込めます。",
            "pro_view": "「壊れにくいスイング」の土台ができています。"
        }
    else:
        c["head"] = {
            "comment": f"移動量は{sway:.1f}%で許容範囲内ですが、疲労時に軸がブレるリスクがあります。",
            "pro_view": "悪くはないですが、もっと「その場」で回れます。"
        }

    # --- 03. Shoulder & X-Factor ---
    if xfactor < 35:
        c["shoulder"] = {
            "comment": "肩と腰が同調して回り、捻転差が作れていません。ゴムを伸ばすような「張り」がないため、腕力に頼りがちです。",
            "pro_view": "身体が硬いのではなく、「分離」できていません。"
        }
        drills.append({"name": "椅子座り捻転", "purpose": "分離動作の習得", "how": "椅子に座り、下半身をロックして胸だけ回す"})
    elif xfactor > 60:
        c["shoulder"] = {
            "comment": "プロ並みの柔軟性がありますが、回りすぎてオーバースイング気味です。戻すタイミングが遅れる原因になります。",
            "pro_view": "柔軟性は武器ですが、今は「回りすぎ」です。"
        }
        drills.append({"name": "3秒トップ停止", "purpose": "トップの収まり確認", "how": "トップで3秒止まり、グラつきがないか確認する"})
    else:
        c["shoulder"] = {
            "comment": "無理なく深い捻転が作れており、理想的なトップの形です。効率よくパワーを出せる状態です。",
            "pro_view": "文句なし。非常に効率の良いエネルギー構造です。"
        }

    # --- 04. Hip Rotation ---
    if hip_rot > 60:
        c["hip"] = {
            "comment": "腰が回りすぎており、上半身との捻転差が消えています。右足で地面を踏ん張り、回転にブレーキをかける意識が必要です。",
            "pro_view": "下半身が緩く、パワーが逃げています。"
        }
        drills.append({"name": "右足ベタ足打ち", "purpose": "腰の開き抑制", "how": "右かかとを上げずにインパクトする"})
        fitting = {"weight": "60g後半〜70g", "flex": "S〜X", "kick": "元調子", "torque": "3.0〜3.5", "note": "重く硬いシャフトで身体の開きを抑える"}
    elif hip_rot < 30:
        c["hip"] = {
            "comment": "腰の回転が止まっており、手打ちの傾向があります。下半身リードでクラブを引っ張る動きが必要です。",
            "pro_view": "下半身を使わず、腕力に頼りすぎています。"
        }
        fitting = {"weight": "40g〜50g前半", "flex": "R〜SR", "kick": "先調子", "torque": "4.5〜5.5", "note": "先が走るシャフトで回転不足を補う"}
    else:
        c["hip"] = {
            "comment": "腰の回転量は45度前後で理想的です。土台としてしっかり機能しています。",
            "pro_view": "プロレベルの安定した下半身使いです。"
        }
        fitting = {"weight": "50g〜60g", "flex": "SR〜S", "kick": "中調子", "torque": "3.8〜4.5", "note": "癖のない挙動で安定性を最大化"}

    # --- 05. Wrist ---
    if cock < 80:
        c["wrist"] = {
            "comment": "コックが深く、タメを作る意識が強いです。タイミングが合えば飛びますが、日によって調子の波が出やすいです。",
            "pro_view": "リストに依存しすぎています。"
        }
    elif cock > 120:
        c["wrist"] = {
            "comment": "ノーコックに近く、ヘッドスピードを上げるための「てこの原理」を使えていません。",
            "pro_view": "安全策を取りすぎて、ポテンシャルを捨てています。"
        }
        if len(drills) < 3:
            drills.append({"name": "連続素振り", "purpose": "手首の柔軟性", "how": "止まらずに連続で振り、遠心力を感じる"})
    else:
        c["wrist"] = {
            "comment": "適度なコック角で、クラブの重みをうまく扱えています。",
            "pro_view": "シンプルで再現性の高い手首使いです。"
        }

    # --- 06. Knee (based on sway logic) ---
    if abs(sway) > 5:
        c["knee"] = {"comment": "スウェーに伴い、膝も一緒に流れてしまっています。これでは地面反力が使えません。"}
    else:
        c["knee"] = {"comment": "膝のブレが少なく、地面をしっかり捉えられています。"}

    # --- Summary & Drills ---
    if len(drills) >= 2:
        summary_data = {
            "strong": ["スイングへの意欲", "ポテンシャルの高さ"],
            "weak": ["各パーツの連動不足", "過剰な代償動作"],
            "note": "「要素を削ぎ落とし、シンプルにする段階」",
            "summary_text": "今は難しく考えすぎているかもしれません。土台（アドレス・軸）を直すだけで、驚くほど楽に打てるようになります。"
        }
    else:
        summary_data = {
            "strong": ["強固なスイング軸", "再現性の高さ"],
            "weak": ["特になし（微調整レベル）"],
            "note": "「完成度が高く、スコアに直結するスイング」",
            "summary_text": "素晴らしいスイングです。大きな改造は不要です。今のリズムを維持し、マネジメントに注力してください。"
        }
        if not drills:
            drills.append({"name": "片手打ち", "purpose": "リズム維持", "how": "片手でウェッジを持ち、ゆったり振る"})

    return c, drills[:3], fitting, summary_data


# ==================================================
# API ROUTES
# ==================================================

@app.route("/report/<report_id>")
def serve_report(report_id):
    """HTMLレポートを返す（変更なし）"""
    return send_from_directory("templates", "report.html")

@app.route("/api/report_data/<report_id>")
def api_report_data(report_id):
    """レポートデータをJSONで返す"""
    try:
        doc = db.collection("reports").document(report_id).get()
        if not doc.exists:
            return jsonify({"error": "not found"}), 404

        d = doc.to_dict()
        return jsonify({
            "status": d.get("status"),
            "report_data": d.get("report_data"),
            "created_at": d.get("created_at")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/submit_swing", methods=["POST"])
def submit_swing():
    """
    スイングデータ(JSON)を受け取り、非同期タスクを作成する
    """
    try:
        data = request.json
        if not data or "mediapipe_data" not in data:
            return jsonify({"error": "Invalid payload: 'mediapipe_data' is required"}), 400

        # レポートIDの発行
        report_id = f"{uuid.uuid4().hex}_{int(datetime.now(timezone.utc).timestamp())}"
        
        # Firestoreに初期状態を保存 (raw_dataとしてmediapipeのリストを保存)
        db.collection("reports").document(report_id).set({
            "status": "pending",
            "raw_data": data["mediapipe_data"], # 骨格座標のリストを想定
            "created_at": datetime.now(timezone.utc).isoformat()
        })

        # Cloud Tasks に処理を依頼
        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(PROJECT_ID, QUEUE_LOCATION, QUEUE_NAME)

        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": TASK_HANDLER_URL,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"report_id": report_id}).encode()
            }
        }
        client.create_task(parent=parent, task=task)

        return jsonify({"report_id": report_id, "status": "queued"})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/task-handler", methods=["POST"])
def handle_task():
    """
    [Worker] バックグラウンドで解析を実行する
    """
    try:
        payload = request.json
        report_id = payload.get("report_id")
        if not report_id:
            return "No report_id", 400

        doc_ref = db.collection("reports").document(report_id)
        doc = doc_ref.get()

        if not doc.exists:
            print(f"Report {report_id} not found.")
            return "Report not found", 404

        # 生データ（骨格座標リスト）を取得
        raw_frames = doc.to_dict().get("raw_data", [])
        
        if not raw_frames:
            print("No raw frames to process.")
            doc_ref.update({"status": "failed", "error": "No raw data"})
            return "No raw data", 200

        # --- 1. 計算実行 (Real Math) ---
        metrics = process_landmarks(raw_frames)
        
        if not metrics:
            doc_ref.update({"status": "failed", "error": "Calculation failed"})
            return "Calculation failed", 200

        # --- 2. 診断テキスト生成 (Pro Text) ---
        comments, drills, fitting, summary = generate_pro_report(metrics)

        # --- 3. フロントエンド互換のJSON構造に整形 ---
        # report.html が期待する構造 ("01"〜"10") に合わせる
        result_json = {
            "01": { # 数値データ
                "frame_count": metrics["frame_count"],
                "shoulder_rotation": metrics["shoulder_rotation"],
                "hip_rotation": metrics["hip_rotation"],
                "cock_angle": metrics["wrist_cock"],
                "head_sway": metrics["sway"],
                "knee_sway": metrics["knee_sway"]
            },
            "02": { # 頭
                "title": "頭の安定性（軸のブレ）",
                "sway": metrics["sway"],
                "comment": comments["head"]["comment"],
                "pro_view": comments["head"]["pro_view"]
            },
            "03": { # 肩
                "title": "肩の回旋（上半身のねじり）",
                "comment": comments["shoulder"]["comment"],
                "pro_view": comments["shoulder"]["pro_view"]
            },
            "04": { # 腰
                "title": "腰の回旋（下半身の動き）",
                "comment": comments["hip"]["comment"],
                "pro_view": comments["hip"]["pro_view"]
            },
            "05": { # 手首
                "title": "手首のメカニクス（クラブ操作）",
                "cock": metrics["wrist_cock"],
                "comment": comments["wrist"]["comment"],
                "pro_view": comments["wrist"]["pro_view"]
            },
            "06": { # 膝
                "title": "下半身の安定性",
                "comment": comments["knee"]["comment"]
            },
            "07": { # 総合診断
                "title": "総合診断",
                "strong": summary["strong"],
                "weak": summary["weak"],
                "note": summary["note"]
            },
            "08": { # ドリル
                "drills": drills
            },
            "09": { # フィッティング
                "fitting": fitting
            },
            "10": { # まとめ
                "summary": summary["summary_text"]
            }
        }

        # Firestore更新
        doc_ref.update({
            "status": "completed",
            "report_data": result_json,
            "updated_at": datetime.now(timezone.utc).isoformat()
        })

        print(f"Report {report_id} completed successfully.")
        return "OK", 200

    except Exception as e:
        print(f"Task failed: {traceback.format_exc()}")
        # リトライを防ぐため200系を返すが、ステータスはfailedにする
        return "Internal Error", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
