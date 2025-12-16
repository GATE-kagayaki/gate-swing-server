import os
import json
import math
import shutil
import traceback
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict

from flask import Flask, request, jsonify, abort, render_template

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore
from google.cloud import tasks_v2
from google.api_core.exceptions import NotFound, PermissionDenied

# ==================================================
# CONFIG
# ==================================================
app = Flask(__name__, template_folder="templates")
app.config["JSON_AS_ASCII"] = False

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")

PROJECT_ID = (
    os.environ.get("PROJECT_ID")
    or os.environ.get("GCP_PROJECT_ID")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("GCP_PROJECT")
    or ""
)

QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")
QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")

SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")

TASK_HANDLER_PATH = "/task-handler"
TASK_HANDLER_URL = f"{SERVICE_HOST_URL}{TASK_HANDLER_PATH}"

db = firestore.Client()
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
tasks_client = tasks_v2.CloudTasksClient()


# ==================================================
# Helpers
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
        "動画を受信しました。\n"
        "AIによるスイング解析を開始します。\n\n"
        "解析完了まで、1〜3分ほどお待ちください。\n"
        "完了次第、結果をお知らせします。\n\n"
        "【進行状況の確認】\n"
        "以下のURLから、解析の進行状況や\n"
        "レポートの準備状況を確認できます。\n"
        f"{SERVICE_HOST_URL}/report/{report_id}\n\n"
        "【料金プラン（プロ評価付きフルレポート）】\n"
        "① 都度会員　500円／1回\n"
        "② 回数券　1,980円／5回\n"
        "③ 月額会員　4,980円／月\n\n"
        "※無料版でも骨格解析と総合評価はご利用いただけます。"
    )


def make_done_push(report_id: str) -> str:
    return (
        "🎉 スイング計測が完了しました！\n\n"
        "以下のリンクから診断レポートを確認できます。\n\n"
        f"{SERVICE_HOST_URL}/report/{report_id}"
    )


def is_premium_user(user_id: str) -> bool:
    # TODO: 決済連動に差し替え
    # いまは「有料版を本格実装して先にテストしたい」方針なので True 固定
    return True


# ==================================================
# Cloud Tasks
# ==================================================
def create_cloud_task(report_id: str, user_id: str, message_id: str) -> str:
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID is empty. Set PROJECT_ID or GCP_PROJECT_ID.")
    if not SERVICE_HOST_URL:
        raise RuntimeError("SERVICE_HOST_URL is empty.")
    if not TASK_SA_EMAIL:
        raise RuntimeError("TASK_SA_EMAIL is empty.")

    parent = tasks_client.queue_path(PROJECT_ID, QUEUE_LOCATION, QUEUE_NAME)

    payload = json.dumps(
        {"report_id": report_id, "user_id": user_id, "message_id": message_id},
        ensure_ascii=False,
    ).encode("utf-8")

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": TASK_HANDLER_URL,
            "headers": {"Content-Type": "application/json"},
            "body": payload,
            "oidc_token": {
                "service_account_email": TASK_SA_EMAIL,
                "audience": SERVICE_HOST_URL,
            },
        }
    }

    resp = tasks_client.create_task(parent=parent, task=task)
    return resp.name


# ==================================================
# MediaPipe analysis
# ==================================================
def analyze_swing_with_mediapipe(video_path: str) -> Dict[str, Any]:
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCVがビデオを読み込めませんでした。形式をご確認ください。")

    frame_count = 0
    max_shoulder = 0.0
    min_hip = 999.0
    max_wrist = 0.0
    max_head = 0.0
    max_knee = 0.0

    def angle(p1, p2, p3):
        ax, ay = p1[0] - p2[0], p1[1] - p2[1]
        bx, by = p3[0] - p2[0], p3[1] - p2[1]
        dot = ax * bx + ay * by
        na = math.hypot(ax, ay)
        nb = math.hypot(bx, by)
        if na * nb == 0:
            return 0.0
        c = max(-1.0, min(1.0, dot / (na * nb)))
        return math.degrees(math.acos(c))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
            except Exception:
                continue

            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark

            def xy(i):  # normalized coords
                return (lm[i].x, lm[i].y)

            LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LH = mp_pose.PoseLandmark.LEFT_HIP.value
            RH = mp_pose.PoseLandmark.RIGHT_HIP.value
            LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
            LW = mp_pose.PoseLandmark.LEFT_WRIST.value
            LI = mp_pose.PoseLandmark.LEFT_INDEX.value
            NO = mp_pose.PoseLandmark.NOSE.value
            LK = mp_pose.PoseLandmark.LEFT_KNEE.value

            max_shoulder = max(max_shoulder, angle(xy(LS), xy(RS), xy(RH)))
            min_hip = min(min_hip, angle(xy(LH), xy(RH), xy(LK)))
            max_wrist = max(max_wrist, angle(xy(LE), xy(LW), xy(LI)))
            max_head = max(max_head, abs(xy(NO)[0] - 0.5))
            max_knee = max(max_knee, abs(xy(LK)[0] - 0.5))

    cap.release()

    if frame_count < 10:
        raise RuntimeError("解析に必要なフレーム数が不足しています。もう少し長めの動画でお試しください。")

    return {
        "frame_count": int(frame_count),
        "max_shoulder_rotation": round(float(max_shoulder), 2),
        "min_hip_rotation": round(float(min_hip), 2),
        "max_wrist_cock": round(float(max_wrist), 2),
        "max_head_drift": round(float(max_head), 4),
        "max_knee_sway": round(float(max_knee), 4),
    }


# ==================================================
# Report content builders (確定仕様)
# ==================================================
def build_01(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "01. 骨格計測データ（AIが測定）",
        "items": [
            {
                "name": "解析フレーム数",
                "value": raw["frame_count"],
                "description": "動画から解析できたフレーム数です。数が多いほど、動作全体を安定して解析できています。",
                "guide": "150〜300 フレーム",
            },
            {
                "name": "最大肩回転角（°）",
                "value": raw["max_shoulder_rotation"],
                "description": "スイング中に肩がどれだけ回転したかを示す角度です。上半身の回旋量の指標になります。",
                "guide": "80〜110°",
            },
            {
                "name": "最小腰回転角（°）",
                "value": raw["min_hip_rotation"],
                "description": "スイング中に腰が最も回転した瞬間の角度です。下半身の回旋量を表します。",
                "guide": "35〜45°",
            },
            {
                "name": "最大手首コック角（°）",
                "value": raw["max_wrist_cock"],
                "description": "スイング中に手首が最も折れた角度です。クラブのコック量の指標になります。",
                "guide": "120〜150°",
            },
            {
                "name": "最大頭部ブレ（Sway）",
                "value": raw["max_head_drift"],
                "description": "スイング中に頭の位置が左右にどれだけ動いたかを示します。スイング軸の安定性を表します。",
                "guide": "0.05〜0.15",
            },
            {
                "name": "最大膝ブレ（Sway）",
                "value": raw["max_knee_sway"],
                "description": "スイング中に膝が左右にどれだけ動いたかを示します。下半身の安定性の指標です。",
                "guide": "0.05〜0.20",
            },
        ],
    }


def _judge_range(val: float, lo: float, hi: float) -> str:
    if val < lo:
        return "low"
    if val > hi:
        return "high"
    return "ok"


def build_02_to_06(raw: Dict[str, Any]) -> Dict[str, Any]:
    # 目安（確定）
    sh = raw["max_shoulder_rotation"]  # 80-110
    hip = raw["min_hip_rotation"]      # 35-45
    cock = raw["max_wrist_cock"]       # 120-150
    head = raw["max_head_drift"]       # 0.05-0.15
    knee = raw["max_knee_sway"]        # 0.05-0.20

    sh_j = _judge_range(sh, 80, 110)
    hip_j = _judge_range(hip, 35, 45)
    cock_j = _judge_range(cock, 120, 150)
    head_j = _judge_range(head, 0.05, 0.15)
    knee_j = _judge_range(knee, 0.05, 0.20)

    # 02 Shoulder
    good2, bad2 = [], []
    if sh_j in ("ok", "high"):
        good2.append("回転量は十分で、パワーを生み出せる動きができています。")
    if sh_j == "high":
        bad2.append("回転量がやや多く、回し過ぎによるブレが出やすい可能性があります。")
    if sh_j == "low":
        bad2.append("回転量がやや不足し、飛距離効率が落ちる可能性があります。")
    pro2 = "肩の回転は強みなので、下半身との同調で再現性を引き上げられます。"

    # 03 Hip
    good3, bad3 = [], []
    if hip_j == "ok":
        good3.append("腰の回転量は目安レンジ内で、下半身の土台が作れています。")
    if hip_j == "high":
        good3.append("腰の回転量が十分で、体全体で動かせる土台があります。")
        bad3.append("回転が大きい分、上半身とズレるとタイミングが崩れやすくなります。")
    if hip_j == "low":
        bad3.append("腰の回転が浅くなりやすく、上半身先行になりやすい可能性があります。")
    pro3 = "腰は安定性の核なので、肩との回転差を縮めるとミスが減ります。"

    # 04 Cock
    good4, bad4 = [], []
    if cock_j in ("ok", "high"):
        good4.append("コック量は確保できており、ヘッドスピードを出しやすい形です。")
    if cock_j == "high":
        bad4.append("角度が大きすぎるため、手首主導になりやすい傾向があります。")
    if cock_j == "low":
        bad4.append("コック量が少なめで、リリースが早くなる可能性があります。")
    pro4 = "体の回転でコックが作れると、タイミングが安定して方向性が整います。"

    # 05 Head
    good5, bad5 = [], []
    if head_j == "ok":
        good5.append("頭部の左右移動は目安範囲で、軸の意識が保てています。")
    if head_j == "high":
        bad5.append("頭部の左右移動がやや大きく、インパクト位置がブレやすい可能性があります。")
    if head_j == "low":
        good5.append("頭部の左右移動が小さく、軸が安定しています。")
    pro5 = "頭の位置が安定すると、フェース向きと打点が揃いやすくなります。"

    # 06 Knee
    good6, bad6 = [], []
    if knee_j == "ok":
        good6.append("膝の左右移動は目安範囲で、下半身は比較的安定しています。")
    if knee_j == "high":
        bad6.append("膝の左右移動が大きく、下半身が流れて軸が崩れる可能性があります。")
    if knee_j == "low":
        good6.append("膝の左右移動が小さく、踏ん張りが効いています。")
    pro6 = "膝が安定すると、上半身の回転量を活かしてもブレにくくなります。"

    # 最大3点に丸め
    def cap3(x): return x[:3]

    return {
        "02": {"title": "02. Shoulder Rotation（肩回転）", "value": sh, "good": cap3(good2), "bad": cap3(bad2), "pro_comment": pro2},
        "03": {"title": "03. Hip Rotation（腰回転）", "value": hip, "good": cap3(good3), "bad": cap3(bad3), "pro_comment": pro3},
        "04": {"title": "04. Wrist Cock（コック角）", "value": cock, "good": cap3(good4), "bad": cap3(bad4), "pro_comment": pro4},
        "05": {"title": "05. Head Stability（頭部ブレ）", "value": head, "good": cap3(good5), "bad": cap3(bad5), "pro_comment": pro5},
        "06": {"title": "06. Knee Stability（膝ブレ）", "value": knee, "good": cap3(good6), "bad": cap3(bad6), "pro_comment": pro6},
    }


def build_07_paid(raw: Dict[str, Any]) -> Dict[str, Any]:
    # 02-06を統合したプロ評価寄りの総合
    sh = raw["max_shoulder_rotation"]
    hip = raw["min_hip_rotation"]
    cock = raw["max_wrist_cock"]
    head = raw["max_head_drift"]
    knee = raw["max_knee_sway"]

    # ざっくり因果（合意済みトーン）
    lines = []
    lines.append("今回のスイング解析では、体全体の回転量は十分に確保されており、飛距離を伸ばせるポテンシャルが確認できました。")

    # 安定性
    if head > 0.15 or knee > 0.20:
        lines.append("一方で、頭部または下半身のブレが大きく、インパクト周辺の再現性が落ちやすい傾向が見られます。")
    else:
        lines.append("頭部・下半身のブレは大きく崩れておらず、安定性の土台は整っています。")

    # 肩と腰
    if sh > 110 and hip < 35:
        lines.append("肩の回転が先行しやすく、腰回転が追いつかないことでタイミングがズレやすい可能性があります。")
    elif sh > 110:
        lines.append("肩回転量が多めのため、下半身と同調させることでブレを抑えやすくなります。")
    elif sh < 80:
        lines.append("肩回転量がやや不足気味のため、上半身の回旋を使えると飛距離効率が上がりやすくなります。")

    # コック
    if cock > 150:
        lines.append("コック角が大きめのため、手首主導にならないよう体の回転でクラブを動かす意識が有効です。")

    lines.append("下半身の安定を活かしながら、体の回転とクラブ動作の連動を整えていくことで、方向性と飛距離の両立が期待できます。")

    return {"title": "07. 総合評価（プロ評価）", "text": lines}


def build_08_drills(raw: Dict[str, Any]) -> Dict[str, Any]:
    sh = raw["max_shoulder_rotation"]
    hip = raw["min_hip_rotation"]
    cock = raw["max_wrist_cock"]
    head = raw["max_head_drift"]
    knee = raw["max_knee_sway"]

    drills = []

    # 条件1：肩が多め or 肩-腰のギャップ
    if sh > 110 or (sh - hip) > 70:
        drills.append({
            "name": "上半身と下半身の同調ドリル（クロスアームターン）",
            "purpose": "肩先行を抑え、体全体で回す感覚を作る",
            "how": "①胸の前で腕を軽くクロス\n②下半身を安定させたまま胸と腰を同時に回す\n③肩だけが先に回らないか確認する"
        })

    # 条件2：コック大
    if cock > 150:
        drills.append({
            "name": "L to L スイング",
            "purpose": "手首主導を抑え、体の回転でクラブを動かす",
            "how": "①腰〜腰の振り幅で構える\n②体の回転でクラブを動かす\n③手先で調整しないリズムで反復する"
        })

    # 条件3：ブレが大きい
    if head > 0.15 or knee > 0.20:
        drills.append({
            "name": "壁チェック（ヘッドステイ＋膝安定）",
            "purpose": "軸ブレを抑え、インパクトの再現性を上げる",
            "how": "①壁の近くでアドレスし頭の位置を基準にする\n②ハーフ〜スリークォーターで素振り\n③頭と膝が左右に流れないか確認する"
        })

    # 最大3つ
    drills = drills[:3]

    # 何も該当しない場合の保険
    if not drills:
        drills.append({
            "name": "テンポ一定ドリル（ハーフスイング）",
            "purpose": "回転量を保ったまま再現性を高める",
            "how": "①ハーフスイングで一定テンポ\n②同じ打点・同じリズムを優先\n③力感を上げずに反復する"
        })

    return {"title": "08. Training Drills（練習ドリル）", "drills": drills}


def build_09_fitting(raw: Dict[str, Any]) -> Dict[str, Any]:
    sh = raw["max_shoulder_rotation"]
    cock = raw["max_wrist_cock"]
    head = raw["max_head_drift"]
    knee = raw["max_knee_sway"]

    stability_risk = (head > 0.15) or (knee > 0.20)
    wrist_risk = (cock > 150)
    rotate_risk = (sh > 110)

    # 方針：レンジを削って答えにする
    weight_guide = "50g台後半〜60g台前半"
    weight_reason = "軽すぎると再現性が落ちやすいため"
    if not stability_risk and not wrist_risk:
        weight_guide = "50g台前半〜60g台前半"
        weight_reason = "振り切りやすさと安定性のバランスを取りやすいため"

    kick_guide = "中調子〜中元調子"
    kick_reason = "タイミングと安定性を取りやすいため"
    if rotate_risk or wrist_risk:
        kick_guide = "中調子〜元調子寄り"
        kick_reason = "挙動を安定させ、回し過ぎ・手首主導の影響を抑えやすいため"

    flex_guide = "R〜SR〜S"
    flex_reason = "柔らかすぎるとタイミングが合いにくいため"

    torque_guide = "3.5〜4.5"
    torque_reason = "フェース挙動を安定させやすいため"
    if stability_risk or wrist_risk:
        torque_guide = "3.0〜4.0"
        torque_reason = "ブレを抑えて方向性を安定させやすいため"

    table = [
        {"item": "①重量（40g台〜70g台）", "guide": weight_guide, "reason": weight_reason},
        {"item": "②キックポイント（先・中・元）", "guide": kick_guide, "reason": kick_reason},
        {"item": "③フレックス（L/A/R/SR/S/X）", "guide": flex_guide, "reason": flex_reason},
        {"item": "④トルク（3.0〜6.0）", "guide": torque_guide, "reason": torque_reason},
    ]

    note = "本結果はあくまでも指標です。ご購入の際は試打を行った上でご検討ください。"
    return {"title": "09. Shaft Fitting Guide（推奨）", "table": table, "note": note}


def build_10_summary() -> Dict[str, Any]:
    return {
        "title": "10. Summary（まとめ）",
        "text": [
            "今回のスイング解析では、回転量を活かせるポテンシャルが確認できました。",
            "体の同調と安定性を高めることで、さらなるレベルアップが期待できます。",
            "練習ドリルとフィッティング指針を参考に、段階的に改善を進めていきましょう。",
            "",
            "あなたのゴルフライフが、より充実したものになることを切に願っています。",
        ],
    }


def build_analysis(raw: Dict[str, Any], is_premium: bool) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {"01": build_01(raw)}

    if not is_premium:
        # 無料版は 01 と 07（無料用）だけ運用するなら、ここに無料07を入れる
        analysis["07"] = {
            "title": "07. 総合評価",
            "text": [
                "骨格データからスイング全体の傾向を評価しました。",
                "安定性と回転量のバランスを整えることで、再現性の向上が期待できます。",
                "",
                "より詳しい分析をご希望の方へ",
                "本レポートでは、スイング全体の傾向を骨格データに基づいて評価しています。",
                "ご自身のスイングを深く理解したい方は、ぜひフルレポートをご活用ください。",
            ],
        }
        return analysis

    # 有料版
    analysis.update(build_02_to_06(raw))
    analysis["07"] = build_07_paid(raw)
    analysis["08"] = build_08_drills(raw)
    analysis["09"] = build_09_fitting(raw)
    analysis["10"] = build_10_summary()

    return analysis


# ==================================================
# Routes
# ==================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "project_id": PROJECT_ID,
            "queue_location": QUEUE_LOCATION,
            "queue_name": QUEUE_NAME,
            "service_host_url": SERVICE_HOST_URL,
            "task_handler_url": TASK_HANDLER_URL,
            "task_sa_email_set": bool(TASK_SA_EMAIL),
        }
    )


@app.route("/webhook", methods=["POST"])
def webhook():
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
    msg = event.message
    report_id = f"{user_id}_{msg.id}"

    premium = is_premium_user(user_id)

    firestore_safe_set(
        report_id,
        {
            "user_id": user_id,
            "status": "PROCESSING",
            "is_premium": premium,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    try:
        task_name = create_cloud_task(report_id, user_id, msg.id)
        firestore_safe_update(report_id, {"task_name": task_name})
        safe_line_reply(event.reply_token, make_initial_reply(report_id))
    except (NotFound, PermissionDenied) as e:
        firestore_safe_update(report_id, {"status": "TASK_FAILED", "error": str(e)})
        safe_line_reply(event.reply_token, "システムエラーが発生しました。時間を置いて再度お試しください。")
    except Exception as e:
        firestore_safe_update(report_id, {"status": "TASK_FAILED", "error": str(e)})
        print("Failed to create task:", traceback.format_exc())
        safe_line_reply(event.reply_token, "システムエラーが発生しました。時間を置いて再度お試しください。")


@app.route("/task-handler", methods=["POST"])
def task_handler():
    d = request.get_json(silent=True) or {}
    report_id = d.get("report_id")
    message_id = d.get("message_id")
    user_id = d.get("user_id")

    if not report_id or not message_id or not user_id:
        return "Invalid payload", 400

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, f"{message_id}.mp4")
    doc_ref = db.collection("reports").document(report_id)

    try:
        doc_ref.set({"status": "IN_PROGRESS"}, merge=True)

        # download from LINE
        content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        # analyze
        raw_data = analyze_swing_with_mediapipe(video_path)

        # build report
        premium = bool(doc_ref.get().to_dict().get("is_premium", False))
        analysis = build_analysis(raw_data, premium)

        doc_ref.set(
            {
                "status": "COMPLETED",
                "raw_data": raw_data,
                "analysis": analysis,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        safe_line_push(user_id, make_done_push(report_id))
        return jsonify({"ok": True}), 200

    except Exception as e:
        print(traceback.format_exc())
        doc_ref.set({"status": "FAILED", "error": str(e)}, merge=True)
        safe_line_push(user_id, "システムエラーが発生し、解析を完了できませんでした。")
        return "Internal Error", 500

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route("/report/<report_id>")
def report_page(report_id: str):
    return render_template("report.html", report_id=report_id)


@app.route("/api/report_data/<report_id>")
def api_report_data(report_id: str):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    d = doc.to_dict() or {}
    return jsonify(
        {
            "status": d.get("status"),
            "analysis": d.get("analysis", {}),
            "raw_data": d.get("raw_data", {}),
            "error": d.get("error"),
            "created_at": d.get("created_at"),
            "is_premium": d.get("is_premium", False),
        }
    )


if __name__ == "__main__":
    # Cloud Run では gunicorn 起動が基本。ローカル用。
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
