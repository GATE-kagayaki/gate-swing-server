import os
import json
import math
import shutil
import traceback
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from flask import Flask, request, jsonify, abort, send_from_directory

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore
from google.cloud import tasks_v2
from google.api_core.exceptions import NotFound, PermissionDenied


# ==================================================
# CONFIG
# ==================================================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")

SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")

# キュー設定（env名はあなたの運用に合わせて統一）
QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", os.environ.get("QUEUE_NAME", "video-analysis-queue"))
QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", os.environ.get("QUEUE_LOCATION", "asia-northeast2"))

TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")

# ✅ PROJECT ID は “確実に” gate-swing-analyzer を拾う
PROJECT_ID = (
    os.environ.get("PROJECT_ID")
    or os.environ.get("GCP_PROJECT_ID")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("GCP_PROJECT")
    or ""
)

TASK_HANDLER_PATH = "/task-handler"
TASK_HANDLER_URL = f"{SERVICE_HOST_URL}{TASK_HANDLER_PATH}"

db = firestore.Client()
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
tasks_client = tasks_v2.CloudTasksClient()


# ==================================================
# Helpers (safe)
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


# ==================================================
# LINE messages
# ==================================================
def make_initial_reply(report_id: str) -> str:
    url = f"{SERVICE_HOST_URL}/report/{report_id}"
    return (
        "✅ 動画を受信しました。\n"
        "AIによるスイング解析を開始します。\n\n"
        "解析完了まで、1〜3分ほどお待ちください。\n"
        "完了次第、結果をお知らせします。\n\n"
        "【進行状況の確認】\n"
        "以下のURLから、解析の進行状況や\n"
        "レポートの準備状況を確認できます。\n"
        f"{url}\n\n"
        "【料金プラン（プロ評価付きフルレポート）】\n"
        "① 都度会員　500円／1回\n"
        "② 回数券　1,980円／5回\n"
        "③ 月額会員　4,980円／月\n\n"
        "※無料版でも骨格解析と総合評価はご利用いただけます。"
    )


def make_done_push(report_id: str) -> str:
    url = f"{SERVICE_HOST_URL}/report/{report_id}"
    return (
        "🎉 スイング計測が完了しました！\n\n"
        "以下のリンクから診断レポートを確認できます。\n\n"
        f"{url}"
    )


# ==================================================
# Premium 판단（まずは安全に：Firestore users/{user_id}.is_premium を参照）
#   - 無ければ False（無料）
#   - テストで強制したいなら FORCE_PREMIUM=true
# ==================================================
def is_premium_user(user_id: str) -> bool:
    if os.environ.get("FORCE_PREMIUM", "").lower() in ("1", "true", "yes"):
        return True
    try:
        doc = db.collection("users").document(user_id).get()
        if doc.exists:
            d = doc.to_dict() or {}
            return bool(d.get("is_premium"))
    except Exception:
        print(traceback.format_exc())
    return False


# ==================================================
# Cloud Tasks
# ==================================================
def create_cloud_task(report_id: str, user_id: str, message_id: str) -> str:
    # ここで落ちたら “必ず” ログ・Firestoreに残す
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID is empty. Set PROJECT_ID or GCP_PROJECT_ID.")
    if not SERVICE_HOST_URL:
        raise RuntimeError("SERVICE_HOST_URL is empty.")
    if not TASK_SA_EMAIL:
        raise RuntimeError("TASK_SA_EMAIL is empty.")
    if not QUEUE_NAME:
        raise RuntimeError("QUEUE_NAME is empty.")
    if not QUEUE_LOCATION:
        raise RuntimeError("QUEUE_LOCATION is empty.")

    queue_path = tasks_client.queue_path(PROJECT_ID, QUEUE_LOCATION, QUEUE_NAME)

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

    # デバッグログ（Cloud Runログで追える）
    print("create_task:", {
        "project": PROJECT_ID,
        "location": QUEUE_LOCATION,
        "queue": QUEUE_NAME,
        "url": TASK_HANDLER_URL,
        "sa": TASK_SA_EMAIL,
        "report_id": report_id,
    })

    resp = tasks_client.create_task(parent=queue_path, task=task)
    return resp.name


# ==================================================
# MediaPipe analysis（遅延importで起動を軽く）
# ==================================================
def analyze_swing_with_mediapipe(video_path: str) -> Dict[str, Any]:
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("OpenCVがビデオファイルを読み込めませんでした。ファイル形式を確認してください。")

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
            except Exception as e:
                print(f"MediaPipe frame error {frame_count}: {e}")
                continue

            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            def xy(i): return (lm[i].x, lm[i].y)

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
        "max_head_drift_x": round(float(max_head), 4),
        "max_knee_sway_x": round(float(max_knee), 4),
    }


# ==================================================
# Report generation (Free/Premium)
# ==================================================
def _in_range(val: float, lo: float, hi: float) -> bool:
    return lo <= val <= hi


def build_section_01(raw: Dict[str, Any]) -> Dict[str, Any]:
    # ユーザー確定仕様（目安は平均レンジのみ）
    items = [
        {
            "name": "解析フレーム数",
            "value": raw["frame_count"],
            "unit": "フレーム",
            "guide": "150～300",
            "desc": "動画から解析できたフレーム数です。数が多いほど、動作全体を安定して解析できています。",
        },
        {
            "name": "最大肩回転角",
            "value": raw["max_shoulder_rotation"],
            "unit": "°",
            "guide": "80～110°",
            "desc": "スイング中に肩がどれだけ回転したかを示す角度です。上半身の回旋量の指標になります。",
        },
        {
            "name": "最小腰回転角",
            "value": raw["min_hip_rotation"],
            "unit": "°",
            "guide": "35～45°",
            "desc": "スイング中に腰が最も回転した瞬間の角度です。下半身の回旋量を表します。",
        },
        {
            "name": "最大手首コック角",
            "value": raw["max_wrist_cock"],
            "unit": "°",
            "guide": "120～150°",
            "desc": "スイング中に手首が最も折れた角度です。クラブのコック量の指標になります。",
        },
        {
            "name": "最大頭部ブレ（Sway）",
            "value": raw["max_head_drift_x"],
            "unit": "",
            "guide": "0.05～0.15",
            "desc": "スイング中に頭の位置が左右にどれだけ動いたかを示します。スイング軸の安定性を表します。",
        },
        {
            "name": "最大膝ブレ（Sway）",
            "value": raw["max_knee_sway_x"],
            "unit": "",
            "guide": "0.05～0.20",
            "desc": "スイング中に膝が左右にどれだけ動いたかを示します。下半身の安定性の指標です。",
        },
    ]
    return {"title": "01. 骨格計測データ（AIが測定）", "items": items}


def build_eval_section(
    key: str,
    title_en: str,
    title_ja: str,
    metric_name: str,
    value: float,
    unit: str,
    guide: Tuple[float, float],
    pro_line: str,
) -> Dict[str, Any]:
    lo, hi = guide
    good: List[str] = []
    bad: List[str] = []

    # 定型ではなく “数値に応じて” 変える（最低限でも根拠が数値）
    if _in_range(value, lo, hi):
        good.append(f"{metric_name}は目安レンジ内で、スイング効率を出しやすい状態です。")
        good.append("動きの再現性が出やすく、調整の効果も反映されやすいです。")
    elif value < lo:
        bad.append(f"{metric_name}が目安より小さく、回旋量（/可動域）が不足している可能性があります。")
        bad.append("飛距離や打ち出し角が伸びきらず、タイミングの取りづらさにつながることがあります。")
        good.append("反面、動きがコンパクトでミートは安定しやすい傾向があります。")
    else:  # value > hi
        bad.append(f"{metric_name}が目安より大きく、回し過ぎ・動き過多でブレが出る可能性があります。")
        bad.append("再現性が落ちると、方向性やミート率が不安定になりやすいです。")
        good.append("一方で、ハマると強い球やパワーを出せる要素は持っています。")

    # 3点上限
    good = good[:3]
    bad = bad[:3]

    return {
        "title": f"{key}. {title_en}（{title_ja}）",
        "value_line": f"{metric_name}: {value}{unit}（目安 {lo}～{hi}{unit}）",
        "good": good,
        "bad": bad,
        "pro": pro_line,
    }


def build_section_07(raw: Dict[str, Any], premium: bool) -> Dict[str, Any]:
    shoulder = float(raw["max_shoulder_rotation"])
    hip = float(raw["min_hip_rotation"])
    wrist = float(raw["max_wrist_cock"])
    head = float(raw["max_head_drift_x"])
    knee = float(raw["max_knee_sway_x"])

    good_parts: List[str] = []
    bad_parts: List[str] = []

    # ざっくり総合（01〜06の数値で分岐）
    if head <= 0.15 and knee <= 0.20:
        good_parts.append("頭部と下半身のブレが小さく、スイング軸は安定傾向です。")
    else:
        bad_parts.append("頭部または下半身のブレが大きく、再現性が落ちやすい傾向があります。")

    if 80 <= shoulder <= 110:
        good_parts.append("肩回転量は目安レンジ内で、上半身の回旋は良好です。")
    elif shoulder < 80:
        bad_parts.append("肩回転量が少なめで、上半身の回旋量が不足している可能性があります。")
    else:
        bad_parts.append("肩回転量が多めで、回し過ぎがブレの原因になる可能性があります。")

    if 35 <= hip <= 45:
        good_parts.append("腰回転（下半身の回旋）は目安レンジ内で、土台は作れています。")
    elif hip < 35:
        bad_parts.append("腰回転が少なめで、下半身主導が作りきれていない可能性があります。")
    else:
        bad_parts.append("腰回転が大きめで、上半身との同調が崩れるとブレが出やすくなります。")

    if 120 <= wrist <= 150:
        good_parts.append("手首コックは目安レンジ内で、溜めとリリースのバランスは良好です。")
    elif wrist < 120:
        bad_parts.append("コック角が小さめで、溜めが作れず球が弱くなる可能性があります。")
    else:
        bad_parts.append("コック角が大きめで、手首主導になりタイミングがズレやすい可能性があります。")

    text: List[str] = []
    text.append("**良い点**")
    for s in good_parts[:3]:
        text.append(f"・{s}")
    text.append("")
    text.append("**改善が期待できる点**")
    for s in bad_parts[:3]:
        text.append(f"・{s}")

    if premium:
        text.append("")
        text.append("**プロ評価（追記）**")
        text.append("・数値は良い要素が出ています。次は「ブレの原因になっている動き」だけを削ると、一気に再現性が上がります。")

    return {"title": "07. 総合評価", "text": text}


def build_section_08(raw: Dict[str, Any]) -> Dict[str, Any]:
    shoulder = float(raw["max_shoulder_rotation"])
    wrist = float(raw["max_wrist_cock"])
    head = float(raw["max_head_drift_x"])
    knee = float(raw["max_knee_sway_x"])

    drills = []

    # 数値条件で可変（最大3）
    if shoulder < 80:
        drills.append({
            "ドリル名": "クロスアームターン",
            "目的": "上半身の回旋量を増やす",
            "やり方": "①胸の前で腕を軽く組む\n②下半身をできるだけ動かさず胸を回す\n③左右交互に10回×2セット",
        })
    if wrist > 150:
        drills.append({
            "ドリル名": "L to L スイング",
            "目的": "手首主導を抑え、体の回転で打つ",
            "やり方": "①腰〜腰の小さい振り幅で構える\n②体の回転でクラブを動かす\n③一定リズムで20球",
        })
    if head > 0.15 or knee > 0.20:
        drills.append({
            "ドリル名": "ウォールターン",
            "目的": "頭・下半身のブレを抑え軸を安定させる",
            "やり方": "①壁を背にしてアドレス\n②頭の位置を固定して肩だけ回す\n③壁との距離が変わらないか確認",
        })

    # 何も引っかからない場合の保険
    if not drills:
        drills.append({
            "ドリル名": "スロー素振り（3秒トップ）",
            "目的": "動作の同調と再現性アップ",
            "やり方": "①ゆっくり上げてトップで3秒止める\n②体の回転で振り下ろす\n③10回×2セット",
        })

    return {"title": "08. 改善戦略と練習ドリル", "drills": drills[:3]}


def build_section_09(raw: Dict[str, Any]) -> Dict[str, Any]:
    # 指示どおり：①重量 ②キック ③フレックス ④トルク
    # 数値条件で可変（簡易ルール）
    shoulder = float(raw["max_shoulder_rotation"])
    wrist = float(raw["max_wrist_cock"])
    head = float(raw["max_head_drift_x"])
    knee = float(raw["max_knee_sway_x"])

    stable = (head <= 0.15 and knee <= 0.20)
    wristy = (wrist > 150)
    low_turn = (shoulder < 80)

    # 重量（40〜70g）
    if stable and not wristy:
        weight = "55〜65g"
        weight_reason = "安定性があるため、少し重量を上げても振り遅れにくい"
    elif wristy:
        weight = "60〜70g"
        weight_reason = "手首主導を抑え、手元の安定を出しやすい"
    else:
        weight = "45〜55g"
        weight_reason = "回旋量不足/ブレがある場合、振り切りやすさを優先"

    # キック（先・中・元）
    if low_turn:
        kick = "先調子"
        kick_reason = "打ち出しを確保しやすく、球の弱さを補いやすい"
    elif wristy:
        kick = "元調子"
        kick_reason = "手元が落ち着きやすく、タイミングを作りやすい"
    else:
        kick = "中調子"
        kick_reason = "クセが少なく、全体バランスが取りやすい"

    # フレックス（HSを取ってないので「目安レンジ」提示）
    # ここは“指標”として幅を出す（ユーザー指示）
    if stable and not wristy:
        flex = "SR〜S"
        flex_reason = "再現性を優先しつつ、振り遅れにくい帯域"
    elif wristy:
        flex = "S〜X"
        flex_reason = "手元の暴れを抑え、当たり負けを防ぎやすい"
    else:
        flex = "R〜SR"
        flex_reason = "振りやすさを優先し、タイミングを合わせやすい"

    # トルク（3.0〜6.0）
    if wristy:
        torque = "3.0〜3.8"
        torque_reason = "手元のねじれを抑え、方向性を安定させやすい"
    elif stable:
        torque = "3.8〜4.8"
        torque_reason = "安定性があるため、振りやすさとのバランスを取りやすい"
    else:
        torque = "4.5〜6.0"
        torque_reason = "しなり感でタイミングを取りやすく、振り抜きやすい"

    table = [
        {"項目": "① 重量", "推奨": weight, "理由": weight_reason},
        {"項目": "② キックポイント", "推奨": kick, "理由": kick_reason},
        {"項目": "③ フレックス", "推奨": flex, "理由": flex_reason},
        {"項目": "④ トルク", "推奨": torque, "理由": torque_reason},
    ]

    note = "本結果はあくまでも指標ですので、ご購入の際は試打をしてからご検討ください。"
    return {"title": "09. シャフトフィッティング（推奨）", "fitting_table": table, "note": note}


def build_section_10() -> Dict[str, Any]:
    text = [
        "今回の計測では、スイングの良い要素がしっかり数値に出ています。",
        "一方で、ブレにつながる要素がある場合は「動きの量を減らす」だけで改善が早まります。",
        "練習ドリルは短時間でも継続するほど効果が出やすいです。",
        "シャフトは指標をもとに、実際の振り心地を優先して選んでください。",
        "あなたのゴルフライフが充実したものになることを切に願っています。",
    ]
    return {"title": "10. まとめ", "text": text}


def build_analysis(raw: Dict[str, Any], premium: bool) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {}
    analysis["01"] = build_section_01(raw)

    # 有料は 02〜06（プロ評価）を出す
    if premium:
        analysis["02"] = build_eval_section(
            "02",
            "Shoulder Rotation",
            "肩回転",
            "肩回転角",
            float(raw["max_shoulder_rotation"]),
            "°",
            (80.0, 110.0),
            "プロ目線：回旋量は『多ければ良い』ではなく、ミートが安定する範囲で再現できることが最優先です。",
        )
        analysis["03"] = build_eval_section(
            "03",
            "Hip Rotation",
            "腰回転",
            "腰回転角",
            float(raw["min_hip_rotation"]),
            "°",
            (35.0, 45.0),
            "プロ目線：腰は『回す』よりも、上半身と同調して回る形が作れると一気に安定します。",
        )
        analysis["04"] = build_eval_section(
            "04",
            "Wrist Cock",
            "コック角",
            "手首コック角",
            float(raw["max_wrist_cock"]),
            "°",
            (120.0, 150.0),
            "プロ目線：コックは“溜め”ですが、過多になると手首主導になりやすいのでリズム管理が鍵です。",
        )
        analysis["05"] = build_eval_section(
            "05",
            "Head Sway",
            "頭部ブレ",
            "頭部ブレ（Sway）",
            float(raw["max_head_drift_x"]),
            "",
            (0.05, 0.15),
            "プロ目線：頭のブレは再現性に直結します。まずは『小さく動く』より『動かさない』が近道です。",
        )
        analysis["06"] = build_eval_section(
            "06",
            "Knee Sway",
            "膝ブレ",
            "膝ブレ（Sway）",
            float(raw["max_knee_sway_x"]),
            "",
            (0.05, 0.20),
            "プロ目線：膝の左右動は体重移動の“量”が出すぎているサインになりやすいです。",
        )

    # 07（無料でも表示、有料は追記あり）
    analysis["07"] = build_section_07(raw, premium=premium)

    # 08/09/10 は有料のみ（あなたの指示）
    if premium:
        analysis["08"] = build_section_08(raw)
        analysis["09"] = build_section_09(raw)
        analysis["10"] = build_section_10()

    return analysis


# ==================================================
# Routes
# ==================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "project_id": PROJECT_ID,
        "queue_location": QUEUE_LOCATION,
        "queue_name": QUEUE_NAME,
        "service_host_url": SERVICE_HOST_URL,
        "task_handler_url": TASK_HANDLER_URL,
        "task_sa_email_set": bool(TASK_SA_EMAIL),
    })


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
def on_video(event: MessageEvent):
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
        print("create_task error:", traceback.format_exc())
        safe_line_reply(event.reply_token, "システムエラーが発生しました。時間を置いて再度お試しください。")
    except Exception as e:
        firestore_safe_update(report_id, {"status": "TASK_FAILED", "error": str(e)})
        print("create_task error:", traceback.format_exc())
        safe_line_reply(event.reply_token, "システムエラーが発生しました。時間を置いて再度お試しください。")


@app.route("/task-handler", methods=["POST"])
def task_handler():
    d = request.get_json(silent=True) or {}
    report_id = d.get("report_id")
    message_id = d.get("message_id")
    user_id = d.get("user_id")

    print("TASK_HANDLER_CALLED", {"report_id": report_id, "message_id": message_id, "user_id": user_id})

    if not report_id or not message_id or not user_id:
        return "Invalid payload", 400

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, f"{message_id}.mp4")
    doc_ref = db.collection("reports").document(report_id)

    try:
        doc_ref.set({"status": "IN_PROGRESS"}, merge=True)

        # 1) download LINE video
        content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        # 2) analyze
        raw_data = analyze_swing_with_mediapipe(video_path)

        # 3) build report
        # premium flag is stored in report doc (from webhook time)
        snap = doc_ref.get()
        premium = False
        if snap.exists:
            premium = bool((snap.to_dict() or {}).get("is_premium"))

        analysis = build_analysis(raw_data, premium=premium)

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
    # ✅ Jinjaを使わず、templates/report.html をそのまま返す（TemplateNotFound事故を根絶）
    return send_from_directory("templates", "report.html")


@app.route("/api/report_data/<report_id>")
def api_report_data(report_id: str):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404

    d = doc.to_dict() or {}
    status_raw = (d.get("status") or "").strip()
    status_norm = status_raw.lower()

    return jsonify(
        {
            # ✅ report.html 側の互換性のため “statusはlower” を返す
            "status": status_norm,
            "status_raw": status_raw,
            "analysis": d.get("analysis", {}),
            "raw_data": d.get("raw_data", {}),
            "error": d.get("error"),
            "created_at": d.get("created_at"),
            "is_premium": bool(d.get("is_premium")),
        }
    )


if __name__ == "__main__":
    # Cloud Run では PORT を必ず使う
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
