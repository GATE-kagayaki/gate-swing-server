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
# [LOGIC] NATURAL & SHARP DIAGNOSIS
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

def generate_natural_diagnosis(metrics):
    """
    適度な分量の解説と、核心を突くプロ評価を生成する
    """
    c = {} 
    drills = []
    fitting = {}
    
    sway = metrics["sway"]
    xfactor = metrics["x_factor"]
    hip_rot = metrics["hip_rotation"]
    cock = metrics["wrist_cock"]
    
    # ---------------------------------------------------------
    # 02. 頭の安定性 (Sway)
    # ---------------------------------------------------------
    if abs(sway) > 8.0:
        c["head_main"] = (
            f"バックスイングで頭が{sway:.1f}%動いています。\n"
            "パワーを溜めようとして、無意識に右へスライドしている状態です。\n"
            "これだとボールとの距離が変わってしまうため、インパクトが安定しません。"
        )
        c["head_pro"] = "「回転」ではなく「横移動」になっています。"
        drills.append({"name": "クローズスタンス打ち", "obj": "その場で回る感覚", "method": "両足を閉じてスイングし、軸ブレを防ぐ"})
        
    elif abs(sway) < 4.0:
        c["head_main"] = (
            f"頭のズレはわずか{sway:.1f}%で、非常に優秀です。\n"
            "軸がしっかり固定されているため、強く振ってもミート率が落ちにくい構造です。\n"
            "この安定感は大きな武器になります。"
        )
        c["head_pro"] = "「壊れにくいスイング」の土台ができています。"
        
    else:
        c["head_main"] = (
            f"移動量は{sway:.1f}%で、許容範囲内です。\n"
            "ただ、疲れてくると右に流れやすくなる傾向が見え隠れしています。\n"
            "ボールを凝視するよりも、「背骨の角度をキープする」意識を持つとさらに良くなります。"
        )
        c["head_pro"] = "悪くはないですが、もっと「その場」で回れます。"

    # ---------------------------------------------------------
    # 03. 肩の回旋 (Shoulder)
    # ---------------------------------------------------------
    if xfactor < 35:
        c["shoulder_main"] = (
            "肩と腰が一緒に回ってしまい、捻転差（Xファクター）が作れていません。\n"
            "ゴムを伸ばすような「張り」がないため、腕力で飛ばそうとしてしまいます。\n"
            "下半身を止めて、雑巾を絞るように上半身だけを回す意識が必要です。"
        )
        c["shoulder_pro"] = "身体が硬いのではなく、「分離」できていません。"
        drills.append({"name": "椅子座り捻転", "obj": "分離動作の習得", "method": "椅子に座り、下半身を固定して胸だけ回す"})
        
    elif xfactor > 60:
        c["shoulder_main"] = (
            "プロ並みの柔軟性があり、深く回せています。\n"
            "ただ、少し回りすぎてオーバースイング気味です。\n"
            "戻すタイミングが遅れやすく、振り遅れの原因になりかねません。"
        )
        c["shoulder_pro"] = "柔軟性は武器ですが、今は「回りすぎ」です。"
        drills.append({"name": "3秒トップ停止", "obj": "トップの収まり", "method": "トップで3秒止まり、グラつきを確認する"})
        
    else:
        c["shoulder_main"] = (
            "無理なく深い捻転が作れており、理想的なトップの形です。\n"
            "下半身との引っ張り合い（捻転差）も十分で、効率よくパワーを出せる状態です。"
        )
        c["shoulder_pro"] = "文句なし。非常に効率の良いエネルギー構造です。"

    # ---------------------------------------------------------
    # 04. 腰の回旋 (Hip)
    # ---------------------------------------------------------
    if hip_rot > 60:
        c["hip_main"] = (
            "腰がクルッと回りすぎています。\n"
            "上半身より先に腰が逃げてしまうので、力がボールに伝わりきりません。\n"
            "「右足の内側」で地面を噛むように踏ん張ると、回転にブレーキがかかり飛びが変わります。"
        )
        c["hip_pro"] = "下半身が緩く、パワーが逃げています。"
        drills.append({"name": "右足ベタ足打ち", "obj": "腰の開き抑制", "method": "右かかとを上げずにインパクトする"})
        fitting = {"weight": "60g後半〜70g", "flex": "S〜X", "kick": "元調子", "torque": "3.0〜3.5", "reason": "重く硬いシャフトで、身体の開きを抑える"}
        
    elif hip_rot < 30:
        c["hip_main"] = (
            "腰の回転が止まっており、腕だけで振っている状態です。\n"
            "これだとコース後半で疲れてきた時に、急にボールが散らばり始めます。\n"
            "もっと足を使って、下半身リードでクラブを引っ張ってくる感覚が必要です。"
        )
        c["hip_pro"] = "下半身を使わず、腕力に頼りすぎています。"
        fitting = {"weight": "40g〜50g前半", "flex": "R〜SR", "kick": "先調子", "torque": "4.5〜5.5", "reason": "先が走るシャフトで、回転不足を補う"}
        
    else:
        c["hip_main"] = (
            "腰の回転量は45度前後で、プロの平均値と同じです。\n"
            "回りすぎず止まりすぎず、土台としてしっかり機能しています。"
        )
        c["hip_pro"] = "プロレベルの安定した下半身使いです。"
        fitting = {"weight": "50g〜60g", "flex": "SR〜S", "kick": "中調子", "torque": "3.8〜4.5", "reason": "癖のない挙動で安定性を最大化"}

    # ---------------------------------------------------------
    # 05. 手首 (Wrist)
    #
