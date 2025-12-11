import os
import tempfile
import shutil
import ffmpeg
import json

from datetime import datetime

# GCP / Firebase / Gemini
from google.cloud import tasks_v2
from google import genai

import firebase_admin
from firebase_admin import credentials, firestore as fb_firestore, initialize_app

# Flask / LINE
from flask import Flask, request, abort, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, VideoMessage

# OpenCV / MediaPipe（本番ではここを実装）
import cv2
import mediapipe as mp
import numpy as np

# ------------------------------------------------
# 環境変数
# ------------------------------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL")
SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL")

if not GCP_PROJECT_ID:
    GCP_PROJECT_ID = "default-gcp-project-id"

TASK_QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION",



