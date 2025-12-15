# server.py の analyze_swing_with_mediapipe 関数全体をこのコードに置き換えてください

# ==================================================
# MediaPipe analysis (クラッシュ対策強化版)
# ==================================================
def analyze_swing_with_mediapipe(video_path: str) -> Dict[str, Any]:
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
        # 角度計算ロジック（変更なし）
        ax, ay = p1[0] - p2[0], p1[1] - p2[1]
        bx, by = p3[0] - p2[0], p3[1] - p2[1]
        dot = ax * bx + ay * by
        na = math.hypot(ax, ay)
        nb = math.hypot(bx, by)
        if na * nb == 0:
            return 0.0
        c = max(-1.0, min(1.0, dot / (na * nb)))
        return math.degrees(math.acos(c))

    # CPUモード実行を明示的に指定してGPUクラッシュを防ぐ
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        model_solution_cpu_mode=True, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            
            # MediaPipeの処理をtry-exceptで囲み、解析エラーでクラッシュするのを防ぐ
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
            except Exception as e:
                 print(f"MediaPipe processing error in frame {frame_count}: {e}")
                 continue # このフレームはスキップ

            if not res.pose_landmarks:
                continue

            # ランドマーク参照を安全に行う
            lm = res.pose_landmarks.landmark
            def xy(i): return (lm[i].x, lm[i].y)

            # ランドマーク定義（変更なし）
            LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LH = mp_pose.PoseLandmark.LEFT_HIP.value
            RH = mp_pose.PoseLandmark.RIGHT_HIP.value
            LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
            LW = mp_pose.PoseLandmark.LEFT_WRIST.value
            LI = mp_pose.PoseLandmark.LEFT_INDEX.value
            NO = mp_pose.PoseLandmark.NOSE.value
            LK = mp_pose.PoseLandmark.LEFT_KNEE.value

            # 角度計算（変更なし）
            max_shoulder = max(max_shoulder, angle(xy(LS), xy(RS), xy(RH)))
            min_hip = min(min_hip, angle(xy(LH), xy(RH), xy(LK)))
            max_wrist = max(max_wrist, angle(xy(LE), xy(LW), xy(LI)))
            max_head = max(max_head, abs(xy(NO)[0] - 0.5))
            max_knee = max(max_knee, abs(xy(LK)[0] - 0.5))

    cap.release()

    if frame_count < 10:
        raise RuntimeError("解析に必要なフレーム数が不足しています。もう少し長めの動画でお試しください。")

    return {
        "frame_count": frame_count,
        "max_shoulder_rotation": round(max_shoulder, 2),
        "min_hip_rotation": round(min_hip, 2),
        "max_wrist_cock": round(max_wrist, 2),
        "max_head_drift_x": round(max_head, 4),
        "max_knee_sway_x": round(max_knee, 4),
            }

