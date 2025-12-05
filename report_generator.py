import cv2
import mediapipe as mp
import numpy as np

# MediaPipe設定
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    3点(a, b, c)から角度bを計算する関数 (2D)
    a, b, c はそれぞれ [x, y] または [x, y, z] の座標リスト/配列
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    # 2D平面での角度計算
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    # 180度以上の場合、反対側の角度を使う
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def get_midpoint(p1, p2):
    """2点の平均座標を計算する"""
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def analyze_swing(video_path):
    """
    動画を解析し、スイングの評価レポート（テキスト）を返します。
    """
    cap = cv2.VideoCapture(video_path)
    
    # 初期値の設定
    max_shoulder_rotation = 0
    max_hip_rotation = 0
    address_spine_angle = None # アドレス時の前傾角度
    
    # MediaPipeモデルの初期化
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose:
        
        frame_count = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # RGB変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 推論実行
            results = pose.process(image)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 必要なランドマークの座標を取得
                # 11:左肩, 12:右肩, 23:左腰, 24:右腰
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                # 中央点の計算
                shoulder_mid = get_midpoint(left_shoulder, right_shoulder)
                hip_mid = get_midpoint(left_hip, right_hip)
                
                # 1. 前傾角度の計算 (Spine Angle)
                # 基準点として、ヒップの中点から垂直に伸びる点を使用 ([x, y - 0.5]でY軸を上に設定)
                vertical_ref = [hip_mid[0], hip_mid[1] - 0.5] 

                current_spine_angle = calculate_angle(
                    vertical_ref, hip_mid, shoulder_mid
                )
                
                # アドレス時の前傾角度の記録 (最初の10フレームの平均を使用)
                if frame_count < 10:
                    if address_spine_angle is None:
                        address_spine_angle = current_spine_angle
                    else:
                        address_spine_angle = (address_spine_angle * frame_count + current_spine_angle) / (frame_count + 1)
                
                # 2. 肩の回転角度 (2Dの簡易的な傾き)
                # 左肩を基準点、右肩のX座標を基にした垂直な点
                shoulder_rotation = calculate_angle(
                    [left_shoulder[0], left_shoulder[1] + 0.1], left_shoulder, right_shoulder
                )
                if shoulder_rotation > max_shoulder_rotation:
                    max_shoulder_rotation = shoulder_rotation

                # 3. 腰の回転角度 (2Dの簡易的な傾き)
                hip_rotation = calculate_angle(
                    [left_hip[0], left_hip[1] + 0.1], left_hip, right_hip
                )
                if hip_rotation > max_hip_rotation:
                    max_hip_rotation = hip_rotation

            frame_count += 1
            
    cap.release()
    
    # --- レポート作成 ---
    report_lines = []
    report_lines.append("🏌️‍♂️ プロ仕様スイング診断レポート ⛳")
    report_lines.append("------------------------------------------")
    
    # 1. アドレスの評価
    if address_spine_angle is not None:
        int_angle = int(address_spine_angle)
        report_lines.append(f"✅ アドレス時の前傾角度: {int_angle}°")
        # 一般的に、ミドルアイアンで30〜40度が推奨されます（カメラアングルに依存）
        if int_angle >= 30 and int_angle <= 45:
            report_lines.append("  → 前傾角度は理想的です！安定した土台ができています。")
        else:
            report_lines.append("  → 前傾が浅すぎるか深すぎる可能性があります。股関節から正しく折る意識を持ちましょう。")
    
    report_lines.append("------------------------------------------")

    # 2. 肩の回転の評価 (バックスイング)
    if max_shoulder_rotation > 0:
        int_angle = int(max_shoulder_rotation)
        report_lines.append(f"✅ 最大肩の回転 (捻転): {int_angle}°")
        # 90度近くが理想
        if int_angle >= 85:
            report_lines.append("  → 非常に深い捻転！パワーを生み出す準備ができています。")
        elif int_angle < 60:
            report_lines.append("  → 回転が浅い傾向です。もっと背中をターゲットに向け、胸をボールから離すように意識しましょう。")
        else:
            report_lines.append("  → 良好な回転量です。")

    # 3. 腰の回転の評価 (バックスイング)
    if max_hip_rotation > 0:
        int_angle = int(max_hip_rotation)
        report_lines.append(f"✅ 最大腰の回転: {int_angle}°")
        # 一般的に30〜45度程度が適切
        if int_angle > 50:
            report_lines.append("  → 腰が回りすぎているかもしれません（オーバースイング）。下半身の安定感を意識し、捻転差を作りましょう。")
        elif int_angle < 20:
            report_lines.append("  → 腰の回転が硬い傾向です。股関節の柔軟性をチェックし、より積極的なヒップターンを目指しましょう。")
        else:
            report_lines.append("  → 適切な範囲の回転です。")
    
    report_lines.append("------------------------------------------")
    report_lines.append(f"総フレーム数: {frame_count} | ご利用ありがとうございました！")
    
    return "\n".join(report_lines)
