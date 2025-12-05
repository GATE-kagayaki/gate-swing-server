import os
import threading
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, VideoMessage
from linebot.utils import extract_content_from_multipart
import requests
import report_generator 
import ffmpeg

# 環境変数から設定値を取得
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set")

# FlaskアプリとLINE Bot APIの設定
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ------------------------------------------------
# メインの解析ロジックを別スレッドで実行する関数
# ------------------------------------------------
def process_video_async(user_id, video_content):
    """
    動画のダウンロード、圧縮、解析、レポート送信をバックグラウンドで実行します。
    """
    # ユーザーから受信したオリジナル動画のパス
    original_video_path = f'/tmp/{user_id}_{os.urandom(8).hex()}_original.mp4'
    # 圧縮後の動画のパス
    compressed_video_path = f'/tmp/{user_id}_{os.urandom(8).hex()}_compressed.mp4'
    
    # 1. オリジナル動画を一時ファイルに保存
    try:
        with open(original_video_path, 'wb') as tf:
            tf.write(video_content)
        app.logger.info(f"オリジナル動画ファイル保存成功: {original_video_path}")
    except Exception as e:
        app.logger.error(f"動画ファイルの保存に失敗: {e}", exc_info=True)
        return

    # 1.5 ★★★ 動画の自動圧縮とリサイズ処理 ★★★
    # 利便性確保のため、どんな動画でも事前に幅 640px にリサイズ＆高圧縮します
    try:
        app.logger.info(f"動画を幅 640px に圧縮・変換開始。")
        # -vf scale=640:-1 で幅640pxにリサイズし、crf 28で高圧縮
        # ★★★ 修正: ffmpegの実行パスを明示的に指定します ★★★
        (
            ffmpeg
            .input(original_video_path)
            .output(compressed_video_path, vf='scale=640:-1', crf=28, vcodec='libx264')
            .overwrite_output()
            .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True) # <-- 修正箇所
        )
        # 解析に使うパスを圧縮後のファイルに設定
        video_to_analyze = compressed_video_path
        app.logger.info(f"動画圧縮・変換成功: {compressed_video_path}")
        
    except ffmpeg.Error as e:
        error_details = e.stderr.decode('utf8') if e.stderr else '詳細不明'
        app.logger.error(f"FFmpegによる動画圧縮に失敗: {error_details}", exc_info=True)
        report_text = f"【動画処理エラー】圧縮に失敗しました。詳細: {error_details[:100]}..."
        line_bot_api.push_message(user_id, TextSendMessage(text=report_text))
        # 圧縮失敗時は元のファイルを解析せずに終了 (負荷を避けるため)
        return
        
    except Exception as e:
        app.logger.error(f"予期せぬ圧縮エラー: {e}", exc_info=True)
        report_text = f"【予期せぬエラー】動画処理で問題が発生しました: {str(e)[:100]}..."
        line_bot_api.push_message(user_id, TextSendMessage(text=report_text))
        return
        
    # 2. 動画の解析を実行
    try:
        # 圧縮されたファイルを解析
        report_text = report_generator.analyze_swing(video_to_analyze)
    except Exception as e:
        report_text = f"【解析エラー】動画処理中に予期せぬエラーが発生しました: {e}"
        app.logger.error(f"解析中の致命的なエラー: {e}", exc_info=True)

    # 3. 結果をユーザーにPUSH通知で返信
    try:
        # 解析完了のメッセージ
        completion_message = f"解析が完了しました。ユーザーID:{user_id}のレポートをLINEに返します。"
        line_bot_api.push_message(user_id, TextSendMessage(text=completion_message))
        
        # レポート本体
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=report_text)
        )
        app.logger.info(f"レポート送信成功: ユーザーID={user_id}")

    except LineBotApiError as e:
        # LINE APIからの明確なエラー応答をログに出力
        app.logger.error(f"LINE APIエラー: Status={e.status_code}, Message={e.message}, Details={e.error_response}", exc_info=True)
    except Exception as e:
        # その他の送信エラーをログに出力
        app.logger.error(f"レポート送信中に予期せぬエラーが発生しました: {e}", exc_info=True)

    # 4. 一時ファイルを削除
    if os.path.exists(original_video_path):
        os.remove(original_video_path)
        app.logger.info(f"一時ファイル削除: {original_video_path}")
        
    if os.path.exists(compressed_video_path):
        os.remove(compressed_video_path)
        app.logger.info(f"一時ファイル削除: {compressed_video_path}")


# ------------------------------------------------
# LINE Webhookのメイン処理 (省略)
# ------------------------------------------------
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.error("Invalid signature. Check your channel secret.")
        abort(400)
    except Exception as e:
        app.logger.error(f"Webhook handling error: {e}", exc_info=True)
        abort(500)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    if event.message.text in ["レポート", "テスト"]:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="動画を送信してください。ゴルフスイングの解析を行います。")
        )
        
@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event):
    user_id = event.source.user_id
    message_id = event.message.id

    # 1. ユーザーへの即時応答（LINEの応答タイムアウト回避）
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="動画を受け付けました。解析を開始します。しばらくお待ちください...")
    )
    
    # 2. 動画コンテンツの取得
    try:
        message_content = line_bot_api.get_message_content(message_id)
        video_content = message_content.content
    except Exception as e:
        app.logger.error(f"動画コンテンツの取得に失敗: {e}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(text="【エラー】動画のダウンロードに失敗しました。"))
        return

    # 3. ★★★ 解析処理を別スレッドで起動（フリーズ回避） ★★★
    app.logger.info(f"動画解析を別スレッドで開始します。ユーザーID: {user_id}")
    thread = threading.Thread(target=process_video_async, args=(user_id, video_content))
    thread.start()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    # Matplotlibのフォントキャッシュ構築を抑制する設定
    os.environ['HOME'] = '/tmp'
    app.run(host='0.0.0.0', port=port)
