from flask import Flask, jsonify
import re

app = Flask(__name__)

# =====================================================
# ダミー診断データ（本番では Firestore / Gemini に置換）
# =====================================================
def get_report(report_id):
    return {
        "status": "COMPLETED",
        "metrics": {
            "解析フレーム数": ("73", "60フレーム以上"),
            "最大肩回転": ("120.0°", "約80°〜100°"),
            "最小腰回転": ("90.0°", "約35°〜45°"),
            "最大コック角": ("108.5°", "約90°〜120°"),
            "最大頭ブレ（Sway）": ("0.0069", "0.05以下"),
            "最大膝ブレ（Sway）": ("0.0146", "0.05以下"),
        },
        "report_md": """
## 02. 頭の安定性（軸のブレ）
**測定値：0.0069**
- スイング中の頭部移動が非常に少ない
- 回転中心が安定し再現性が高い
- ミート率向上に直結しやすい

プロ評価：
頭の安定性は非常に高く、スイング全体の土台が整っています。

## 03. 肩の回旋（上半身のねじり）
**測定値：120.0°**
- 肩の可動域が大きい
- パワーを生み出せる素地がある
- オーバースイング傾向に注意

プロ評価：
高い回旋量は武器ですが、再現性とのバランスが鍵です。

## 07. 総合診断
- 安定している点
  - 軸と下半身の安定性が高い
- 改善が期待される点
  - 回転量の最適化

## 08. 改善戦略とドリル
| ドリル名 | 目的 | やり方 |
|---|---|---|
| ハーフスイング | 回転量制御 | ①腰〜腰 ②静止 ③一定 |
| クロスアーム | 肩回旋改善 | ①腕交差 ②胸回旋 ③左右 |
| テンポ練習 | 再現性向上 | ①一定 ②一定 ③一定 |

## 09. スイング傾向補正型フィッティング（ドライバー）
| 項目 | 推奨 | 理由 |
|---|---|---|
| シャフト重量 | 50g台 | 操作性と安定性 |
| フレックス | SR | タイミング重視 |
| キックポイント | 中 | 弾道安定 |
| トルク | 4.5 | フェース管理 |

本診断は骨格分析に基づく傾向提案です。  
リシャフトについては、お客様ご自身で実際に試打した上でご検討ください。

## 10. まとめ
現在の安定性は非常に高い水準です。  
回転量を整えることで、再現性と飛距離の両立が可能になります。

お客様のゴルフライフが、より充実したものになることを切に願っています。
"""
    }

# =====================================================
# API
# =====================================================
@app.route("/api/report_data/<report_id>")
def api_report(report_id):
    return jsonify(get_report(report_id))

# =====================================================
# レポート画面
# =====================================================
@app.route("/report/<report_id>")
def report_view(report_id):
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GATE AIスイング診断</title>
<style>
body{{font-family:system-ui;background:#f3f4f6}}
.card{{background:#fff;border-radius:12px;padding:20px;margin:16px auto;max-width:900px}}
h2{{border-bottom:2px solid #ccc;padding-bottom:6px}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}
th,td{{border:1px solid #ccc;padding:8px}}
th{{background:#f9fafb}}
ul{{padding-left:20px}}
</style>
</head>
<body>

<div class="card">
<h1>GATE AIスイングドクター</h1>
<p>レポートID：{report_id}</p>
</div>

<div class="card" id="content">読み込み中...</div>

<script>
fetch("/api/report_data/{report_id}")
.then(r=>r.json())
.then(d=>{
  let html = "<h2>01. 骨格計測データ（AIが測った数値）</h2>";
  html += "<table><tr><th>計測項目</th><th>測定値</th><th>理想の目安</th></tr>";
  for(const k in d.metrics){{
    html += `<tr><td>${{k}}</td><td>${{d.metrics[k][0]}}</td><td>${{d.metrics[k][1]}}</td></tr>`;
  }}
  html += "</table>";

  html += "<h3>各数値の見方（簡単な説明）</h3><ul>";
  html += "<li><b>解析フレーム数</b>：解析精度の目安</li>";
  html += "<li><b>最大肩回転</b>：上半身の捻転量</li>";
  html += "<li><b>最小腰回転</b>：下半身の回旋量</li>";
  html += "<li><b>最大コック角</b>：タメの深さ</li>";
  html += "<li><b>最大頭ブレ</b>：軸の安定性</li>";
  html += "<li><b>最大膝ブレ</b>：下半身の安定性</li>";
  html += "</ul>";

  let md = d.report_md;
  md = md.replace(/^## (.*)$/gm,"<h2>$1</h2>");
  md = md.replace(/\\*\\*(.*?)\\*\\*/g,"<b>$1</b>");
  md = md.replace(/^\\|([\\s\\S]*?)\\n\\n/g, block=>{
    const lines = block.trim().split("\\n");
    const head = lines[0].split("|").slice(1,-1);
    const rows = lines.slice(2).map(l=>l.split("|").slice(1,-1));
    let t = "<table><tr>"+head.map(h=>"<th>"+h+"</th>").join("")+"</tr>";
    rows.forEach(r=>t+="<tr>"+r.map(c=>"<td>"+c+"</td>").join("")+"</tr>");
    return t+"</table>";
  });
  md = md.replace(/^- (.*)$/gm,"<li>$1</li>");
  md = md.replace(/(<li>.*<\\/li>)/gs,"<ul>$1</ul>");

  document.getElementById("content").innerHTML = html + md;
});
</script>

</body>
</html>
"""

# =====================================================
# 起動
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
