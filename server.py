@app.route("/report/<report_id>", methods=["GET"])
def get_report_web(report_id):
    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>GATE AIスイングドクター 診断レポート</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://cdn.tailwindcss.com"></script>
</head>

<body class="bg-gray-100 text-gray-800">
  <div class="max-w-4xl mx-auto p-6 bg-white shadow-lg my-6 rounded-lg">
    <h1 class="text-3xl font-extrabold text-center mb-2 text-emerald-600">
      GATE AIスイングドクター
    </h1>
    <p class="text-center text-sm text-gray-500 mb-6">
      診断レポートID：{report_id}
    </p>

    <div id="loading" class="text-center py-10 text-gray-500">
      レポートを読み込み中…
    </div>

    <div id="content" class="hidden">

      <!-- 骨格データ -->
      <h2 class="text-2xl font-bold border-b-4 border-emerald-500 mb-4">
        01. 骨格計測データ
      </h2>

      <div id="metrics" class="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8"></div>

      <!-- AI レポート -->
      <h2 class="text-2xl font-bold border-b-4 border-emerald-500 mb-4">
        AIスイング診断レポート
      </h2>

      <div id="ai-report" class="prose max-w-none"></div>

    </div>
  </div>

<script>
const reportId = "{report_id}";
const apiUrl = "/api/report_data/" + reportId;

function card(title, value, unit="") {{
  return `
    <div class="bg-gray-50 border rounded-lg p-4 text-center shadow">
      <div class="text-sm text-gray-500 mb-1">${{title}}</div>
      <div class="text-2xl font-bold text-gray-800">${{value}}${{unit}}</div>
    </div>
  `;
}}

fetch(apiUrl)
  .then(res => res.json())
  .then(data => {{
    document.getElementById("loading").classList.add("hidden");
    document.getElementById("content").classList.remove("hidden");

    const m = data.mediapipe_data || {{}};
    const metrics = document.getElementById("metrics");

    metrics.innerHTML =
      card("解析フレーム数", m.frame_count ?? "N/A") +
      card("最大肩回転", m.max_shoulder_rotation ?? "N/A", "°") +
      card("最小腰回転", m.min_hip_rotation ?? "N/A", "°") +
      card("最大コック角", m.max_wrist_cock ?? "N/A", "°") +
      card("最大頭ブレ", m.max_head_drift_x ?? "N/A") +
      card("最大膝ブレ", m.max_knee_sway_x ?? "N/A");

    // Markdown簡易変換
    let report = data.ai_report_text || data.summary || "";
    report = report
      .replace(/\\n## (.*)/g, "<h2 class='text-xl font-bold mt-6 mb-2'>$1</h2>")
      .replace(/\\n### (.*)/g, "<h3 class='text-lg font-semibold mt-4 mb-1'>$1</h3>")
      .replace(/\\n\\n/g, "<br><br>")
      .replace(/\\n/g, "<br>");

    document.getElementById("ai-report").innerHTML = report;
  })
  .catch(err => {{
    document.getElementById("loading").innerText = "レポートの読み込みに失敗しました。";
  }});
</script>

</body>
</html>
"""
    return html, 200




