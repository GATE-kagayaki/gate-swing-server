<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GATE AIスイングドクター｜SWING ANALYSIS REPORT</title>

<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;700&family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">

<style>
    body { font-family: 'Noto Sans JP', sans-serif; background:#f3f4f6; }
    .paper { background:#fff; max-width:900px; margin:20px auto; padding:32px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    h2 { border-left:4px solid #111827; padding-left:12px; margin-top:32px; font-weight:700; font-family: 'Noto Serif JP', serif; }
    table { width:100%; border-collapse:collapse; margin-top:12px; table-layout: fixed; }
    th,td { border:1px solid #e5e7eb; padding:8px; font-size: 0.9rem; vertical-align: top; }
    th { background:#f9fafb; font-weight: 700; width: 30%; }
    .premium-alert { background: #fef2f2; color: #dc2626; border: 1px dashed #f87171; padding: 16px; margin-top: 10px; text-align: center; }
    .loading { text-align: center; padding: 100px 0; }
</style>
</head>

<body>
<div class="paper">

    <h1 class="text-2xl font-bold text-center mb-2">GATE AIスイングドクター</h1>
    <p class="text-center text-sm mb-6 text-gray-600">SWING ANALYSIS REPORT</p>

    <div id="loading" class="loading">
        <div class="animate-spin h-8 w-8 border-4 border-gray-500 rounded-full border-t-transparent mx-auto"></div>
        <p class="mt-4 text-gray-500 text-sm">レポートデータを読み込み中...</p>
    </div>

    <div id="report-content" class="hidden">
        
        <p class="text-xs text-gray-500 mb-6">
            REPORT ID：<span id="reportId"></span>
        </p>

        <section id="section-01">
            <h2>01. 骨格計測データ（AIが測った数値）</h2>
            <table>
                <tbody id="data-01">
                    </tbody>
            </table>
        </section>

        </div>

</div>

<script>
const reportId = location.pathname.split("/").pop();
document.getElementById("reportId").innerText = reportId;

const reportContent = document.getElementById("report-content");
const loadingElement = document.getElementById("loading");

function renderSection(key, sectionData) {
    // 01セクションは既にHTMLに存在するためスキップ
    if (key === '01') {
        let tableBody = '';
        Object.entries(sectionData.data).forEach(([k, v]) => {
            let unit = '';
            // 単位の付与ロジック
            if (k.includes('回転') || k.includes('コック')) unit = '°';
            if (k.includes('Sway')) unit = '%';

            // 数値が '-' や空の場合は単位を付けない
            const displayValue = (v === null || v === undefined || v === '') ? '-' : (v + unit);
            
            tableBody += `<tr><th>${k}</th><td>${displayValue}</td></tr>`;
        });
        document.getElementById('data-01').innerHTML = tableBody;
        return;
    }

    const el = document.createElement('section');
    el.id = `section-${key}`;
    el.innerHTML += `<h2>${sectionData.title}</h2>`;

    if (sectionData.text) {
        // 02〜07, 10: テキストコンテンツ
        if (sectionData.title.includes('(有料限定)')) {
            el.innerHTML += `<div class="premium-alert">${sectionData.text.join('<br>')}</div>`;
        } else {
             el.innerHTML += sectionData.text.map(t => `<p class="text-gray-700 whitespace-pre-line">${t}</p>`).join("");
        }
    } else if (sectionData.drills) {
        // 08: ドリルテーブル
        let tableHtml = '<table><thead><tr><th>ドリル名</th><th>目的</th><th>やり方</th></tr></thead><tbody>';
        sectionData.drills.forEach(d => {
            tableHtml += `<tr><td>${d["ドリル名"]}</td><td>${d["目的"]}</td><td>${d["やり方"]}</td></tr>`;
        });
        tableHtml += '</tbody></table>';
        el.innerHTML += tableHtml;

    } else if (sectionData.fitting) {
        // 09: フィッティングテーブル
        let tableHtml = '<table><thead><tr><th>項目</th><th>推奨</th></tr></thead><tbody>';
        Object.entries(sectionData.fitting).forEach(([k, v]) => {
             // 理由/備考をまとめて表示するため、ここでは理由/備考をスキップ
             if (k !== '備考' && k !== '理由') {
                 tableHtml += `<tr><th>${k}</th><td>${v}</td></tr>`;
             }
        });
        // 備考欄を一つのセルで表示
        const note = sectionData.fitting['備考'] || sectionData.fitting['理由'] || 'ー';
        tableHtml += `<tr><th>備考</th><td colspan="1">${note}</td></tr>`;
        tableHtml += '</tbody></table>';
        el.innerHTML += tableHtml;
    }
    
    reportContent.appendChild(el);
}

fetch(`/api/report_data/${reportId}`)
    .then(r => {
        if (!r.ok) throw new Error("API通信エラー: " + r.status);
        return r.json();
    })
    .then(data => {
        loadingElement.classList.add("hidden");

        if (data.status === "completed" && data.analysis) {
            reportContent.classList.remove("hidden");
            const analysis = data.analysis;

            // キー（01, 02, 03...）の昇順でレンダリング
            Object.keys(analysis).sort().forEach(key => {
                const sectionData = analysis[key];
                if (sectionData) {
                    renderSection(key, sectionData);
                }
            });

        } else if (data.status === "pending" || data.status === "PROCESSING") {
             loadingElement.querySelector('p').innerText = "解析処理中です。完了までしばらくお待ちください...";
             loadingElement.classList.remove("hidden");
        } else {
            loadingElement.innerHTML = '<p class="text-red-600 font-bold">⚠️ レポートデータが見つからないか、解析に失敗しました。</p>';
        }
    })
    .catch(error => {
        loadingElement.innerHTML = `<p class="text-red-600 font-bold">⚠️ サーバーとの通信エラーが発生しました。<br>F12キーを押してコンソールを確認してください。</p>`;
        loadingElement.classList.remove("hidden");
        console.error('Fetch Error:', error);
    });
</script>
</body>
</html>

