"""Collect already authorized acceptance artifacts; no training, inference or network."""
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

home = Path(sys.argv[1]).resolve()
conversation = Path(sys.argv[2]).resolve()
delivery = home.parent / 'delivery'
delivery.mkdir(exist_ok=True)
reports = json.loads((home / 'reports.json').read_text())
links = []
for report in reports:
    label = 'lda' if report['modelId'] == 'lda' else 'stm-diagnostic'
    target = delivery / label
    shutil.copytree(Path(report['reportPath']).parent, target, dirs_exist_ok=True)
    links.append((label, 'LDA · 真实本地训练' if label == 'lda' else 'STM · 原结果诊断（按用户要求未重跑）'))
# A clearly marked cloud-model fixture tests the same presentation contract.
fixture = subprocess.run([sys.executable, str(Path(__file__).with_name('fixture-report.py')), str(home.parent / 'cloud-fixture'), 'job-simulated-theta', 'theta'], capture_output=True, text=True, check=True)
cloud = json.loads(fixture.stdout)['report']
shutil.copytree(Path(cloud['reportPath']).parent, delivery / 'theta-simulated', dirs_exist_ok=True)
links.append(('theta-simulated', 'THETA · 模拟产物（无真实云调用）'))
transcript = json.loads(conversation.read_text())
shutil.copy2(conversation, delivery / 'conversation.json')
shutil.copy2(home / 'outside-data/support feedback.csv', delivery / 'sample.csv')
if (home / 'synthesis.json').exists():
    synthesis = json.loads((home / 'synthesis.json').read_text())
    shutil.copy2(synthesis['documentPath'], delivery / 'interpretation.md')
seconds = [turn['elapsedMs'] / 1000 for turn in transcript['transcript']]
summary = {'scope': '功能闭环验收，不评定模型优劣', 'dialogueTurns': len(seconds), 'checks': transcript['checks'],
           'averageTurnSeconds': round(sum(seconds)/len(seconds), 2), 'maximumTurnSeconds': round(max(seconds), 2),
           'realTrainingJobs': 2, 'stmRerun': False, 'realCloudEmbeddingRequests': 0,
           'reports': [{'directory': value, 'label': label} for value, label in links]}
(delivery / 'acceptance.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
content = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>THETA 功能闭环验收</title>',
           '<style>body{font:16px/1.7 system-ui;max-width:980px;margin:50px auto;padding:0 24px;color:#17303a}a{color:#096c8a}section{border:1px solid #c8d7dc;border-radius:10px;padding:20px;margin:20px 0}small{color:#52616a}</style>',
           '<h1>THETA · 功能闭环验收</h1><p>验证上传、理解、提案、确认/拒绝/改案、训练、结果交付、综合解读和历史恢复。这里不比较模型优劣。全部输入为36条合成客服文本。</p>',
           f'<p>本轮真实对话：{len(seconds)}轮；平均 {summary["averageTurnSeconds"]} 秒/轮，最长 {summary["maximumTurnSeconds"]} 秒。计时包含供应商响应，不能视为性能保证。</p>',
           '<p>真实训练共2次（LDA、STM）；STM原结果保留诊断，修复后未重跑。云 embedding 只做模拟权限/预算验证，没有真实云费用。</p>',
           '<h2>完整图表与表格</h2>']
for directory, label in links:
    content.append(f'<section><h3>{html.escape(label)}</h3><a href="{directory}/index.html">打开报告：全部图表与完整CSV</a><br><a href="{directory}/manifest.json">来源与文件校验清单</a></section>')
content += ['<h2>复验材料</h2><ul><li><a href="sample.csv">合成输入数据</a></li><li><a href="conversation.json">完整用户对话、逐次调用和耗时</a></li><li><a href="interpretation.md">结果解读落盘示例</a></li><li><a href="acceptance.json">验收摘要</a></li></ul>', '<h2>通过的功能检查</h2><ul>']
content.extend('<li>' + html.escape(check) + '</li>' for check in transcript['checks'])
content.append('</ul></html>')
(delivery / 'index.html').write_text('\n'.join(content))
print(json.dumps({'delivery': str(delivery / 'index.html'), **summary}, ensure_ascii=False))
