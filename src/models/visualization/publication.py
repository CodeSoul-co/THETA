"""Shared, offline publication styling/export for every native model visualizer."""
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar
import json
import html
import logging
from urllib.parse import quote

import matplotlib as mpl
from matplotlib import font_manager, pyplot as plt
from matplotlib.text import Text
from cycler import cycler

# Stable topic identity across figures; avoid red/green-only distinctions.
COLORS = ['#2F668A', '#BA6946', '#408571', '#A9758E', '#A48035',
          '#558DA2', '#796B98', '#5B6B72', '#8C564B', '#AA8D28']

_figure_edit = ContextVar('native_figure_edit', default=None)


@contextmanager
def figure_edit(chart, relative_path, output_dir, spec):
    """Run the existing plotting code with scoped edits to ONE native output."""
    state = dict(chart=chart, path=Path(relative_path).with_suffix(''),
                 root=Path(output_dir).resolve(), spec=spec, files=[])
    token = _figure_edit.set(state)
    try:
        yield state
    finally:
        _figure_edit.reset(token)


def selected_chart(name):
    edit = _figure_edit.get()
    return edit is None or edit['chart'] == name


def selected_figure(filename):
    edit = _figure_edit.get()
    if edit is None:
        return True
    return Path(filename).resolve().relative_to(edit['root']).with_suffix('') == edit['path']


@lru_cache(maxsize=1)
def chinese_font():
    for name in ['Noto Sans CJK SC', 'Source Han Sans SC', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC', 'WenQuanYi Zen Hei']:
        try:
            return font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            pass
    for file in ['/System/Library/Fonts/STHeiti Light.ttc', '/System/Library/Fonts/PingFang.ttc',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 'C:/Windows/Fonts/msyh.ttc']:
        if Path(file).is_file():
            font_manager.fontManager.addfont(file)
            return file
    return None


def setup_style(language='en'):
    logging.getLogger('fontTools.subset').setLevel(logging.WARNING)
    font = chinese_font()
    families = ['Arial', 'Helvetica', 'DejaVu Sans']
    families = [name for name in families if any(f.name == name for f in font_manager.fontManager.ttflist)]
    if font:
        families.insert(0, font_manager.FontProperties(fname=font).get_name())
    elif language == 'zh':
        raise ValueError('Chinese publication export requires a CJK font (e.g. Noto Sans CJK SC).')
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': families,
        'font.size': 8, 'axes.titlesize': 10, 'axes.labelsize': 8,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
        'legend.frameon': False, 'axes.grid': False, 'axes.titlelocation': 'left',
        'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.width': .6,
        'ytick.major.width': .6, 'xtick.major.size': 3, 'ytick.major.size': 3,
        'lines.linewidth': 1, 'lines.markersize': 3, 'text.color': '#202020',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#333333', 'axes.linewidth': .6, 'axes.axisbelow': True,
        'axes.prop_cycle': cycler(color=COLORS), 'grid.color': '#DFE4E8', 'grid.linewidth': .6,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.unicode_minus': False, 'image.cmap': 'cividis'})
    return font


def validate_export(dpi, formats):
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 72 <= dpi <= 1200:
        raise ValueError('dpi must be an integer between 72 and 1200')
    if isinstance(formats, str): formats = formats.split(',')
    formats = tuple(dict.fromkeys(formats))
    if not formats or any(fmt not in {'png', 'pdf', 'svg'} for fmt in formats):
        raise ValueError('formats must select png, pdf and/or svg')
    return formats


def save_figure(fig, filename, *, dpi=300, formats=('png', 'pdf', 'svg'), **kwargs):
    if not selected_figure(filename):
        return []
    edit = _figure_edit.get()
    if edit is not None:
        # Edit the Figure produced by the original plotting function, before its
        # normal export. Never translate numeric columns into a different chart.
        title = getattr(fig, '_suptitle', None)
        if title is None:
            titles = [ax._left_title if ax.get_title(loc='left') else ax.title
                      for ax in fig.axes if ax.get_title(loc='left') or ax.get_title()]
            if len(titles) != 1:
                raise ValueError('原图没有唯一总标题，无法安全修改；请明确子图标题')
            title = titles[0]
        title.set_text(edit['spec']['title'])
    formats = validate_export(dpi, formats)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the visualizer's topic-dependent canvas. Shrinking every figure
    # to a journal column while retaining font sizes makes dense plots overlap.
    latin = next((name for name in mpl.rcParams['font.sans-serif'] if name in {'Arial','Helvetica','DejaVu Sans'}), 'DejaVu Sans')
    for text in fig.findobj(Text):
        # Older Matplotlib versions cannot fall back within a mixed CJK/Latin string.
        cjk = any('\u2e80' <= char <= '\ua4cf' for char in text.get_text())
        font = chinese_font() if cjk else None
        text.set_fontfamily(font_manager.FontProperties(fname=font).get_name() if font else 'DejaVu Sans' if any('\u2190' <= c <= '\u22ff' for c in text.get_text()) else latin)
        text.set_fontsize(min(11, max(7, text.get_fontsize())) if text.get_gid() != 'metric-value' else 20)
    panels = [ax for ax in fig.axes if ax.axison and ax.get_label() != '<colorbar>' and getattr(ax, 'get_subplotspec', lambda: None)() is not None]
    for ax in fig.axes:
        ax.grid(False)
        ax.tick_params(direction='out', width=.6, length=3, labelsize=7.5)
        ax.title.set_fontsize(10)
        ax.set_title(ax.get_title(loc='left'), loc='left', fontsize=10)
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
        legend = ax.get_legend()
        if legend:
            legend.set_frame_on(False)
            for text in legend.get_texts(): text.set_fontsize(7.5); text.set_color('#202020')
        for spine in ax.spines.values(): spine.set_linewidth(.6)
    if len(panels) > 1 and not getattr(fig, '_theta_no_panel_labels', False):
        for i, ax in enumerate(panels):
            if not any(t.get_gid() == 'publication-panel' for t in ax.texts):
                letter = ax.text(-.12, 1.05, chr(97 + i) if i < 26 else str(i+1),
                    transform=ax.transAxes, size=8, weight='bold', va='bottom', color='#202020')
                letter.set_gid('publication-panel')
    # Reflow after setting typography without discarding adaptive panel sizes.
    if getattr(fig, 'get_layout_engine', lambda: None)() is None:
        try: fig.tight_layout(pad=1.1, w_pad=2, h_pad=2)
        except (ValueError, RuntimeError): pass
    kwargs.update(dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=.12)
    paths = []
    for fmt in formats:
        target = path.with_suffix('.' + fmt)
        fig.savefig(target, format=fmt, **kwargs)
        paths.append(str(target))
    if edit is not None:
        edit['files'].extend(paths)
    return paths


def export_manifest(output_dir, dpi, formats, data=None):
    """Inventory actual artifacts, not counts of attempted chart functions."""
    root = Path(output_dir)
    files = sorted(str(f.relative_to(root)) for f in root.rglob('*')
                   if f.is_file() and f.name != 'index.html' and f.suffix.lower() in {'.png', '.pdf', '.svg', '.csv', '.html'})
    data = data or {}
    payload = {'schema': 'theta.publication.v1', 'dpi': dpi, 'style': 'adaptive', 'editableSvgText': True, 'formats': list(formats), 'files': files,
               'source': data.get('source_metadata'), 'plotScope': data.get('plot_scope'),
               'note': 'Vector text/lines in PDF and SVG; dense scatter and word clouds may be rasterized.'}
    chart_status=[]
    for name in ['chart-status.json','additional-chart-status.json']:
        path=root/name
        if path.exists():chart_status.extend(json.loads(path.read_text()))
    payload['chartStatus']=chart_status
    unavailable={}
    labels={'topic_network_circular':'主题相关性网络 · 环形','wordcloud_grid':'矩形词云拼图','year_topic_sankey':'年份 → 主题桑基图','source_topic_sankey':'来源 → 主题桑基图','kl_divergence':'KL 散度变化','sankey_diagram':'桑基图','topic_significance_chart':'主题显著性',
            'topic_num_evaluation':'主题数 K 评估','generate_topic_word_dist_change':'逐主题词分布变化',
            'generate_topic_word_sense':'词义轨迹','training_convergence':'训练曲线',
            'temporal_charts':'时序图','group_charts':'来源分组图'}
    reasons={'kl_divergence':'缺少分时主题词矩阵；不能用随机扰动生成变化。',
             'topic_significance_chart':'缺少有效的统计检验依据；平均权重不是显著性。',
             'topic_num_evaluation':'需要至少两种 K 的真实评估结果。',
             'generate_topic_word_dist_change':'缺少分时主题词矩阵；DTM 的实际分时词图由专用入口生成。',
             'generate_topic_word_sense':'缺少实际分时词向量及词义对应。',
             'training_convergence':'没有保存训练历史，无法还原损失或困惑度曲线。',
             'temporal_charts':'没有可验证的日期标签。','group_charts':'没有可验证的来源或分组标签。'}
    for item in chart_status:
        if item['status']=='generated':continue
        key=(item['chart'].split(':')[0],item['status'])
        unavailable.setdefault(key,[]).append(item)
    rows=[]
    for (key,status),items in unavailable.items():
        label=labels.get(key,key)+(f' × {len(items)}' if len(items)>1 else '')
        detail=reasons.get(key,items[0].get('detail') or '当前模型或数据不适用，详见生成记录。') if status=='skipped' else items[0].get('detail','')
        rows.append('<tr><td>'+html.escape(label)+'</td><td>'+('未生成' if status=='skipped' else '失败')+'</td><td>'+html.escape(detail)+'</td></tr>')
    coverage=('<details class="note" open><summary>未生成的图表与原因（'+str(len(rows))+' 类）</summary><table><thead><tr><th>图表</th><th>状态</th><th>原因</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></details>') if rows else ''
    (root / 'publication-manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    cards = []
    for stem in sorted({str(Path(f).with_suffix('')) for f in files if Path(f).suffix in {'.png', '.svg', '.pdf'}}):
        variants = [f'{stem}.{fmt}' for fmt in formats if f'{stem}.{fmt}' in files]
        preview = next((f for f in variants if f.endswith(('.png', '.svg'))), None)
        links = ' · '.join(f'<a href="{quote(f)}" download>{Path(f).suffix[1:].upper()}</a>' for f in variants)
        picture = f'<a href="{quote(preview)}"><img loading="lazy" src="{quote(preview)}" alt="{html.escape(stem)}"></a>' if preview else ''
        name = Path(stem).name
        fallback = '矩形词云拼图 · ' + name.rsplit('_', 1)[-1] if name.startswith('topic_wordcloud_grid_') else stem
        title = html.escape(labels.get(name, fallback))
        cards.append(f'<article><h2>{title}</h2>{picture}<p>{links}</p></article>')
    auxiliary = ' · '.join(f'<a href="{quote(f)}">{html.escape(f)}</a>' for f in files if f.endswith(('.csv', '.html')))
    scope = html.escape(data.get('plot_scope') or '全局图使用全部模型文档；时序图仅使用可验证的有效日期。')
    (root / 'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>THETA · 图表与数据</title>
<style>body{font:16px/1.65 system-ui,sans-serif;color:#203440;background:#f5f7f9;margin:0}main{max-width:1200px;margin:auto;padding:40px 24px}h1{font-size:32px;margin-bottom:8px}h2{font-size:15px;overflow-wrap:anywhere}a{color:#0072b2}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}article{background:white;border:1px solid #dbe3e8;padding:18px;border-radius:8px}img{width:100%;height:300px;object-fit:contain}p{overflow-wrap:anywhere}.note{padding:16px;background:#eaf1f5;margin:24px 0}input{font:inherit;padding:10px;width:min(500px,90%);border:1px solid #9cb0bf;margin:12px 0 24px}article[hidden]{display:none}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:10px;border-bottom:1px solid #d1dde5;vertical-align:top}summary{cursor:pointer;font-weight:600}</style>
<main><h1>THETA · 图表与数据</h1>''' +
        f'<p>PNG：{dpi} DPI · PDF/SVG：矢量文字与线条，散点与词云可能含栅格图层。</p>' +
        f'<div class="note">{scope}<br>日期缺失、训练历史缺失、无跨期实体关联或无分时 beta 时，不补造曲线、迁移或显著性证据。完整图用于方法检查，投稿请结合 README 中的适用范围选择。</div>' +
        coverage + '<input id="filter" aria-label="筛选图表" placeholder="按主题或图表名称筛选"><section>' + ''.join(cards) +
        '</section><h2>CSV 与交互图</h2><p>' + auxiliary + '</p></main>' +
        '''<script>document.querySelector('#filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('article').forEach(a=>a.hidden=!a.querySelector('h2').textContent.toLowerCase().includes(q))});</script></html>''', encoding='utf-8')
    return payload
