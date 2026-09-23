"""Canvas, statistics, data loading and reusable plotting primitives."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors as mcolors
from scipy.stats import gaussian_kde
ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'{path}: CSV is empty')
    return rows

def nums(rows, key):
    try:
        a = np.array([float(r[key]) for r in rows])
    except (KeyError, ValueError) as e:
        raise ValueError(f'Invalid or missing numeric column {key!r}') from e
    if not np.isfinite(a).all():
        raise ValueError(f'{key}: NaN and infinity are not allowed')
    return a

def subset(rows, **filters):
    return [r for r in rows if all((r[k] == str(v) for k, v in filters.items()))]

def unique(rows, key):
    return list(dict.fromkeys((r[key] for r in rows)))

def col(c, alpha):
    return mcolors.to_rgba(c, alpha)

def configure(s):
    for filename in ['Arial.ttf', 'Arial Bold.ttf', 'Arial Italic.ttf', 'Arial Bold Italic.ttf']:
        p = Path('/System/Library/Fonts/Supplemental') / filename
        if p.exists():
            font_manager.fontManager.addfont(p)
    families = {f.name for f in font_manager.fontManager.ttflist}
    family = s['font'] if s['font'] in families else 'DejaVu Sans'
    plt.rcParams.update({'font.family': [family, 'DejaVu Sans'], 'font.size': s['font_size'], 'axes.linewidth': s['line_width'], 'axes.labelsize': s['font_size'], 'xtick.labelsize': s['font_size'], 'ytick.labelsize': s['font_size'], 'xtick.major.width': s['line_width'], 'ytick.major.width': s['line_width'], 'xtick.major.size': 5, 'ytick.major.size': 5, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': True, 'mathtext.fontset': 'dejavusans', 'savefig.facecolor': 'white'})
    return family

class Figure:

    def __init__(self, s, data_dir, annotations, summary):
        self.s = s
        self.data_dir = Path(data_dir)
        self.annotations = annotations
        self.summary = summary
        self.font = configure(s)
        self.fig = plt.figure(figsize=(s['width'] / s['dpi'], s['height'] / s['dpi']), dpi=s['dpi'], facecolor='white')
        self.rng = np.random.default_rng(s.get('seed', 190913))

    def ax(self, box=None, **kw):
        x, y, w, h = box or self.s['axes']
        W, H = (self.s['width'], self.s['height'])
        return self.fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)

    def text(self, x, y, text, **kw):
        return self.fig.text(x / self.s['width'], 1 - y / self.s['height'], text, **kw)

    def load(self, name):
        return read_csv(self.data_dir / name)

def clean(ax, full=False, color='black'):
    for name, spine in ax.spines.items():
        spine.set_visible(full or name in ['left', 'bottom'])
        spine.set_color(color)
    ax.tick_params(colors='black')

def label_axes(ax, s):
    if 'xlim' in s:
        ax.set_xlim(s['xlim'])
    if 'ylim' in s:
        ax.set_ylim(s['ylim'])
    if 'xlabel' in s:
        ax.set_xlabel(s['xlabel'])
    if 'ylabel' in s:
        ax.set_ylabel(s['ylabel'])
    if 'yticks' in s:
        ax.set_yticks(s['yticks'])

def bracket(ax, a, b, y, text, height=0, lw=1.5, fontsize=None):
    ax.plot([a, a, b, b], [y - height, y, y, y - height], c='black', lw=lw, clip_on=False)
    ax.annotate(text, ((a + b) / 2, y), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=fontsize, annotation_clip=False)

def summary(rows, mode):
    vals = nums(rows, 'value')
    if mode == 'samples':
        return (vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
    means = nums(rows, 'mean')
    errors = nums(rows, 'error')
    if not np.all(means == means[0]) or not np.all(errors == errors[0]):
        raise ValueError('Repeated mean/error fields must agree within each category and group')
    if errors[0] < 0:
        raise ValueError('Error must be nonnegative')
    return (means[0], errors[0])

def boxes(ax, arrays, positions, colors, vert=True, width=0.65, points=False, rng=None, showfliers=True, lw=1.2):
    if any((len(a) == 0 for a in arrays)):
        raise ValueError('Boxplot group has no observations')
    bp = ax.boxplot(arrays, positions=positions, orientation='vertical' if vert else 'horizontal', widths=width, patch_artist=True, manage_ticks=False, showfliers=showfliers, boxprops={'linewidth': lw, 'edgecolor': '#333'}, medianprops={'color': '#333', 'linewidth': lw}, whiskerprops={'color': '#777', 'linewidth': lw}, capprops={'linewidth': 0}, flierprops={'marker': '.', 'markersize': 3, 'markerfacecolor': '#555', 'markeredgecolor': '#555'})
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    if points:
        for a, p in zip(arrays, positions):
            jitter = rng.uniform(-width * 0.42, width * 0.42, len(a))
            ax.scatter(p + jitter if vert else a, a if vert else p + jitter, s=4, c='black', alpha=0.65, zorder=3, lw=0)
    return bp

def density(a, grid, bw=None):
    a = np.asarray(a, dtype=float)
    if not len(a):
        raise ValueError('Density group has no observations')
    if len(a) < 2 or np.ptp(a) < 1e-10:
        sd = max(np.ptp(grid) / 100, 0.001)
        return np.exp(-0.5 * ((grid - np.mean(a)) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    return gaussian_kde(a, bw_method=bw)(grid)

def grouped_bars(c, rows, width, spacing, open_points=True):
    s = c.s
    ax = c.ax()
    clean(ax)
    label_axes(ax, s)
    locations = {}
    for i, category in enumerate(s['categories']):
        for j, (g, cc) in enumerate(zip(s['groups'], s['colors'])):
            x = i * spacing + j
            rr = subset(rows, category=category, group=g)
            if not rr:
                raise ValueError(f'Missing observations for {category} / {g}')
            mean, error = summary(rr, c.summary)
            v = nums(rr, 'value')
            locations[i, j] = (x, mean, error, v)
            ax.bar(x, mean, width, fc=cc, ec='#333', lw=1.5)
            ax.errorbar(x, mean, yerr=error, fmt='none', ecolor='#444', elinewidth=1.4, capsize=7, capthick=1.2)
            ax.scatter(x + c.rng.uniform(-width * 0.18, width * 0.18, len(v)), v, s=43, c='white' if open_points else cc, edgecolors='#222', lw=1.5, zorder=3)
    ax.set_xlim(-1, (len(s['categories']) - 1) * spacing + len(s['groups']))
    ax.set_xticks(np.arange(len(s['categories'])) * spacing + (len(s['groups']) - 1) / 2, s['categories'])
    return (ax, locations)

def tree_layout(rows, leaves, start, gap):
    """Validate explicit edge table, then return angles and root distances."""
    nodes = {r['node']: r for r in rows}
    if len(nodes) != len(rows):
        raise ValueError('Tree node IDs must be unique')
    if 'root' in nodes:
        raise ValueError('root is reserved for the implicit tree root')
    children = {'root': []}
    for r in rows:
        if r['parent'] != 'root' and r['parent'] not in nodes:
            raise ValueError(f"Unknown tree parent {r['parent']}")
        if float(r['length']) < 0 or not np.isfinite(float(r['length'])):
            raise ValueError('Invalid branch length')
        children.setdefault(r['parent'], []).append(r['node'])
        children.setdefault(r['node'], [])
    terminal = {node for node in nodes if not children[node]}
    if len({r['node'] for r in leaves}) != len(leaves) or terminal != {r['node'] for r in leaves}:
        raise ValueError('Leaf metadata must name every terminal node exactly once')
    leaves = sorted(leaves, key=lambda r: float(r['order']))
    if len({r['order'] for r in leaves}) != len(leaves):
        raise ValueError('Leaf order values must be unique')
    angle = {r['node']: np.deg2rad(start + i * (360 - gap) / len(leaves)) for i, r in enumerate(leaves)}
    distances = {'root': 0.0}
    active = set()
    visited = set()

    def visit(node):
        if node in active:
            raise ValueError('Tree contains a cycle')
        active.add(node)
        for child in children[node]:
            distances[child] = distances[node] + float(nodes[child]['length'])
            visit(child)
        if children[node]:
            angle[node] = np.mean([angle[ch] for ch in children[node]])
        active.remove(node)
        visited.add(node)
    visit('root')
    if len(visited) != len(nodes) + 1:
        raise ValueError('Tree has a disconnected component or cycle')
    return (nodes, children, leaves, angle, distances)

def line_data(rows, xkey='x', ykey='density'):
    if not rows:
        raise ValueError('Curve group has no observations')
    if ykey not in rows[0]:
        values = nums(rows, 'value' if 'value' in rows[0] else xkey)
        pad = max(np.ptp(values) * 0.15, abs(values.mean()) * 0.01, 0.01)
        x = np.linspace(values.min() - pad, values.max() + pad, 300)
        y = density(values, x)
        height = float(rows[0].get('height', 1.4))
        if not np.isfinite(height) or height <= 0:
            raise ValueError('Density display height must be positive')
        return (x, y / y.max() * height)
    x = nums(rows, xkey)
    y = nums(rows, ykey)
    if np.any(y < 0):
        raise ValueError('Density height must be nonnegative')
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError('Curve x values must be distinct and have at least two points')
    return (x, y)
