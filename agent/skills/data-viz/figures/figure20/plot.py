"""Text positioned in a shared two-dimensional space with optional vector mesh."""
from collections import defaultdict
from matplotlib.collections import LineCollection
from vizlib.common import nums


def draw(c, rows):
    s = c.s
    ax = c.ax()
    ax.set_xlim(s['xlim'])
    ax.set_ylim(s['ylim'][::-1])
    ax.set_axis_off()
    x, y, size = (nums(rows, key) for key in ('x', 'y', 'font_size'))
    if (size <= 0).any() or len({r['word_id'] for r in rows}) != len(rows):
        raise ValueError('Words need unique IDs and positive font sizes')
    if s.get('show_mesh', True):
        paths = defaultdict(list)
        mesh = c.load(s['mesh_file'])
        mx, my, order = (nums(mesh, key) for key in ('x', 'y', 'order'))
        for row, px, py, rank in zip(mesh, mx, my, order):
            paths[row['path_id']].append((rank, px, py))
        segments = []
        for points in paths.values():
            if len(points) < 2 or len({p[0] for p in points}) != len(points):
                raise ValueError('Mesh paths need at least two distinct ordered vertices')
            segments.append([(px, py) for _, px, py in sorted(points)])
        ax.add_collection(LineCollection(segments, colors=s['mesh_color'],
                                        linewidths=s['mesh_width'], alpha=s['mesh_alpha'], zorder=0))
    for row, px, py, fs in zip(rows, x, y, size):
        if row['role'] not in ('word', 'highlight') or not row['text'].strip():
            raise ValueError('Every label needs text and role word/highlight')
        highlight = row['role'] == 'highlight'
        ax.text(px, py, row['text'], fontsize=fs, ha='center', va='center',
                color=s['highlight_color'] if highlight else s['word_color'],
                alpha=s['text_alpha'], clip_on=True, zorder=2,
                fontweight='normal')


if __name__ == '__main__':
    from render import main
    main(default_figure=20)
