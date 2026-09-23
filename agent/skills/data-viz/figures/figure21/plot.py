"""Rectangular hierarchy whose terminal labels are small stacks of words."""
import numpy as np
from vizlib.common import nums


def layout(rows, leaves):
    nodes = {row['node']: row for row in rows}
    labels = {row['node']: row for row in leaves}
    if len(nodes) != len(rows) or len(labels) != len(leaves) or nodes.keys() & labels.keys():
        raise ValueError('Internal and terminal node IDs must be unique and disjoint')
    order = nums(leaves, 'order')
    heights = nums(rows, 'height')
    if len(leaves) < 2 or len(set(order)) != len(leaves) or (heights < 0).any():
        raise ValueError('Need two or more ordered leaves and nonnegative heights')
    children = [r[k] for r in rows for k in ('left', 'right')]
    if len(set(children)) != len(children) or any(ch not in nodes and ch not in labels for ch in children):
        raise ValueError('Children must exist and have exactly one parent')
    roots = set(nodes) - set(children)
    if len(roots) != 1 or set(labels) - set(children):
        raise ValueError('Hierarchy must have a single root and include all leaves')
    leaves = [leaves[i] for i in np.argsort(order)]
    positions = {row['node']: (float(i), 0.) for i, row in enumerate(leaves)}
    active, seen, branches = set(), set(), []

    def visit(node):
        if node in labels:
            seen.add(node)
            return positions[node]
        if node in active:
            raise ValueError('Hierarchy contains a cycle')
        active.add(node)
        row = nodes[node]
        lx, lh = visit(row['left'])
        rx, rh = visit(row['right'])
        height = float(row['height'])
        if height < max(lh, rh):
            raise ValueError('Merge height must be at least both child heights')
        positions[node] = ((lx + rx) / 2, height)
        branches.append(([lx, lx, rx, rx], [lh, height, height, rh]))
        active.remove(node)
        seen.add(node)
        return positions[node]
    visit(next(iter(roots)))
    if seen != set(nodes) | set(labels):
        raise ValueError('Disconnected hierarchy')
    return leaves, branches


def draw(c, rows):
    s = c.s
    leaves, branches = layout(rows, c.load(s['labels_file']))
    if not np.isfinite(s['height_limit']) or s['height_limit'] <= 0 or nums(rows, 'height').max() > s['height_limit']:
        raise ValueError('height_limit must be positive and contain all merge heights')
    ax = c.ax()
    ax.set_axis_off()
    ax.set_xlim(0, len(leaves)-1)
    ax.set_ylim(0, s['height_limit'])
    for xs, ys in branches:
        ax.plot(xs, ys, color=s['branch_color'], lw=s['branch_width'],
                solid_capstyle='butt', solid_joinstyle='miter', clip_on=False)
    for i, row in enumerate(leaves):
        if not row['words'].strip():
            raise ValueError('Leaf labels must not be empty')
        ax.annotate(row['words'].replace('|', '\n'), (i, 0), xytext=(0, -s['label_gap']),
                    textcoords='offset points', ha='center', va='top',
                    fontsize=s['label_size'], linespacing=s['line_spacing'], annotation_clip=False)
    if s.get('heading'):
        c.text(s['title_x'], s['title_y'], s['heading'], ha='center', va='top',
               fontsize=s['title_size'], linespacing=1.0)
    if s.get('panel_label'):
        c.text(s['panel_x'], s['panel_y'], s['panel_label'], ha='left', va='top',
               fontsize=s['panel_size'], fontweight='bold')


if __name__ == '__main__':
    from render import main
    main(default_figure=21)
