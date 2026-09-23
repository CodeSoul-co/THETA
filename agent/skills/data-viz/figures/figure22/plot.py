"""Document-to-word association curves and optional aligned box summaries."""
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle
from vizlib.common import nums


def bands(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate {key}')
    y0, y1 = nums(rows, 'y0'), nums(rows, 'y1')
    if (y1 <= y0).any():
        raise ValueError('Band bottom must exceed top')
    ranges = sorted(zip(y0, y1))
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError('Bands may touch but must not overlap')
    return result


def draw(c, rows):
    s = c.s
    ax = c.ax()
    ax.set_xlim(0, s['width'])
    ax.set_ylim(s['height'], 0)
    ax.set_axis_off()
    documents = bands(c.load(s['documents_file']), 'document_id')
    topics = bands(c.load(s['topics_file']), 'topic_id')
    if not 0 <= s['bundle_strength'] <= 1 or s['flow_width'] <= 0 or s['band_padding'] < 0:
        raise ValueError('Invalid bundle strength, flow width or padding')
    for band in [*documents.values(), *topics.values()]:
        if float(band['y1'])-float(band['y0']) <= 2*s['band_padding']:
            raise ValueError('Band must be taller than twice the internal padding')
    weight, source_pos, target_pos = (nums(rows, k) for k in ('weight', 'source_position', 'target_position'))
    if (weight <= 0).any() or any(((v < 0) | (v > 1)).any() for v in (source_pos, target_pos)):
        raise ValueError('Link weights must be positive; positions must lie in [0,1]')
    if len({r['link_id'] for r in rows}) != len(rows):
        raise ValueError('Link IDs must be unique')
    for row, w, u, v in zip(rows, weight, source_pos, target_pos):
        if row['document_id'] not in documents or row['topic_id'] not in topics:
            raise ValueError('Every link must refer to a known document and topic')
        doc, topic = documents[row['document_id']], topics[row['topic_id']]
        y0 = float(doc['y0']) + s['band_padding'] + u * (float(doc['y1'])-float(doc['y0'])-2*s['band_padding'])
        y1 = float(topic['y0']) + s['band_padding'] + v * (float(topic['y1'])-float(topic['y0'])-2*s['band_padding'])
        strength = s['bundle_strength']
        sy = y0*(1-strength) + (float(doc['y0'])+float(doc['y1']))/2*strength
        ty = y1*(1-strength) + (float(topic['y0'])+float(topic['y1']))/2*strength
        sx, tx = s['source_bundle_x'], s['target_bundle_x']
        start, end = s['flow_start'], s['flow_end']
        points = np.array([[start,y0], [start+30,y0], [sx-25,sy], [sx,sy],
                           [sx+55,sy], [tx-60,ty], [tx,ty],
                           [tx+40,ty], [end-30,y1], [end,y1]])
        path = Path(points, [Path.MOVETO]+[Path.CURVE4]*9)
        ax.add_patch(PathPatch(path, facecolor='none', edgecolor=topic['color'],
                               lw=s['flow_width']*w, alpha=s['flow_alpha'], zorder=1))
        if s.get('show_nodes', True):
            xy = np.array([[start+40,(y0+sy)/2], [sx,sy], [tx,ty], [end-40,(ty+y1)/2]])
            ax.plot(xy[:, 0], xy[:, 1], color=s['node_color'], linestyle='none',
                    alpha=s['node_alpha'], marker='s', ms=s['node_size'], mec='none', zorder=2)
    for left, right, heading in [(s['document_left'], s['document_right'],s['document_heading']),
                                 (s['topic_left'],s['topic_right'],s['topic_heading'])]:
        ax.text((left+right)/2, s['header_y'], heading, ha='center', va='bottom', fontsize=s['heading_size'], fontweight='bold')
        ax.plot([left,right],[s['top'],s['top']], c=s['rule_color'], lw=s['rule_width'])
        ax.plot([left,right],[s['bottom'],s['bottom']], c=s['rule_color'], lw=s['rule_width'])
    for doc in documents.values():
        y0,y1 = float(doc['y0']),float(doc['y1'])
        mid=(y0+y1)/2
        ax.text(s['document_label_x'],mid,doc['label'].replace('|','\n'),ha='right',va='center',fontsize=s['label_size'],linespacing=1.18)
        if y1 < s['bottom']:
            ax.plot([s['document_left'],s['document_label_x'],s['flow_start']-12,s['flow_start']],
                    [y1,y1,mid,mid],c=s['rule_color'],lw=s['inner_rule_width'],alpha=.9,zorder=3)
    for topic in topics.values():
        y0,y1 = float(topic['y0']),float(topic['y1'])
        ax.text(s['topic_label_x'],(y0+y1)/2,topic['label'].replace('|','\n'),ha='left',va='center',fontsize=s['label_size'],linespacing=1.18)
        if y1 < s['bottom']:
            ax.plot([s['topic_left'],s['flow_end'],s['topic_label_x']-6,s['topic_right']],
                    [y1,y1,(y0+y1)/2,(y0+y1)/2],c=s['rule_color'],lw=s['inner_rule_width'],alpha=.9,zorder=3)
    if s.get('show_dissemination', True):
        summaries = c.load(s['summary_file'])
        keys=['minimum','q1','median','q3','maximum']
        values=np.column_stack([nums(summaries,k) for k in keys])
        if (np.diff(values,axis=1)<0).any() or len({r['topic_id'] for r in summaries}) != len(summaries):
            raise ValueError('One ordered five-number summary is allowed per topic')
        lo,hi=s['summary_xlim']
        if hi <= lo or (values<lo).any() or (values>hi).any():
            raise ValueError('Summary values must fall inside summary_xlim')
        left,right=s['summary_left'],s['summary_right']
        convert=lambda value: left+(value-lo)/(hi-lo)*(right-left)
        ax.text((left+right)/2,s['header_y'],s['summary_heading'],ha='center',va='bottom',fontsize=s['heading_size'],fontweight='bold')
        for y in (s['top'],s['bottom']):
            ax.plot([left,right],[y,y],c=s['rule_color'],lw=s['rule_width'])
        for row, vals in zip(summaries,values):
            if row['topic_id'] not in topics:
                raise ValueError('Summary topic is missing from the topic table')
            topic=topics[row['topic_id']]
            y=(float(topic['y0'])+float(topic['y1']))/2
            a,b,med,d,e=map(convert,vals)
            h=s['box_height']
            ax.plot([a,e],[y,y],c=s['rule_color'],lw=s['box_width'])
            for x in (a,e):
                ax.plot([x,x],[y-h*.23,y+h*.23],c=s['rule_color'],lw=s['box_width'])
            ax.add_patch(Rectangle((b,y-h/2),d-b,h,facecolor='white',edgecolor=s['rule_color'],lw=s['box_width'],zorder=4))
            ax.plot([med,med],[y-h/2,y+h/2],c=s['median_color'],lw=s['median_width'],zorder=5)
        for tick in s['summary_ticks']:
            x=convert(tick)
            ax.plot([x,x],[s['bottom'],s['bottom']-2],c=s['rule_color'],lw=s['box_width'])
            ax.text(x,s['bottom']+3,f'{tick:.1f}',ha='center',va='top',fontsize=s['tick_size'])


if __name__ == '__main__':
    from render import main
    main(default_figure=22)
