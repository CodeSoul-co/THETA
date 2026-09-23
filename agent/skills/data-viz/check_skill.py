"""Exercise a self-contained skill workspace with non-biological data."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.prepare_workspace import ROOT, prepare


def main():
    source = ROOT / 'figures' / 'figure07' / 'data' / 'figure07.csv'
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='data-viz-skill-check-') as temp:
        temp = Path(temp)
        project = prepare(temp / 'latency', [7])
        assert sorted(p.name for p in (project / 'figures').glob('figure*')) == ['figure07']
        assert not (project / '.git').exists() and not (project / 'references').exists()
        for numbers in ([7], []):
            try:
                prepare(project, numbers)
            except FileExistsError:
                pass
            else:
                raise AssertionError('Existing workspace was overwritten')
        try:
            prepare(temp / 'invalid', [99999])
        except ValueError:
            assert not (temp / 'invalid').exists()
        else:
            raise AssertionError('Unknown template was accepted')
        style = json.loads((project / 'figures/figure07/style.json').read_text())
        style.update(title='Service latency', title_en='Service latency',
                     groups=['Region A', 'Region B', 'Region C'],
                     categories=['Read', 'Write', 'Search', 'Export'],
                     ylabel='Latency (ms)', ylim=[0, 100])
        (project / 'my_style.json').write_text(json.dumps(style))
        (project / 'my_data').mkdir()
        with (project / 'my_data/figure07.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['category', 'group', 'sample', 'value', 'mean', 'error'])
            writer.writeheader()
            for c, category in enumerate(style['categories']):
                for g, group in enumerate(style['groups']):
                    for sample, offset in enumerate([-3, 0, 3], 1):
                        mean = 25 + c * 10 + g * 5
                        writer.writerow(dict(category=category, group=group, sample=sample,
                                             value=mean+offset, mean=mean, error=3))
        subprocess.run([sys.executable, 'render.py', '--figure', '7', '--data-dir', 'my_data',
                        '--style', 'my_style.json', '--format', 'png', 'svg', 'pdf',
                        '--annotations', 'none', '--summary', 'samples'], cwd=project, check=True)
        log = json.loads((project / 'output/render_log.json').read_text())[0]
        assert log['reference_annotations'] is False and log['example_data_unchanged'] is False
        assert log['summary_mode'] == 'samples'
        for fmt in ['png', 'svg', 'pdf']:
            assert (project / f'output/figure07.{fmt}').stat().st_size > 1000
        svg = (project / 'output/figure07.svg').read_text()
        assert 'Latency (ms)' in svg and 'Region A' in svg and 'Relative_mRNA' not in svg
        blank = prepare(temp / 'new-chart')
        subprocess.run([sys.executable, '-m', 'scripts.new_figure', '--id', '20',
                        '--title', 'New chart', '--title-en', 'New chart'], cwd=blank, check=True)
        subprocess.run([sys.executable, '-m', 'figures.figure20.plot', '--format', 'png', 'svg'], cwd=blank, check=True)
        subprocess.run([sys.executable, '-m', 'scripts.build_gallery'], cwd=blank, check=True)
        assert (blank / 'output/figure20.png').stat().st_size > 1000
        assert (blank / 'docs/GALLERY.en.md').is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    print('PASS: isolated template reuse with service data, new-chart rendering, and overwrite protection.')


if __name__ == '__main__':
    main()
