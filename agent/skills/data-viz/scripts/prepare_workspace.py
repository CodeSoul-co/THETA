"""Copy selected templates into a fresh, standalone visualization workspace."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination, figures=(), root=ROOT):
    root, destination = Path(root), Path(destination)
    selected = []
    for number in dict.fromkeys(figures):
        folder = root / 'figures' / f'figure{number:02d}'
        if not (folder / 'plot.py').is_file() or not (folder / 'style.json').is_file():
            raise ValueError(f'Unknown figure: {number}')
        selected.append(folder)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('render.py', 'requirements.txt', 'LICENSE'):
        shutil.copy2(root / name, destination / name)
    shutil.copytree(root / 'vizlib', destination / 'vizlib', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (destination / 'figures').mkdir()
    shutil.copy2(root / 'figures' / '__init__.py', destination / 'figures' / '__init__.py')
    for folder in selected:
        shutil.copytree(folder, destination / 'figures' / folder.name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (destination / 'scripts').mkdir()
    for name in ('__init__.py', 'new_figure.py', 'build_gallery.py'):
        shutil.copy2(root / 'scripts' / name, destination / 'scripts' / name)
    next_id = max([0] + [int(folder.name.removeprefix('figure')) for folder in selected]) + 1
    for lang, name in [('zh', 'README.md'), ('en', 'README.en.md')]:
        zh = lang == 'zh'
        content = '# ' + ('可视化工作目录' if zh else 'Visualization workspace') + '\n\n'
        content += '[中文](README.md) | [English](README.en.md)\n\n'
        content += ('由 Data Viz Skill 创建。随附 CSV 是模板演示数据；请接入自己的数据、更新样式，并记录数据映射与处理过程。\n\n' if zh else 'Created by Data Viz Skill. Included CSVs are template demonstrations; supply your data, update styles, and document mappings and preparation.\n\n')
        content += '```bash\npython3 -m venv .venv\n.venv/bin/python -m pip install -r requirements.txt\n.venv/bin/python render.py --list\n```\n\n'
        content += ('替换数据用 `--data-dir`，替换样式用 `--style`，并按需调整 plot.py 的布局。新数据使用 `--annotations none`；需要分组均值与 SEM 时使用 `--summary samples`，已有可信汇总时使用 `--summary provided` 并注明误差定义。\n\n' if zh else 'Use `--data-dir` for replacement data and `--style` for custom styles; adapt plot.py layouts as needed. Use `--annotations none` for new data, `--summary samples` for group means and SEM, or `--summary provided` for intentional precomputed summaries with documented error definitions.\n\n')
        content += f'```bash\npython -m scripts.new_figure --id {next_id} --title "New chart" --title-en "New chart"\npython -m scripts.build_gallery\n```\n\n'
        content += '<!-- FIGURES:START -->\n\n'
        for folder in selected:
            suffix = '' if zh else '.en'
            content += f'- [{folder.name}](figures/{folder.name}/README{suffix}.md)\n'
        content += '\n<!-- FIGURES:END -->\n'
        (destination / name).write_text(content, encoding='utf-8')
    (destination / '.gitignore').write_text('.venv/\n__pycache__/\n*.pyc\nmy_data/\noutput/\n.env\n', encoding='utf-8')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New destination; never overwrites an existing directory')
    parser.add_argument('--figures', type=int, nargs='+', default=[], help='Template IDs to copy; omit for a new chart')
    args = parser.parse_args()
    try:
        workspace = prepare(args.out, args.figures)
    except (ValueError, FileExistsError) as error:
        parser.exit(2, f'{error}\n')
    print(f'Created {workspace}. Run render.py or python -m scripts.new_figure inside it.')
