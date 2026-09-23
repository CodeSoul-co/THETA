"""Add an independently runnable figure folder without editing a registry."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PLOT = '''"""A small, replaceable example: replicate values by group."""
from vizlib.common import clean, label_axes, nums, subset


def draw(c, rows):
    s = c.s
    ax = c.ax()
    clean(ax)
    label_axes(ax, s)
    for i, group in enumerate(s["groups"]):
        values = nums(subset(rows, group=group), "value")
        ax.bar(i, values.mean(), width=.6, color=s["colors"][i], alpha=.6)
        ax.scatter(i + c.rng.uniform(-.15, .15, len(values)), values, color=s["colors"][i])
    ax.set_xticks(range(len(s["groups"])), s["groups"])


if __name__ == "__main__":
    from render import main
    main(default_figure=__NUMBER__)
'''


def create(number, title, title_en, root=ROOT):
    if number < 1:
        raise ValueError("Figure ID must be positive")
    folder = Path(root) / "figures" / f"figure{number:02d}"
    folder.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing figure.
    (folder / "data").mkdir()
    data_file = f"figure{number:02d}.csv"
    rows = [{"group": group, "value": value} for group, values in [("A", [1, 1.2, .8]), ("B", [2, 1.8, 2.2])] for value in values]
    with (folder / "data" / data_file).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "value"])
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256((folder / "data" / data_file).read_bytes()).hexdigest()
    style = {"id": number, "title": title, "title_en": title_en, "width": 800, "height": 600,
             "dpi": 100, "font": "DejaVu Sans", "font_size": 14, "line_width": 1.2,
             "seed": 190913, "axes": [100, 60, 640, 460], "data_file": data_file,
             "groups": ["A", "B"], "colors": ["#549f98", "#c79852"], "ylim": [0, 3],
             "ylabel": "Value (arbitrary units)", "provenance": "Synthetic starter observations; not experimental data."}
    provenance = {"figure": number, "title": {"zh": title, "en": title_en}, "classification": "synthetic",
                  "original_raw_data_available": False, "source_publication": None,
                  "provenance": {"zh": "手工指定的演示值，不是实验数据。", "en": "Hand-specified demonstration values, not experimental data."},
                  "raw_data_required": {"zh": ["每个独立样本的 ID、分组、测量值和物理单位；导入真实数据时更新此说明。"],
                                        "en": ["Independent sample IDs, groups, measurements and physical units; update this contract when importing actual data."]},
                  "preprocessing": {"zh": "value 直接绘制，柱高取组均值。", "en": "Values are plotted directly; bars show group means."},
                  "files": [{"path": "data/" + data_file, "classification": "synthetic", "row_count": len(rows), "sha256": digest,
                             "fields": [{"name": "group", "type": "string", "unit": "label", "description": {"zh": "A/B 分组", "en": "A/B group"}},
                                        {"name": "value", "type": "number", "unit": "arbitrary", "description": {"zh": "单次测量演示值", "en": "One demonstration measurement"}}]}]}
    for name, value in [("style.json", style), ("provenance.json", provenance), ("example_hashes.json", {data_file: digest})]:
        (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (folder / "plot.py").write_text(PLOT.replace("__NUMBER__", str(number)), encoding="utf-8")
    for lang, name in [("zh", "README.md"), ("en", "README.en.md")]:
        zh = lang == "zh"
        text = f"# {number:02d} · {title if zh else title_en}\n\n[中文](README.md) | [English](README.en.md)\n\n"
        text += ("当前是两个分组的模拟观测示例。新增实际图式时修改 plot.py、style.json、数据与 provenance.json，并同步更新这两份说明。\n\n## 原始数据要求\n\n" if zh else "This starter contains synthetic observations for two groups. Replace plot.py, style.json, data and provenance.json for your chart, and update both READMEs.\n\n## Raw-data requirements\n\n")
        text += provenance["raw_data_required"][lang][0] + "\n\n"
        text += f"| Column | Meaning |\n|---|---|\n| group | A/B |\n| value | {'模拟测量值，无实际单位' if zh else 'Synthetic measurement, arbitrary units'} |\n\n"
        text += f"```bash\npython -m figures.figure{number:02d}.plot --format png svg\npython -m scripts.build_gallery\n```\n"
        (folder / name).write_text(text, encoding="utf-8")
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--title-en", required=True)
    args = parser.parse_args()
    print(create(args.id, args.title, args.title_en).relative_to(ROOT))
