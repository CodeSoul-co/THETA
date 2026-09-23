# Built-in visualization skill

**English** | [中文](data-viz-skill.md)

THETA includes [AdamsukS/data-viz-skill](https://github.com/AdamsukS/data-viz-skill), a collection of editable Python templates for PNG, SVG, and PDF figures. It can visualize verified topic-model results, surveys, and other structured data.

Files are under `agent/skills/data-viz/`, pinned to commit `a01848fcfacfcde4dfdf116611030eb9212d467e`. The snapshot contains 22 templates. Its MIT license remains in that directory; provenance and per-file hashes are in `agent/skills/data-viz.source.json`. Startup does not download a new copy.

## Use with the Agent

Ask: “Use the built-in data-viz skill to inspect my data, recommend a suitable figure template, and prepare an editable plotting project.”

| Tool | Purpose |
| --- | --- |
| `skills_list` | Discover bundled skills, versions, and files |
| `skills_read` | Read `SKILL.md`, template selection guidance, and a chosen template's README, style, and plotting code |
| `skills_prepare` | Copy selected templates into a new project and deliver an archive with labeled example previews |

The tools work in both CLI and Web modes. Preparing a project does not render user data, install dependencies, or run computation. Rendering requires an authorized execution environment or the local commands below. A skill does not grant execution permission.

## Render locally

Extract the project and enter its directory. With Python 3.10+:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python render.py --list
.venv/bin/python render.py --figure 7 --data-dir ./my_data --style ./my_style.json \
  --out ./output --format png svg pdf --annotations none --summary samples
```

Prepare `my_data` and `my_style.json` according to the selected template. You can also create a workspace directly:

```sh
python3 agent/skills/data-viz/scripts/prepare_workspace.py --out ./visualization --figures 7
```

Replace data and labels first, then edit the copied plotting code as needed. Do not modify bundled templates. Omit unsupported panels or uncertainty instead of inventing samples, zero error, or example p-values. `--summary samples` computes summaries from observations; `--summary provided` uses supplied summaries whose SD, SEM, or CI definition must be stated.

## Deliverables

Keep titles, units, denominators, exclusions, group ordering, uncertainty, and statistical methods traceable. Inspect labels, clipping, colors, and legends. Deliver figures together with source code, data mapping, reproduction commands, and interpretation.

Template CSV files and previews are illustrative examples, not findings from a user's data or a way to reconstruct published numerical results. Upstream updates require review of licensing, hashes, rendering, and data contracts.
