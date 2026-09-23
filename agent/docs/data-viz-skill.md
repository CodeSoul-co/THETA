# 推荐并内置：Data Viz Skill

推荐 [AdamsukS/data-viz-skill](https://github.com/AdamsukS/data-viz-skill)：适合把已验证数据转成论文图，复用实际 Python 模板、布局、配色与数据契约，输出 PNG/SVG/PDF 和可修改源码。它适合主题模型结果、调查和业务数据，不限于生物学案例。

THETA 已随代码内置完整技能，位于 `agent/skills/data-viz/`，无需首次启动联网下载。固定上游 commit 为 `a01848fcfacfcde4dfdf116611030eb9212d467e`，当前快照实际包含 22 个模板，包括自然语言相关图式。许可证为 MIT，原文保留在技能目录；逐文件 SHA-256 与来源见 `agent/skills/data-viz.source.json`。上游 README 的模板数量可能滞后，以实际目录为准。

## 在 THETA 中使用

可直接说：“请用内置 data-viz skill，检查我的数据，推荐最合适的论文图模板，给我可修改的绘图工程。”

Agent 的工具：

| 工具 | 用法与作用 |
|---|---|
| `skills_list` | `{}`：发现内置技能、固定版本、文件目录 |
| `skills_read` | `{"skill":"data-viz","file":"SKILL.md"}`：读取指令；再读 `docs/TEMPLATE_SELECTION.md` 和选中模板的 README、style.json、plot.py；支持 offset/limit |
| `skills_prepare` | `{"skill":"data-viz","figures":[7]}`：建立新的可修改工作目录，交付 tar.gz、说明和明确标注为示例的预览 |

这些工具在 CLI 和网页 Agent 中相同。`skills_prepare` 只复制模板，**不渲染、不安装依赖、不计算、不附带用户数据**。Agent 根据当前宿主是否提供已授权的代码 worker 决定后续执行；没有该 worker 时交付工程与下面的本地命令，不能声称已绘制用户结果。技能不是授权，也不能绕过现有计算确认。

## 本地运行

解压工程，进入其目录，然后执行（Python 3.10+）：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python render.py --list
.venv/bin/python render.py --figure 7 --data-dir ./my_data --style ./my_style.json \
  --out ./output --format png svg pdf --annotations none --summary samples
```

`my_data` 和 `my_style.json` 必须先按选定模板的实际契约准备，不是安装后自动存在的文件。也可以直接使用内置 helper：

```sh
python3 agent/skills/data-viz/scripts/prepare_workspace.py --out ./visualization --figures 7
```

优先替换数据与标签，再局部修改复制出来的 `plot.py`；不要修改内置模板。对缺少的不确定性或辅助面板，应删除对应元素，不要制造零误差、模拟样本或示例 P 值。`--summary samples` 表示从样本计算摘要；使用已有摘要时显式指定 `--summary provided` 并说明误差是 SD、SEM 还是 CI。需要本地修改图代码时，仍以实际数据语义和用户要求为准。

## 论文交付要求

图题、单位、分母、缺失处理、分组顺序、误差定义和对应统计方法必须可追溯。图形不负责产生主题权重、语义嵌入或因果证据；文本图需要先有已计算且行来源可靠的结构化输入。渲染后检查标签、裁切、颜色映射和图例；交付图片、源码、数据映射、复现命令和实质解释。

模板中的 CSV 与预览属于估读或模拟案例，不能作为用户的真实分析结果，也不能用于复原原论文数值。更新上游时重新核查差异、许可证、哈希、渲染与数据契约；不会静默跟随 main 更新。
