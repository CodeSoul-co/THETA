import { mkdirSync, readdirSync, lstatSync, linkSync, copyFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { pythonExecutable } from '../src/adapters/python-worker.js';
import { repositoryRoot } from '../src/environment.js';

/** Python's ZIP writer records UTF-8 names consistently on macOS and Linux. */
export async function createDeliveryZip(stage: string, archive: string): Promise<void> {
  await promisify(execFile)(pythonExecutable(), ['-c', `
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import sys
root = Path(sys.argv[1])
with ZipFile(sys.argv[2], 'w', compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
    for file in sorted(root.rglob('*')):
        if file.is_symlink():
            raise ValueError('交付包不允许符号链接')
        if file.is_file():
            archive.write(file, file.relative_to(root).as_posix())
`, stage, archive], { timeout: 300_000 });
}

/** Shared final delivery only; manual and conversation keep independent records. */
export function stageDelivery(stage: string, entries: Array<{ name: string; path: string }>, root = repositoryRoot()) {
  const copy = (source: string, name: string) => {
    const relative = name.replaceAll('\\', '/');
    if (path.isAbsolute(relative) || relative.split('/').some(part => part === '..' || !part)) throw new Error('产物路径无效');
    if (!lstatSync(source).isFile()) throw new Error('产物必须是普通文件');
    const destination = path.join(stage, relative);
    mkdirSync(path.dirname(destination), { recursive: true });
    try { linkSync(source, destination); } catch { copyFileSync(source, destination); }
  };
  for (const entry of entries) copy(entry.path, entry.name);
  const codeRoot = path.join(root, 'src', 'models');
  const sourceFiles = (folder: string): void => {
    for (const entry of readdirSync(folder, { withFileTypes: true })) {
      if (entry.isSymbolicLink() || entry.name.startsWith('.') || entry.name === '__pycache__') continue;
      const file = path.join(folder, entry.name);
      if (entry.isDirectory()) sourceFiles(file);
      else if (entry.name.endsWith('.py') || (file.startsWith(path.join(codeRoot, 'resources') + path.sep) && entry.name.endsWith('.txt'))) {
        copy(file, '代码/src/models/' + path.relative(codeRoot, file).split(path.sep).join('/'));
      }
    }
  };
  sourceFiles(codeRoot);
  writeFileSync(path.join(stage, '使用方法.md'), `# 结果与模型使用说明

此包包含当前选定模型的实际图表、绘图数据、训练产物及当前版本源代码，不包含其他项目数据、API 密钥或基础嵌入模型权重。

## 固定模型加载入口（全部 12 种模型）

1. 找到包中的 model_delivery.json，它所在的目录就是模型目录。
2. 使用该文件记录的 Python 版本建立独立环境，按同目录 requirements-model.txt 安装依赖。
3. 在解压目录执行（将模型目录替换为实际路径）：

    PYTHONPATH=代码/src/models python 代码/src/models/model_delivery.py 模型目录 --trust-model

在自己的程序中，全部模型统一通过以下方法加载：

    from model_delivery import load_trained_model
    model = load_trained_model("模型目录", trusted=True)

加载不会重新训练、下载模型或调用外部服务。神经模型导出为 CPU 参数。具体模型的推理方法、真实参数签名和原始方法说明见 model_delivery.json 的 methods；不是所有模型都接受同样的输入矩阵。

## 新数据推理与复现绘图

- 词袋输入必须使用训练时的词表与相同列顺序，不能重新拟合词表；使用随模型提供的 vocab.json。
- THETA、CTM、BERTopic 的文本嵌入须使用训练时相同的基础模型与维度；云端嵌入需要用户单独配置和授权，密钥从不随包导出。
- DTM 要提供匹配的时间片；STM 要沿用训练期协变量编码。相关元数据以本包实际产物为准。
- 原始矩阵与 CSV 是重新绘图的数据；完整绘图实现位于代码/src/models/visualization，入口 run_visualization.py。按该入口 --help 选择模型、结果目录和中文语言，不需要重新训练。
- 缺少 model_delivery.json 的历史任务没有统一可加载模型包，不能只凭 theta/beta 矩阵还原训练模型；需要保留原生加载方法或重新训练。

## 安全与解释边界

joblib/pickle 反序列化可以执行代码，仅加载可信来源，校验和只能检测意外损坏，不能证明来源安全。不要把 trusted=True 用于陌生文件。
训练完成不等于模型已收敛或研究结论有效；请结合日志、指标、代表文本与任务目标核查结果。
`, 'utf8');
}
