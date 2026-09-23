# 结果与模型使用说明

此包包含选定任务的图表、绘图数据、训练模型及源代码，不包含 API 密钥或基础嵌入模型权重。

## 固定加载入口（全部 12 种模型）

找到 `model_delivery.json`，其所在目录就是模型目录。使用文件记录的 Python 版本建立独立环境，并安装同目录的 `requirements-model.txt`。

在解压目录运行：

```sh
PYTHONPATH=代码/src/models python 代码/src/models/model_delivery.py 模型目录 --trust-model
```

全部模型统一使用以下入口，不会重新训练、下载权重或调用外部服务：

```python
from model_delivery import load_trained_model
model = load_trained_model("模型目录", trusted=True)
beta = model.get_beta()
```

具体推理方法及真实参数签名见 `model_delivery.json` 的 `methods`。神经模型参数保存为 CPU 格式。

## 推理与绘图

- 词袋必须沿用模型目录的 `vocab.json` 及列顺序，不能重新拟合词表。
- THETA、CTM、BERTopic 必须使用训练时相同的嵌入模型和维度。云端嵌入需要单独配置、授权及额度。
- DTM 的 `get_beta()` 返回时间×主题×词表；可用 `get_beta(time_index=0)` 读取一个时间片。
- STM 必须沿用训练时的协变量及编码，不能把关联解释为因果效应。
- 图表数据见实际 CSV 与矩阵；绘图代码位于 `代码/src/models/visualization`，使用 `run_visualization.py --help` 查看参数，选择已有模型结果目录即可，无须重新训练。
- 没有 `model_delivery.json` 的历史任务不能仅靠主题矩阵还原模型。

## 安全与研究边界

joblib/pickle 反序列化可能执行代码，只加载可信来源。校验和只能检查文件完整性，不能证明来源可信。不要对陌生模型使用 `trusted=True`。

训练结束不等于收敛或研究结论有效；需结合指标、日志及代表文本核查。固定样本和少量迭代仅用于流程验收，不能当作全量质量结论。
