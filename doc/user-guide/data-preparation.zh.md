# 数据准备

[English](https://github.com/CodeSoul-co/THETA/blob/main/doc/user-guide/data-preparation.md) | **中文**

---

本指南涵盖数据格式要求和清洗流程。

---

## 数据格式要求

THETA接受具有特定列要求的CSV文件。预处理流程识别几个标准的文本内容列名。

**可接受的文本列名：**
- `text`
- `content`
- `cleaned_content`
- `clean_text`

**可选列：**
- `label` 或 `category` - 有监督模式必需
- `year`、`timestamp` 或 `date` - DTM（时间分析）必需

示例CSV结构：

```csv
text,label,year
"关于可再生能源和太阳能电池板的文档。",环境,2020
"讨论机器学习应用的文章。",技术,2021
"关于医疗改革的政策文件。",医疗,2022
```

---

## 数据清洗

原始文本通常包含降低主题质量的噪声。数据清洗模块处理英文和中文文本中的常见问题。

### 英文数据清洗

```bash
cd ./THETA

python -m dataclean.main \
    --input ./data/raw_data.csv \
    --output ./data/cleaned_data.csv \
    --language english
```

清洗过程移除：
- HTML标签和标记
- URL和电子邮件地址
- 特殊字符和符号
- 多余空白
- 不可打印字符

### 中文数据清洗

中文文本需要专门处理以实现正确的分词和清洗。

```bash
python -m dataclean.main \
    --input ./data/raw_data.csv \
    --output ./data/cleaned_data.csv \
    --language chinese
```

中文的额外步骤：
- 移除繁体标点符号
- 处理全角和半角字符
- 保留中文词边界

### 批量清洗

处理目录中的多个文件：

```bash
python -m dataclean.main \
    --input ./data/raw/ \
    --output ./data/cleaned/ \
    --language english
```

输入目录中的所有CSV文件都将被处理并以相同的文件名保存到输出目录。

## 工作台中的训练集、验证集与测试集

在数据选择卡中开启“自定义训练 / 验证 / 测试集划分”，可选择：

- **按比例划分**：三份比例都大于零且合计为 100%，初始为 70% / 20% / 10%。支持随机与顺序划分；随机种子可调整，顺序划分依次取训练、验证、测试记录。
- **分别上传**：选择或上传三个不同文件，各自支持表格或文档格式，分别选择正文、时间、标签和元数据列。文档按正文内容分段，全量数据为三份文件的合并结果。

关闭开关时，默认随机按 70% / 30% 划分训练与验证集，并使用全量数据测试。此时测试包含训练和验证记录，**不是独立留出测试成绩**。预处理后训练集至少保留两条有效正文，各评估集至少保留一条。词表建立与模型拟合仅使用训练记录，所有结果分组使用同一个已训练模型。

在“研究结果 → 数据集结果”中切换全量、训练、验证和测试集，查看各组指标与平均主题权重，下载逐条文档结果及主题矩阵。历史任务没有划分记录时会明确提示，不会补造分组。Web 与桌面端共用此流程；命令行智能体的训练方案支持通过 `dataSplit` 设置 `enabled`、`mode: "ratio"`、`ratios`、`method` 和 `seed`。
