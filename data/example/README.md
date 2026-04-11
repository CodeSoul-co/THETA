# 数据上传说明

将你的数据集放入 `data/{数据集名称}/` 目录下，按以下规范命名列。

## 目录结构

```
data/
├── example/                  ← 本示例文件夹
│   └── example_cleaned.csv
└── {你的数据集}/
    └── {数据集名称}_cleaned.csv
```

## 必需列

| 列名 | 说明 | 是否必填 |
|------|------|----------|
| `text` | 正文内容（字符串） | ✅ 必填 |
| `timestamp` | 发布日期，如 `2026-04-01` | DTM 必填 |
| `cov_*` | 协变量，前缀 `cov_`，如 `cov_platform` | STM 必填 |
| `label` | 分类标签（有监督模式） | 可选 |

## 支持的文件格式

- `.csv`（推荐，需包含 `text` 列）
- `.xlsx`（会自动转换）
- `.txt` / `.docx` / `.pdf`（每个文件视为一篇文档）

## 日期格式

`timestamp` 列支持以下格式，DTM 分析统一转为年份：

```
2026          ← 年份
2026-04-01    ← 日期（推荐）
2026-04-01 14:30:00  ← 完整时间
```

## 快速启动

```bash
# 将数据放入 data/{数据集名}/ 后执行：
conda run -n theta bash scripts/quick_start.sh {数据集名}

# 示例运行本 example 数据集：
conda run -n theta bash scripts/quick_start.sh example
```

## 参考示例

见本目录 `example_cleaned.csv`，包含：
- 10 条中英文混合文本
- `text`、`timestamp`、`cov_platform` 三列
- 覆盖微信公众号、twitter、知乎、reddit、小红书五个平台
