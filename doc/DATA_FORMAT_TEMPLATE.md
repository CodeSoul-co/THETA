# THETA 数据格式要求模板 (严格模式)

> 本文档说明 THETA 主题模型框架对输入数据的**严格列名规范**

---

## 📋 列名规范总览

| 用途 | 列名 | 是否必需 | 说明 |
|------|------|---------|------|
| **文本** | `text` | ✅ 必需 | 文档文本内容 |
| **时间** | `timestamp` | DTM 必需 | 支持年份/日期/时间戳格式 |
| **协变量** | `cov_*` | STM 必需 | 必须以 `cov_` 前缀命名 |
| **标签** | `label` | 有监督必需 | 文档分类标签 |

---

## 1️⃣ 文本列: `text`

**必须命名为 `text`**

```csv
text
"这是第一篇文档的内容..."
"这是第二篇文档的内容..."
```

---

## 2️⃣ 时间列: `timestamp`

**必须命名为 `timestamp`**，用于 DTM (Dynamic Topic Model) 时序分析。

### 支持的时间格式

| 格式 | 示例 | 处理方式 |
|------|------|---------|
| **年份** | `2026` | 直接使用 |
| **日期** | `2026-10-17` 或 `2026/10/17` | 提取年份 → `2026` |
| **时间戳** | `2026-10-17 14:30:00` | 提取年份 → `2026` |

### 示例

```csv
text,timestamp
"2020年的文档内容...",2020
"2021年的文档内容...",2021-06-15
"2022年的文档内容...",2022-03-20 10:30:00
```

### ⚠️ 重要说明

- **DTM 最终只使用年份**，不支持月/日级别的时间切片
- 无论输入 `2026-10-17` 还是 `2026`，最终都会转换为年份 `2026`
- 至少需要 **2 个不同的年份** 才能运行 DTM

---

## 3️⃣ 协变量列: `cov_*` 前缀

**必须以 `cov_` 前缀命名**，用于 STM (Structural Topic Model) 分析。

### 命名规范

| 原列名 | 规范列名 |
|--------|---------|
| `province` | `cov_province` |
| `category` | `cov_category` |
| `source` | `cov_source` |
| `author` | `cov_author` |
| `region` | `cov_region` |

### 示例

```csv
text,timestamp,cov_province,cov_category,cov_source
"文档内容...",2023,北京,政策,政府网站
"文档内容...",2023,上海,新闻,媒体
"文档内容...",2024,广东,报告,统计局
```

### 协变量要求

- **唯一值数量**: 2-50 个（不能太少或太多）
- **数据类型**: 分类变量（不是连续数值）
- **至少需要 1 个协变量列** 才能运行 STM

---

## 4️⃣ 标签列: `label`

**必须命名为 `label`**，用于有监督主题建模。

```csv
text,label
"关于环境保护的文章...",环境
"关于人工智能的论文...",科技
"关于医疗改革的报告...",医疗
```

---

## 📝 完整数据模板

### 模板 1: 基础主题建模 (LDA/CTM/ETM)

```csv
text
"文档内容1..."
"文档内容2..."
"文档内容3..."
```

### 模板 2: 有监督主题建模

```csv
text,label
"文档内容1...",类别A
"文档内容2...",类别B
"文档内容3...",类别A
```

### 模板 3: 时序主题建模 (DTM)

```csv
text,timestamp
"2020年文档...",2020
"2021年文档...",2021
"2022年文档...",2022
```

### 模板 4: 结构化主题建模 (STM)

```csv
text,cov_province,cov_category
"文档内容...",北京,政策
"文档内容...",上海,新闻
"文档内容...",广东,报告
```

### 模板 5: 完整元数据 (DTM + STM)

```csv
text,timestamp,cov_province,cov_category,cov_source,label
"文档内容...",2023,北京,政策,政府网站,环境
"文档内容...",2023,上海,新闻,媒体,科技
"文档内容...",2024,广东,报告,统计局,经济
```

---

## 🔍 系统检测输出示例

运行 `prepare_data.py` 时会显示：

```
============================================================
Column Auto-Detection Results (Strict Mode)
============================================================

[Naming Convention]
  - Text column: 'text'
  - Time column: 'timestamp' (supports: 2026, 2026-10-17, 2026-10-17 14:30:00)
  - Covariate columns: 'cov_<name>' (e.g., cov_province, cov_category)
  - Label column: 'label'

[Time Column] ✓ 'timestamp' (type: year)
  Sample values: [2020, 2021, 2022, 2023, 2024]
  Note: All formats will be converted to YEAR for DTM analysis

[Covariate Columns] ✓ Detected 3 columns:
  - cov_province: 34 unique values, e.g., ['北京', '上海', '广东']
  - cov_category: 5 unique values, e.g., ['政策', '新闻', '报告']
  - cov_source: 8 unique values, e.g., ['政府网站', '媒体', '学术期刊']
============================================================
```

---

## ⚠️ 遗留列名兼容

为保持向后兼容，系统仍会识别以下遗留列名，但会显示警告：

### 文本列（遗留）
- `cleaned_content`, `clean_text`, `content`, `Text`
- **建议**: 重命名为 `text`

### 时间列（遗留）
- `year`, `date`, `time`, `created_at`, `年份`
- **建议**: 重命名为 `timestamp`

### 协变量列（遗留）
- 任何符合条件的分类列（2-50 个唯一值）
- **建议**: 添加 `cov_` 前缀

---

## 🛠️ 数据准备命令

```bash
cd THETA/src/models

# 基础模型
python prepare_data.py --dataset my_data --model baseline

# DTM（需要 timestamp 列）
python prepare_data.py --dataset my_data --model dtm

# STM（需要 cov_* 列）
python prepare_data.py --dataset my_data --model baseline
```

---

## ✅ 数据质量检查清单

- [ ] CSV 文件使用 UTF-8 编码
- [ ] 文本列命名为 `text`
- [ ] 时间列命名为 `timestamp`（如需 DTM）
- [ ] 协变量列以 `cov_` 前缀命名（如需 STM）
- [ ] 标签列命名为 `label`（如需有监督学习）
- [ ] 每个文档至少 20 个词
- [ ] 总文档数至少 100 篇

---

**最后更新**: 2026-04-11
