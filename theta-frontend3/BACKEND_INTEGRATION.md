# 前端与后端对接说明

本文档说明 theta-frontend3 与 langgraph_agent 后端的对接逻辑，以及已知的不合理或需注意之处。

## 一、核心流程（仪表盘 → 自动流水线）

1. **新建项目**：用户输入项目名、分析模式、主题数；前端用项目名生成 `datasetName`（`getDatasetName(projectName)`：去空格、特殊字符、转小写）。
2. **上传数据**：在新标签页中上传文件；调用 `POST /api/datasets/upload`，**必须使用后端返回的 `dataset_name` 做后续所有调用**（前后端对名称的 sanitize 可能不一致）。
3. **上传后**：前端等待约 1.5 秒再调用预处理检查，避免文件尚未落盘。
4. **预处理/向量化**：`GET /api/preprocessing/check/{dataset}` → 若未就绪则 `POST /api/preprocessing/start` → 轮询 `GET /api/preprocessing/{job_id}` 直至 `status === "completed"` 或 `"failed"`。
5. **训练与评估**：`POST /api/tasks` 创建任务，轮询 `GET /api/tasks/{task_id}` 直至 `status === "completed"` 或 `"failed"`。

## 二、主要 API 与前端用法

| 接口 | 说明 | 前端注意 |
|------|------|----------|
| `POST /api/datasets/upload` | 上传文件到 `DATA_DIR/{dataset_name}/` | 使用返回的 `dataset_name` 作为后续所有 `dataset` 参数；支持任意格式，但**分析需至少一个 CSV**（见下）。 |
| `GET /api/datasets` | 列出数据集 | 后端仅列出**至少包含一个 `.csv` 的目录**。 |
| `GET /api/preprocessing/check/{dataset}` | 是否已预处理（BOW+嵌入） | 返回 `ready_for_training`；不校验是否有 CSV。 |
| `POST /api/preprocessing/start` | 启动预处理（BOW+向量化） | **⚠️ 见下「不合理处」**：目录下**必须已有至少一个 `.csv`**，否则 404。 |
| `GET /api/preprocessing/{job_id}` | 预处理任务状态 | `status` 可能为 `pending`、`bow_generating`、`bow_completed`、`embedding_generating`、`embedding_completed`、`completed`、`failed`；前端按 `completed`/`failed` 判断结束。 |
| `POST /api/tasks` | 创建训练任务 | 传入 `dataset`（即上传返回的 `dataset_name`）、`mode`、`num_topics`。 |
| `GET /api/tasks/{task_id}` | 任务状态 | `current_step`、`progress`、`message`、`error_message`；前端用 `status === "completed"`/`"failed"` 结束轮询。 |

## 三、不合理处与前端应对

### 1. 预处理强制要求 CSV（⚠️ 不合理）

- **现象**：`POST /api/preprocessing/start` 要求 `DATA_DIR/{dataset_name}/` 下**至少有一个 `.csv` 文件**，否则返回 **404**，`detail`: `"No CSV files found in dataset '{dataset}'"`。
- **矛盾**：上传接口接受任意格式（PDF、TXT、DOCX 等），文档也提到“可通过数据清洗转为 CSV”，但若用户**只上传非 CSV**，无法直接进入预处理，且错误是 404 而非 4xx + 明确业务错误码。
- **前端应对**：
  - 上传区文案提示：分析需至少一个 CSV 文件（含文本列如 text、content、cleaned_content）；若仅上传非 CSV，需先做数据清洗生成 CSV。
  - 若 `startPreprocessing` 返回错误且 `message` 包含 `"No CSV files found"`，则展示友好提示：当前数据集需要包含至少一个 CSV 文件才能进行分析，请上传包含文本列的 CSV 或先使用数据清洗转为 CSV。

### 2. 数据集命名一致性

- **建议**：上传后**仅使用后端返回的 `dataset_name`** 调用预处理、任务等接口，不要再用前端生成的名称，避免前后端 sanitize 规则不一致导致 404 或目录不匹配。

### 3. 上传后立即检查预处理

- **建议**：上传成功后延迟约 1.5 秒再调用 `checkPreprocessingStatus` / `startPreprocessing`，减少因文件尚未完全落盘导致的“未找到 CSV”等偶发问题。

## 四、类型与错误处理（前端）

- **预处理任务状态**：后端 `PreprocessingStatus` 含 `status` 细粒度值（如 `bow_generating`、`embedding_completed`）；前端 `PreprocessingJob` 已扩展为包含这些状态及 `error_message`。
- **错误信息**：后端 4xx/5xx 返回 `{ detail: string }`；前端 `fetchApi` 会抛出 `Error(error.detail)`，可在 catch 中根据 `message` 判断是否为“No CSV files found”并做上述友好提示。

## 五、小结

- 上传 → **用返回的 `dataset_name`** → 短暂延迟 → 预处理检查 → 若无 CSV 则 `startPreprocessing` 会 404，前端捕获并提示 CSV 要求。
- 预处理/任务状态已按后端细粒度状态与错误信息做了兼容；若后端后续增加新的 `status` 或字段，前端只需在类型和判断上做小幅扩展即可。
