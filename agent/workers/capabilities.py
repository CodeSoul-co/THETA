"""Portable data/model capabilities. No Agent prompts or numerical model implementation."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import math
from collections import Counter
from statistics import mean, median

from .dataset.explorer import explore_dataset
from .dataset.readers import SUPPORTED_SUFFIXES, load_dataset
from .results_reader import file_hash
from .model_contract import contract, validate_parameters
from . import runtime_environments

MODELS = {
    "etm": "原始 ETM：BOW 与词级语义表示",
    "nvdm": "神经文档变分表示，不直接等同主题概率",
    "gsm": "Gaussian Softmax 神经主题模型",
    "prodlda": "Product-of-experts 神经主题模型",
    "lda": "词袋主题模型，适合作为可解释基线",
    "btm": "基于词对的短文本主题模型",
    "hdp": "HDP 截断主题探索，需识别低权重主题",
    "dtm": "按时间片建模，需要有效时间列",
    "stm": "结构主题模型，需要协变量",
    "ctm": "结合语义 embedding 与词袋的主题模型，需要本地 SBERT",
    "bertopic": "语义聚类主题模型，需要本地 SBERT",
    "theta": "THETA 主题模型；zero_shot 支持本地 Qwen 或显式批准的云 embedding，微调需要本地 Qwen",
}


def engine_root() -> Path:
    root = Path(os.environ.get("THETA_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    if not (root / "src/models/run_pipeline.py").is_file():
        raise ValueError("THETA_PROJECT_ROOT 必须指向包含 src/models 的计算仓库")
    return root


def model_inspect(payload: dict) -> dict:
    model = payload["modelId"].lower()
    if model not in MODELS:
        raise ValueError("不支持的模型")
    parameters = contract(engine_root(), model)
    return {"modelId": model, "description": MODELS[model], "runtimeProfile": runtime_environments.model_profile(model),
            "requiresTime": model == "dtm", "requiresCovariates": model == "stm",
            "requiresWeights": model in {"theta", "ctm", "bertopic"},
            "embeddingOptions": (["local", "cloud_zero_shot_requires_approval"] if model == 'theta'
                                 else ["local_sbert"] if model in {'ctm', 'bertopic'} else ["not_required"]),
            "parameterNotes": {"max_iter": "LDA/STM 的训练迭代上限", "epochs": "神经模型的训练轮数；不是 LDA/STM 的迭代数",
                               "covariates": "STM 必须选择真实分组/解释变量；输出关联不等于因果效应",
                               "num_topics": "用于可解释的小规模比较；不能只按样本数自动确定业务主题数"},
            "supportedParameters": sorted(parameters),
            "parameters": parameters,
            "parameterChoices": {key: info["choices"] for key, info in parameters.items() if info.get("choices") is not None},
            "languageNote": "language 仅影响图表标题、坐标轴和图例。分词按文字体系自动处理混合文本，加载检测到的语言停用词；可用 text.stopwords 替换本次任务的内置词表。",
            "engine": "src/models/run_pipeline.py"}


def runtime_check(payload: dict) -> dict:
    model = model_inspect(payload)["modelId"]
    cloud = payload.get('embeddingProvider') == 'cloud'
    mode_supported = not cloud or model == 'theta' and payload.get('mode', 'zero_shot') == 'zero_shot'
    # prepare_data imports the complete dataclean converter even for CSV input.
    # Keep the probe aligned with the code path that compute.submit will execute,
    # otherwise readiness can report a false positive and fail before training.
    modules = [
        "numpy", "pandas", "scipy", "sklearn", "tqdm", "torch", "gensim", "jieba", "langdetect", "nltk",
        "docx", "PyPDF2", "pdfminer", "pdf2docx", "openpyxl",
    ]
    if model in {"ctm", "bertopic"}:
        modules += ["sentence_transformers", "transformers"]
    elif model == "theta" and payload.get("embeddingProvider") != "cloud":
        modules += ["transformers"]
    if model == "bertopic":
        modules += ["bertopic", "umap", "hdbscan"]
    if model == 'theta' and payload.get('mode') in {'supervised', 'unsupervised'}: modules += ['peft']
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    asset_key = "QWEN_MODEL_0_6B" if model == "theta" else "SBERT_MODEL_PATH"
    if model == "theta" and payload.get("modelSize") in {"4B", "8B"}:
        asset_key = "QWEN_MODEL_" + payload["modelSize"]
    needs = model in {"theta", "ctm", "bertopic"} and not (model == 'theta' and cloud and mode_supported)
    asset = payload.get("modelAssets", {}).get(asset_key, os.environ.get(asset_key, ""))
    assets_ready = not needs or (bool(asset) and Path(asset).is_dir() and (Path(asset) / 'config.json').is_file()
        and any(file.stat().st_size > 0 for pattern in ['*.safetensors', 'pytorch_model*.bin'] for file in Path(asset).glob(pattern) if file.is_file()))
    gpu_ready = True
    if (payload.get('device') or 'cpu').startswith('cuda:'):
        try:
            import torch
            gpu_ready = torch.cuda.is_available() and int(payload['device'].split(':')[1]) < torch.cuda.device_count()
        except (ImportError, ValueError): gpu_ready = False
    issues = []
    if not assets_ready:
        kind = "CTM / BERTopic 兼容模型" if model in {"ctm", "bertopic"} else "本地嵌入模型"
        issues.append(f"{kind}尚未准备好。请在设置的嵌入模型页面下载兼容模型，并选择包含 config.json、分词器与权重的完整目录；安装包不包含模型权重。")
    if missing:
        issues.append("计算依赖缺失：" + ", ".join(missing) + "。桌面用户请重新安装完整安装包；源码用户请检查计算环境依赖。")
    if not gpu_ready:
        issues.append("所选 GPU 不可用。请选择 CPU，或检查 GPU 与驱动配置后重试。")
    if not mode_supported:
        issues.append("当前模型或训练模式不支持云端嵌入。请选择兼容的本地模型；云端嵌入适用于 THETA 零样本分析。")
    return {"ready": not missing and assets_ready and gpu_ready and mode_supported, "issues": issues, "embeddingModeSupported": mode_supported, "deviceReady": gpu_ready, "modelId": model, "missingDependencies": missing,
            "modelAssetsReady": assets_ready, "requiredAssetVariable": asset_key if needs else None,
            "modelAsset": {"name": Path(asset).name, "path": str(Path(asset).resolve())} if needs and asset else None,
            "python": sys.executable, "enginePresent": True,
            "runtime": runtime_environments.identity(runtime_environments.model_profile(model)),
            "limitations": ["Readiness is a dependency/cache probe; actual model compatibility is checked during loading."]}


def dataset_import(payload: dict) -> dict:
    source = Path(payload["filePath"]).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("请选择支持的数据文件")
    if source.stat().st_size > 200 * 1024 * 1024:
        raise ValueError("本地导入限制为 200 MiB；更大数据请使用数据服务")
    digest = file_hash(source)
    target = Path(payload["uploadDir"]).resolve() / digest / ("data" + source.suffix.lower())
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        try:
            shutil.copyfile(source, temp)
            if file_hash(temp) != digest:
                raise ValueError("导入过程中源文件发生变化，请重新导入")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    if file_hash(target) != digest:
        raise ValueError("托管数据校验失败")
    return {"datasetRef": "dataset-" + digest, "sha256": digest, "fileName": source.name,
            "managedPath": str(target), "sizeBytes": target.stat().st_size}


def verify_dataset(dataset: dict) -> Path:
    file = Path(dataset["managedPath"])
    if file.is_symlink() or file_hash(file) != dataset["sha256"]:
        raise ValueError("数据版本已变化；重新导入并确认方案后才能继续")
    return file


def dataset_preview(payload: dict) -> dict:
    """Explicit manual-workbench preview, never exposed as an Agent tool."""
    file = verify_dataset(payload['dataset'])
    table = load_dataset(file, profile_limit=5)
    text_input = file.suffix.lower() in {'.txt', '.md', '.pdf', '.docx'}
    # Preview the beginning, not the reservoir sample used for profiling large files.
    rows = table.head_rows[:5]
    result = {'columns': table.columns, 'rows': [[str(row.get(column) if row.get(column) is not None else '') for column in table.columns] for row in rows],
              'inputKind': 'text' if text_input else 'table', 'totalRecords': table.row_count}
    if text_input:
        if not table.row_count:
            raise ValueError('未读取到可分析的正文。扫描版 PDF 请先完成文字识别，或上传含文字的 TXT、Word 文件。')
        result.update(textColumn='text', segments=[{key: row[key] for key in ['text', 'page', 'paragraph'] if key in row} for row in rows])
    return result


def dataset_profile(payload: dict) -> dict:
    dataset = payload["dataset"]
    file = verify_dataset(dataset)
    result = explore_dataset({"filePath": str(file), "datasetRef": dataset["datasetRef"],
                              "datasetHash": dataset["sha256"], "fileName": dataset["fileName"]})
    # This boundary sends aggregates only, even when the LLM requests raw examples.
    result.pop("sampleRows", None)
    for item in result.get("columnProfiles", []):
        for key in list(item):
            if key.lower() in {"examples", "samplevalues", "topvalues", "values", "samples"}:
                item.pop(key)
    columns = payload.get("columns") or ([payload["column"]] if payload.get("column") else [])
    if columns:
        unknown = set(columns) - set(result["columns"])
        if unknown:
            raise ValueError("数据中不存在请求的列")
        result["columnProfiles"] = [item for item in result["columnProfiles"] if item.get("name") in columns]
    operation = payload.get("operation", "overview")
    result["operation"] = operation
    if operation in {"text_profile", "categorical_profile", "duplicates", "relationships"}:
        table = load_dataset(file)
        chosen = columns or [entry["name"] for entry in result.get("candidateRoles", {}).get("text", [])[:1]]
        if not chosen:
            raise ValueError("请指定需要分析的列")
        values = [str(row.get(chosen[0]) or "").strip() for row in table.rows]
        nonempty = [value for value in values if value]
        detail = {"columns": chosen, "profileRows": len(values), "sampled": table.rows_truncated}
        if operation == "text_profile":
            lengths = [len(value) for value in nonempty]
            detail.update({"nonEmptyCount": len(nonempty), "averageLength": mean(lengths) if lengths else 0,
                           "medianLength": median(lengths) if lengths else 0, "maximumLength": max(lengths, default=0)})
        elif operation == "categorical_profile":
            counts = Counter(nonempty)
            detail.update({"categoryCount": len(counts), "largestCategoryCounts": sorted(counts.values(), reverse=True)[:20],
                           "categoryLabelsWithheld": True})
        elif operation == "duplicates":
            detail.update({"nonEmptyCount": len(nonempty), "duplicateCount": len(nonempty) - len(set(nonempty)),
                           "duplicateRatio": (len(nonempty) - len(set(nonempty))) / max(1, len(nonempty))})
        else:
            if len(chosen) != 2:
                raise ValueError("relationships 需要指定两列；当前支持数值列 Pearson 相关")
            pairs = []
            for row in table.rows:
                try:
                    x, y = float(row[chosen[0]]), float(row[chosen[1]])
                    if math.isfinite(x) and math.isfinite(y):
                        pairs.append((x, y))
                except (ValueError, TypeError):
                    continue
            coefficient = None
            if len(pairs) >= 2:
                mx, my = mean(x for x, _ in pairs), mean(y for _, y in pairs)
                denominator = math.sqrt(sum((x-mx)**2 for x, _ in pairs) * sum((y-my)**2 for _, y in pairs))
                if denominator:
                    coefficient = sum((x-mx)*(y-my) for x, y in pairs) / denominator
            detail.update({"numericPairCount": len(pairs), "pearsonCorrelation": coefficient,
                           "limitation": "数值相关不代表因果关系；非数值或常量列不返回系数"})
        result["detail"] = detail
    elif operation == "missingness":
        result["detail"] = [{"column": item["name"], "missingRatio": item["missingRatio"]} for item in result["columnProfiles"]]
    elif operation == "time_profile":
        result["detail"] = result["timeCoverage"]
    verify_dataset(dataset)
    return result


def validate_plan(payload: dict) -> dict:
    plan, dataset = payload["plan"], payload["dataset"]
    model_inspect({"modelId": plan["modelId"]})
    profile = dataset_profile({"dataset": dataset})
    columns = profile["columns"]
    selected = [plan["textColumn"], *plan.get("covariates", [])]
    if plan.get("labelColumn"):
        selected.append(plan["labelColumn"])
    if plan.get("timeColumn"):
        selected.append(plan["timeColumn"])
    if any(column not in columns for column in selected):
        raise ValueError("方案中的列不存在于数据中")
    if profile["rowCount"] < 2:
        raise ValueError("训练至少需要两条文本")
    if plan["modelId"] == "dtm" and not plan.get("timeColumn"):
        raise ValueError("DTM 需要明确的时间列")
    if plan["modelId"] == "stm" and not plan.get("covariates"):
        raise ValueError("STM 需要明确的协变量列")
    validate_parameters(engine_root(), plan)
    if 'prepare.time_slices' in plan['params']:
        if not plan.get('timeColumn'): raise ValueError('time_slices 必须对应已选时间列')
        table = load_dataset(verify_dataset(dataset), profile_limit=1_000_000)
        if table.rows_truncated: raise ValueError('无法完整核实时间片，不会使用画像样本代替')
        count = len({str(row.get(plan['timeColumn'])) for row in table.rows if row.get(plan['timeColumn']) is not None})
        if count != plan['params']['prepare.time_slices']: raise ValueError('time_slices 必须等于时间列现有片数；原入口不重新分桶，请先选好实际分箱列')
    from .execution_policy import execution_policy
    execution_policy(plan)
    return {"valid": True, "rowCount": profile["rowCount"], "columns": columns}
