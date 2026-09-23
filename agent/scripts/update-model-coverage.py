"""Audit every published entry/API parameter against the Agent's executable contract."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers.capabilities import engine_root
from workers.model_contract import contract, cli_parameters, api_parameters, config_parameters, MODEL_CLASSES, MANAGED_CLI, DERIVED

root = engine_root()
contracts = {model: contract(root, model) for model in MODEL_CLASSES}
rows = []

def add(source, name, model=None, candidates=(), reason=None):
    targets = [(m, key) for m, entries in contracts.items() if not model or m == model for key in candidates if key in entries]
    if targets: state, detail = '可配置', '; '.join(f'{m}: params.{key}' for m, key in targets)
    elif reason: state, detail = reason
    elif name in MANAGED_CLI: state, detail = '宿主管理', MANAGED_CLI[name]
    elif name in DERIVED or name in {'time_slices', 'word_embedding_dim', 'data_dir', 'data_exp_dir', 'result_dir', 'output_dir', 'workspace_dir', 'user_id', 'input', 'output', 'model_path', 'embedding_dim', 'num_topics', 'vocab_size'}: state, detail = '数据/任务绑定', '由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定'
    else: state, detail = '待处理', ''
    rows.append({'source': source, 'parameter': name, 'state': state, 'mapping': detail})

for file in ['run_pipeline.py', 'prepare_data.py', 'config.py', 'main.py']:
    for name in cli_parameters(root, file):
        special = ('原实现无效', '声明但无消费端，不伪装为可生效参数') if name == 'no_wordcloud' else None
        add(file, name, candidates=(name, 'prepare.' + name, 'main.' + name), reason=special)
for name in config_parameters(root):
    add('ModelConfig', name, 'theta', (name, 'config.' + name), ('映射', 'params.kl_warmup') if name == 'kl_warmup_epochs' else None)
for name in config_parameters(root, 'EmbeddingConfig'):
    aliases = {'mode': 'mode', 'model_path': 'embedding.model_path', 'batch_size': 'prepare.batch_size', 'max_length': 'prepare.max_length',
               'provider': 'embedding_provider', 'cloud_provider': 'embedding_cloud_provider', 'model': 'embedding_model', 'api_base': 'embedding_api_base',
               'api_key_env': 'embedding_api_key_env', 'dimensions': 'embedding_dimensions', 'normalize': 'embedding.normalize'}
    add('EmbeddingConfig', name, 'theta', (aliases.get(name, ''),))
add('PipelineConfig', 'seed', 'theta', ('pipeline.seed',))
for model, (file, cls) in MODEL_CLASSES.items():
    file = 'model/' + (file if model == 'theta' else 'baseline/' + file) + '.py'
    for method in ['__init__', 'fit']:
        for name in api_parameters(root, file, cls, method):
            if name == 'self': continue
            reason = None
            if name == 'T' and model == 'hdp': reason = ('原实现覆盖', '使用 params.max_topics；原 T 被 max_topics 覆盖')
            if name == 'train_word_embeddings' and model in {'lda', 'ctm'}: reason = ('兼容占位', '本模型不消费词嵌入冻结选项')
            if name == 'kl_weight' and model == 'etm': reason = ('原训练器覆盖', 'ETM 训练器自算 KL 退火，构造参数不控制训练器目标；不接收无效覆盖')
            if name == 'embedding_model': reason = ('映射', 'params.embedding.model_path；使用批准的本地权重，预处理与训练保持一致')
            add(f'{model}.{method}', name, model, (name, 'model.' + name, 'fit.' + name, 'config.' + name), reason)
for model in MODEL_CLASSES:
    if model == 'theta': continue
    for name in api_parameters(root, 'model/baseline_trainer.py', 'BaselineTrainer', 'train_' + model):
        if name == 'self': continue
        add('BaselineTrainer.train_' + model, name, model, (name, 'trainer.' + name), ('映射', 'params.patience') if name == 'early_stopping_patience' else None)
for method in ['__init__', '_train_neural_topic_model', 'train_all']:
    for name in api_parameters(root, 'model/baseline_trainer.py', 'BaselineTrainer', method):
        if name == 'self': continue
        reason = ('宿主管理', 'plan.device / 按独立模型研究及确认执行') if name in {'device', 'models'} else None
        add('BaselineTrainer.' + method, name, candidates=(name,), reason=reason)
for name in api_parameters(root, 'model/baseline/etm.py', None, 'train_word2vec_embeddings'):
    add('train_word2vec_embeddings', name, 'etm', (name, 'word2vec.' + name))

unresolved = [row for row in rows if row['state'] == '待处理']
if unresolved:
    print(json.dumps(unresolved, ensure_ascii=False, indent=2)); sys.exit(1)
out = root / 'agent/knowledge/documents/execution-coverage.md'
lines = ['# Agent 模型与参数执行覆盖核对', '', '核对日期：2026-09-05。由当前源码签名和 Agent 参数路由生成；不代表已对全部模型完成真实训练。', '',
         '12 个模型均已注册。params 内不带前缀的是统一入口选项；model./trainer./fit./prepare./main./config./pipeline./word2vec./embedding. 分别连接明确接收端。', '',
         '所有执行仍通过独立宿主确认。本地适配器支持扩展参数；旧 Go worker 未登记扩展协议时会明确拒绝，不丢弃参数。', '',
         '数据/任务绑定字段由宿主提供；原实现无效或覆盖的字段明确列出，不宣称可以调整。DTM 的 time_slices 只能与既有分箱相符，不做隐式重分桶；分布式 rank/world_size 属于启动器，当前本地启动器为单进程 CPU 或单张 CUDA。', '',
         '统一 CLI 未暴露不等于 Agent 不可调：例如 NVDM/GSM 的 model.dropout、ProdLDA 的 model.variance、ETM 的 model.train_embeddings 已通过原类 API 接入。使用下方对应入口章节核实；请勿把旧清单中的统一 CLI 限制当作当前 Agent 限制。', '']
from itertools import groupby
for section, items in groupby(rows, key=lambda row: row['source']):
    lines += [f'## {section}', '', '| 来源/入口 | 参数 | 状态 | 实际传递/原因 |', '|---|---|---|---|']
    lines += [f"| {row['source']} | {row['parameter']} | {row['state']} | {row['mapping']} |" for row in items]
    lines.append('')
content = '\n'.join(lines)+'\n'
if '--check' in sys.argv:
    if not out.is_file() or out.read_text() != content: raise SystemExit('执行覆盖文档已过期，请运行 python3 agent/scripts/update-model-coverage.py')
else: out.write_text(content)
print(json.dumps({'rows':len(rows),'models':len(contracts),'parameters':{m:len(c) for m,c in contracts.items()},'output':str(out)},ensure_ascii=False))
