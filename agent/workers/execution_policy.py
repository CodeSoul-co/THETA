"""Resolve host configuration without sending requests; bind it to one approved task."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
from contextlib import closing
import sys
import time
from urllib.parse import urlsplit
from . import runtime_environments


def embedding_settings(params=None):
    params = params or {}
    from .capabilities import engine_root
    name = '_theta_agent_embedding_settings'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, engine_root() / 'src/models/model/embedding_providers.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    settings = sys.modules[name].resolve_embedding_settings(provider=params.get('embedding_provider', 'cloud'), **{name: params['embedding_' + name] for name in ['cloud_provider', 'model', 'api_base', 'api_key_env', 'dimensions'] if 'embedding_' + name in params})
    if 'embedding.normalize' in params: settings.normalize = params['embedding.normalize']
    settings.api_base = settings.api_base.rstrip('/').removesuffix('/embeddings')
    return settings


def execution_policy(plan: dict) -> dict:
    selected = str(plan['params'].get('embedding_provider', 'local'))
    device = plan.get('device', 'cpu')
    if device != 'cpu' and not re.fullmatch(r'cuda:[0-9]+', device): raise ValueError('device 必须为 cpu 或 cuda:非负编号')
    assets = {}
    if plan['params'].get('embedding.model_path'):
        key = ('QWEN_MODEL_' + str(plan['params'].get('model_size', '0.6B')).replace('.', '_')) if plan['modelId'] == 'theta' else 'SBERT_MODEL_PATH'
        assets[key] = str(Path(plan['params']['embedding.model_path']).expanduser().resolve(strict=True))
        if not (Path(assets[key]) / 'config.json').is_file(): raise ValueError('本地模型路径缺少 config.json')
    base = {'timeoutSeconds': plan['timeoutSeconds'], 'device': device, 'modelAssets': assets,
            'runtime': runtime_environments.identity(runtime_environments.model_profile(plan['modelId']))}
    if selected in {'local', 'qwen'}:
        return {**base, 'embedding': {'mode': 'local'}, 'maxExternalRequests': 0}
    if selected not in {'cloud', 'openai', 'dashscope', 'siliconflow', 'zhipu', 'volcengine', 'openai_compatible'} or plan['modelId'] != 'theta' or plan['params'].get('mode', 'zero_shot') != 'zero_shot':
        raise ValueError('云 embedding 仅支持 THETA zero_shot；微调和其他模型请使用本地模型')
    settings = embedding_settings(plan['params'])
    target = urlsplit(settings.api_base)
    if not (target.scheme == 'https' or target.scheme == 'http' and target.hostname in {'localhost', '127.0.0.1', '::1'}):
        raise ValueError('Embedding 端点需要 HTTPS；仅本地测试允许 HTTP')
    if target.username or target.password or target.query or target.fragment:
        raise ValueError('Embedding 地址不得包含凭据或查询参数')
    key_env = 'EMBEDDING_API_KEY' if os.environ.get('EMBEDDING_API_KEY') else settings.api_key_env
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key_env):
        raise ValueError('EMBEDDING_API_KEY_ENV 必须填写环境变量名，密钥内容应放在 EMBEDDING_API_KEY 或对应的密钥变量中')
    if not settings.api_key or not settings.model:
        raise ValueError('云 embedding 的模型、地址或密钥尚未完整配置')
    limit = plan.get('externalRequestLimit', 100)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError('外部请求上限必须为 1–1000 的整数')
    return {**base, 'embedding': {'mode': 'cloud', 'provider': settings.cloud_provider,
            'endpoint': settings.api_base.rstrip('/') + '/embeddings', 'model': settings.model,
            'keyEnv': key_env, 'credentialFingerprint': hashlib.sha256(settings.api_key.encode()).hexdigest(),
            'dimensions': settings.dimensions, 'normalize': settings.normalize},
            'maxExternalRequests': limit, 'timeoutSeconds': plan['timeoutSeconds']}


def configuration_summary(_: dict) -> dict:
    # Never return credentials or their fingerprint to the conversation.
    settings = embedding_settings()
    url = urlsplit(settings.api_base)
    public_endpoint = f'{url.scheme}://{url.hostname or ""}{":" + str(url.port) if url.port else ""}{url.path}'
    valid_key_name = bool(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', settings.api_key_env))
    return {'embedding': {'preferredMode': os.environ.get('EMBEDDING_PROVIDER', 'local'),
            'provider': settings.cloud_provider, 'endpoint': public_endpoint, 'model': settings.model,
            'configured': bool(settings.api_key and settings.model and settings.api_base),
            'configurationIssues': (["EMBEDDING_API_KEY_ENV 必须是环境变量名，不能填密钥内容"] if not valid_key_name else
                                    ["密钥变量未设置或为空"] if not settings.api_key else []),
            'requiresConfirmation': True, 'supportedCloudMode': 'THETA zero_shot'},
            'agentInferenceRequiresConfirmation': False,
            'workerEnvironments': runtime_environments.catalog({}),
            'policy': '配置表示能力可用，不代表授权。模型下载和未登记外部服务不会自动调用。'}


def preview(payload: dict) -> dict:
    from .capabilities import validate_plan, runtime_check
    validation = validate_plan(payload)
    policy = execution_policy(payload['plan'])
    ready = runtime_check({'modelId': payload['plan']['modelId'], 'modelSize': payload['plan']['params'].get('model_size'),
                           'embeddingProvider': policy['embedding']['mode'], 'mode': payload['plan']['params'].get('mode'), 'device': payload['plan'].get('device'), 'modelAssets': policy.get('modelAssets', {})})
    return {'execution': policy, 'readiness': ready, 'rowCount': validation['rowCount']}


def assert_authorized(payload: dict) -> dict:
    receipt = payload.get('authorization') or {}
    database = Path(payload['home']).resolve() / 'research.sqlite'
    if not database.is_file() or not receipt.get('id'):
        raise ValueError('计算启动缺少宿主用户确认凭据')
    with closing(sqlite3.connect(database)) as db, db:
        row = db.execute('SELECT value FROM records WHERE kind=? AND id=?', ('effect-approval', receipt['id'])).fetchone()
    if not row:
        raise ValueError('授权记录不存在')
    approval = json.loads(row[0])
    expected = {key: value for key, value in payload.items() if key not in {'authorization', 'home'}}
    if (approval['status'] != 'approved' or approval['hash'] != receipt.get('hash')
            or approval['action'] != 'compute.submit' or approval['target'] != 'local'
            or approval['payload'] != expected or time.time() * 1000 > approval['expiresAt']):
        raise ValueError('计算操作未批准、已过期或内容发生变化')
    if expected.get('execution') != execution_policy(expected['plan']):
        raise ValueError('运行环境、外部服务配置或密钥已变化，请重新确认')
    return approval


def training_environment(policy: dict) -> dict:
    environment = dict(os.environ)
    embedding = policy['embedding']
    credential = os.environ.get(embedding.get('keyEnv', ''), '')
    # Ambient main .env must never enable cloud work accidentally.
    for key in list(environment):
        upper = key.upper()
        if upper.startswith(('EMBEDDING_', 'THETA_INFERENCE_', 'REDIS_', 'OBJECT_STORAGE_', 'AWS_')) or any(token in upper for token in ('API_KEY', 'SECRET', 'TOKEN', 'PASSWORD')):
            environment.pop(key, None)
    environment.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '',
                        'EMBEDDING_PROVIDER': 'local'})
    device = policy.get('device', 'cpu')
    environment['CUDA_VISIBLE_DEVICES'] = device.split(':')[1] if device.startswith('cuda:') else ''
    environment.update(policy.get('modelAssets', {}))
    if embedding['mode'] == 'cloud':
        environment.update({'EMBEDDING_PROVIDER': 'cloud', 'EMBEDDING_CLOUD_PROVIDER': embedding['provider'],
                            'EMBEDDING_API_BASE': embedding['endpoint'].removesuffix('/embeddings'),
                            'EMBEDDING_MODEL': embedding['model'], 'EMBEDDING_API_KEY': credential,
                            'EMBEDDING_API_KEY_ENV': 'EMBEDDING_API_KEY', 'EMBEDDING_MAX_RETRIES': '2',
                            'EMBEDDING_NORMALIZE': str(embedding['normalize']).lower()})
        if embedding.get('dimensions'):
            environment['EMBEDDING_DIMENSIONS'] = str(embedding['dimensions'])
    return environment
