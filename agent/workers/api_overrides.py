"""Pass approved API/config options into existing functions; never implement training here."""
import functools
import importlib
import inspect
from pathlib import Path
import subprocess
import sys

from .model_contract import MODEL_CLASSES, cli_parameters


def scoped(params, prefix):
    return {key[len(prefix) + 1:]: value for key, value in params.items() if key.startswith(prefix + '.')}


def override(function, values):
    signature = inspect.signature(function)
    variadic = next((name for name, item in signature.parameters.items() if item.kind == item.VAR_KEYWORD), None)
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        for name, value in values.items():
            if name in signature.parameters: bound.arguments[name] = value
            elif variadic: bound.arguments.setdefault(variadic, {})[name] = value
            else: raise ValueError(f'参数没有真实接收端：{name}')
        return function(*bound.args, **bound.kwargs)
    return wrapped


def install(plan, script, root):
    params = plan['params']; model = plan['modelId']
    if 'text.stopwords' in params:
        from utils.stopword_manager import configure_stopwords
        configure_stopwords(params['text.stopwords'])
    if script.name == 'prepare_data.py':
        if model == 'etm' and scoped(params, 'word2vec'):
            from gensim.models import Word2Vec
            values = scoped(params, 'word2vec')
            Word2Vec.__init__ = override(Word2Vec.__init__, values)
        if model in {'ctm', 'bertopic', 'dtm'} and 'prepare.max_length' in params:
            from sentence_transformers import SentenceTransformer
            original_init = SentenceTransformer.__init__
            @functools.wraps(original_init)
            def initialize(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                instance.max_seq_length = params['prepare.max_length']
            SentenceTransformer.__init__ = initialize
        return
    if script.name == 'run_pipeline.py' and model != 'theta':
        trainer = importlib.import_module('model.baseline_trainer')
        method = '_train_neural_topic_model' if model in {'nvdm', 'gsm', 'prodlda'} else 'train_' + model
        values = scoped(params, 'trainer')
        if values: setattr(trainer.BaselineTrainer, method, override(getattr(trainer.BaselineTrainer, method), values))
        # CUDA_VISIBLE_DEVICES maps the requested physical GPU to local cuda:0.
        trainer.BaselineTrainer.__init__ = override(trainer.BaselineTrainer.__init__, {'device': 'cuda:0' if plan.get('device', 'cpu').startswith('cuda:') else 'cpu'})
        values = scoped(params, 'word2vec')
        if values: trainer.train_word2vec_embeddings = override(trainer.train_word2vec_embeddings, values)
    if script.name == 'main.py':
        config = importlib.import_module('config')
        original = config.config_from_args
        @functools.wraps(original)
        def configure(args):
            result = original(args)
            if 'pipeline.seed' in params: result.seed = params['pipeline.seed']
            for name, value in scoped(params, 'config').items(): setattr(result.model, name, value)
            # Preserve explicitly approved zero values despite the legacy parser's `or default` resolution.
            for name in ['kl_start', 'kl_end']:
                if name in params: setattr(result.model, name, params[name])
            if 'kl_warmup' in params: result.model.kl_warmup_epochs = params['kl_warmup']
            if 'patience' in params: result.model.patience = params['patience']
            for name in ['lora_dropout']:
                if 'main.' + name in params: setattr(result.model, name, params['main.' + name])
            if 'config.test_ratio' in params:
                result.model.val_ratio = 1 - result.model.train_ratio - params['config.test_ratio']
            return result
        config.config_from_args = configure
    # Patch the original class, preserving identity, methods, exports and numerical implementation.
    if script.name in {'run_pipeline.py', 'main.py'} and (model != 'theta' or script.name == 'main.py'):
        file, cls_name = MODEL_CLASSES[model]
        module = importlib.import_module('model.' + (file.replace('/', '.') if model == 'theta' else 'baseline.' + file))
        cls = getattr(module, cls_name)
        values = scoped(params, 'model')
        if model == 'theta' and 'config.encoder_activation' in params: values['encoder_activation'] = params['config.encoder_activation']
        if values: cls.__init__ = override(cls.__init__, values)
        freeze_key = 'model.train_word_embeddings' if model == 'dtm' else 'model.train_embeddings' if model == 'etm' else None
        if model == 'etm' and freeze_key not in params and 'trainer.train_embeddings' in params: freeze_key = 'trainer.train_embeddings'
        if freeze_key and freeze_key in params:
            original_init = cls.__init__
            @functools.wraps(original_init)
            def initialize(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                (instance.decoder.word_embeddings if model == 'dtm' else instance.rho).requires_grad_(params[freeze_key])
            cls.__init__ = initialize
        fit = scoped(params, 'fit')
        if fit: cls.fit = override(cls.fit, fit)
    if script.name == 'run_pipeline.py' and model == 'theta':
        from .pipeline_adapter import append_arguments
        original_run = subprocess.run
        @functools.wraps(original_run)
        def run(command, *args, **kwargs):
            if isinstance(command, list) and len(command) >= 2 and Path(command[1]).name == 'main.py':
                command = list(command)
                main = scoped(params, 'main')
                if main.get('train_word_embeddings') is False:
                    main.pop('train_word_embeddings'); main['no_train_word_embeddings'] = True
                append_arguments(command, main, cli_parameters(root, 'config.py'))
                if plan.get('labelColumn'): command.extend(['--label_col', 'label'])
                if plan.get('timeColumn'): command.extend(['--timestamp_column', 'timestamp'])
                command = [command[0], str(Path(__file__).with_name('engine_entry.py')), str(root / 'src/models/main.py'), *command[2:]]
            return original_run(command, *args, **kwargs)
        subprocess.run = run
