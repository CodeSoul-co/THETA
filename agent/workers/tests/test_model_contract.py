"""Parameter routing checks reuse source signatures and don't run numerical training."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace, ModuleType
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from workers.capabilities import engine_root, model_inspect, MODELS, validate_plan, dataset_import
from workers.model_contract import contract, validate_parameters, MODEL_CLASSES, cli_parameters, api_parameters
from workers.api_overrides import override, install
from workers.local_compute import normalize_dataset
sys.path.insert(0, str(engine_root() / 'trainning'))
from worker.config import WorkerConfig
from worker.protocol import ExecutionSpec
from worker.pipeline import JobPaths
from workers.pipeline_adapter import AgentThetaPipeline


class ModelContractTests(unittest.TestCase):
    def test_all_twelve_models_and_model_specific_routes(self):
        self.assertEqual(len(MODELS), 12)
        for model in MODELS:
            info = model_inspect({'modelId': model})
            self.assertTrue(info['parameters'])
            validate_parameters(engine_root(), {'modelId': model, 'params': {}, 'timeoutSeconds': 30})
        self.assertIn('model.alpha', contract(engine_root(), 'lda'))
        self.assertNotIn('epochs', contract(engine_root(), 'lda'))
        self.assertNotIn('alpha', contract(engine_root(), 'lda'))
        self.assertNotIn('num_topics', contract(engine_root(), 'hdp'))
        for model, params in [('bertopic', {'num_topics': 2}), ('ctm', {'hidden_sizes': [64, 32]}),
                              ('lda', {'model.alpha': .2, 'model.eta': None, 'trainer.learning_method': 'online'}),
                              ('theta', {'hidden_sizes': [48, 24], 'dropout': .1, 'kl_warmup': 0, 'patience': 0, 'config.stage1_lr': .0003, 'main.lora_dropout': 0., 'config.encoder_activation': 'relu'}),
                              ('btm', {'model.window_size': 9, 'fit.verbose': False}), ('prodlda', {'model.variance': .8})]:
            with self.subTest(model=model): validate_parameters(engine_root(), {'modelId': model, 'params': params, 'timeoutSeconds': 30})
        for model, params in [('lda', {'epochs': 3}), ('lda', {'num_topics': 1}), ('lda', {'num_topics': 101}), ('lda', {'alpha': 1}),
                              ('etm', {'dropout': 1}), ('ctm', {'hidden_sizes': [64, 'bad']}), ('gsm', {'hidden_sizes': [64] * 11}), ('nvdm', {'hidden_sizes': [2]}),
                              ('theta', {'embedding_provider': 'cloud', 'mode': 'supervised'}), ('theta', {'embedding_provider': 'openai_compatible', 'embedding_api_base': 'http://remote.invalid/v1'}),
                              ('btm', {'model.window_size': True}), ('lda', {'max_iter': float('nan')})]:
            with self.subTest(params=params), self.assertRaises(ValueError): validate_parameters(engine_root(), {'modelId': model, 'params': params, 'timeoutSeconds': 30})

    def test_parameters_reach_original_argument_slots_including_positional_and_variadic_calls(self):
        def native(self, max_iter=100, learning_method='batch'): return max_iter, learning_method
        wrapped = override(native, {'max_iter': 3, 'learning_method': 'online'})
        self.assertEqual(wrapped(None, 99, learning_method='batch'), (3, 'online'))
        def neural(self, epochs=100, **kwargs): return epochs, kwargs
        self.assertEqual(override(neural, {'learning_rate': .02})(None, 4), (4, {'learning_rate': .02}))
        with self.assertRaises(ValueError): override(native, {'invented': 1})(None)

    def test_pipeline_commands_all_models_keep_scopes_and_bind_columns(self):
        with tempfile.TemporaryDirectory() as home:
            config = replace(WorkerConfig.load(), project_root=engine_root(), job_root=Path(home), resource_class='cpu', gpu_id=None)
            pipeline = AgentThetaPipeline(config, None, None)
            for model in MODELS:
                pipeline.plan = {'modelId': model, 'covariates': ['channel'] if model == 'stm' else [], 'timeColumn': 'year' if model == 'dtm' else None, 'labelColumn': 'label' if model == 'theta' else None}
                params = {'language': 'chinese', 'prepare.vocab_size': 321}
                if model == 'lda': params['model.alpha'] = .2
                if model in {'etm', 'ctm', 'dtm', 'nvdm', 'gsm', 'prodlda', 'theta'}: params['hidden_sizes'] = [48, 24]
                if model == 'etm': params['embedding_dim'] = 48
                if model == 'theta': params.update({'main.stage1_epochs': 3, 'embedding_provider': 'local'})
                spec = SimpleNamespace(model=SimpleNamespace(name=model), params=params, task_id=1, attempt=1, user_id='local', dataset=SimpleNamespace())
                # Real immutable spec shape; replace() in the adapter must accept its params field.
                from dataclasses import make_dataclass
                Spec = make_dataclass('Spec', [(key, object) for key in vars(spec)])
                spec = Spec(**vars(spec))
                paths = SimpleNamespace(workspace=Path(home)/'workspace', dataset_file=Path(home)/'data.csv')
                prepare = pipeline.build_prepare_command(spec, paths)
                train = pipeline.build_train_command(spec, paths, Path(home)/'prepared')
                self.assertEqual(prepare[prepare.index('--vocab_size')+1], '321')
                self.assertEqual(train[train.index('--models')+1], model)
                self.assertFalse(any('model.alpha' in arg or 'stage1_epochs' in arg for arg in train))
                if model == 'etm': self.assertIn('--bow-only', prepare); self.assertEqual(train[train.index('--embedding_dim')+1], '48')
                if 'hidden_sizes' in params: self.assertEqual(train[train.index('--hidden_sizes')+1:train.index('--hidden_sizes')+3], ['48', '24'])
                if model == 'stm': self.assertIn('--covariate_columns', prepare)
                if model == 'theta': self.assertIn('--label_col', prepare); self.assertEqual(train[train.index('--language')+1], 'zh')
            alias_spec = Spec(**{**vars(spec), 'params': {'language': 'en'}})
            prepare = pipeline.build_prepare_command(alias_spec, paths)
            self.assertNotIn('--language', prepare)

    def test_supervised_label_mapping_and_missing_labels_fail_before_submission(self):
        with tempfile.TemporaryDirectory() as home:
            file = Path(home)/'data.csv'; file.write_text('body,target,year\nfirst text,A,2020\nsecond text,B,2021\n')
            dataset = dataset_import({'filePath': str(file), 'uploadDir': str(Path(home)/'uploads')})
            plan = {'modelId': 'theta', 'textColumn': 'body', 'labelColumn': 'target', 'params': {'mode': 'supervised'}, 'timeoutSeconds': 30}
            validate_plan({'plan': plan, 'dataset': dataset})
            output = Path(home)/'normalized.csv'; normalize_dataset({'plan': plan, 'dataset': dataset}, output)
            self.assertIn('text,label', output.read_text()); self.assertIn('first text,A', output.read_text())
            del plan['labelColumn']
            with self.assertRaisesRegex(ValueError, 'labelColumn'): validate_plan({'plan': plan, 'dataset': dataset})

    def test_approved_namespaces_reach_original_model_and_trainer_without_training(self):
        class NativeModel:
            def __init__(self, window_size=15): self.window = window_size
            def fit(self, bow, verbose=True): return bow, verbose
        class NativeTrainer:
            def __init__(self, device='auto'): self.device = device
            def train_btm(self, n_iter=100): return n_iter
        modules = {'model.baseline_trainer': SimpleNamespace(BaselineTrainer=NativeTrainer),
                   'model.baseline.btm': SimpleNamespace(BTM=NativeModel)}
        with patch('workers.api_overrides.importlib.import_module', side_effect=modules.__getitem__):
            install({'modelId': 'btm', 'device': 'cuda:2', 'params': {'model.window_size': 7, 'fit.verbose': False}}, Path('run_pipeline.py'), engine_root())
        self.assertEqual(NativeModel(99).window, 7)
        self.assertEqual(NativeModel().fit('fixture', True), ('fixture', False))
        self.assertEqual(NativeTrainer('auto').device, 'cuda:0')

    def test_theta_config_zero_values_and_explicit_test_split_reach_native_configuration(self):
        config = SimpleNamespace(config_from_args=lambda args: SimpleNamespace(seed=42, model=SimpleNamespace(train_ratio=.8, val_ratio=.1, test_ratio=.1)))
        class NativeModel:
            def __init__(self, encoder_activation='softplus'): self.activation = encoder_activation
        modules = {'config': config, 'model.theta.etm': SimpleNamespace(ETM=NativeModel)}
        with patch('workers.api_overrides.importlib.import_module', side_effect=modules.__getitem__):
            install({'modelId': 'theta', 'params': {'pipeline.seed': 123, 'config.test_ratio': .15, 'config.encoder_activation': 'relu', 'kl_start': 0., 'kl_warmup': 0, 'patience': 0, 'main.lora_dropout': 0.}}, Path('main.py'), engine_root())
        value = config.config_from_args(None)
        self.assertEqual(value.seed, 123); self.assertEqual(value.model.kl_start, 0.); self.assertEqual(value.model.lora_dropout, 0.)
        self.assertEqual(value.model.kl_warmup_epochs, 0); self.assertEqual(value.model.patience, 0)
        self.assertAlmostEqual(value.model.val_ratio, .05); self.assertEqual(NativeModel().activation, 'relu')

    def test_four_new_native_classes_receive_options_without_fitting(self):
        from contextlib import ExitStack
        import importlib
        sys.path.insert(0, str(engine_root() / 'src/models'))
        trainer = importlib.import_module('model.baseline_trainer')
        for model in ['etm', 'nvdm', 'gsm', 'prodlda']:
            module = importlib.import_module('model.baseline.' + model)
            cls = getattr(module, MODEL_CLASSES[model][1])
            params = {'model.train_embeddings': False, 'trainer.use_pretrained_embeddings': False} if model == 'etm' else {'model.dropout': .31}
            if model == 'prodlda': params['model.variance'] = .8
            method = 'train_etm' if model == 'etm' else '_train_neural_topic_model'
            with self.subTest(model=model), ExitStack() as stack:
                # Restore every installed hook; only construct tiny native models, never fit.
                stack.enter_context(patch.object(cls, '__init__', cls.__init__))
                stack.enter_context(patch.object(trainer.BaselineTrainer, '__init__', trainer.BaselineTrainer.__init__))
                stack.enter_context(patch.object(trainer.BaselineTrainer, method, getattr(trainer.BaselineTrainer, method)))
                install({'modelId': model, 'params': params}, Path('run_pipeline.py'), engine_root())
                native = cls(vocab_size=8, num_topics=2, hidden_dim=4, hidden_sizes=[6, 4], **({'embedding_dim': 3} if model == 'etm' else {}))
                self.assertEqual(native.hidden_sizes, [6, 4])
                if model == 'etm': self.assertFalse(native.rho.requires_grad); self.assertEqual(list(native.rho.shape), [8, 3])
                else: self.assertEqual(native.dropout_rate, .31)
                if model == 'prodlda': self.assertAlmostEqual(native.prior_var[0, 0].item(), .8)
