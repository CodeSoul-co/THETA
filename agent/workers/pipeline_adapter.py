"""Extend only argument transport; reuse the original worker's lifecycle and artifact handling."""
from dataclasses import replace
from worker.pipeline import ThetaPipeline, PARAM_FLAGS
from .model_contract import cli_parameters


def append_arguments(command, values, definitions):
    for name, value in values.items():
        info = definitions[name]
        flag = info['flag']
        # An explicit false can disable a flag added by the shared pipeline.
        while flag in command:
            index = command.index(flag)
            del command[index:index + (1 if info.get('action') in {'store_true', 'store_false'} else 2)]
        if info.get('action') in {'store_true', 'store_false'}:
            if value == (info['action'] == 'store_true'): command.append(flag)
        elif isinstance(value, (list, tuple)):
            command.append(flag)
            command.extend(str(item) for item in value)
        elif value is not None:
            command.extend([flag, str(value)])
    return command


class AgentThetaPipeline(ThetaPipeline):
    def build_prepare_command(self, spec, paths):
        prepare = cli_parameters(self.config.project_root, 'prepare_data.py')
        common = {key: value for key, value in spec.params.items() if key in PARAM_FLAGS and key != 'language'}
        command = super().build_prepare_command(replace(spec, params=common), paths)
        extra = {key.removeprefix('prepare.'): value for key, value in spec.params.items() if key.startswith('prepare.')}
        extra.update({key: value for key, value in spec.params.items() if key.startswith('embedding_') and key in prepare})
        if spec.model.name == 'dtm' and extra.get('skip_sbert'): extra['bow_only'] = True
        # The shared prepare helper allocates a fixed 300-column matrix. Other dimensions
        # must use the existing trainer's dimension-aware Word2Vec helper instead.
        if spec.model.name == 'etm' and (spec.params.get('trainer.use_pretrained_embeddings') is False or spec.params.get('embedding_dim', 300) != 300): extra['bow_only'] = True
        append_arguments(command, extra, prepare)
        if self.plan.get('timeColumn'): command.extend(['--time_column', 'year'])
        if self.plan.get('labelColumn'): command.extend(['--label_col', 'label'])
        if self.plan.get('covariates'): command.extend(['--covariate_columns', *[f'cov_{i}' for i in range(len(self.plan['covariates']))]])
        if extra.get('clean'): command.extend(['--raw-input', str(paths.dataset_file)])
        if self.config.gpu_id is not None: command.extend(['--gpu', str(self.config.gpu_id)])
        return command

    def build_train_command(self, spec, paths, data_exp):
        common = {key: value for key, value in spec.params.items() if key in PARAM_FLAGS}
        if spec.model.name == 'theta' and 'language' in common:
            common['language'] = {'chinese': 'zh', 'english': 'en'}.get(common['language'], common['language'])
        command = super().build_train_command(replace(spec, params=common), paths, data_exp)
        definitions = cli_parameters(self.config.project_root, 'run_pipeline.py')
        extra = {key: value for key, value in spec.params.items() if key in definitions and key not in PARAM_FLAGS}
        return append_arguments(command, extra, definitions)
