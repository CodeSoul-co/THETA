import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/models'))
from data.dataset_split import assignments, current, export_results

class DatasetSplitTests(unittest.TestCase):
    def test_default_test_reuses_all_rows(self):
        spec = assignments(10)
        self.assertEqual([len(spec['groups'][k]) for k in ('train', 'validation', 'test')], [7, 3, 10])
        self.assertTrue(spec['testUsesAllData'])
        self.assertFalse(set(spec['groups']['train']) & set(spec['groups']['validation']))

    def test_ordered_random_and_independent_uploads(self):
        options = {'enabled': True, 'ratios': [.6, .2, .2], 'method': 'sequential'}
        spec = assignments(10, options)
        self.assertEqual(spec['groups'], {'train': list(range(6)), 'validation': [6,7], 'test': [8,9]})
        random = {**options, 'method': 'random', 'seed': 123}
        self.assertEqual(assignments(100, random), assignments(100, random))
        self.assertNotEqual(assignments(100, random), assignments(100, {**random, 'seed': 42}))
        external = assignments(5, {'enabled': True, 'mode': 'upload'}, ['test','train','validation','train','train'])
        self.assertEqual(external['groups']['train'], [1,3,4])
        for bad in [[.8,.2,.1], [1,0,0], [float('nan'),.2,.1]]:
            with self.assertRaises(ValueError): assignments(100, {**options, 'ratios': bad})
        with self.assertRaises(ValueError): assignments(2)

    def test_preprocessing_mapping_and_real_lda_training(self):
        import numpy as np
        from prepare_data import generate_bow
        from model.baseline_trainer import BaselineTrainer
        from model.baseline.lda import SklearnLDA
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec = assignments(10, {'enabled': True, 'method': 'sequential', 'ratios': [.6,.2,.2]})
            manifest = root / 'split.json'; manifest.write_text(json.dumps(spec))
            texts = ['apple fruit orchard garden', 'robot machine code computer']*3 + ['heldout exclusiveword fruit']*4
            with patch.dict(os.environ, {'THETA_DATA_SPLIT_FILE': str(manifest)}):
                bow, vocab = generate_bow(texts, 100, root)
                self.assertNotIn('exclusiveword', vocab)
                trainer = BaselineTrainer(dataset='test', num_topics=2, workspace_dir=str(root), output_dir=str(root/'outputs'), device='cpu')
                trainer.bow_matrix = bow.toarray(); trainer.vocab = vocab; trainer.vocab_size = len(vocab)
                fitted = []
                original = SklearnLDA.fit
                def fit(model, values, *args, **kwargs):
                    fitted.append(values.copy()); return original(model, values, *args, **kwargs)
                with patch.object(SklearnLDA, 'fit', fit): result = trainer.train_lda(max_iter=2)
                np.testing.assert_array_equal(fitted[0], bow.toarray()[:6])
                self.assertEqual(result['theta'].shape, (10,2))
                export_results(root/'report', result['theta'], result['beta'], bow, vocab, 'lda')
                report = json.loads((root/'report/split_results.json').read_text())
                self.assertEqual([report['groups'][k]['count'] for k in ('all','train','validation','test')], [10,6,2,2])
                self.assertEqual(np.load(root/'report/splits/test/theta.npy').shape, (2,2))
                self.assertEqual(current(9, list(range(9)))['groups']['test'], [8])
                with self.assertRaises(ValueError): current(8, list(range(8)))

    def test_mixed_uploads_are_normalized_with_exact_roles(self):
        from workers.capabilities import dataset_import
        from workers.local_compute import normalize_dataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = {'train': ('train.csv', 'body\napple fruit\nrobot code\ngarden fruit\n'), 'validation': ('val.txt', 'validation content'), 'test': ('test.json', '[{"message":"test content"}]')}
            sources = {}
            for role, (name, text) in files.items():
                path = root/name; path.write_text(text)
                dataset = dataset_import({'filePath': str(path), 'uploadDir': str(root/'uploads')})
                sources[role] = {'dataset': dataset, 'textColumn': {'train':'body','validation':'text','test':'message'}[role]}
            payload = {'dataset': sources['train']['dataset'], 'plan': {'textColumn':'body', 'dataSplit': {'enabled':True, 'mode':'upload','sources':sources}}}
            target = root/'data.csv'; normalize_dataset(payload, target)
            spec = json.loads(target.with_suffix('.split.json').read_text())
            self.assertEqual(spec['groups'], {'train':[0,1,2], 'validation':[3], 'test':[4]})
            self.assertIn('source_file', target.read_text())

    def test_neural_training_loaders_exclude_held_out_rows(self):
        import numpy as np
        import torch
        from model.baseline_trainer import BaselineTrainer
        from data.dataloader import create_dataloader as original_loader
        def loader(*args, **kwargs):
            kwargs.update(num_workers=0, persistent_workers=False, pin_memory=False)
            kwargs.pop('prefetch_factor', None)
            return original_loader(*args, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root/'split.json'
            manifest.write_text(json.dumps(assignments(20, {'enabled': True, 'ratios': [.6,.2,.2], 'method':'sequential'})))
            (root/'time_slices.json').write_text(json.dumps({'num_time_slices': 2, 'unique_times':[2020,2021]}))
            np.save(root/'time_indices.npy', np.arange(20)%2)
            bow = np.random.default_rng(42).integers(1,8,(20,12)).astype('float32')
            with patch.dict(os.environ, {'THETA_DATA_SPLIT_FILE': str(manifest)}), patch('model.baseline_trainer.create_dataloader', loader):
                for name in ['ctm', 'etm', 'nvdm', 'gsm', 'prodlda', 'dtm']:
                    with self.subTest(model=name):
                        trainer = BaselineTrainer(dataset='fixture', num_topics=2, workspace_dir=str(root), output_dir=str(root/name), device='cpu')
                        trainer.bow_matrix=bow; trainer.vocab=[str(i) for i in range(12)]; trainer.vocab_size=12
                        trainer.sbert_embeddings=np.random.default_rng(7).normal(size=(20,8)).astype('float32')
                        seen=[]
                        original_subset=__import__('model.baseline_trainer', fromlist=['split_subset']).split_subset
                        def observed(dataset, role, rows):
                            result = original_subset(dataset, role, rows)
                            if role=='train': seen.extend(result.indices)
                            return result
                        with patch('model.baseline_trainer.split_subset', observed):
                            options = {'epochs': 1, 'batch_size': 4}
                            if name=='etm': options.update(use_pretrained_embeddings=False, embedding_dim=8, hidden_dim=16)
                            result=getattr(trainer, 'train_'+name)(**options)
                        self.assertEqual(seen, list(range(12)))
                        self.assertEqual(result['theta'].shape, (20,2))
                        self.assertTrue(np.isfinite(result['theta']).all())

    def test_complete_batches_keep_small_sets_and_merge_singletons(self):
        import torch
        from data.dataloader import create_dataloader
        for count, batch_size in [(3, 256), (5, 2), (9, 4), (10, 4)]:
            batches = list(create_dataloader(torch.arange(count), batch_size=batch_size))
            self.assertTrue(all(len(batch) >= 2 for batch in batches))
            self.assertEqual(sorted(torch.cat(batches).tolist()), list(range(count)))
        batches = list(create_dataloader(torch.arange(1), batch_size=256, shuffle=False))
        self.assertEqual(batches[0].tolist(), [0])

    def test_theta_frozen_embedding_mode_really_trains_with_nonzero_loss(self):
        import logging
        import numpy as np
        import torch
        from config import PipelineConfig
        from main import train_etm
        from model.theta.etm import ETM
        rng = np.random.default_rng(17)
        config = PipelineConfig()
        config.embedding.mode = 'zero_shot'
        config.model.epochs = 2
        config.model.batch_size = 256
        config.model.num_workers = 0
        config.model.num_topics = 2
        config.model.doc_embedding_dim = 8
        config.model.word_embedding_dim = 8
        config.model.hidden_dim = 8
        config.model.hidden_sizes = [8, 8]
        config.model.train_word_embeddings = False
        config.model.early_stopping = False
        snapshots = []
        original = ETM.__init__
        def capture(model, *args, **kwargs):
            original(model, *args, **kwargs)
            snapshots.append({name: value.detach().clone() for name, value in model.named_parameters()})
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder)/'split.json'
            manifest.write_text(json.dumps(assignments(10, {'enabled':True, 'method':'sequential', 'ratios':[.6,.2,.2]})))
            with patch.dict(os.environ, {'THETA_DATA_SPLIT_FILE': str(manifest)}), patch.object(ETM, '__init__', capture), patch('main.load_labels_for_supervised', return_value=(None, 0, None)):
                result = train_etm(rng.normal(size=(10,8)).astype('float32'), rng.integers(1,8,(10,12)).astype('float32'), rng.normal(size=(12,8)).astype('float32'), config, logging.getLogger('theta-test'), torch.device('cpu'))
            history = result['history']
            for name in ['train_loss', 'val_loss', 'recon_loss']:
                self.assertEqual(len(history[name]), 2)
                self.assertTrue(all(np.isfinite(v) and v > 0 for v in history[name]))
            self.assertTrue(np.isfinite(result['test_loss']) and result['test_loss'] > 0)
            parameters = dict(result['model'].named_parameters())
            self.assertFalse(torch.equal(snapshots[0]['encoder.mu_layer.weight'], parameters['encoder.mu_layer.weight']))
            self.assertFalse(torch.equal(snapshots[0]['decoder.topic_embeddings'], parameters['decoder.topic_embeddings']))
            self.assertTrue(torch.equal(snapshots[0]['decoder.word_embeddings'], parameters['decoder.word_embeddings']))

    def test_visualization_rows_metrics_and_metadata_follow_saved_split(self):
        import csv
        import numpy as np
        import pandas as pd
        from visualization.dataset_views import split_view, render_split_views
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); report=root/'split_results.json'
            theta=np.array([[.9,.1],[.8,.2],[.2,.8],[.1,.9]])
            data={'theta':theta,'beta':np.eye(2),'source_rows':np.array([2,4,6,8]), 'bow_matrix':np.arange(8).reshape(4,2),
                  'timestamps':list(range(4)), 'source_frame':pd.DataFrame({'id':[2,4,6,8]}), 'training_history':{'train_loss':[2,1]}, 'split_report_path':report}
            groups={'all':{'count':4,'metrics':{'PPL':10}}}
            for role,indices in {'train':[0,1],'validation':[2],'test':[3]}.items():
                target=root/'splits'/role;target.mkdir(parents=True)
                np.save(target/'theta.npy',theta[indices])
                with (target/'documents.csv').open('w') as f:
                    writer=csv.writer(f);writer.writerow(['source_row']);writer.writerows([[int(data['source_rows'][i])+1] for i in indices])
                groups[role]={'count':len(indices),'metrics':{'PPL':len(indices)}}
            report.write_text(json.dumps({'groups':groups,'testUsesAllData':False}))
            view=split_view(data,report,'validation')
            np.testing.assert_array_equal(view['theta'],theta[[2]])
            np.testing.assert_array_equal(view['bow_matrix'],data['bow_matrix'][[2]])
            self.assertEqual(view['source_frame']['id'].tolist(),[6]);self.assertEqual(view['timestamps'],[2])
            self.assertEqual(view['metrics'],{'PPL':1});self.assertIsNone(view['training_history'])
            rendered=[]
            output=root/'plots';output.mkdir()
            render_split_views(data,output,lambda subset,target: rendered.append((subset['theta'].copy(),target)))
            self.assertEqual([len(matrix) for matrix,target in rendered],[2,1,1])
            np.save(root/'splits/test/theta.npy',theta[[0]])
            with self.assertRaises(ValueError):split_view(data,report,'test')

if __name__ == '__main__': unittest.main()
