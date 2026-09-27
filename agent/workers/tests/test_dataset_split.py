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

if __name__ == '__main__': unittest.main()
