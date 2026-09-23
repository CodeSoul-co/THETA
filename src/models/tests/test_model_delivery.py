"""Synthetic artifact/dispatch checks, not trained-model or product acceptance."""
import ast
import io
import json
import logging
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifact_utils import export_baseline_metadata, find_topic_matrix_pair, sha256_file, validate_topic_matrices
from visualization import run_visualization as native, visualization_generator, topic_visualizer


class StoredModel:
    def save_model(self, path): Path(path).write_text('synthetic checkpoint contract only')


class DeliveryTests(unittest.TestCase):
    def test_eleven_baseline_exports_and_native_loaders_preserve_axes_and_values(self):
        from sklearn.feature_extraction.text import CountVectorizer
        for name in ['lda', 'hdp', 'stm', 'btm', 'etm', 'ctm', 'dtm', 'nvdm', 'gsm', 'prodlda', 'bertopic']:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as home, redirect_stdout(io.StringIO()):
                root=Path(home)/'exp_fixture'; output=root/name/'model'; output.mkdir(parents=True)
                workspace=Path(home)/'workspace'; workspace.mkdir()
                theta=np.array([[.8,.2],[.2,.8],[.6,.4],[.5,.5]])
                if name=='nvdm': theta-=.5
                if name=='bertopic': theta[-1]=0
                beta=np.array([[.9,.1],[.1,.9]])
                np.save(output/'theta_k2.npy',theta);np.save(output/'beta_k2.npy',beta)
                digest=sha256_file(output/'theta_k2.npy')
                vocab=['refund','shipping']; texts=['refund shipping']*4
                (workspace/'vocab.json').write_text(json.dumps(vocab))
                (workspace/'texts.json').write_text(json.dumps(texts))
                np.save(workspace/'bow_matrix.npy',np.ones((4,2)))
                np.save(workspace/'source_rows.npy',np.arange(4))
                model=StoredModel()
                if name=='bertopic':
                    model._beta_vocab=vocab[::-1]
                    model.model=SimpleNamespace(vectorizer_model=CountVectorizer(vocabulary={word:i for i,word in enumerate(vocab)}))
                    model.get_topics=lambda:[0,1,0,-1]
                result={'model':model,'theta':theta,'beta':beta,'training_history':[{'epoch':1,'loss':2.},{'epoch':2,'loss':1.}]}
                meta=export_baseline_metadata(root,name,result,SimpleNamespace(vocab=vocab,texts=texts))
                delivery_vocab=json.loads((root/meta['beta']).with_name('vocab.json').read_text())
                self.assertEqual(delivery_vocab, model._beta_vocab if name=='bertopic' else vocab)
                if name=='dtm':
                    np.save(workspace/'time_indices.npy',np.array([0,1,0,1]))
                    (workspace/'time_slices.json').write_text(json.dumps({'unique_times':[2024,2025]}))
                    np.save(output/'beta_over_time_k2.npy',np.stack([beta,beta]))
                loaded=native.load_baseline_data(str(root),'fixture',name,99,workspace_dir=workspace)
                np.testing.assert_array_equal(loaded['theta'],theta);np.testing.assert_array_equal(loaded['beta'],beta)
                self.assertEqual(sha256_file(output/'theta_k2.npy'),digest)
                self.assertEqual(meta['topicCount'],2);self.assertEqual(loaded['training_history']['train_loss'],[2.,1.])
                if name=='bertopic':
                    self.assertEqual(loaded['vocab'],vocab[::-1]);self.assertEqual(loaded['document_topics'][-1],-1)
                if name=='dtm':
                    self.assertEqual(loaded['timestamps'][1].year,2025)
                    with patch.object(native,'_generate_topic_word_evolution') as evolution:
                        native._run_dtm_specific_visualizations(loaded,Path(home)/'plots','en',50)
                        evolution.assert_called_once()

    def test_bertopic_padding_uses_real_fitted_word_evidence_or_fails(self):
        from model.baseline.bertopic import BERTopicModel
        vectorizer = SimpleNamespace(get_feature_names_out=lambda: np.array(['refund', 'shipping']))
        fitted = SimpleNamespace(vectorizer_model=vectorizer,
            get_topics=lambda: {-1: [('refund', .1)], 0: [('refund', .8), ('', .00001)], 1: [('', .00001)]},
            c_tf_idf_=sparse.csr_matrix([[.1, .2], [.8, .2], [0., .9]]))
        wrapper = SimpleNamespace(model=fitted)
        beta = BERTopicModel.get_beta(wrapper)
        self.assertEqual(wrapper._beta_vocab, ['refund', 'shipping'])
        np.testing.assert_allclose(beta, np.eye(2))
        fitted.c_tf_idf_ = sparse.csr_matrix([[.1,.2],[.8,.2],[0.,0.]])
        with self.assertRaisesRegex(ValueError, 'no valid word evidence'):
            BERTopicModel.get_beta(wrapper)

    def test_bertopic_keeps_fitted_state_if_word_export_fails(self):
        from model.baseline import bertopic
        from model.baseline_trainer import BaselineTrainer
        with tempfile.TemporaryDirectory() as home, redirect_stdout(io.StringIO()):
            trainer = SimpleNamespace(vocab_size=2, num_topics=2, texts=['refund', 'shipping'],
                sbert_embeddings=np.eye(2), output_dir=home)
            fitted = SimpleNamespace(num_topics=2, outlier_count=1, fit=lambda *args, **kwargs: None,
                get_theta=lambda: np.array([[.5,.1],[.1,.2]]), get_topics=lambda: [0,-1])
            def invalid_words(): raise ValueError('no valid word evidence')
            fitted.get_beta = invalid_words
            def checkpoint(model, filename):
                self.assertIs(model, fitted); Path(filename).write_text('synthetic saved-state fixture')
            with patch.object(bertopic, 'BERTopicModel', return_value=fitted), patch('joblib.dump', side_effect=checkpoint):
                with self.assertRaisesRegex(ValueError, 'no valid word evidence'):
                    BaselineTrainer.train_bertopic(trainer)
            output = Path(home) / 'bertopic'
            np.testing.assert_array_equal(np.load(output / 'document_topics.npy'), [0,-1])
            np.testing.assert_allclose(np.load(output / 'theta_k2.npy'), [[.5,.1],[.1,.2]])
            self.assertTrue((output / 'model.joblib').is_file())

    def test_pipeline_export_exception_has_nonzero_exit(self):
        import run_pipeline
        with patch.object(sys, 'argv', ['run_pipeline.py', '--dataset', 'fixture', '--models', 'bertopic']), patch.object(run_pipeline, 'run_baseline', side_effect=KeyError('')), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:
                run_pipeline.main()
            self.assertEqual(failure.exception.code, 1)
        for outcome in [{'error': 'missing workspace'}, {'train_status': 'failed'}]:
            with patch.object(sys, 'argv', ['run_pipeline.py', '--dataset', 'fixture', '--models', 'bertopic']), patch.object(run_pipeline, 'run_baseline', return_value=outcome), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    run_pipeline.main()
                self.assertEqual(failure.exception.code, 1)

    def test_original_pipeline_publishes_model_manifest_before_reporting_success(self):
        import run_pipeline
        from model import baseline_trainer
        with tempfile.TemporaryDirectory() as home, redirect_stdout(io.StringIO()):
            root=Path(home)/'result'; workspace=Path(home)/'workspace'; workspace.mkdir()
            theta=np.eye(2); beta=np.eye(2)
            trainer=SimpleNamespace(vocab=['refund','shipping'],texts=['refund','shipping'],load_from_workspace=lambda:None)
            def train(**kwargs):
                np.save(root/'theta_k2.npy',theta); np.save(root/'beta_k2.npy',beta)
                return {'model':StoredModel(),'theta':theta,'beta':beta}
            trainer.train_lda=train
            with patch.object(sys,'argv',['run_pipeline.py','--dataset','fixture','--models','lda','--num_topics','2','--skip-eval','--skip-viz']),patch.object(run_pipeline,'find_workspace_dir',return_value=workspace),patch.object(run_pipeline,'get_result_path',return_value=root),patch.object(baseline_trainer,'BaselineTrainer',return_value=trainer):
                result=run_pipeline.run_baseline('lda',run_pipeline.parse_args())
            self.assertEqual(result['train_status'],'completed')
            self.assertEqual(json.loads((root/'result_manifest.json').read_text())['topicCount'],2)

    def test_matrix_validation_and_ambiguous_experiments_fail(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home)
            for k in [2,3]:
                np.save(root/f'theta_k{k}.npy',np.ones((2,k)))
                np.save(root/f'beta_k{k}.npy',np.ones((k,2)))
            with self.assertRaisesRegex(ValueError,'Select a single'):find_topic_matrix_pair(root)
            with self.assertRaisesRegex(ValueError,'axes'):validate_topic_matrices(np.ones((2,2)),np.ones((3,2)),['a','b'],'lda')
            with self.assertRaises(ValueError):validate_topic_matrices(np.full((2,2),np.nan),np.eye(2),['a','b'],'lda')

    def test_bertopic_dispatch_preserves_noise_and_raw_membership(self):
        theta=np.array([[.4,.1],[.1,.4],[0.,0.]])
        data={'theta':theta,'beta':np.eye(2),'vocab':['refund','shipping'],
              'topic_words':[(0,[('refund',1.)]),(1,[('shipping',1.)])], 'document_topics':np.array([0,1,-1])}
        with tempfile.TemporaryDirectory() as home, patch.object(visualization_generator,'VisualizationGenerator') as generator, patch.object(topic_visualizer,'TopicVisualizer') as visualizer, patch.object(topic_visualizer,'generate_pyldavis_visualization') as pyldavis, redirect_stdout(io.StringIO()):
            native.run_baseline_visualization(home,'fixture','bertopic',2,output_dir=home,language='en',data=data)
            np.testing.assert_allclose(generator.call_args.kwargs['theta'],[[.8,.2],[.2,.8]])
            np.testing.assert_array_equal(visualizer.return_value.visualize_document_topics.call_args.kwargs['labels'],[0,1,-1])
            np.testing.assert_array_equal(theta,[[.4,.1],[.1,.4],[0.,0.]])
            pyldavis.assert_not_called();generator.return_value.generate_all.assert_called_once()
            self.assertIn('2/3 assigned',(Path(home)/'README.md').read_text())

    def test_theta_saver_uses_supplied_embeddings_and_saves_label_mapping(self):
        import torch
        from typing import Dict, List, Optional
        tree=ast.parse((Path(__file__).resolve().parents[1]/'main.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_results')
        namespace={'PipelineConfig':object,'np':np,'os':__import__('os'),'json':json,'torch':torch,'sparse':sparse,'datetime':datetime,'logging':logging,'Dict':Dict,'List':List,'Optional':Optional}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'main.py','exec'),namespace)
        used=[]
        model=SimpleNamespace(eval=lambda:None,get_theta=lambda x:used.append(x.numpy()) or x,
            get_beta=lambda:torch.eye(2),get_topic_embeddings=lambda:torch.eye(2),
            get_topic_words=lambda **kw:[(0,[('refund',1.)]),(1,[('shipping',1.)])],state_dict=lambda:{})
        with tempfile.TemporaryDirectory() as home:
            root=Path(home)/'fixture'/'0.6B'/'theta'/'exp_fixture';root.mkdir(parents=True)
            config=SimpleNamespace(model_dir=str(root/'theta'),bow_dir=str(root/'data/bow'),evaluation_dir=str(root),
                visualization_dir=str(root/'visualization'),result_dir=str(root),exp_dir=str(root),embedding=SimpleNamespace(mode='supervised'),
                save=lambda path:Path(path).write_text('{}'))
            (root/'data/bow').mkdir(parents=True);(root/'data/bow/vocab.json').write_text('["refund","shipping"]')
            sparse.save_npz(root/'data/bow/bow_matrix.npz',sparse.csr_matrix(np.eye(2)))
            emb=np.array([[.7,.3],[.2,.8]],dtype=np.float32)
            with patch('model_delivery.save_trained_model'):
                namespace['save_results'](model,{'train_loss':[2.,1.]},['refund','shipping'],emb,sparse.csr_matrix(np.eye(2)),np.eye(2),config,logging.getLogger('test'),torch.device('cpu'),label_encoder=SimpleNamespace(classes_=['A','B']))
            np.testing.assert_array_equal(used[0],emb)
            self.assertEqual(json.loads((root/'theta/label_mapping.json').read_text())['num_classes'],2)
            self.assertEqual(json.loads((root/'result_manifest.json').read_text())['topicCount'],2)
            with redirect_stdout(io.StringIO()):data=native.load_visualization_data(str(root.parents[3]),root.parents[2].name,mode='supervised',model_size='0.6B',model_exp=root.name,model_type='theta')
            np.testing.assert_array_equal(data['theta'],emb)

    def test_failed_metrics_are_omitted_and_latent_perplexity_is_never_computed(self):
        from evaluation import unified_evaluator as module
        evaluator=module.UnifiedEvaluator(np.eye(2),np.array([[-1.,1.],[1.,-1.]]),np.eye(2),['refund','shipping'],model_name='nvdm',num_topics=2)
        with patch.object(module,'compute_topic_diversity',return_value=.8),patch.object(module,'compute_topic_diversity_inverted_rbo',return_value=.7),patch.object(module,'compute_topic_coherence_npmi',return_value=(.3,[.2,.4])),patch.object(module,'compute_topic_coherence_cv',side_effect=ValueError('no C_V')),patch.object(module,'compute_topic_coherence_umass',return_value=(-1.,[-1.,-1.])),patch.object(module,'compute_topic_exclusivity',return_value=(.5,[.5,.5])),patch.object(module,'compute_perplexity',side_effect=AssertionError('latent PPL')),redirect_stdout(io.StringIO()):
            metrics=evaluator.compute_all_metrics()
            self.assertNotIn('C_V',metrics);self.assertNotIn('PPL',metrics);self.assertIn('C_V',metrics['unavailable'])
            self.assertNotIn('PPL',evaluator.get_metrics_dict())


if __name__=='__main__':unittest.main()
