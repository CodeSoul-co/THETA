"""Rendering/semantic regressions on small fixtures; never fit models or call APIs."""
import contextlib
import ast
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/models'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from visualization.publication import setup_style, save_figure, validate_export
from visualization.visualization_generator import VisualizationGenerator
from visualization.topic_visualizer import TopicVisualizer, generate_pyldavis_visualization
from visualization.run_visualization import run_baseline_visualization, run_all_visualizations, attach_source_metadata, _run_dtm_specific_visualizations, _export_chart_source_data, load_baseline_data


class PublicationTests(unittest.TestCase):
    def data(self):
        return {'theta': np.array([[.8,.2],[.1,.9],[.4,.6]]),
                'beta': np.array([[.8,.15,.05],[.1,.3,.6]]),
                'vocab': ['产品','服务','政策'],
                'topic_words': [(0,[('产品',.8),('服务',.15)]),(1,[('政策',.6),('服务',.3)])],
                'bow_matrix': np.array([[8,2,1],[1,3,6],[2,2,5]]), 'metrics': None,
                'training_history': None, 'timestamps': None}

    def test_resolution_vectors_and_format_selection(self):
        setup_style('zh')
        with tempfile.TemporaryDirectory() as home:
            fig, ax = plt.subplots(figsize=(4,3));ax.bar([0,1],[.2,.8]);ax.set_title('主题权重')
            before = [p.get_height() for p in ax.patches]
            paths = save_figure(fig, Path(home)/'test.png', dpi=300)
            self.assertEqual({Path(p).suffix for p in paths},{'.png','.pdf','.svg'})
            with Image.open(paths[0]) as im:
                self.assertAlmostEqual(im.info['dpi'][0],300,delta=.1)
                self.assertGreater(im.width,900)
            self.assertTrue(Path(paths[1]).read_bytes().startswith(b'%PDF'))
            self.assertIn('<path',Path(paths[2]).read_text())
            self.assertEqual(before,[p.get_height() for p in ax.patches])
            selected = save_figure(fig,Path(home)/'only.png',dpi=600,formats=['pdf'])
            self.assertEqual([Path(p).suffix for p in selected],['.pdf'])
            self.assertFalse((Path(home)/'only.png').exists());plt.close(fig)
        for value in [0,71,1201,300.5,True]:
            with self.assertRaises(ValueError):validate_export(value,['png'])

    def test_coherence_preserves_metric_scales_and_topic_identity(self):
        data = self.data()
        data['metrics'] = {'NPMI_per_topic': [.15, .4], 'UMass_per_topic': [-12., -3.]}
        with tempfile.TemporaryDirectory() as home:
            generator = VisualizationGenerator(**data, output_dir=home, formats=['svg'])
            captured = []
            with patch.object(generator, '_save', side_effect=lambda *a, **k: captured.append(plt.gcf())):
                generator.generate_topic_coherence_chart()
            axes = captured[0].axes
            self.assertEqual(len(axes), 2)
            self.assertEqual([p.get_width() for p in axes[0].patches], [.15, .4])
            self.assertEqual([p.get_width() for p in axes[1].patches], [-12., -3.])
            self.assertEqual([t.get_text() for t in axes[0].get_yticklabels()], ['T1', 'T2'])
            self.assertGreater(axes[0].get_xlim()[1], .4)
            self.assertLess(axes[1].get_xlim()[0], -12)
            table = pd.read_csv(Path(home)/'global/topic_coherence.csv')
            self.assertEqual(table.topic.tolist(), [1, 2, 1, 2])
            self.assertEqual(table.value.tolist(), [.15, .4, -12., -3.])

    def test_chart_exports_include_machine_readable_plot_data(self):
        data = self.data()
        data['metrics'] = {'topic_exclusivity_per_topic': [.7, .5]}
        data['timestamps'] = pd.to_datetime(['2020-01-01', '2021-01-01', '2021-02-01'])
        with tempfile.TemporaryDirectory() as home:
            generator = VisualizationGenerator(**data, output_dir=home, formats=['svg'])
            _export_chart_source_data(data, home)
            with patch.object(generator, '_save'):
                generator.generate_topic_word_importance(0)
                generator.generate_topic_evolution(0)
                generator.generate_topic_diversity_chart()

            word_importance = pd.read_csv(Path(home)/'topic/topic_1/word_importance.csv')
            evolution = pd.read_csv(Path(home)/'topic/topic_1/topic_evolution.csv')
            exclusivity = pd.read_csv(Path(home)/'global/topic_exclusivity.csv')
            topic_words = pd.read_csv(Path(home)/'global/topic_word_weights.csv')
            topic_similarity = pd.read_csv(Path(home)/'global/topic_beta_similarity.csv')
            intertopic = pd.read_csv(Path(home)/'global/intertopic_distance.csv')

            self.assertEqual(word_importance.columns.tolist(), ['rank', 'word', 'weight'])
            self.assertEqual(word_importance.word.tolist(), ['产品', '服务'])
            self.assertEqual(evolution.columns.tolist(), ['year', 'mean_topic_weight', 'n_documents'])
            self.assertEqual(evolution.n_documents.tolist(), [1, 2])
            self.assertEqual(exclusivity.to_dict('records'), [
                {'topic_id': 1, 'exclusivity': .7},
                {'topic_id': 2, 'exclusivity': .5},
            ])
            self.assertEqual(topic_words.columns.tolist(), ['topic_id', 'rank', 'word', 'weight'])
            self.assertEqual(sorted(topic_words.topic_id.unique().tolist()), [1, 2])
            self.assertEqual(topic_similarity.shape, (2, 3))
            self.assertEqual(intertopic.columns.tolist(), ['topic_id', 'pc1', 'pc2', 'mean_topic_weight'])

    def test_adaptive_export_keeps_editable_cjk_text_and_clean_axes(self):
        from matplotlib import font_manager, ft2font
        setup_style('zh')
        with tempfile.TemporaryDirectory() as home:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax in axes:
                ax.plot([1,2], [.2,.5]); ax.set_xlabel('主题权重'); ax.set_title('Area ∝ weight'); ax.grid(True)
            save_figure(fig, Path(home)/'journal.png', formats=['svg'])
            svg = (Path(home)/'journal.svg').read_text()
            self.assertIn('<text', svg)
            self.assertIn('主题权重', svg)
            np.testing.assert_allclose(fig.get_size_inches(), [12, 5])
            for ax in axes:
                self.assertFalse(any(line.get_visible() for line in ax.get_xgridlines()))
                title=ax._left_title
                title_font=font_manager.findfont(title.get_fontproperties())
                self.assertIn(ord('∝'),ft2font.FT2Font(title_font).get_charmap())
                font = font_manager.findfont(ax.xaxis.label.get_fontproperties())
                chars = ft2font.FT2Font(font).get_charmap()
                self.assertTrue(all(ord(char) in chars for char in '主题权重'))
            self.assertEqual([t.get_text() for ax in axes for t in ax.texts if t.get_gid()=='publication-panel'], ['a','b'])
            plt.close(fig)

    def test_sankey_mass_conservation_dates_and_source_scope(self):
        data=self.data();data['timestamps']=pd.to_datetime(['2020-01-01',None,'2022-01-01'])
        data['dimension_values']=np.array(['A','A','B'])
        original=data['theta'].copy()
        with tempfile.TemporaryDirectory() as home:
            gen=VisualizationGenerator(**data,output_dir=home,formats=['svg'])
            gen.generate_sankey_diagram()
            year=pd.read_csv(Path(home)/'global/year_topic_sankey.csv')
            source=pd.read_csv(Path(home)/'global/source_topic_sankey.csv')
            self.assertAlmostEqual(year.weight_mass.sum(),2)
            self.assertAlmostEqual(source.weight_mass.sum(),3)
            np.testing.assert_allclose(year.groupby('topic_id').weight_mass.sum(),original[[0,2]].sum(axis=0))
            np.testing.assert_allclose(source.groupby('topic_id').weight_mass.sum(),original.sum(axis=0))
            self.assertEqual(set(year.group),{2020,2022})
            for kind in ['year','source']:
                self.assertTrue((Path(home)/f'global/{kind}_topic_sankey.svg').exists())
                markup=(Path(home)/f'global/{kind}_topic_sankey.html').read_text()
                self.assertIn('they do not represent migration',markup)
                self.assertNotIn('<script src=',markup)
            np.testing.assert_array_equal(original,data['theta'])

    def test_sankey_large_float32_weights_keep_total_precision(self):
        data=self.data()
        data['theta']=np.tile(np.array([[.1234567,.8765433]],dtype=np.float32),(32601,1))
        data['dimension_values']=np.array(['A']*32601)
        with tempfile.TemporaryDirectory() as home:
            gen=VisualizationGenerator(**data,output_dir=home,formats=['svg'])
            gen.generate_sankey_diagram()
            source=pd.read_csv(Path(home)/'global/source_topic_sankey.csv')
            np.testing.assert_allclose(source.groupby('topic_id').weight_mass.sum(),
                                       data['theta'].sum(axis=0,dtype=np.float64),rtol=1e-12)

    def test_arc_network_keeps_exact_signed_edges(self):
        with tempfile.TemporaryDirectory() as home:
            gen=VisualizationGenerator(**self.data(),output_dir=home,formats=['svg'])
            gen.generate_topic_network()
            edges=pd.read_csv(Path(home)/'global/topic_network_edges.csv')
            self.assertEqual(edges[['source_topic','target_topic']].values.tolist(),[[1,2]])
            self.assertAlmostEqual(edges.pearson_r.iloc[0],-1)
            self.assertTrue((Path(home)/'global/Topic Correlation Network.svg').exists())

    def test_reference_grid_paginates_and_keeps_titles_outside_images(self):
        with tempfile.TemporaryDirectory() as home:
            viz=TopicVisualizer(output_dir=home,dpi=96,formats=['svg'],language='zh')
            words=[(i,[('产品',.8),('服务',.4),('政策',.2)]) for i in range(8)]
            real_save=viz._save_figure
            pages=[]
            def checked_save(fig,path,**kwargs):
                paths=real_save(fig,path,**kwargs)
                fig.canvas.draw();renderer=fig.canvas.get_renderer()
                active=[ax for ax in fig.axes if ax.get_visible()]
                pages.append(len(active))
                for ax in active:
                    self.assertGreater(ax.title.get_window_extent(renderer).y0,
                                       ax.images[0].get_window_extent(renderer).y1)
                    self.assertFalse(fig._suptitle.get_window_extent(renderer).overlaps(ax.title.get_window_extent(renderer)))
                return paths
            with patch.object(viz,'_save_figure',side_effect=checked_save):
                paths=viz.visualize_wordcloud_grid(words)
            self.assertEqual(pages,[6,2]);self.assertEqual(len(paths),2)
            cloud=viz._make_wordcloud(dict(words[0][1]),3,shape='rectangle')
            self.assertIsNone(cloud.mask)
            np.testing.assert_allclose([cloud.words_[w] for w in ['产品','服务','政策']],[1,.5,.25])

    def test_circular_network_keeps_edges_and_separates_key(self):
        with tempfile.TemporaryDirectory() as home:
            gen=VisualizationGenerator(**self.data(),output_dir=home,dpi=96,formats=['svg'])
            gen.generate_topic_network()
            real_save=gen._save
            def checked_save(path,**kwargs):
                result=real_save(path,**kwargs)
                fig=plt.gcf();fig.canvas.draw();renderer=fig.canvas.get_renderer()
                network,key=fig.axes
                self.assertLess(network.get_window_extent(renderer).x1,key.get_window_extent(renderer).x0)
                for txt in key.texts:
                    self.assertFalse(txt.get_window_extent(renderer).overlaps(network.get_window_extent(renderer)))
                return result
            with patch.object(gen,'_save',side_effect=checked_save):gen.generate_topic_network(layout='circular')
            pd.testing.assert_frame_equal(pd.read_csv(Path(home)/'global/topic_network_edges.csv'),
                                          pd.read_csv(Path(home)/'global/topic_network_circular_edges.csv'))
            with self.assertRaises(ValueError):gen.generate_topic_network(layout='spring')

    def test_gallery_discloses_skipped_chart_reasons(self):
        from visualization.publication import export_manifest
        with tempfile.TemporaryDirectory() as home:
            root=Path(home)
            (root/'chart-status.json').write_text(json.dumps([{'chart':'training_convergence','status':'skipped','detail':'no history'}]))
            payload=export_manifest(root,300,['svg'])
            self.assertEqual(payload['chartStatus'][0]['status'],'skipped')
            self.assertIn('训练曲线',(root/'index.html').read_text())
            self.assertIn('没有保存训练历史',(root/'index.html').read_text())

    def test_wordcloud_layout_preserves_weights_words_and_topic_identity(self):
        with tempfile.TemporaryDirectory() as home:
            visualizer=TopicVisualizer(output_dir=home,dpi=96,formats=['svg'])
            weights={'产品':.9,'数据':.6,'服务':.4,'客户':.2,'agent':.1}
            original=weights.copy()
            cloud=visualizer._make_wordcloud(weights,5,topic_idx=4)
            self.assertEqual(weights,original)
            self.assertEqual({w[0][0] for w in cloud.layout_},set(weights))
            self.assertTrue(all(w[3] is None for w in cloud.layout_))
            np.testing.assert_allclose([cloud.words_[w] for w in weights],[v/.9 for v in weights.values()])
            self.assertEqual(cloud.layout_,visualizer._make_wordcloud(weights,5,topic_idx=4).layout_)
            with self.assertRaises(ValueError):visualizer._make_wordcloud({'empty':0},5)
            # Legacy and combined APIs must also produce readable CJK exports.
            visualizer.visualize_topic_words([(4,list(weights.items()))],as_wordcloud=True,filename='legacy.svg')
            visualizer.visualize_combined_wordcloud([(4,list(weights.items()))],filename='combined.svg')
            self.assertTrue((Path(home)/'legacy_topic5.svg').exists())
            self.assertTrue((Path(home)/'combined.svg').exists())

    def test_projection_detail_retains_all_coordinates_in_overview(self):
        from visualization.topic_visualizer import draw_document_projection
        rng=np.random.default_rng(14)
        coords=np.vstack([rng.normal(size=(100,2)),[[0,-90],[45,0]]])
        original=coords.copy();topics=np.arange(len(coords))%3;topics[-1]=-1
        fig=draw_document_projection(coords,topics,method='UMAP',language='en',total_count=200)
        overview,detail=fig.axes
        points=np.concatenate([c.get_offsets() for c in overview.collections])
        self.assertEqual(sorted(map(tuple,points)),sorted(map(tuple,coords)))
        self.assertLess(detail.get_ylim()[0],0)
        self.assertGreater(detail.get_ylim()[0],-90)
        self.assertTrue(any('IQR' in t.get_text() for t in fig.texts))
        np.testing.assert_array_equal(coords,original);plt.close(fig)

    def test_small_multiple_limits_include_every_observed_value_and_keep_gaps(self):
        data=self.data();data['timestamps']=pd.to_datetime(['2020-01-01','2022-01-01','2022-02-01'])
        with tempfile.TemporaryDirectory() as home:
            generator=VisualizationGenerator(**data,output_dir=home,formats=['svg'])
            for method in [generator.generate_representative_topic_evolution,generator.generate_vocab_evolution]:
                figures=[]
                with patch.object(generator,'_save',side_effect=lambda *a,**kw:figures.append(plt.gcf())):method()
                for ax in figures[0].axes:
                    for line in ax.lines:
                        y=np.asarray(line.get_ydata(),dtype=float)
                        self.assertTrue(np.isnan(y).any())
                        self.assertGreaterEqual(ax.get_ylim()[1],np.nanmax(y))
                        self.assertLessEqual(ax.get_ylim()[0],np.nanmin(y))
            counts=pd.read_csv(Path(home)/'global/word_counts_by_year.csv',index_col=0)
            np.testing.assert_allclose(counts.loc[2022,['产品','服务','政策']].values,[3,5,11])

    def test_correlation_palette_displays_negative_values(self):
        with tempfile.TemporaryDirectory() as home:
            visualizer=TopicVisualizer(output_dir=home,formats=['svg'])
            with patch.object(visualizer,'_save_or_show',side_effect=lambda fig,*a:fig):
                fig=visualizer.visualize_topic_similarity(np.array([[1.,2.,3.],[3.,2.,1.]]),metric='correlation')
            self.assertEqual(fig.axes[0].collections[0].get_clim(),(-1,1))
            self.assertAlmostEqual(np.min(fig.axes[0].collections[0].get_array()),-1)
            plt.close(fig)

    def test_dtm_paginates_every_topic_with_deterministic_word_order(self):
        from visualization.run_visualization import _generate_topic_word_evolution
        with tempfile.TemporaryDirectory() as home:
            evolution = {str(k): {'2022':[('政策',.1),('服务',.2)], '2023':[('政策',.3),('服务',.1)]}
                         for k in range(1,9)}
            _generate_topic_word_evolution(evolution, home, dpi=96, formats=['svg'])
            self.assertTrue((Path(home)/'topic_word_evolution.svg').exists())
            self.assertTrue((Path(home)/'topic_word_evolution_2.svg').exists())
            values=pd.read_csv(Path(home)/'dtm_word_evolution.csv')
            self.assertEqual(set(values.topic),set(range(1,9)))
            self.assertEqual(values[values.topic==8].weight.tolist(),[.1,.3,.2,.1])

    def test_renderer_failure_is_reported_not_silently_swallowed(self):
        from visualization.run_visualization import _run_additional_visualizations
        with tempfile.TemporaryDirectory() as home, \
                patch.object(TopicVisualizer,'visualize_topic_words',side_effect=ValueError('broken word evidence')), \
                patch.object(TopicVisualizer,'visualize_topic_similarity'), \
                patch.object(TopicVisualizer,'visualize_document_topics'), \
                patch.object(TopicVisualizer,'visualize_all_wordclouds'):
            with self.assertRaisesRegex(ValueError,'Native chart export failed'):
                _run_additional_visualizations(self.data(), home, 'en', 96, ['svg'], 'nvdm')
            status=json.loads((Path(home)/'additional-chart-status.json').read_text())
            self.assertEqual(status[0]['status'],'failed')
            self.assertIn('broken word evidence',status[0]['detail'])

    def test_all_registered_model_routes_forward_shared_export_options(self):
        # Full model fitting is deliberately outside this test. One real topic
        # table render per route tests the shared delivery boundary.
        def table_only(generator): generator.generate_topic_table()
        with tempfile.TemporaryDirectory() as home, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(VisualizationGenerator,'generate_all',table_only), \
                patch.object(TopicVisualizer,'visualize_topic_words'), \
                patch.object(TopicVisualizer,'visualize_topic_similarity'), \
                patch.object(TopicVisualizer,'visualize_document_topics'), \
                patch.object(TopicVisualizer,'visualize_all_wordclouds'), \
                patch.object(TopicVisualizer,'visualize_intertopic_distance'), \
                patch.object(TopicVisualizer,'visualize_topic_word_frequency'), \
                patch('visualization.topic_visualizer.generate_pyldavis_visualization',return_value=None):
            models=['lda','hdp','btm','stm','dtm','etm','nvdm','gsm','prodlda','ctm','bertopic','theta']
            for model in models:
                with self.subTest(model=model):
                    data=self.data();out=Path(home)/model
                    if model=='bertopic':data['document_topics']=np.array([0,1,-1])
                    if model=='nvdm':data['theta']=np.array([[-.2,.5],[.7,-.1],[.2,.3]])
                    original=data['theta'].copy()
                    if model=='theta':
                        run_all_visualizations(home,'fixture','zero_shot',data=data,output_dir=out,dpi=96,formats=['svg'])
                    else:
                        run_baseline_visualization(home,'fixture',model,data=data,output_dir=out,language='en',dpi=96,formats=['svg'])
                    self.assertTrue((out/'global/topic_table_1.svg').is_file())
                    self.assertFalse(list(out.rglob('*.png')))
                    self.assertTrue((out/'index.html').is_file())
                    self.assertEqual(json.loads((out/'publication-manifest.json').read_text())['formats'],['svg'])
                    np.testing.assert_array_equal(data['theta'],original)

    def test_real_text_alignment_and_missing_dates(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home)
            source=root/'source.csv';training=root/'training.csv'
            pd.DataFrame({'正文':['a','b','c'],'日期':['2025/1/1',None,'2026-01-01']}).to_csv(source,index=False)
            pd.DataFrame({'text':['a','b','c']}).to_csv(training,index=False)
            data=attach_source_metadata(self.data(),source,training,'正文','日期')
            self.assertEqual(data['source_metadata']['missingTimeRows'],1)
            self.assertEqual(len(data['theta']),3)
            pd.DataFrame({'text':['c','b','a']}).to_csv(training,index=False)
            with self.assertRaisesRegex(ValueError,'row by row'):
                attach_source_metadata(self.data(),source,training,'正文','日期')

    def test_dtm_uses_observed_weights_even_below_top_terms(self):
        data=self.data()
        data['beta_over_time']=np.stack([data['beta'],data['beta'][:,::-1]])
        data['time_slices_info']={'unique_times':[2025,2026]}
        with tempfile.TemporaryDirectory() as home, patch('visualization.run_visualization._generate_topic_word_evolution') as render:
            _run_dtm_specific_visualizations(data,home,formats=['svg'])
            evolution=render.call_args.args[0]
            for topic in range(2):
                for time,year in enumerate([2025,2026]):
                    for word,weight in evolution[str(topic+1)][str(year)]:
                        self.assertEqual(weight,data['beta_over_time'][time,topic,data['vocab'].index(word)])
            self.assertEqual(render.call_args.args[-1],['svg'])

    def test_observed_temporal_beta_kl_and_word_tables(self):
        from scipy.special import rel_entr
        data=self.data();dynamic=np.stack([data['beta'],data['beta'][:,::-1]])
        data.update(beta_over_time=dynamic,time_slices_info={'unique_times':[2025,2026]})
        with tempfile.TemporaryDirectory() as home:
            gen=VisualizationGenerator(**data,output_dir=home,formats=['svg'])
            gen.generate_kl_divergence();gen.generate_topic_word_dist_change(0)
            observed=pd.read_csv(Path(home)/'global/temporal_topic_kl.csv',index_col=0)
            np.testing.assert_allclose(observed.values,rel_entr(dynamic[1:],dynamic[:-1]).sum(axis=2))
            frame=pd.read_csv(Path(home)/'topic/topic_1/word_distribution_by_time.csv',index_col=0)
            for word in frame.index:np.testing.assert_allclose(frame.loc[word].values,dynamic[:,0,data['vocab'].index(word)])
            np.testing.assert_array_equal(dynamic,data['beta_over_time'])
            gen.beta_over_time=dynamic[:,:,:2]
            with self.assertRaises(ValueError):gen.generate_kl_divergence()

    def test_temporal_kl_skips_zero_support_instead_of_reporting_chart_failure(self):
        data=self.data();dynamic=np.stack([data['beta'],data['beta'].copy()])
        dynamic[0,0,0]=0
        dynamic[0,0]/=dynamic[0,0].sum()
        dynamic[1,0,0]=.1
        dynamic[1,0]/=dynamic[1,0].sum()
        data.update(beta_over_time=dynamic,time_slices_info={'unique_times':[2025,2026]})
        with tempfile.TemporaryDirectory() as home:
            gen=VisualizationGenerator(**data,output_dir=home,formats=['svg'])
            log=io.StringIO()
            with contextlib.redirect_stdout(log):gen.generate_kl_divergence()
            self.assertIn('[SKIP] kl_divergence: infinite for zero-support transitions',log.getvalue())
            self.assertFalse((Path(home)/'global/temporal_topic_kl.csv').exists())

    def test_training_curves_preserve_values_and_resolve_scale(self):
        data=self.data();history={'recon_loss':[10,8,6],'kl_loss':[.1,.05,.02],
                                 'train_ppl':[1e7,1e5,1e3],'val_ppl':[2e6,2e4,2e3]}
        data['training_history']=history
        with tempfile.TemporaryDirectory() as home:
            gen=VisualizationGenerator(**data,output_dir=home,formats=['svg'])
            scales=[];real_save=gen._save
            def capture(path,**kwargs):
                fig=plt.gcf()
                scales.append([ax.get_yscale() for ax in fig.axes])
                return real_save(path,**kwargs)
            with patch.object(gen,'_save',side_effect=capture):gen.generate_training_convergence()
            self.assertEqual(scales,[['linear','linear'],['log']])
            table=pd.read_csv(Path(home)/'global/training_curves.csv',index_col=0)
            for key,values in history.items():np.testing.assert_array_equal(table[key],values)

    def test_stm_leaf_result_loads_covariate_sidecars(self):
        data=self.data()
        with tempfile.TemporaryDirectory() as home:
            root=Path(home);leaf=root/'stm'/'model';leaf.mkdir(parents=True)
            np.save(leaf/'theta_k2.npy',data['theta']);np.save(leaf/'beta_k2.npy',data['beta'])
            (leaf/'vocab.json').write_text(json.dumps(data['vocab']))
            gamma=np.array([[0.],[.2]]);cov=np.array([[0.],[1.],[0.]])
            np.save(leaf/'Gamma_k2.npy',gamma);np.save(leaf/'covariates_k2.npy',cov)
            (leaf/'covariate_info_k2.json').write_text(json.dumps({'covariate_names':['source']}))
            with contextlib.redirect_stdout(io.StringIO()):loaded=load_baseline_data(leaf,'test','stm',2)
            np.testing.assert_array_equal(loaded['Gamma'],gamma)
            np.testing.assert_array_equal(loaded['covariates'],cov)
            self.assertEqual(loaded['covariate_names'],['source'])

    def test_stm_gradient_uses_word_evidence(self):
        from scipy.special import softmax
        from scipy.optimize import approx_fprime
        tree=ast.parse((Path(__file__).resolve().parents[1]/'src/models/model/baseline/stm.py').read_text())
        method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_document_gradient')
        env={'np':np};exec(compile(ast.Module(body=[method],type_ignores=[]),'stm-gradient','exec'),env)
        beta=np.array([[.7,.2,.1],[.1,.5,.4],[.3,.3,.4]])
        bow=np.array([8,2,1]);eta=np.array([.2,-.4]);mu=np.array([.1,.1]);precision=np.eye(2)
        def likelihood(x):return bow@np.log(softmax(np.r_[x,0])@beta)-.5*np.sum((x-mu)**2)
        np.testing.assert_allclose(env['_document_gradient'](bow,beta,softmax(np.r_[eta,0]),eta,mu,precision),
                                   approx_fprime(eta,likelihood,1e-6),rtol=1e-5)

    def test_no_fabricated_frequency_or_random_evolution(self):
        with tempfile.TemporaryDirectory() as home, contextlib.redirect_stdout(io.StringIO()):
            data=self.data()
            gen=VisualizationGenerator(**data,output_dir=home,formats=['svg'])
            before=data['theta'].copy()
            gen.generate_kl_divergence();gen.generate_sankey_diagram()
            gen.generate_topic_word_dist_change(0);gen.generate_topic_word_sense(0)
            gen.generate_topic_significance_chart()
            self.assertFalse(list(Path(home).rglob('*.svg')))
            viz=TopicVisualizer(output_dir=home,formats=['svg'])
            with patch.object(viz,'_save_or_show') as capture:
                viz.visualize_topic_word_frequency(data['beta'],data['topic_words'])
                fig=capture.call_args.args[0]
                self.assertEqual([p.get_width() for p in fig.axes[0].patches],[.8,.15])
                plt.close(fig)
            np.testing.assert_array_equal(before,data['theta'])

    def test_pyldavis_never_repairs_invalid_rows_or_mutates_input(self):
        import importlib.util
        if importlib.util.find_spec('pyLDAvis') is None:self.skipTest('optional pyLDAvis not installed')
        data=self.data();data['theta'][0]=0
        with tempfile.TemporaryDirectory() as home, self.assertRaisesRegex(ValueError,'empty rows'):
            generate_pyldavis_visualization(data['theta'],data['beta'],data['bow_matrix'],data['vocab'],str(Path(home)/'test.html'))
        self.assertEqual(data['theta'][0].sum(),0)

    def test_all_baseline_loaders_preserve_original_matrix_axes(self):
        with tempfile.TemporaryDirectory() as home, contextlib.redirect_stdout(io.StringIO()):
            root=Path(home)
            for model in ['lda','hdp','btm','stm','dtm','etm','nvdm','gsm','prodlda','ctm','bertopic']:
                directory=root/model;directory.mkdir()
                data=self.data()
                if model=='nvdm':data['theta']=data['theta']-.5
                np.save(directory/'theta_k2.npy',data['theta']);np.save(directory/'beta_k2.npy',data['beta'])
                np.save(directory/'bow_matrix.npy',data['bow_matrix'])
                (directory/'vocab.json').write_text(json.dumps(data['vocab']))
                if model=='bertopic':np.save(directory/'document_topics.npy',np.array([0,1,-1]))
                loaded=load_baseline_data(directory,'fixture',model,99)
                np.testing.assert_array_equal(loaded['theta'],data['theta'])
                np.testing.assert_array_equal(loaded['beta'],data['beta'])
                self.assertEqual(loaded['vocab'],data['vocab'])

    def test_bertopic_exports_fitted_vocabulary_without_padding(self):
        # Execute only the export method, never import or fit a transformer.
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'src/models/model/baseline/bertopic.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BERTopicModel')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_beta')
        env={'np':np}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'export-test','exec'),env)
        vectorizer=SimpleNamespace(get_feature_names_out=lambda:np.array(['refund','shipping']))
        fitted=SimpleNamespace(get_topics=lambda:{-1:[('noise',.2)],0:[('refund',.5),('',0)],1:[('shipping',.8)]},
                               vectorizer_model=vectorizer)
        model=SimpleNamespace(model=fitted)
        beta=env['get_beta'](model)
        self.assertEqual(model._beta_vocab,['refund','shipping'])
        np.testing.assert_array_equal(beta,np.eye(2))


if __name__=='__main__':unittest.main()
