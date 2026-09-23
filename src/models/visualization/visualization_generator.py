"""
Visualization Generator

Supports global charts and per-topic charts with bilingual labels (English/Chinese).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from pathlib import Path
from datetime import datetime, timedelta
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# High quality DPI
from visualization.publication import COLORS, setup_style, save_figure, validate_export, selected_chart

DPI = 300


class VisualizationGenerator:
    """
    Visualization chart generator with bilingual support (English/Chinese).

    Supports:
    - Global charts: topic table, network graph, clustering heatmap, etc.
    - Per-topic charts: word importance, evolution, word distribution changes
    - Bilingual labels (English/Chinese)
    """

    def __init__(self, theta, beta, vocab, topic_words,
                 topic_embeddings=None, timestamps=None, dimension_values=None,
                 bow_matrix=None, training_history=None, metrics=None,
                 output_dir='./visualization', language='en', dpi=300, formats=('png', 'pdf', 'svg'),
                 beta_over_time=None, time_slices_info=None):
        """
        Initialize visualization generator.

        Args:
            theta: Document-topic distribution matrix (n_docs, n_topics)
            beta: Topic-word distribution matrix (n_topics, n_vocab)
            vocab: Vocabulary list (actual words, not word_0, word_1...)
            topic_words: Topic words list [(topic_id, [(word, weight), ...]), ...]
            topic_embeddings: Topic embedding vectors (n_topics, embedding_dim) (optional)
            timestamps: Document timestamp array (optional, for temporal charts)
            dimension_values: Dimension value array, e.g., region (optional, for dimension heatmap)
            bow_matrix: Bag-of-words matrix (n_docs, n_vocab) (optional, for pyLDAvis)
            training_history: Training history dict (optional, for convergence curves)
            metrics: Evaluation metrics dict (optional, for metrics display)
            output_dir: Output directory
            language: Language 'en' or 'zh'
            dpi: Image resolution
        """
        self.theta = theta
        self.beta = beta
        self.beta_over_time = beta_over_time
        self.time_slices_info = time_slices_info or {}
        self.vocab = vocab
        self.topic_words = topic_words
        self.topic_embeddings = topic_embeddings
        self.timestamps = None if timestamps is None else pd.DatetimeIndex(timestamps).to_pydatetime()
        self.dimension_values = dimension_values
        self.bow_matrix = bow_matrix
        self.training_history = training_history
        self.metrics = dict(metrics) if metrics is not None else None
        if self.metrics is not None:
            aliases = {'NPMI': 'topic_coherence_npmi', 'C_V': 'topic_coherence_cv',
                       'UMass': 'topic_coherence_umass', 'Exclusivity': 'topic_exclusivity'}
            for native, chart in aliases.items():
                for suffix, target in [('', '_avg'), ('_per_topic', '_per_topic')]:
                    if self.metrics.get(native + suffix) is not None:
                        self.metrics.setdefault(chart + target, self.metrics[native + suffix])
        self.output_dir = Path(output_dir)
        self.language = language
        self.dpi = dpi
        self.formats = validate_export(dpi, formats)
        self._projection = None
        self.exported_files = []

        self.n_docs, self.n_topics = theta.shape
        self.n_vocab = beta.shape[1]

        # Setup fonts
        self._setup_fonts()

        # Create output directories
        self._create_output_dirs()

    def _setup_fonts(self):
        setup_style(self.language)

    def _save(self, filename, **kwargs):
        paths = save_figure(plt.gcf(), filename, formats=self.formats, **kwargs)
        self.exported_files.extend(paths)
        return paths

    def _document_projection(self):
        if self._projection is None:
            from sklearn.decomposition import PCA
            indices = np.sort(np.random.default_rng(42).choice(self.n_docs, min(5000, self.n_docs), replace=False))
            sample = self.theta[indices]
            if min(sample.shape) < 2:
                self._projection_method = 'Coordinate'
                coords = np.column_stack((sample[:, 0], np.zeros(len(sample))))
            elif len(sample) < 5:
                self._projection_method = 'PCA'
                coords = PCA(n_components=2).fit_transform(sample)
            else:
                self._projection_method = 'UMAP'
                import umap
                coords = umap.UMAP(n_components=2, random_state=42, n_jobs=1,
                    n_neighbors=min(30, len(sample)-1), min_dist=.3, metric='cosine').fit_transform(sample)
            self._projection = indices, sample, coords
            pd.DataFrame({'matrix_row': indices, 'x': coords[:, 0], 'y': coords[:, 1],
                          'dominant_topic': sample.argmax(axis=1)+1}).to_csv(self.global_dir / 'document_projection.csv', index=False)
        return self._projection

    def _create_output_dirs(self):
        """Create output directory structure.

        New Structure (language is handled at parent level):
            {output_dir}/
            ├── global/         # Global charts
            └── topic/          # Per-topic charts
                ├── topic_1/
                ├── topic_2/
                └── ...
        """
        # Output directly to output_dir (language directory is handled by run_pipeline)
        self.global_dir = self.output_dir / 'global'
        self.topics_dir = self.output_dir / 'topic'

        self.global_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.n_topics):
            (self.topics_dir / f'topic_{i+1}').mkdir(parents=True, exist_ok=True)

    def _get_label(self, key):
        """Get label based on language."""
        labels = {
            'topic': {'en': 'Topic', 'zh': '主题'},
            'topic_name': {'en': 'Topic Name', 'zh': '主题名称'},
            'topic_id': {'en': 'Topic ID', 'zh': '主题ID'},
            'word': {'en': 'Word', 'zh': '词'},
            'weight': {'en': 'Weight', 'zh': '权重'},
            'proportion': {'en': 'Proportion', 'zh': '比例'},
            'year': {'en': 'Year', 'zh': '年份'},
            'time_period': {'en': 'Time Period', 'zh': '时间段'},
            'correlation': {'en': 'Correlation', 'zh': '相关系数'},
            'kl_divergence': {'en': 'KL Divergence', 'zh': 'KL散度'},
            'dimension': {'en': 'Dimension', 'zh': '维度'},
            'strength': {'en': 'Strength', 'zh': '强度'},
            'keywords': {'en': 'Keywords', 'zh': '关键词'},
            'umap_dim1': {'en': 'UMAP Dimension 1', 'zh': 'UMAP 维度 1'},
            'umap_dim2': {'en': 'UMAP Dimension 2', 'zh': 'UMAP 维度 2'},
            'umap1': {'en': 'UMAP 1', 'zh': 'UMAP 1'},
            'umap2': {'en': 'UMAP 2', 'zh': 'UMAP 2'},
            'outliers': {'en': 'Outliers', 'zh': '离群点'},
            'document_count': {'en': 'Document Count', 'zh': '文档数量'},
            'loss': {'en': 'Loss', 'zh': '损失'},
            'epoch': {'en': 'Epoch', 'zh': '轮次'},
            'coherence': {'en': 'Coherence', 'zh': '一致性'},
            'exclusivity': {'en': 'Exclusivity', 'zh': '排他性'},
            'frequency': {'en': 'Frequency', 'zh': '频率'},
            'word_frequency': {'en': 'Word Frequency', 'zh': '词频'},
            'similarity': {'en': 'Similarity', 'zh': '相似度'},
            'cosine_similarity': {'en': 'Cosine Similarity', 'zh': '余弦相似度'},
            'others': {'en': 'Others', 'zh': '其他'},
            'train_loss': {'en': 'Train Loss', 'zh': '训练损失'},
            'val_loss': {'en': 'Validation Loss', 'zh': '验证损失'},
            'recon_loss': {'en': 'Reconstruction Loss', 'zh': '重构损失'},
            'train_perplexity': {'en': 'Training perplexity', 'zh': '训练困惑度'},
            'val_perplexity': {'en': 'Validation perplexity', 'zh': '验证困惑度'},
            'kl_loss': {'en': 'KL Loss', 'zh': 'KL损失'},
            'perplexity': {'en': 'Perplexity', 'zh': '困惑度'},
            'score': {'en': 'Score', 'zh': '分数'},
            'significance': {'en': 'Significance', 'zh': '显著性'},
            'significance_score': {'en': 'Significance Score', 'zh': '显著性分数'},
            'num_topics': {'en': 'Number of Topics (K)', 'zh': '主题数 (K)'},
            'mean': {'en': 'Mean', 'zh': '均值'},
            'figure': {'en': 'Figure', 'zh': '图'},
            'doc_clustering_caption': {'en': 'Document clustering by dominant topic', 'zh': '按主导主题的文档聚类'},
            'doc_clusters_outliers_caption': {'en': 'Document clusters with outlier detection', 'zh': '带离群点检测的文档聚类'},
            'training_val_loss': {'en': 'Training & Validation Loss', 'zh': '训练与验证损失'},
            'recon_kl_loss': {'en': 'Reconstruction & KL Loss', 'zh': '重构与KL损失'},
            'final_train_loss': {'en': 'Final Train Loss', 'zh': '最终训练损失'},
            'final_val_loss': {'en': 'Final Validation Loss', 'zh': '最终验证损失'},
            'final_perplexity': {'en': 'Final Perplexity', 'zh': '最终困惑度'},
            'coherence_score': {'en': 'Coherence Score', 'zh': '一致性分数'},
            'topic_coherence_by_metric': {'en': 'Topic Coherence by Metric', 'zh': '各指标主题一致性'},
            'per_topic_exclusivity': {'en': 'Per-Topic Exclusivity', 'zh': '各主题排他性'},
            'topic_significance_ranking': {'en': 'Topic Significance Ranking', 'zh': '主题显著性排名'},
            'topic_num_evaluation': {'en': 'Topic Number Evaluation', 'zh': '主题数评估'},
            'top_words_freq_evolution': {'en': 'Top Words Frequency Evolution', 'zh': '高频词演变'},
            'topic_similarity_change': {'en': 'Topic Similarity Change (KL Divergence)', 'zh': '主题相似度变化 (KL散度)'},
            'dim_topic_heatmap': {'en': 'Dimension-Topic Distribution Heatmap', 'zh': '维度-主题分布热力图'},
            'domain_topic_over_time': {'en': 'Domain Topic Distribution Over Time', 'zh': '领域主题分布时序变化'},
            'topic_dist_similarity_evolution': {'en': 'Topic Distribution Similarity Evolution', 'zh': '主题分布相似度演化'},
            'word_dist_change': {'en': 'Word Distribution Change', 'zh': '词分布变化'},
            'word_sense_evolution': {'en': 'Word Semantic Evolution', 'zh': '词语义演化'},
        }
        return labels.get(key, {}).get(self.language, key)

    def _get_filename(self, key):
        """Get filename based on language (using chart title as filename)."""
        filenames = {
            'topic_table': {'en': 'Topic Identification Results.png', 'zh': '主题识别结果.png'},
            'topic_network': {'en': 'Topic Correlation Network.png', 'zh': '主题相关性网络.png'},
            'doc_clusters': {'en': 'Document Clustering by Dominant Topic.png', 'zh': '文档主题聚类.png'},
            'clustering_heatmap': {'en': 'Topic Clustering Heatmap with Dendrogram.png', 'zh': '主题聚类热力图.png'},
            'clusters_outliers': {'en': 'Document Clusters with Outlier Detection.png', 'zh': '文档聚类与离群点检测.png'},
            'topic_proportion_pie': {'en': 'Topic Proportion Distribution.png', 'zh': '主题占比分布.png'},
            'doc_volume': {'en': 'Document Volume Over Time.png', 'zh': '文档数量时序变化.png'},
            'representative_topic_evolution': {'en': 'Representative Topic Evolution.png', 'zh': '代表性主题演化.png'},
            'kl_divergence': {'en': 'Topic Distribution KL Divergence Over Time.png', 'zh': '主题分布KL散度时序变化.png'},
            'vocab_evolution': {'en': 'High-Frequency Word Evolution.png', 'zh': '高频词演变.png'},
            'topic_similarity_evolution': {'en': 'Topic Distribution Similarity Evolution.png', 'zh': '主题分布相似度演化.png'},
            'all_topics_strength_table': {'en': 'Topic Strength by Year.png', 'zh': '各年度主题强度.png'},
            'dim_heatmap': {'en': 'Dimension-Topic Heatmap.png', 'zh': '维度-主题热力图.png'},
            'domain_topic_distribution': {'en': 'Domain Topic Distribution Over Time.png', 'zh': '领域主题分布时序变化.png'},
            'training_loss': {'en': 'Training Loss.png', 'zh': '训练损失.png'},
            'training_recon_kl': {'en': 'Reconstruction and KL Loss.png', 'zh': '重构损失与KL损失.png'},
            'training_summary': {'en': 'Training Summary.png', 'zh': '训练总结.png'},
            'training_perplexity': {'en': 'Training Perplexity.png', 'zh': '训练困惑度.png'},
            '7_core_metrics': {'en': '7 Core Metrics.png', 'zh': '7项核心指标.png'},
            'topic_coherence': {'en': 'Topic Coherence.png', 'zh': '主题一致性.png'},
            'topic_exclusivity': {'en': 'Topic Exclusivity.png', 'zh': '主题排他性.png'},
            'topic_significance': {'en': 'Topic Significance Ranking.png', 'zh': '主题显著性排名.png'},
            'topic_num_evaluation': {'en': 'Topic Number Evaluation.png', 'zh': '主题数评估.png'},
            'word_importance': {'en': 'Word Importance.png', 'zh': '词重要性.png'},
            'word_cloud': {'en': 'Word Cloud.png', 'zh': '词云.png'},
            'topic_evolution': {'en': 'Topic Evolution.png', 'zh': '主题演化.png'},
            'word_distribution_change': {'en': 'Word Distribution Change.png', 'zh': '词分布变化.png'},
            'word_sense_evolution': {'en': 'Word Semantic Evolution.png', 'zh': '词语义演化.png'},
        }
        return filenames.get(key, {}).get(self.language, f'{key}.png')

    def _get_title(self, key):
        """Get chart title based on language."""
        titles = {
            'topic_table': {'en': 'Topic Identification Results', 'zh': '主题识别结果'},
            'topic_network': {'en': 'Topic Correlation Network', 'zh': '主题相关性网络'},
            'doc_clusters': {'en': 'Document Clustering by Dominant Topic', 'zh': '文档主题聚类'},
            'clustering_heatmap': {'en': 'Topic Clustering Heatmap with Dendrogram', 'zh': '主题聚类热力图'},
            'clusters_outliers': {'en': 'Document Clusters with Outlier Detection', 'zh': '文档聚类与离群点检测'},
            'topic_proportion_pie': {'en': 'Topic Proportion Distribution', 'zh': '主题占比分布'},
            'doc_volume': {'en': 'Document Volume Over Time', 'zh': '文档数量时序变化'},
            'representative_topic_evolution': {'en': 'Representative Topic Evolution', 'zh': '代表性主题演化'},
            'kl_divergence': {'en': 'Topic Distribution KL Divergence Over Time', 'zh': '主题分布KL散度时序变化'},
            'vocab_evolution': {'en': 'High-Frequency Word Evolution', 'zh': '高频词演变'},
            'topic_similarity_evolution': {'en': 'Topic Distribution Similarity Evolution', 'zh': '主题分布相似度演化'},
            'all_topics_strength_table': {'en': 'Topic Strength by Year', 'zh': '各年度主题强度'},
            'dim_heatmap': {'en': 'Dimension-Topic Heatmap', 'zh': '维度-主题热力图'},
            'domain_topic_distribution': {'en': 'Domain Topic Distribution Over Time', 'zh': '领域主题分布时序变化'},
            'training_loss': {'en': 'Training Loss', 'zh': '训练损失'},
            'training_recon_kl': {'en': 'Reconstruction and KL Loss', 'zh': '重构损失与KL损失'},
            'training_summary': {'en': 'Training Summary', 'zh': '训练总结'},
            'topic_coherence': {'en': 'Topic Coherence', 'zh': '主题一致性'},
            'topic_exclusivity': {'en': 'Topic Exclusivity', 'zh': '主题排他性'},
            'word_importance': {'en': 'Word Importance', 'zh': '词重要性'},
            'word_cloud': {'en': 'Word Cloud', 'zh': '词云'},
            'topic_evolution': {'en': 'Topic Evolution', 'zh': '主题演化'},
            'word_distribution_change': {'en': 'Word Distribution Change', 'zh': '词分布变化'},
        }
        return titles.get(key, {}).get(self.language, key)

    # ========== GLOBAL CHARTS ==========

    def generate_topic_table(self, strength_label='strength'):
        """Export full topic table and readable figure pages."""
        # Prepare data for CSV
        csv_data = []
        for topic_id, words in self.topic_words:
            top_words = [w[0] for w in words[:10]]
            strength = self.theta[:, topic_id].mean()
            topic_name = f"{top_words[0]}, {top_words[1]}" if len(top_words) >= 2 else top_words[0]

            csv_data.append({
                'topic_id': topic_id + 1,
                'topic_name': topic_name,
                strength_label: strength,
                'keywords': ', '.join(top_words)
            })

        # Save CSV to global directory
        csv_df = pd.DataFrame(csv_data)
        csv_filename = '主题表.csv' if self.language == 'zh' else 'topic_table.csv'
        csv_path = self.global_dir / csv_filename
        csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ {csv_filename}")
        for start in range(0, self.n_topics, 12):
            entries = csv_data[start:start+12]
            values = [[f"T{row['topic_id']}", ' · '.join(row['keywords'].split(', ')[:5]), f"{row[strength_label]:.4f}"] for row in entries]
            fig, ax = plt.subplots(figsize=(7.1, .34*len(entries)+.65))
            ax.axis('off')
            headers = ['主题', '代表词（模型排序）', '平均权重'] if self.language=='zh' else ['Topic','Top exported terms','Mean weight']
            if strength_label == 'mean_latent_coordinate': headers[-1] = 'Mean coordinate'
            table = ax.table(cellText=values,colLabels=headers,colWidths=[.1,.7,.2],cellLoc='left',bbox=[0,0,1,1])
            table.auto_set_font_size(False);table.set_fontsize(9)
            for (row,col), cell in table.get_celld().items():
                cell.visible_edges='B';cell.set_edgecolor('#DFE4E8');cell.set_linewidth(.5)
                if col==2:cell.get_text().set_ha('right')
                if row>0 and col==0:cell.set_text_props(color=COLORS[(entries[row-1]['topic_id']-1)%len(COLORS)],weight='bold')
                cell.set_facecolor('white')
                if row==0: cell.set_text_props(weight='bold',color='#164563')
            self._save(self.global_dir/f'topic_table_{start//12+1}.png',dpi=self.dpi)
            plt.close(fig)

    def generate_topic_network(self, layout="arc", threshold=.3):
        """Fixed-order arc network: layout does not imply distance or clusters."""
        from matplotlib.path import Path as MplPath
        from matplotlib.patches import PathPatch
        from matplotlib.lines import Line2D
        if layout not in {'arc','circular'} or not 0 <= threshold < 1:
            raise ValueError('Use arc/circular layout and 0 <= threshold < 1')
        if self.n_topics < 2:
            print('[SKIP] topic_network: correlation requires at least two topics')
            return
        corr=np.corrcoef(self.theta.T)
        edges=[(i,j,float(corr[i,j])) for i in range(self.n_topics) for j in range(i+1,self.n_topics)
               if np.isfinite(corr[i,j]) and abs(corr[i,j])>threshold]
        if layout=='circular':
            self._draw_circular_network(edges,threshold)
            return
        fig,ax=plt.subplots(figsize=(6.5,max(3,self.n_topics*.43)))
        widest=max([.45+.22*(j-i) for i,j,_ in edges],default=.6)
        for i,j,value in sorted(edges,key=lambda e:-(e[1]-e[0])):
            y1,y2=self.n_topics-1-i,self.n_topics-1-j
            bend=.45+.22*(j-i)
            curve=MplPath([(0,y1),(-bend,y1),(-bend,y2),(0,y2)],
                          [MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4])
            color=COLORS[0] if value>0 else COLORS[1]
            ax.add_patch(PathPatch(curve,fill=False,color=color,lw=.7+abs(value)*2.5,
                                  linestyle='-' if value>0 else '--',alpha=.85))
            if len(edges)<=16:
                ax.text(-.75*bend,(y1+y2)/2,f'{value:+.2f}',ha='center',va='center',fontsize=8,
                        color=color,bbox=dict(facecolor='white',edgecolor='none',pad=1.8))
        terms=dict(self.topic_words)
        for i in range(self.n_topics):
            y=self.n_topics-1-i;color=COLORS[i%len(COLORS)]
            ax.scatter([0],[y],s=70,c='white',edgecolors=color,linewidths=1.8,zorder=4)
            ax.text(.18,y,f'T{i+1}',va='center',color=color,weight='bold',fontsize=9)
            ax.text(.65,y,' · '.join(w for w,_ in terms.get(i,[])[:3]),va='center',fontsize=9)
        ax.set_xlim(-widest-.25,3.2);ax.set_ylim(-.65,self.n_topics-.35);ax.axis('off')
        title='主题相关网络' if self.language=='zh' else 'Topic correlation network'
        ax.set_title(f'{title} · Pearson |r| > {threshold:g}',pad=24)
        handles=[Line2D([],[],color=COLORS[0],label='正相关' if self.language=='zh' else 'Positive'),
                 Line2D([],[],color=COLORS[1],linestyle='--',label='负相关' if self.language=='zh' else 'Negative')]
        ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,-.11),ncol=2)
        ax.text(0,-.06,('固定主题顺序；弧线位置不表示距离。' if self.language=='zh' else 'Fixed topic order; arc position does not encode distance.')+
                (f' 共 {len(edges)} 条边。' if self.language=='zh' else f' {len(edges)} edges.'),transform=ax.transAxes,fontsize=7,color='#667C8A')
        pd.DataFrame(corr,index=np.arange(self.n_topics)+1,columns=np.arange(self.n_topics)+1).to_csv(self.global_dir/'topic_correlations.csv',index_label='topic_id')
        pd.DataFrame([(i+1,j+1,r) for i,j,r in edges],columns=['source_topic','target_topic','pearson_r']).to_csv(self.global_dir/'topic_network_edges.csv',index=False)
        self._save(self.global_dir/self._get_filename('topic_network'),dpi=self.dpi);plt.close(fig)

    def _draw_circular_network(self, edges, threshold):
        """Fixed circular geometry with a separate key: no labels on crossing edges."""
        from matplotlib.lines import Line2D
        from textwrap import fill
        fig,(ax,key)=plt.subplots(1,2,figsize=(7.15,min(6.5,max(4.3,self.n_topics*.32))),
                                  width_ratios=[1.65,1],layout='constrained')
        angles=np.pi/2+np.arange(self.n_topics)*2*np.pi/self.n_topics
        positions=np.c_[np.cos(angles),np.sin(angles)]
        for i,j,value in edges:
            ax.plot(*positions[[i,j]].T,color=COLORS[0] if value>0 else COLORS[1],
                    lw=.6+abs(value)*2,alpha=.55,linestyle='-' if value>0 else '--',zorder=1)
        size=max(100,550*min(1,8/self.n_topics))
        for i,(x,y) in enumerate(positions):
            ax.scatter([x],[y],s=size,color=COLORS[i%len(COLORS)],edgecolors='white',linewidths=1.5,zorder=3)
            ax.text(x,y,f'T{i+1}',ha='center',va='center',fontsize=8,color='white',zorder=4)
        ax.set(xlim=(-1.22,1.22),ylim=(-1.22,1.22),aspect='equal');ax.axis('off');key.axis('off')
        rows=[f'T{i+1}  '+fill(' · '.join(w for w,_ in dict(self.topic_words).get(i,[])[:3]),22)
              for i in range(self.n_topics)]
        # Edge values occupy their own block, never the network's dense intersections.
        visible=sorted(edges,key=lambda e:-abs(e[2]))[:10]
        labels=[f'T{i+1}–T{j+1}   r = {value:+.3f}' for i,j,value in visible]
        heading='相关系数' if self.language=='zh' else 'Pearson correlation'
        if len(edges)>10:heading+=' · '+('前 10 条；全部见 CSV' if self.language=='zh' else 'top 10; all in CSV')
        content='\n\n'.join(rows)+'\n\n'+heading+'\n'+'\n'.join(labels or ['—'])
        # Use the existing companion topic table for large K instead of shrinking text.
        if self.n_topics>10:
            content=('主题说明见主题表；全部相关系数见 CSV。' if self.language=='zh' else 'Topic descriptions: topic table. All correlations: CSV.')
        key.text(0,1,content,transform=key.transAxes,va='top',fontsize=7,linespacing=1.25)
        fig.suptitle(('主题相关性网络 · 环形' if self.language=='zh' else 'Topic correlation network · circular')+
                     f' · Pearson |r| > {threshold:g}',fontsize=10)
        handles=[Line2D([],[],color=COLORS[0],label='正相关' if self.language=='zh' else 'Positive'),
                 Line2D([],[],color=COLORS[1],linestyle='--',label='负相关' if self.language=='zh' else 'Negative')]
        fig.legend(handles=handles,loc='outside lower center',ncol=2)
        pd.DataFrame([(i+1,j+1,r) for i,j,r in edges],columns=['source_topic','target_topic','pearson_r']).to_csv(
            self.global_dir/'topic_network_circular_edges.csv',index=False)
        self._save(self.global_dir/'topic_network_circular.png',dpi=self.dpi);plt.close(fig)

    def generate_doc_clusters(self):
        from visualization.topic_visualizer import draw_document_projection
        indices, sample, coords = self._document_projection()
        fig = draw_document_projection(coords, np.argmax(sample,axis=1), method=self._projection_method,
                                       language=self.language,total_count=self.n_docs)
        self._save(self.global_dir/self._get_filename('doc_clusters'),dpi=self.dpi)
        plt.close(fig)

    def generate_clustering_heatmap(self):
        """
        Generate hierarchical clustering heatmap with dendrogram.

        Layout matches reference style:
        - Left dendrogram with topic labels on left side of heatmap
        - Top dendrogram with topic labels on bottom of heatmap
        - Right colorbar aligned with heatmap height
        - No white grid lines - smooth color gradient
        """
        import seaborn as sns
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform

        # Filter out zero-variance topics (e.g., HDP may have empty topics)
        # This prevents NaN in correlation matrix
        theta_filtered = self.theta.copy()
        topic_variances = np.var(theta_filtered, axis=0)
        valid_topics = topic_variances > 1e-10
        n_valid = valid_topics.sum()

        if n_valid < 2:
            print(f"  ⚠ clustering_heatmap skipped (only {n_valid} valid topics)")
            return

        if n_valid < self.n_topics:
            print(f"  ⚠ Filtering {self.n_topics - n_valid} zero-variance topics for heatmap")
            theta_filtered = theta_filtered[:, valid_topics]

        topic_corr = np.corrcoef(theta_filtered.T)

        # Handle any remaining NaN/Inf values
        topic_corr = np.nan_to_num(topic_corr, nan=0.0, posinf=1.0, neginf=-1.0)

        # Create topic labels based on language (only for valid topics)
        valid_indices = np.where(valid_topics)[0]
        if n_valid > 20:
            topic_labels = [f"{self._get_label('topic')}{valid_indices[i]+1}" if self.language == 'zh' else f"T{valid_indices[i]+1}" for i in range(n_valid)]
        else:
            topic_labels = [f"{self._get_label('topic')} {valid_indices[i]+1}" if self.language == 'zh' else f"Topic {valid_indices[i]+1}" for i in range(n_valid)]

        # Compute linkage for hierarchical clustering
        # Convert correlation to distance (1 - correlation), ensure symmetric
        dist_matrix = 1 - topic_corr
        dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Ensure symmetry
        np.fill_diagonal(dist_matrix, 0)

        # Ensure no NaN/Inf in distance matrix
        dist_matrix = np.nan_to_num(dist_matrix, nan=1.0, posinf=2.0, neginf=0.0)
        dist_matrix = np.clip(dist_matrix, 0, 2)  # Correlation distance should be in [0, 2]

        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='average')

        fig = plt.figure(figsize=(6.1,4.7),layout='constrained')
        fig._theta_no_panel_labels=True
        grid=fig.add_gridspec(1,3,width_ratios=[1,5,.22])
        tree=fig.add_subplot(grid[0,0]);ax=fig.add_subplot(grid[0,1]);bar=fig.add_subplot(grid[0,2])
        leaves=dendrogram(linkage_matrix,orientation='left',ax=tree,no_labels=True,
                          color_threshold=0,above_threshold_color='#667C8A')['leaves']
        tree.set_ylim(n_valid*10,0);tree.axis('off')
        ordered=topic_corr[np.ix_(leaves,leaves)]
        sns.heatmap(ordered,ax=ax,cbar_ax=bar,cmap='RdBu_r',vmin=-1,vmax=1,linewidths=.6,
                    annot=bool(n_valid<=12),fmt='.2f',annot_kws={'fontsize':7.5},
                    xticklabels=[f'T{valid_indices[i]+1}' for i in leaves],
                    yticklabels=[f'T{valid_indices[i]+1}' for i in leaves],
                    cbar_kws={'label':'Pearson r'})
        ax.tick_params(axis='both',rotation=0)
        ax.set_title('主题权重相关性' if self.language=='zh' else 'Topic-weight correlations')
        self._save(self.global_dir/self._get_filename('clustering_heatmap'),dpi=self.dpi);plt.close(fig)

    def generate_clusters_with_outliers(self):
        from visualization.topic_visualizer import draw_document_projection
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        indices, sample, coords = self._document_projection()
        distances, _ = NearestNeighbors(n_neighbors=min(10,len(coords))).fit(coords).kneighbors(coords)
        labels = DBSCAN(eps=max(float(np.percentile(distances[:,-1],90)),1e-8),min_samples=5).fit_predict(coords)
        topics = np.argmax(sample,axis=1); topics[labels == -1] = -1
        title = '投影 DBSCAN 诊断（非模型离群标签）' if self.language=='zh' else 'Projection DBSCAN diagnostic (not model outliers)'
        fig = draw_document_projection(coords,topics,method=self._projection_method,language=self.language,
                                       total_count=self.n_docs,title=title)
        self._save(self.global_dir/self._get_filename('clusters_outliers'),dpi=self.dpi)
        plt.close(fig)

    def generate_topic_proportion_pie(self):
        """Ranked bars avoid crowded pie labels and preserve exact mean membership."""
        values = self.theta.mean(axis=0)
        indices = np.argsort(-values)[:20]
        labels = [f'T{i+1}  ' + ' · '.join(w for w, _ in self.topic_words[i][1][:2]) for i in indices]
        fig, ax = plt.subplots(figsize=(6.4, max(2.6, .30 * len(indices))))
        ax.hlines(range(len(indices)), 0, values[indices]*100, color='#C8CED3', linewidth=1.2)
        ax.scatter(values[indices]*100, range(len(indices)), color=[COLORS[i % len(COLORS)] for i in indices], s=26, zorder=3)
        ax.set_yticks(range(len(indices)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values[indices])*108)
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
        for y, value in enumerate(values[indices]):
            ax.text(1.01,y,f'{value:.1%}',transform=ax.get_yaxis_transform(),va='center')
        ax.set_xlabel('平均文档主题权重（%）' if self.language == 'zh' else 'Mean document-topic weight (%)')
        ax.set_title(f'K={self.n_topics} · N={self.n_docs:,}')
        ax.grid(axis='x', alpha=.5)
        pd.DataFrame({'topic_id': np.arange(self.n_topics)+1, 'mean_weight': values}).to_csv(self.global_dir/'topic_proportions.csv',index=False)
        fig.tight_layout()
        self._save(self.global_dir / self._get_filename('topic_proportion_pie'), dpi=self.dpi)
        plt.close(fig)

    def generate_representative_topic_evolution(self):
        if self.timestamps is None: return
        from matplotlib.ticker import PercentFormatter, MaxNLocator
        years = np.array([t.year for t in self.timestamps]); periods=np.arange(years.min(),years.max()+1)
        if len(periods)<2: return
        topics=np.argsort(-self.theta.mean(axis=0))[:5]
        fig,axes=plt.subplots(len(topics),1,figsize=(6.6, max(2,1.1*len(topics))),sharex=True,sharey=True,layout='constrained',squeeze=False)
        fig._theta_no_panel_labels=True
        for ax,topic in zip(axes[:,0],topics):
            values=[self.theta[years==y,topic].mean() if (years==y).any() else np.nan for y in periods]
            ax.plot(periods,values,'o-',color=COLORS[topic%len(COLORS)],markersize=3)
            ax.set_title(f'T{topic+1}  '+ ' · '.join(w for w,_ in self.topic_words[topic][1][:2]),fontsize=9)
            ax.yaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_locator(MaxNLocator(nbins=3));ax.set_ylim(bottom=0)
        maximum=max(np.max(self.theta[years==y][:,topics].mean(axis=0)) for y in np.unique(years))
        axes[0,0].set_ylim(0,max(.01,maximum*1.12))
        axes[-1,0].set_xticks(periods[::max(1,int(np.ceil(len(periods)/7)))])
        axes[-1,0].set_xlabel(self._get_label('year'))
        fig.supylabel('平均主题权重' if self.language=='zh' else 'Mean topic weight',fontsize=8)
        self._save(self.global_dir/self._get_filename('representative_topic_evolution'),dpi=self.dpi);plt.close(fig)

    def generate_topic_similarity_evolution(self):
        """Generate topic similarity evolution over time."""
        if self.timestamps is None:
            print("  [SKIP] topic_similarity_evolution (no timestamps)")
            return

        years = np.array([t.year for t in self.timestamps])
        unique_years = sorted(set(years))

        if len(unique_years) < 2:
            return

        year_topic_dists = []
        for year in unique_years:
            mask = years == year
            year_topic_dists.append(self.theta[mask].mean(axis=0))

        from scipy.spatial.distance import cosine
        similarities = []
        year_pairs = []

        for i in range(len(unique_years) - 1):
            sim = 1 - cosine(year_topic_dists[i], year_topic_dists[i+1])
            similarities.append(sim)
            year_pairs.append(f"{unique_years[i]}-{unique_years[i+1]}")

        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        x = range(len(similarities))
        ax.plot(x,similarities,'o-',color=COLORS[0],markersize=4)
        year_pairs=[f'{a}→{str(b)[-2:]}' for a,b in zip(unique_years[:-1],unique_years[1:])]

        ax.set_xticks(x)
        ax.set_xticklabels(year_pairs, rotation=45, ha='right')
        ax.set_xlabel(self._get_label('time_period'), fontsize=12)
        ax.set_ylabel(self._get_label('cosine_similarity'), fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save(self.global_dir / self._get_filename('topic_similarity_evolution'), dpi=self.dpi,
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('topic_similarity_evolution')}")

    def generate_all_topics_strength_table(self):
        if self.timestamps is None: return
        from matplotlib.ticker import PercentFormatter
        years=np.array([t.year for t in self.timestamps]);periods=sorted(set(years))
        values=np.array([self.theta[years==year].mean(axis=0) for year in periods])
        frame=pd.DataFrame(values.T,index=[f'T{i+1}' for i in range(self.n_topics)],columns=periods)
        frame.to_csv(self.global_dir/'topic_weights_by_year.csv',index_label='topic_id')
        fig,ax=plt.subplots(figsize=(6.5,max(3,len(periods)*.32)))
        sns.heatmap(values,ax=ax,cmap='Blues',vmin=0,linewidths=.8,linecolor='white',
                    annot=len(periods)<=20 and self.n_topics<=16,fmt='.1%',annot_kws={'fontsize':8},
                    xticklabels=frame.index,yticklabels=[f'{y}   (n={(years==y).sum():,})' for y in periods],
                    cbar_kws={'label':'平均权重' if self.language=='zh' else 'Mean weight','format':PercentFormatter(1)})
        ax.tick_params(axis='both',rotation=0);ax.set_xlabel(self._get_label('topic'))
        ax.set_title('年度主题构成' if self.language=='zh' else 'Topic composition by year')
        self._save(self.global_dir/self._get_filename('all_topics_strength_table'),dpi=self.dpi);plt.close(fig)

    def generate_domain_topic_distribution(self):
        if self.dimension_values is None or self.timestamps is None: return
        from matplotlib.ticker import MaxNLocator, PercentFormatter
        from textwrap import fill
        years = np.array([t.year for t in self.timestamps])
        unique_years = np.arange(years.min(), years.max()+1)
        dimensions = pd.Series(self.dimension_values).value_counts().head(4).index
        topics = np.argsort(-self.theta.mean(axis=0))[:3]
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
        rows = []
        for ax, dimension in zip(axes.flat, dimensions):
            for j,topic in enumerate(topics):
                values = []
                for year in unique_years:
                    mask = (years == year) & (self.dimension_values == dimension)
                    value = self.theta[mask, topic].mean() if mask.any() else np.nan
                    values.append(value)
                    rows.append([dimension, int(year), int(topic)+1, int(mask.sum()), value])
                ax.plot(unique_years, values, marker=['o','s','^'][j],linestyle='-', color=COLORS[topic % len(COLORS)], label=f'T{topic+1}', markersize=3)
            ax.set_title(fill(str(dimension), 22)+f' · n={(self.dimension_values==dimension).sum():,}')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='y', alpha=.5)
            ax.set_xlabel(self._get_label('year')); ax.set_ylabel(self._get_label('proportion'))
        for ax in list(axes.flat)[len(dimensions):]: ax.set_visible(False)
        axes[0,0].legend(loc='best', frameon=False)
        pd.DataFrame(rows,columns=['group','year','topic_id','n_documents','mean_weight']).to_csv(self.global_dir/'group_topic_over_time.csv',index=False)
        fig.tight_layout()
        self._save(self.global_dir / self._get_filename('domain_topic_distribution'),dpi=self.dpi)
        plt.close(fig)


    def generate_doc_volume(self):
        if self.timestamps is None: return
        counts=pd.Series([t.year for t in self.timestamps]).value_counts().sort_index()
        fig,ax=plt.subplots(figsize=(5.8,max(3,len(counts)*.29)))
        ax.barh(range(len(counts)),counts.values,color=COLORS[0],height=.55)
        ax.set_yticks(range(len(counts)),counts.index);ax.invert_yaxis()
        for i,value in enumerate(counts): ax.text(1.01,i,f'{value:,}',transform=ax.get_yaxis_transform(),va='center')
        ax.set_xlim(0,max(counts)*1.05);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.set_xlabel(self._get_label('document_count'))
        ax.set_title('年度样本量（按现有数据）' if self.language=='zh' else 'Annual sample coverage (observed data)')
        counts.rename_axis('year').to_csv(self.global_dir/'document_counts_by_year.csv',header=['n_documents'])
        self._save(self.global_dir/self._get_filename('doc_volume'),dpi=self.dpi);plt.close(fig)

    def generate_kl_divergence(self):
        """Adjacent-time KL from actual aligned topic-word probabilities."""
        evidence=self._temporal_beta()
        if evidence is None:
            print('[SKIP] kl_divergence: no aligned time-specific beta')
            return
        beta,times=evidence
        # Relative entropy is undefined when q=0 but p>0; report this rather than hiding infinity.
        from scipy.special import rel_entr
        values=rel_entr(beta[1:],beta[:-1]).sum(axis=2)
        if not np.isfinite(values).all():
            print('[SKIP] kl_divergence: infinite for zero-support transitions; inspect beta')
            return
        labels=[f'{a}→{b}' for a,b in zip(times[:-1],times[1:])]
        fig,ax=plt.subplots(figsize=(7.15,max(3,len(labels)*.3)),layout='constrained')
        sns.heatmap(values,ax=ax,cmap='Blues',vmin=0,linewidths=.5,linecolor='white',
                    xticklabels=[f'T{i+1}' for i in range(self.n_topics)],yticklabels=labels,
                    cbar_kws={'label':'KL (nats)'})
        ax.tick_params(axis='both',rotation=0)
        ax.set_title('相邻时间片主题词分布变化' if self.language=='zh' else 'Adjacent-time topic-word divergence')
        ax.set_xlabel('KL(beta[t] || beta[t-1]) · '+('相同主题的完整词表' if self.language=='zh' else 'full vocabulary, same topic'))
        pd.DataFrame(values,index=labels,columns=[f'T{i+1}' for i in range(self.n_topics)]).to_csv(self.global_dir/'temporal_topic_kl.csv',index_label='transition')
        self._save(self.global_dir/self._get_filename('kl_divergence'),dpi=self.dpi);plt.close(fig)

    def generate_dimension_heatmap(self):
        if self.dimension_values is None: return
        from textwrap import fill
        from matplotlib.ticker import PercentFormatter
        groups=pd.Series(self.dimension_values).value_counts().index
        values=np.array([self.theta[self.dimension_values==g].mean(axis=0) for g in groups])
        fig,ax=plt.subplots(figsize=(7,max(3,len(groups)*.45)))
        sns.heatmap(values,ax=ax,cmap='Blues',vmin=0,linewidths=.8,linecolor='white',
                    annot=len(groups)<=16 and self.n_topics<=16,fmt='.1%',annot_kws={'fontsize':8},
                    xticklabels=[f'T{i+1}' for i in range(self.n_topics)],
                    yticklabels=[fill(str(g),18)+f' (n={(self.dimension_values==g).sum():,})' for g in groups],
                    cbar_kws={'label':'平均权重' if self.language=='zh' else 'Mean weight','format':PercentFormatter(1)})
        ax.tick_params(axis='both',rotation=0);ax.set_xlabel(self._get_label('topic'))
        ax.set_title('来源与主题构成' if self.language=='zh' else 'Topic composition by source')
        pd.DataFrame(values,index=groups,columns=[f'T{i+1}' for i in range(self.n_topics)]).to_csv(self.global_dir/'topic_weights_by_group.csv',index_label='group')
        self._save(self.global_dir/self._get_filename('dim_heatmap'),dpi=self.dpi);plt.close(fig)

    def generate_topic_word_importance(self, topic_idx):
        """Generate word importance bar chart for a single topic."""
        topic_dir = self.topics_dir / f'topic_{topic_idx + 1}'

        if topic_idx < len(self.topic_words):
            words_weights = self.topic_words[topic_idx][1][:15]
        else:
            top_indices = np.argsort(self.beta[topic_idx])[-15:][::-1]
            words_weights = [(self.vocab[i] if i < len(self.vocab) else f'word_{i}',
                             self.beta[topic_idx, i]) for i in top_indices]

        words = [w[0] for w in words_weights]
        weights = [w[1] for w in words_weights]
        topic_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'rank': np.arange(1, len(words) + 1),
            'word': words,
            'weight': weights,
        }).to_csv(topic_dir / 'word_importance.csv', index=False)

        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = range(len(words))
        ax.barh(y_pos, weights, color=COLORS[0], alpha=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()

        ax.set_xlabel(self._get_label('weight'), fontsize=12)

        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        self._save(topic_dir / self._get_filename('word_importance'), dpi=self.dpi,
                   bbox_inches='tight', facecolor='white')
        plt.close()

    def generate_topic_evolution(self, topic_idx):
        if self.timestamps is None: return
        from matplotlib.ticker import PercentFormatter
        years=np.array([t.year for t in self.timestamps]);periods=np.arange(years.min(),years.max()+1)
        counts=np.array([(years==y).sum() for y in periods])
        values=[self.theta[years==y,topic_idx].mean() if (years==y).any() else np.nan for y in periods]
        fig,(ax,sample)=plt.subplots(2,1,figsize=(6.3,3.7),height_ratios=[3,1],sharex=True,layout='constrained')
        fig._theta_no_panel_labels=True
        color=COLORS[topic_idx%len(COLORS)]
        ax.plot(periods,values,'o-',color=color,markersize=4,lw=1.5)
        ax.set_ylim(bottom=0);ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel('平均主题权重' if self.language=='zh' else 'Mean topic weight')
        ax.set_title(f'T{topic_idx+1}  '+ ' · '.join(w for w,_ in self.topic_words[topic_idx][1][:2]))
        ax.annotate(f'{values[-1]:.1%}',(periods[-1],values[-1]),xytext=(6,0),textcoords='offset points',va='center',color=color)
        ax.margins(x=.07)
        sample.bar(periods,counts,width=.55,color='#CDD9E0')
        sample.set_ylim(0,max(counts)*1.6);sample.set_yticks([]);sample.set_ylabel('样本 n' if self.language=='zh' else 'Sample n')
        for year,count in zip(periods,counts):sample.text(year,count,f'{count:,}',ha='center',va='bottom',fontsize=7)
        sample.set_xticks(periods[::max(1,int(np.ceil(len(periods)/7)))])
        sample.set_xlabel(self._get_label('year'))
        directory=self.topics_dir/f'topic_{topic_idx+1}';directory.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'year':periods,'mean_topic_weight':values,'n_documents':counts}).to_csv(directory/'topic_evolution.csv',index=False)
        self._save(directory/self._get_filename('topic_evolution'),dpi=self.dpi);plt.close(fig)

    def generate_topic_word_dist_change(self, topic_idx):
        """Same words at each actual time slice; no synthetic perturbations."""
        evidence=self._temporal_beta()
        if evidence is None:
            print('[SKIP] word_distribution_change: no aligned time-specific beta')
            return
        beta,times=evidence
        selected=np.argsort(-beta[:,topic_idx,:].mean(axis=0))[:8]
        frame=pd.DataFrame(beta[:,topic_idx,selected].T,index=[self.vocab[i] for i in selected],columns=times)
        fig,ax=plt.subplots(figsize=(7.15,3.4),layout='constrained')
        sns.heatmap(frame,ax=ax,cmap='Blues',vmin=0,linewidths=.5,linecolor='white',
                    cbar_kws={'label':'词概率' if self.language=='zh' else 'Word probability'})
        ax.tick_params(axis='y',rotation=0);ax.tick_params(axis='x',rotation=45)
        ax.set_title(f'T{topic_idx+1} · '+('分时词概率' if self.language=='zh' else 'Time-specific word probabilities'))
        ax.set_ylabel('');ax.set_xlabel('全时段均值最高的 8 个词；同一色阶' if self.language=='zh' else 'Top 8 words by time-mean probability; one color scale')
        directory=self.topics_dir/f'topic_{topic_idx+1}';directory.mkdir(exist_ok=True)
        frame.to_csv(directory/'word_distribution_by_time.csv',index_label='word')
        self._save(directory/'word_distribution_by_time.png',dpi=self.dpi);plt.close(fig)

    def _temporal_beta(self):
        times=self.time_slices_info.get('unique_times',[])
        if self.beta_over_time is None or len(times)<2:return None
        beta=np.asarray(self.beta_over_time,dtype=float)
        if beta.shape!=(len(times),self.n_topics,self.n_vocab) or not np.isfinite(beta).all() or (beta<0).any():
            raise ValueError('Time-specific beta must align to time/topic/vocabulary axes and be finite/nonnegative')
        mass=beta.sum(axis=2,keepdims=True)
        if (mass<=0).any():raise ValueError('Time-specific beta contains an empty topic')
        return beta/mass,list(times)

    def generate_topic_word_sense(self, topic_idx):
        """Do not present synthetic trajectories as fitted model results."""
        print("  [SKIP] word_sense_evolution: no observed semantic trajectories; simulated values are not evidence")

    def generate_training_convergence(self):
        """Generate training convergence curves (split into separate figures)."""
        if self.training_history is None:
            print("  [SKIP] training_convergence (no training_history)")
            return

        # Keep the two native loss histories separate: they optimize different objectives.
        stages = [key for key in ('stage1', 'stage2') if isinstance(self.training_history.get(key), dict)]
        if stages:
            original_history, original_dir = self.training_history, self.global_dir
            try:
                for stage in stages:
                    self.training_history = original_history[stage]
                    self.global_dir = original_dir / stage
                    self.global_dir.mkdir(parents=True, exist_ok=True)
                    self.generate_training_convergence()
            finally:
                self.training_history, self.global_dir = original_history, original_dir
            return

        from matplotlib.ticker import MaxNLocator
        series={k:np.asarray(v,dtype=float) for k,v in self.training_history.items()
                if k in {'train_loss','val_loss','recon_loss','kl_loss','perplexity','train_ppl','val_ppl'} and len(v)}
        # A short/sparse run can have an infinite perplexity while its loss
        # curves remain valid. Omit only the invalid series so one diagnostic
        # does not suppress every persisted training chart.
        series={k:v for k,v in series.items() if np.isfinite(v).all()}
        if not series:
            raise ValueError('Training curves require saved finite, nonempty histories')
        pd.DataFrame({k:pd.Series(v,index=np.arange(1,len(v)+1)) for k,v in series.items()}).to_csv(
            self.global_dir/'training_curves.csv',index_label='epoch')
        for keys,filename in [(['train_loss','val_loss'],'training_loss'),
                              (['recon_loss','kl_loss'],'training_recon_kl'),
                              (['perplexity','train_ppl','val_ppl'],'training_perplexity')]:
            keys=[k for k in keys if k in series]
            if not keys:continue
            split=filename=='training_recon_kl'
            fig,axes=plt.subplots(1,len(keys) if split else 1,figsize=(7.1,3) if split else (6.2,3.2),
                                  squeeze=False,layout='constrained')
            for i,key in enumerate(keys):
                ax=axes.flat[i] if split else axes.flat[0]
                values=series[key]
                label=self._get_label({'train_ppl':'train_perplexity','val_ppl':'val_perplexity'}.get(key,key))
                ax.plot(range(1,len(values)+1),values,marker=['o','s','^'][i],linestyle='-' if i==0 else '--',
                        color=COLORS[i],markersize=3,lw=1.2,label=label)
                ax.set_xlabel(self._get_label('epoch'));ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if split:ax.set_title(label);ax.set_ylabel(self._get_label('loss'))
            if not split:
                ax=axes.flat[0]
                ylabel=self._get_label('perplexity' if filename=='training_perplexity' else 'loss')
                combined=np.concatenate([series[k] for k in keys])
                if filename=='training_perplexity' and (combined>0).all() and combined.max()/combined.min()>100:
                    ax.set_yscale('log');ylabel+=('（对数坐标）' if self.language=='zh' else ' (log scale)')
                ax.set_ylabel(ylabel)
                fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=len(keys))
            self._save(self.global_dir/self._get_filename(filename),dpi=self.dpi);plt.close(fig)

    def generate_vocab_evolution(self):
        if self.timestamps is None or self.bow_matrix is None:return
        from matplotlib.ticker import MaxNLocator
        years=np.array([t.year for t in self.timestamps]);periods=np.arange(years.min(),years.max()+1)
        if len(periods)<2:return
        totals=np.asarray(self.bow_matrix.sum(axis=0)).ravel();ids=np.argsort(totals)[-10:][::-1]
        words=[self.vocab[i] for i in ids]
        values=np.array([np.asarray(self.bow_matrix[years==y][:,ids].sum(axis=0)).ravel() if (years==y).any() else np.full(len(ids),np.nan) for y in periods])
        fig,axes=plt.subplots(int(np.ceil(len(ids)/2)),2,figsize=(7,6.5),sharex=True,sharey=True,layout='constrained',squeeze=False)
        fig._theta_no_panel_labels=True
        for i,(ax,word) in enumerate(zip(axes.flat,words)):
            ax.plot(periods,values[:,i],'o-',color=COLORS[0],markersize=2.5)
            ax.set_title(word,fontsize=9);ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=4));ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axes[0,0].set_ylim(0,max(1,float(np.nanmax(values))*1.12))
        for ax in list(axes.flat)[len(ids):]:ax.set_visible(False)
        fig.supylabel('原始词频（未按样本量归一化）' if self.language=='zh' else 'Raw word counts (not sample-normalized)',fontsize=8)
        fig.supxlabel(self._get_label('year'),fontsize=8)
        pd.DataFrame(values,index=periods,columns=words).to_csv(self.global_dir/'word_counts_by_year.csv',index_label='year')
        self._save(self.global_dir/self._get_filename('vocab_evolution'),dpi=self.dpi);plt.close(fig)

    def generate_topic_coherence_chart(self):
        """Aligned topic rows, independent metric axes, unmodified observations."""
        keys=[('topic_coherence_npmi_per_topic','NPMI'),('topic_coherence_cv_per_topic','C_V'),('topic_coherence_umass_per_topic','UMass')]
        available=[(key,name) for key,name in keys if (self.metrics or {}).get(key) is not None]
        if not available:return
        fig,axes=plt.subplots(1,len(available),figsize=(max(3,len(available)*2.3),max(2.6,self.n_topics*.3)),sharey=True,squeeze=False,layout='constrained')
        records=[];y=np.arange(self.n_topics)
        for ax,(key,name) in zip(axes[0],available):
            values=np.asarray(self.metrics[key],dtype=float)
            if values.shape!=(self.n_topics,):
                plt.close(fig);raise ValueError(f'{name} must contain one value per topic')
            ax.barh(y,values,color=[COLORS[i%len(COLORS)] for i in y],height=.45)
            ax.set_title(name);ax.axvline(0,color='#9BA3A8',lw=.6)
            ax.set_yticks(y,[f'T{i+1}' for i in y]);ax.invert_yaxis() if not ax.yaxis_inverted() else None
            ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);ax.locator_params(axis='x',nbins=4)
            records.extend({'topic':i+1,'metric':name,'value':v} for i,v in enumerate(values))
        self._save(self.global_dir/self._get_filename('topic_coherence'),dpi=self.dpi);plt.close(fig)
        pd.DataFrame(records).to_csv(self.global_dir/'topic_coherence.csv',index=False)

    def generate_topic_diversity_chart(self):
        if not self.metrics or 'topic_exclusivity_per_topic' not in self.metrics:return
        values=np.asarray(self.metrics['topic_exclusivity_per_topic']);y=np.arange(len(values))
        fig,ax=plt.subplots(figsize=(5.2,max(2.5,len(values)*.29)))
        ax.hlines(y,0,values,color='#D9E0E5',lw=1.5)
        ax.scatter(values,y,color=[COLORS[i%len(COLORS)] for i in y],s=32,zorder=3)
        ax.set_yticks(y,[f'T{i+1}' for i in y]);ax.invert_yaxis();ax.set_xlim(0,1.06)
        for i,value in enumerate(values):ax.text(1.01,i,f'{value:.3f}',transform=ax.get_yaxis_transform(),va='center')
        ax.axvline(values.mean(),color='#727B82',linestyle='--',lw=.8,label=f'{self._get_label("mean")}: {values.mean():.3f}')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.18));ax.set_xlabel(self._get_label('exclusivity'))
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
        pd.DataFrame({'topic_id': y + 1, 'exclusivity': values}).to_csv(self.global_dir/'topic_exclusivity.csv',index=False)
        self._save(self.global_dir/self._get_filename('topic_exclusivity'),dpi=self.dpi);plt.close(fig)

    def generate_metrics_summary(self):
        """Deprecated: Overlaps with generate_7_core_metrics_chart. Skipped."""
        print("  [SKIP] metrics_summary (replaced by 7_core_metrics_chart)")
        return

    def generate_7_core_metrics_chart(self):
        if not self.metrics:return
        names=['TD','iRBO','NPMI','C_V','UMass','Exclusivity','PPL']
        items=[(name,float(self.metrics[name])) for name in names if isinstance(self.metrics.get(name),(float,int)) and np.isfinite(self.metrics[name])]
        if not items:return
        cols=min(4,len(items));rows=int(np.ceil(len(items)/cols))
        fig,axes=plt.subplots(rows,cols,figsize=(6.6,rows*1.25),squeeze=False,layout='constrained')
        for ax,(name,value) in zip(axes.flat,items):
            ax.axis('off');ax.plot([0,1],[.95,.95],transform=ax.transAxes,color='#D8E0E6',lw=.8)
            ax.text(0,.7,name,transform=ax.transAxes,color='#5B6B72',fontsize=9)
            text=ax.text(0,.24,f'{value:.4g}',transform=ax.transAxes,color=COLORS[0],fontsize=20);text.set_gid('metric-value')
        for ax in list(axes.flat)[len(items):]:ax.axis('off')
        fig.suptitle('模型评估 · 各指标保持原始量纲' if self.language=='zh' else 'Model evaluation · original metric scales',x=.02,ha='left',fontsize=10)
        pd.DataFrame(items,columns=['metric','value']).to_csv(self.global_dir/'evaluation_metrics.csv',index=False)
        self._save(self.global_dir/self._get_filename('7_core_metrics'),dpi=self.dpi);plt.close(fig)

    def generate_topic_significance_chart(self):
        # Mean topic mass is not a significance test, and an undocumented score
        # must not be presented as statistical evidence.
        print('  [SKIP] topic_significance: no documented inferential test; use the observed mean-topic-weight chart')


    def generate_topic_num_evaluation(self, k_evaluation_data: dict = None):
        """Generate topic number evaluation chart showing metrics across different K values.

        Args:
            k_evaluation_data: Dict with K values as keys and evaluation metrics as values.
                              Format: {k: {'coherence': float, 'exclusivity': float, 'perplexity': float}, ...}
                              If None, will try to load from evaluation files in parent directory.
        """
        current_k = self.n_topics

        # Try to load real evaluation data
        if k_evaluation_data is None:
            k_evaluation_data = self._load_k_evaluation_data()

        if k_evaluation_data is None or len(k_evaluation_data) < 2:
            print("  [SKIP] topic_num_evaluation (need evaluation results for at least 2 different K values)")
            print("         To generate this chart, run training with different topic numbers (K)")
            print("         and save evaluation results, then re-run visualization.")
            return

        k_values = sorted(k_evaluation_data)
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
        for ax, key, label, color in zip(axes, ['coherence','exclusivity','perplexity'],
                                        ['NPMI','Exclusivity','Perplexity'], COLORS):
            values = [k_evaluation_data[k].get(key, np.nan) for k in k_values]
            ax.plot(k_values, values, 'o-', color=color, markersize=3, linewidth=1)
            if current_k in k_values: ax.axvline(current_k, color='#999999', linestyle='--', linewidth=.6)
            ax.set_xlabel('K'); ax.set_ylabel(label); ax.set_xticks(k_values)
        pd.DataFrame.from_dict(k_evaluation_data,orient='index').sort_index().to_csv(self.global_dir/'k_evaluation.csv',index_label='K')
        self._save(self.global_dir / self._get_filename('topic_num_evaluation'), dpi=self.dpi)
        plt.close(fig)

    def _load_k_evaluation_data(self) -> dict:
        """Load evaluation data for different K values from evaluation directory.

        Looks for evaluation JSON files in the parent evaluation directory
        with format: evaluation_k{K}.json or in subdirectories named k{K}/

        Returns:
            Dict with K values as keys and metrics as values, or None if not found.
        """
        import json

        # Try to find evaluation directory (go up from visualization output dir)
        eval_dir = None
        current_dir = self.output_dir

        # Navigate up to find evaluation directory
        for _ in range(5):  # Max 5 levels up
            parent = current_dir.parent
            potential_eval = parent / 'evaluation'
            if potential_eval.exists():
                eval_dir = potential_eval
                break
            # Also check for k_evaluation subdirectory
            potential_k_eval = parent / 'k_evaluation'
            if potential_k_eval.exists():
                eval_dir = potential_k_eval
                break
            current_dir = parent

        if eval_dir is None:
            return None

        k_data = {}

        # Pattern 1: Look for evaluation_k{K}.json files
        for eval_file in eval_dir.glob('evaluation_k*.json'):
            try:
                k_str = eval_file.stem.replace('evaluation_k', '')
                k = int(k_str)
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                k_data[k] = {
                    'coherence': data.get('topic_coherence_npmi_avg', data.get('coherence', 0)),
                    'exclusivity': data.get('topic_exclusivity_avg', data.get('exclusivity', 0)),
                    'perplexity': data.get('perplexity', data.get('ppl', 0))
                }
            except (ValueError, json.JSONDecodeError):
                continue

        # Pattern 2: Look for k{K}/ subdirectories with evaluation.json
        for k_dir in eval_dir.glob('k*'):
            if k_dir.is_dir():
                try:
                    k_str = k_dir.name.replace('k', '')
                    k = int(k_str)
                    eval_file = k_dir / 'evaluation.json'
                    if eval_file.exists():
                        with open(eval_file, 'r') as f:
                            data = json.load(f)
                        k_data[k] = {
                            'coherence': data.get('topic_coherence_npmi_avg', data.get('coherence', 0)),
                            'exclusivity': data.get('topic_exclusivity_avg', data.get('exclusivity', 0)),
                            'perplexity': data.get('perplexity', data.get('ppl', 0))
                        }
                except (ValueError, json.JSONDecodeError):
                    continue

        # Pattern 3: Look in parent's sibling directories (for different K trainings)
        # e.g., /result/0.6B/FCPB/unsupervised_k10/, /result/0.6B/FCPB/unsupervised_k20/
        mode_dir = self.output_dir
        for _ in range(4):
            mode_dir = mode_dir.parent
            if mode_dir.name in ['unsupervised', 'supervised', 'zero_shot']:
                break

        if mode_dir.parent.exists():
            base_mode = mode_dir.name
            for sibling in mode_dir.parent.glob(f'{base_mode}_k*'):
                if sibling.is_dir():
                    try:
                        k_str = sibling.name.replace(f'{base_mode}_k', '')
                        k = int(k_str)
                        eval_file = sibling / 'evaluation' / 'evaluation.json'
                        if eval_file.exists():
                            with open(eval_file, 'r') as f:
                                data = json.load(f)
                            k_data[k] = {
                                'coherence': data.get('topic_coherence_npmi_avg', data.get('coherence', 0)),
                                'exclusivity': data.get('topic_exclusivity_avg', data.get('exclusivity', 0)),
                                'perplexity': data.get('perplexity', data.get('ppl', 0))
                            }
                    except (ValueError, json.JSONDecodeError):
                        continue

        return k_data if k_data else None

    def generate_sankey_diagram(self):
        """Observed group-to-topic allocations, never inferred cross-year migrations."""
        scopes=[]
        if self.timestamps is not None:
            valid=~pd.isna(self.timestamps)
            if valid.any():
                years=np.array([str(t.year) for t in np.asarray(self.timestamps)[valid]])
                scopes.append(('year',years,self.theta[valid],sorted(set(years))))
        if self.dimension_values is not None:
            groups=np.asarray([str(g) for g in self.dimension_values])
            scopes.append(('source',groups,self.theta,list(pd.Series(groups).value_counts().index)))
        if not scopes:
            print('[SKIP] sankey_diagram: needs verified dates or source labels')
            return
        for kind,groups,theta,labels in scopes:
            if not np.isfinite(theta).all() or (theta<0).any() or theta.sum()<=0:
                raise ValueError('Sankey allocation requires finite nonnegative topic weights')
            mass=np.array([theta[groups==group].sum(axis=0,dtype=np.float64) for group in labels])
            counts=np.array([(groups==group).sum() for group in labels])
            title=('年份 → 主题' if kind=='year' else '来源 → 主题') if self.language=='zh' else ('Year → topic' if kind=='year' else 'Source → topic')
            title+=(' · 权重分配桑基图' if self.language=='zh' else ' · weight allocation Sankey')
            records=[{'group':group,'n_documents':int(counts[i]),'topic_id':j+1,'weight_mass':float(mass[i,j])}
                     for i,group in enumerate(labels) for j in range(self.n_topics)]
            pd.DataFrame(records).to_csv(self.global_dir/f'{kind}_topic_sankey.csv',index=False)
            self._generate_sankey_matplotlib(mass,labels,counts,title,kind)
            # Preserve the repository's interactive Plotly delivery, with real link masses.
            import plotly.graph_objects as go
            source=[];target=[];values=[];colors=[]
            for i in range(len(labels)):
                for j in range(self.n_topics):
                    if mass[i,j]>0:
                        source.append(i);target.append(len(labels)+j);values.append(float(mass[i,j]))
                        r,g,b=matplotlib.colors.to_rgb(COLORS[j%len(COLORS)])
                        colors.append(f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.45)')
            nodes=[f'{g} (n={n:,})' for g,n in zip(labels,counts)]+[f'T{j+1}' for j in range(self.n_topics)]
            # Keep the same year/topic order as the static figure; Plotly's auto layout reorders both.
            gap=min(.045,.5/max(max(mass.shape)-1,1));scale=1-gap*(max(mass.shape)-1)
            centers=[]
            for heights in [mass.sum(axis=1)/mass.sum()*scale,mass.sum(axis=0)/mass.sum()*scale]:
                occupied=heights.sum()+gap*(len(heights)-1)
                centers.extend(((1-occupied)/2+np.r_[0,np.cumsum(heights[:-1]+gap)]+heights/2).tolist())
            fig=go.Figure(go.Sankey(arrangement='fixed',node=dict(label=nodes,pad=gap*(800-90-30),thickness=15,
                          x=[.02]*len(labels)+[.98]*self.n_topics,y=centers,
                          color=['#96A6B1']*len(labels)+[COLORS[j%len(COLORS)] for j in range(self.n_topics)]),
                          link=dict(source=source,target=target,value=values,color=colors,
                                    hovertemplate='%{source.label} → %{target.label}<br>Weight mass: %{value:.4f}<extra></extra>')))
            note='每条带表示同组文档的主题权重合计，不表示跨期迁移。' if self.language=='zh' else 'Links sum topic weights within a group; they do not represent migration.'
            fig.update_layout(title=title+'<br><sup>'+note+'</sup>',font_size=12,height=800,paper_bgcolor='white',margin=dict(l=30,r=30,t=90,b=30))
            path=self.global_dir/f'{kind}_topic_sankey.html';fig.write_html(str(path),include_plotlyjs=True)
            self.exported_files.append(str(path))

    def _generate_sankey_matplotlib(self,mass,labels,counts,title,kind):
        """Native Bezier ribbons, with identical conserved mass on both sides."""
        from matplotlib.path import Path as MplPath
        from matplotlib.patches import PathPatch,Rectangle
        from textwrap import fill
        total=mass.sum();flow=mass/total
        gap=min(.045,.5/max(max(mass.shape)-1,1));scale=1-gap*(max(mass.shape)-1)
        left=flow.sum(axis=1)*scale;right=flow.sum(axis=0)*scale
        def positions(heights):
            occupied=heights.sum()+gap*(len(heights)-1)
            return (1+occupied)/2-np.r_[0,np.cumsum(heights[:-1]+gap)]
        ly=positions(left);ry=positions(right);lo=ly.copy();ro=ry.copy()
        fig,ax=plt.subplots(figsize=(7.2,max(4,min(6.5,len(labels)*.4))))
        for i in range(len(labels)):
            for j in range(self.n_topics):
                width=flow[i,j]*scale
                if width<=0:continue
                y1,y2=lo[i],ro[j]
                vertices=[(.025,y1),(.4,y1),(.6,y2),(.975,y2),(.975,y2-width),(.6,y2-width),(.4,y1-width),(.025,y1-width),(.025,y1)]
                path=MplPath(vertices,[MplPath.MOVETO]+[MplPath.CURVE4]*3+[MplPath.LINETO]+[MplPath.CURVE4]*3+[MplPath.CLOSEPOLY])
                ax.add_patch(PathPatch(path,facecolor=COLORS[j%len(COLORS)],edgecolor='none',alpha=.40))
                lo[i]-=width;ro[j]-=width
        for i,label in enumerate(labels):
            ax.add_patch(Rectangle((0,ly[i]-left[i]),.025,left[i],facecolor='#738897',edgecolor='none'))
            ax.text(-.025,ly[i]-left[i]/2,fill(str(label),16)+f' · n={counts[i]:,}',ha='right',va='center',fontsize=7)
        terms=dict(self.topic_words)
        for j in range(self.n_topics):
            ax.add_patch(Rectangle((.975,ry[j]-right[j]),.025,right[j],facecolor=COLORS[j%len(COLORS)],edgecolor='none'))
            label=f'T{j+1}  '+ ' · '.join(w for w,_ in terms.get(j,[])[:2])
            ax.text(1.025,ry[j]-right[j]/2,label+f'  {flow[:,j].sum():.1%}',va='center',fontsize=7)
        ax.set_xlim(-.38,1.65);ax.set_ylim(-.06,1.06);ax.axis('off');ax.set_title(title,pad=15)
        note='带宽 = 主题权重合计；不表示跨期迁移。' if self.language=='zh' else 'Width = sum of topic weights; not cross-period migration.'
        ax.text(0,-.025,note,transform=ax.transAxes,fontsize=7,color='#667C8A')
        self._save(self.global_dir/f'{kind}_topic_sankey.png',dpi=self.dpi);plt.close(fig)

    # ========== MAIN GENERATION ==========

    def generate_all(self):
        """Generate all visualizations."""
        print(f"\n{'='*60}")
        print(f"Generating visualizations ({self.language.upper()})")
        print(f"Output: {self.output_dir}")
        print(f"Topics: {self.n_topics}")
        print(f"Data available:")
        print(f"  - timestamps: {'Yes' if self.timestamps is not None else 'No'}")
        print(f"  - bow_matrix: {'Yes' if self.bow_matrix is not None else 'No'}")
        print(f"  - training_history: {'Yes' if self.training_history is not None else 'No'}")
        print(f"  - metrics: {'Yes' if self.metrics is not None else 'No'}")
        print(f"{'='*60}")

        chart_count = 0
        temporal = None
        if self.timestamps is not None:
            from copy import copy
            valid = ~pd.isna(self.timestamps)
            if valid.any():
                temporal = copy(self)
                temporal.theta = self.theta[valid]
                temporal.timestamps = np.asarray(self.timestamps)[valid]
                temporal.n_docs = int(valid.sum())
                if self.bow_matrix is not None: temporal.bow_matrix = self.bow_matrix[valid]
                if self.dimension_values is not None: temporal.dimension_values = np.asarray(self.dimension_values)[valid]
            (self.output_dir / 'temporal-scope.json').write_text(json.dumps({
                'allDocuments': self.n_docs, 'datedDocuments': int(valid.sum()),
                'excludedMissingDate': int((~valid).sum()),
                'note': 'Only temporal charts exclude missing dates; global charts use all model rows.'}, indent=2))


        statuses = []
        def safe_generate(func, name):
            if not selected_chart(name):
                return
            from contextlib import redirect_stdout
            from io import StringIO
            before = len(self.exported_files)
            output = StringIO()
            try:
                with redirect_stdout(output): func()
                created = self.exported_files[before:]
                statuses.append({'chart': name, 'status': 'generated' if created else 'skipped',
                                 'files': created, 'detail': output.getvalue().strip()})
            except Exception as exc:
                statuses.append({'chart': name, 'status': 'failed', 'detail': str(exc)})
                plt.close('all')
                print(f'  [Error] {name}: {exc}')
            print(output.getvalue(), end='')

        # ========== Basic Global Charts (always available) ==========
        print(f"\n[Basic Global Charts]")
        safe_generate(self.generate_topic_table, 'topic_table')
        safe_generate(self.generate_topic_network, 'topic_network')
        safe_generate(lambda: self.generate_topic_network(layout='circular'), 'topic_network_circular')
        safe_generate(self.generate_doc_clusters, 'doc_clusters')
        safe_generate(self.generate_clustering_heatmap, 'clustering_heatmap')
        safe_generate(self.generate_clusters_with_outliers, 'clusters_with_outliers')
        safe_generate(self.generate_topic_proportion_pie, 'topic_proportion_pie')
        safe_generate(self.generate_sankey_diagram, 'sankey_diagram')

        # ========== Temporal Global Charts (need timestamps) ==========
        print(f"\n[Temporal Global Charts]")
        if temporal is not None:
            safe_generate(temporal.generate_doc_volume, 'doc_volume')
            safe_generate(temporal.generate_representative_topic_evolution, 'representative_topic_evolution')
            safe_generate(temporal.generate_kl_divergence, 'kl_divergence')
            if self.bow_matrix is not None:
                safe_generate(temporal.generate_vocab_evolution, 'vocab_evolution')
            safe_generate(temporal.generate_topic_similarity_evolution, 'topic_similarity_evolution')
            safe_generate(temporal.generate_all_topics_strength_table, 'all_topics_strength_table')
        else:
            statuses.append({'chart':'temporal_charts','status':'skipped','detail':'No verified dates: yearly volume, topic trends and yearly heatmap are unavailable.'})

        # ========== Dimension Charts (need dimension_values) ==========
        print(f"\n[Dimension Charts]")
        if self.dimension_values is not None:
            safe_generate(self.generate_dimension_heatmap, 'dimension_heatmap')
            if temporal is not None:
                safe_generate(temporal.generate_domain_topic_distribution, 'domain_topic_distribution')
        else:
            statuses.append({'chart':'group_charts','status':'skipped','detail':'No verified source/group labels: group heatmap and trends are unavailable.'})

        # ========== Training & Metrics Charts ==========
        print(f"\n[Training & Metrics Charts]")
        if self.training_history is not None:
            safe_generate(self.generate_training_convergence, 'training_convergence')
        else:
            statuses.append({'chart':'training_convergence','status':'skipped','detail':'No saved training history; loss and perplexity curves cannot be reconstructed.'})
        if self.metrics is not None:
            safe_generate(self.generate_topic_coherence_chart, 'topic_coherence_chart')
            safe_generate(self.generate_topic_diversity_chart, 'topic_diversity_chart')
            safe_generate(self.generate_7_core_metrics_chart, '7_core_metrics_chart')
            safe_generate(self.generate_topic_significance_chart, 'topic_significance_chart')
        safe_generate(self.generate_topic_num_evaluation, 'topic_num_evaluation')

        # ========== Per-Topic Charts ==========
        print(f"\n[Per-Topic Charts]")
        for topic_idx in range(self.n_topics):
            print(f"  Topic {topic_idx + 1}/{self.n_topics}...", end=' ')
            # Skip word_importance chart (duplicate of topic word distribution from TopicVisualizer)
            # try:
            #     self.generate_topic_word_importance(topic_idx)
            #     chart_count += 1
            # except Exception as e:
            #     print(f"[Error] word_importance: {e}", end=' ')
            if temporal is not None:
                for name in ['generate_topic_evolution', 'generate_topic_word_dist_change', 'generate_topic_word_sense']:
                    safe_generate(lambda name=name, topic_idx=topic_idx: getattr(temporal, name)(topic_idx), f'{name}:T{topic_idx+1}')
            print("✓")

        print(f"\n{'='*60}")
        (self.output_dir / 'chart-status.json').write_text(json.dumps(statuses, ensure_ascii=False, indent=2))
        print(f"Done! Actual PNG/PDF/SVG files: {sum(1 for f in self.output_dir.rglob('*') if f.suffix in {'.png', '.pdf', '.svg'})}")
        print(f"{'='*60}")

        if any(item['status'] == 'failed' for item in statuses):
            raise RuntimeError('One or more visualizations failed; inspect chart-status.json')


def load_model_data(model_dir, bow_dir=None, result_dir=None):
    """
    Load model data from directory.

    Args:
        model_dir: Directory containing ETM model outputs (theta, beta, topic_words, etc.)
        bow_dir: Directory containing BOW data (bow_matrix.npz, vocab.txt).
                 If None, will try to find it relative to model_dir.
        result_dir: Base result directory for timestamps.npy.
                    If None, will try to find it relative to model_dir.

    Returns:
        dict with keys: theta, beta, topic_embeddings, topic_words, vocab,
                       bow_matrix, timestamps, config
    """
    from scipy import sparse

    model_dir = Path(model_dir)

    def find_latest(directory, pattern):
        files = list(Path(directory).glob(pattern))
        return max(files, key=lambda x: x.stat().st_mtime) if files else None

    data = {}

    # ========== Load from model_dir ==========
    # Load theta
    theta_file = find_latest(model_dir, "theta_*.npy")
    if theta_file:
        data['theta'] = np.load(theta_file)
        print(f"  Loaded theta: {data['theta'].shape}")

    # Load beta
    beta_file = find_latest(model_dir, "beta_*.npy")
    if beta_file:
        data['beta'] = np.load(beta_file)
        print(f"  Loaded beta: {data['beta'].shape}")

    # Load topic embeddings
    emb_file = find_latest(model_dir, "topic_embeddings_*.npy")
    if emb_file:
        data['topic_embeddings'] = np.load(emb_file)
        print(f"  Loaded topic_embeddings: {data['topic_embeddings'].shape}")

    # Load topic words
    words_file = find_latest(model_dir, "topic_words_*.json")
    if words_file:
        with open(words_file, 'r', encoding='utf-8') as f:
            topic_words_dict = json.load(f)
        data['topic_words'] = [
            (int(k), [(item[0], item[1]) for item in v])
            for k, v in sorted(topic_words_dict.items(), key=lambda x: int(x[0]))
        ]
        print(f"  Loaded topic_words: {len(data['topic_words'])} topics")

    # Load config
    config_file = find_latest(model_dir, "config_*.json")
    if config_file:
        with open(config_file, 'r', encoding='utf-8') as f:
            data['config'] = json.load(f)
        print(f"  Loaded config")

    # Load training history
    history_file = find_latest(model_dir, "training_history_*.json")
    if history_file:
        with open(history_file, 'r', encoding='utf-8') as f:
            data['training_history'] = json.load(f)
        print(f"  Loaded training_history")

    # ========== Load from bow_dir ==========
    # Try to find bow_dir if not specified
    if bow_dir is None:
        # model_dir is typically: result/{dataset}/{mode}/model
        # bow_dir is typically: result/{dataset}/bow
        possible_bow_dir = model_dir.parent.parent / 'bow'
        if possible_bow_dir.exists():
            bow_dir = possible_bow_dir

    if bow_dir and Path(bow_dir).exists():
        bow_dir = Path(bow_dir)

        # Load BOW matrix
        bow_file = bow_dir / 'bow_matrix.npy'
        if bow_file.exists():
            data['bow_matrix'] = np.load(bow_file)
            print(f"  Loaded bow_matrix: {data['bow_matrix'].shape}")

        # Load vocab (real vocabulary, not word_0, word_1, ...)
        vocab_file = bow_dir / 'vocab.txt'
        if vocab_file.exists():
            with open(vocab_file, 'r', encoding='utf-8') as f:
                data['vocab'] = [line.strip() for line in f.readlines()]
            print(f"  Loaded vocab: {len(data['vocab'])} words")

        # Load vocab embeddings
        vocab_emb_file = bow_dir / 'vocab_embeddings.npy'
        if vocab_emb_file.exists():
            data['vocab_embeddings'] = np.load(vocab_emb_file)
            print(f"  Loaded vocab_embeddings: {data['vocab_embeddings'].shape}")

    # ========== Load from result_dir ==========
    # Try to find result_dir if not specified
    if result_dir is None:
        # model_dir is typically: result/{dataset}/{mode}/model
        # result_dir is typically: result/{dataset}/{mode}
        possible_result_dir = model_dir.parent
        if possible_result_dir.exists():
            result_dir = possible_result_dir

    if result_dir and Path(result_dir).exists():
        result_dir = Path(result_dir)

        # Load timestamps
        ts_file = result_dir / 'timestamps.npy'
        if ts_file.exists():
            data['timestamps'] = np.load(ts_file, allow_pickle=True)
            print(f"  Loaded timestamps: {len(data['timestamps'])} dates")

    # ========== Fallback for vocab ==========
    # If vocab not loaded from bow_dir, generate placeholder
    if 'vocab' not in data and 'beta' in data:
        data['vocab'] = [f"word_{i}" for i in range(data['beta'].shape[1])]
        print(f"  Generated placeholder vocab: {len(data['vocab'])} words")

    return data


def load_complete_data(dataset_dir):
    """
    Load complete data from a dataset directory structure.

    Expected structure:
        dataset_dir/
        ├── model/          # theta, beta, topic_words, config, etc.
        ├── bow/            # bow_matrix.npz, vocab.txt
        ├── evaluation/     # metrics.json
        └── timestamps.npy  # (optional)

    Args:
        dataset_dir: Path to dataset directory (e.g., real_data/hatespeech_supervised)
                     or mode directory (e.g., result/hatespeech/supervised)

    Returns:
        dict with all available data
    """
    from scipy import sparse

    dataset_dir = Path(dataset_dir)
    data = {}

    print(f"\nLoading data from: {dataset_dir}")

    # Determine directory structure
    model_dir = dataset_dir / 'model'
    bow_dir = dataset_dir / 'bow'
    evaluation_dir = dataset_dir / 'evaluation'

    # If model_dir doesn't exist, check if this is already the model dir
    if not model_dir.exists() and (dataset_dir / 'theta_*.npy').exists():
        model_dir = dataset_dir
        bow_dir = dataset_dir.parent / 'bow'

    # Load model data
    if model_dir.exists():
        model_data = load_model_data(model_dir, bow_dir, dataset_dir)
        data.update(model_data)

    # Load evaluation metrics
    if evaluation_dir.exists():
        metrics_files = list(evaluation_dir.glob('metrics_*.json'))
        if metrics_files:
            latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
            with open(latest_metrics, 'r', encoding='utf-8') as f:
                data['metrics'] = json.load(f)
            print(f"  Loaded metrics")

    # Summary
    print(f"\nData loaded:")
    for key in data:
        if isinstance(data[key], np.ndarray):
            print(f"  {key}: {data[key].shape}")
        elif isinstance(data[key], list):
            print(f"  {key}: {len(data[key])} items")
        elif hasattr(data[key], 'shape'):
            print(f"  {key}: {data[key].shape}")
        else:
            print(f"  {key}: loaded")

    return data


if __name__ == "__main__":
    # Example usage
    print("VisualizationGenerator - Example Usage")
    print("="*60)
    print("""
    # Method 1: Load from model directory only (basic)
    from visualization_generator import VisualizationGenerator, load_model_data

    data = load_model_data('/path/to/model')

    generator = VisualizationGenerator(
        theta=data['theta'],
        beta=data['beta'],
        vocab=data['vocab'],
        topic_words=data['topic_words'],
        output_dir='./visualization',
        language='zh',
        dpi=600
    )
    generator.generate_all()

    # Method 2: Load complete data (recommended)
    from visualization_generator import VisualizationGenerator, load_complete_data

    # This loads: theta, beta, vocab, topic_words, topic_embeddings,
    #             bow_matrix, timestamps, training_history, metrics
    data = load_complete_data('/path/to/dataset')

    generator = VisualizationGenerator(
        theta=data['theta'],
        beta=data['beta'],
        vocab=data['vocab'],
        topic_words=data['topic_words'],
        topic_embeddings=data.get('topic_embeddings'),
        timestamps=data.get('timestamps'),        # For temporal charts
        bow_matrix=data.get('bow_matrix'),        # For vocab evolution
        training_history=data.get('training_history'),  # For convergence curve
        metrics=data.get('metrics'),              # For coherence/diversity charts
        dimension_values=None,                    # User-provided if available
        output_dir='./visualization',
        language='zh',  # or 'en'
        dpi=600
    )
    generator.generate_all()

    # Available Charts:
    # ================
    # Basic Global (always available):
    #
    # Temporal Global (need timestamps):
    #
    # Dimension (need dimension_values):
    #
    # Training & Metrics:
    #
    # Per-Topic:
    """)
