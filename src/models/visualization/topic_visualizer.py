"""
Topic Visualization Tools for ETM

This module provides visualization tools for ETM results:
- Topic word clouds
- Topic similarity heatmap
- Document-topic distribution visualization
- Topic evolution over time (if timestamps available)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
import logging

# Try to import wordcloud, but don't fail if it's not available
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    logging.warning("WordCloud package not available. Install with 'pip install wordcloud'")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
import matplotlib
from visualization.publication import COLORS, setup_style, save_figure, validate_export, selected_figure

logger = logging.getLogger(__name__)


def draw_document_projection(coords, topics, *, method, language, total_count, title=None):
    """Show every sampled point plus an explicitly bounded detail view; never move points."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    coords, topics = np.asarray(coords), np.asarray(topics)
    q1, q3 = np.quantile(coords, [.25, .75], axis=0)
    low = np.maximum(coords.min(axis=0), q1-1.5*(q3-q1))
    high = np.minimum(coords.max(axis=0), q3+1.5*(q3-q1))
    span = high - low
    low -= np.where(span > 0, 0, .5)
    high += np.where(span > 0, 0, .5)
    inside = np.all((coords >= low) & (coords <= high), axis=1)
    zoom = not inside.all()
    unique = np.unique(topics)
    legend_height = max(0, int(np.ceil(len(unique) / 6)) - 2) * .20
    fig, axes = plt.subplots(1,2 if zoom else 1,figsize=(7.15,3.5 + legend_height) if zoom else (5.4,3.7 + legend_height),
                             width_ratios=[1,1.8] if zoom else [1],layout='constrained',squeeze=False)
    axes=axes.ravel()
    for ax in axes:
        for topic in unique:
            mask = topics == topic
            ax.scatter(*coords[mask].T, s=5 if len(coords)>1000 else 20, alpha=.65 if len(coords)>1000 else .85, linewidths=0,
                       c=COLORS[int(topic) % len(COLORS)] if topic >= 0 else '#9BA3A8', rasterized=True)
        ax.set_xlabel(f'{method} 1'); ax.set_ylabel(f'{method} 2')
        ax.locator_params(nbins=4)
    if zoom:
        axes[0].add_patch(Rectangle(low, *(high-low), fill=False, edgecolor='#333333', linewidth=.8, linestyle='--'))
        axes[1].set_xlim(low[0], high[0]); axes[1].set_ylim(low[1], high[1])
        axes[0].set_title('全景')
        axes[1].set_title(f'局部 · 文档数={inside.sum():,}')
    handles = [Line2D([], [], marker='o', linestyle='', color=COLORS[int(t)%len(COLORS)] if t>=0 else '#9BA3A8',
               label=f'T{int(t)+1}' if t>=0 else ('未分配 / 诊断离群' if language=='zh' else 'Unassigned / diagnostic noise')) for t in unique]
    fig.legend(handles=handles, loc='outside lower center', ncol=min(6,len(handles)), fontsize=7)
    note = '局部：四分位范围外扩 1.5×IQR；全景保留全部点' if language=='zh' else 'Detail: axis-wise quartiles ± 1.5×IQR; overview retains all points'
    default_title = f'文档主题投影 · {method} · 文档 {len(coords):,}/{total_count:,} · 随机种子 42' if language == 'zh' else f'{method} · n={len(coords):,}/{total_count:,} · seed=42'
    fig.suptitle((title or default_title)+('\n'+note if zoom else ''), fontsize=9, ha='left', x=.02)
    return fig


class TopicVisualizer:
    """
    Visualization tools for ETM results.
    
    Provides methods to visualize:
    - Topic word clouds
    - Topic similarity heatmap
    - Document-topic distribution
    - Topic embeddings in 2D space
    """
    
    def __init__(
        self,
        output_dir: str = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        cmap: str = "viridis",
        random_state: int = 42,
        language: str = 'en',
        formats=('png', 'pdf', 'svg')
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            figsize: Default figure size
            dpi: Default figure DPI
            cmap: Default colormap
            random_state: Random state for reproducibility
            language: Language for labels ('en' or 'zh')
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
        self.formats = validate_export(dpi, formats)
        self.cmap = cmap
        self.random_state = random_state
        self.language = language
        
        self._labels = {
            'topic': {'en': 'Topic', 'zh': '主题'},
            'word': {'en': 'Word', 'zh': '词'},
            'value': {'en': 'Value', 'zh': '值'},
            'frequency': {'en': 'Frequency', 'zh': '频率'},
            'similarity': {'en': 'Similarity', 'zh': '相似度'},
            'proportion': {'en': 'Proportion', 'zh': '比例'},
            'average_proportion': {'en': 'Average Proportion', 'zh': '平均比例'},
            'epoch': {'en': 'Epoch', 'zh': '轮次'},
            'loss': {'en': 'Loss', 'zh': '损失'},
            'perplexity': {'en': 'Perplexity', 'zh': '困惑度'},
            'train_loss': {'en': 'Train Loss', 'zh': '训练损失'},
            'val_loss': {'en': 'Val Loss', 'zh': '验证损失'},
            'train_perplexity': {'en': 'Train Perplexity', 'zh': '训练困惑度'},
            'val_perplexity': {'en': 'Validation Perplexity', 'zh': '验证困惑度'},
            'top_words_per_topic': {'en': 'Top 10 Words per Topic', 'zh': '每个主题的前10个词'},
            'topic_word_clouds': {'en': 'Topic Word Clouds', 'zh': '主题词云'},
            'etm_metrics': {'en': 'Evaluation Metrics', 'zh': '评估指标'},
            'topic_similarity_matrix': {'en': 'Topic Similarity Matrix', 'zh': '主题相似度矩阵'},
            'doc_topic_dist': {'en': 'Document-topic distribution', 'zh': '文档-主题分布'},
            'topic_proportions': {'en': 'Topic Proportions Across All Documents', 'zh': '所有文档的主题比例'},
            'train_val_loss': {'en': 'Training & Validation Loss', 'zh': '训练与验证损失'},
            'perplexity_training': {'en': 'Perplexity During Training', 'zh': '训练过程中的困惑度'},
            'intertopic_distance': {'en': 'Intertopic Distance Map', 'zh': '主题间距离图'},
            'via_mds': {'en': '(via multidimensional scaling)', 'zh': '(通过多维缩放)'},
            'marginal_topic_dist': {'en': 'Marginal topic distribution', 'zh': '边际主题分布'},
            'top_salient_terms': {'en': 'Most Salient Terms', 'zh': '最显著词汇'},
            'overall_term_freq': {'en': 'Overall term frequency', 'zh': '整体词频'},
            'estimated_term_freq': {'en': 'Estimated term frequency within the selected topic', 'zh': '所选主题内的估计词频'},
            'diversity_td': {'en': 'Diversity (TD)', 'zh': '多样性 (TD)'},
            'diversity_irbo': {'en': 'Diversity (iRBO)', 'zh': '多样性 (iRBO)'},
            'coherence_npmi': {'en': 'Coherence (NPMI)', 'zh': '一致性 (NPMI)'},
            'coherence_cv': {'en': 'Coherence (C_V)', 'zh': '一致性 (C_V)'},
            'exclusivity': {'en': 'Exclusivity', 'zh': '排他性'},
        }
        
        # Set plot style
        
        self._setup_chinese_fonts()

    def _setup_chinese_fonts(self):
        self.chinese_font_path = setup_style(self.language)

    def _save_figure(self, fig, filename, **kwargs):
        return save_figure(fig, filename, formats=self.formats, **kwargs)
    
    def _get_label(self, key: str) -> str:
        """获取双语标签"""
        if key in self._labels:
            return self._labels[key].get(self.language, self._labels[key].get('en', key))
        return key
    
    def _get_topic_label(self, topic_idx: int, short: bool = False) -> str:
        """获取主题标签"""
        if self.language == 'zh':
            return f"主题{topic_idx + 1}" if short else f"主题 {topic_idx + 1}"
        else:
            return f"T{topic_idx + 1}" if short else f"Topic {topic_idx + 1}"
        
    def _save_or_show(self, fig, filename=None):
        """Save figure to file or show it"""
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            filepath = self._save_figure(fig, filepath, dpi=self.dpi, bbox_inches='tight')[0]
            plt.close(fig)
            logger.info(f"Figure saved to {filepath}")
            return filepath
        else:
            plt.show()
            return None
    
    def _make_wordcloud(self, frequencies, num_words, topic_idx=0, color_func=None, shape="ellipse"):
        """One native layout for model, legacy and combined word-cloud entry points."""
        frequencies = {word: float(weight) for word, weight in frequencies.items()
                       if word.strip() and np.isfinite(weight) and weight > 0}
        if not frequencies:
            raise ValueError('Word cloud requires positive, finite word weights')
        ranked = sorted(frequencies, key=frequencies.get, reverse=True)
        ranks = {word: rank for rank, word in enumerate(ranked)}
        accent = np.array(matplotlib.colors.to_rgb(COLORS[topic_idx % len(COLORS)]))
        def topic_color(word, **kwargs):
            rank = ranks[word]
            color = accent * .70 if rank < 3 else accent if rank < 8 else accent*.60 + np.array([.48,.52,.55])*.40
            return matplotlib.colors.to_hex(color)
        font = self.chinese_font_path
        # Prefer the installed medium face where available, without downloading fonts.
        if font:
            medium = Path(font).with_name(Path(font).name.replace('Light', 'Medium'))
            if medium.is_file(): font = str(medium)
        if shape not in {'ellipse','rectangle'}: raise ValueError('shape must be ellipse or rectangle')
        y, x = np.ogrid[-1:1:620j, -1:1:1100j]
        mask = np.where(x*x + y*y > 1, 255, 0).astype(np.uint8)
        return WordCloud(mask=mask if shape=='ellipse' else None, width=1100, height=780, font_path=font, background_color='white',
                         prefer_horizontal=1.0 if shape=="ellipse" else .90, margin=12 if shape=="ellipse" else 4, max_font_size=280,
                         min_font_size=16, relative_scaling=.5, repeat=False,
                         scale=max(1, self.dpi/180), max_words=num_words,
                         random_state=self.random_state, color_func=color_func or topic_color
                         ).generate_from_frequencies(frequencies)

    def visualize_topic_words(
        self,
        topic_words: List[Tuple[int, List[Tuple[str, float]]]],
        num_topics: int = None,
        num_words: int = 10,
        as_wordcloud: bool = False,
        filename: str = None
    ) -> Union[plt.Figure, List[plt.Figure]]:
        """
        Visualize top words for each topic.
        
        Args:
            topic_words: List of (topic_idx, [(word, prob), ...])
            num_topics: Number of topics to visualize (None for all)
            num_words: Number of words per topic
            as_wordcloud: Whether to use word clouds
            filename: Filename to save visualization
            
        Returns:
            Figure or list of figures
        """
        if num_topics is None:
            num_topics = len(topic_words)
        else:
            num_topics = min(num_topics, len(topic_words))
        
        if as_wordcloud and not WORDCLOUD_AVAILABLE:
            logger.warning("WordCloud package not available, falling back to bar plots")
            as_wordcloud = False
        
        if as_wordcloud:
            # Create a word cloud for each topic
            figs = []
            for topic_idx, words in topic_words[:num_topics]:
                # Create word frequency dictionary
                word_freq = {word: prob for word, prob in words[:num_words*2]}
                
                # Create word cloud
                fig, ax = plt.subplots(figsize=(5.6, 3.5))
                wc = self._make_wordcloud(word_freq, num_words, topic_idx)

                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                
                figs.append(fig)
                
                # Save or show
                if filename:
                    base, ext = os.path.splitext(filename)
                    topic_filename = f"{base}_topic{topic_idx + 1}{ext}"
                    self._save_or_show(fig, topic_filename)
            
            return figs
        else:
            # Generate one individual chart per topic
            colors = [COLORS[i % len(COLORS)] for i in range(num_topics)]
            saved_paths = []
            
            for i, (topic_idx, words) in enumerate(topic_words[:num_topics]):
                fig, ax = plt.subplots(figsize=(5.2, 3.1), facecolor='white')
                
                top_words = [word for word, _ in words[:num_words]]
                top_probs = [prob for _, prob in words[:num_words]]
                
                y_pos = np.arange(len(top_words))
                ax.barh(y_pos, top_probs, align='center', color=COLORS[topic_idx % len(COLORS)], height=.55, linewidth=0)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([f'{j+1:02d}  {w}' for j,w in enumerate(top_words)], fontsize=9)
                for j, value in enumerate(top_probs):
                    ax.text(1.01, j, f'{value:.4g}', transform=ax.get_yaxis_transform(), va='center', fontsize=8)
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='y', length=0)
                ax.set_xlim(0, max(top_probs)*1.06)
                ax.invert_yaxis()
                ax.set_xlabel('词权重' if self.language == 'zh' else 'Word weight')
                ax.set_title(f'主题 {topic_idx + 1} · 主题词权重', loc='left', weight='bold')
                ax.tick_params(axis='x', labelsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                
                # Save to corresponding topic folder instead of global folder
                if self.language == 'zh':
                    topic_filename = f'主题{topic_idx + 1} 词语分布.png'
                else:
                    topic_filename = f'Topic {topic_idx + 1} Word Distribution.png'
                
                # Create topic directory path
                topic_dir = os.path.join(self.output_dir, '..', 'topic', f'topic_{topic_idx + 1}')
                os.makedirs(topic_dir, exist_ok=True)
                topic_filepath = os.path.join(topic_dir, topic_filename)
                
                self._save_figure(fig, topic_filepath, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Figure saved to {topic_filepath}")
                saved_paths.append(topic_filepath)
                plt.close(fig)
            
            return saved_paths
    
    def visualize_all_wordclouds(
        self,
        topic_words: List[Tuple[int, List[Tuple[str, float]]]],
        num_words: int = 30,
        filename: str = None
    ) -> List[str]:
        """
        Generate individual wordcloud figures for each topic.
        
        Args:
            topic_words: List of (topic_idx, [(word, prob), ...])
            num_words: Number of words per wordcloud
            filename: Filename to save visualization (ignored, individual files created)
            
        Returns:
            List of saved file paths
        """
        if not WORDCLOUD_AVAILABLE:
            logger.warning("WordCloud package not available")
            return []
        
        saved_paths = []
        
        for i, (topic_idx, words) in enumerate(topic_words):
            # Create individual figure for each topic
            fig, ax = plt.subplots(figsize=(5.6, 3.5), facecolor='white')
            
            word_freq = {word: prob for word, prob in words[:num_words] if word.strip() and np.isfinite(prob) and prob > 0}
            
            try:
                wc = self._make_wordcloud(word_freq, num_words, topic_idx)

                ax.imshow(wc, interpolation='bilinear')
                ax.set_title(f'主题 {topic_idx+1} · 主题关键词', fontsize=10)
            except Exception as e:
                plt.close(fig)
                raise ValueError(f'Word cloud T{topic_idx+1} failed: {e}') from e
            
            ax.axis('off')
            
            # Save to corresponding topic folder
            topic_dir = os.path.join(self.output_dir, '..', 'topic', f'topic_{topic_idx + 1}')
            os.makedirs(topic_dir, exist_ok=True)
            
            if self.language == 'zh':
                topic_filename = f'主题{topic_idx + 1} 词云.png'
            else:
                topic_filename = f'Topic {topic_idx + 1} Word Cloud.png'
            
            topic_filepath = os.path.join(topic_dir, topic_filename)
            self._save_figure(fig, topic_filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Word cloud saved to {topic_filepath}")
            saved_paths.append(topic_filepath)
            plt.close(fig)
        
        return saved_paths
    
    def visualize_wordcloud_grid(self, topic_words, num_words=80, columns=3, topics_per_page=6):
        """Rectangular topic panels, paginated with titles outside the word images."""
        if not WORDCLOUD_AVAILABLE:
            raise RuntimeError('WordCloud package is required')
        if not 1 <= columns <= 3 or not 1 <= topics_per_page <= 6:
            raise ValueError('Use 1–3 columns and 1–6 topics per page to preserve readable type')
        paths=[]
        for start in range(0,len(topic_words),topics_per_page):
            path=Path(self.output_dir)/f'topic_wordcloud_grid_{start//topics_per_page+1}.png'
            if not selected_figure(path):
                continue
            page=topic_words[start:start+topics_per_page]
            ncols=min(columns,len(page));nrows=int(np.ceil(len(page)/ncols))
            fig,axes=plt.subplots(nrows,ncols,figsize=(7.15,2.35*nrows+.45),squeeze=False,layout='constrained')
            for ax,(topic_idx,words) in zip(axes.flat,page):
                frequencies={word:weight for word,weight in words[:num_words]}
                cloud=self._make_wordcloud(frequencies,num_words,topic_idx,shape='rectangle')
                ax.imshow(cloud,interpolation='bilinear');ax.axis('off')
                # A separate two-line header stays outside the image's extent.
                from textwrap import fill
                ax.set_title(f'主题 {topic_idx+1}\n'+fill(' · '.join(w for w,_ in words[:2]),18),fontsize=8,pad=8)
            for ax in list(axes.flat)[len(page):]:ax.set_visible(False)
            fig.suptitle('各主题词云'+
                         f' · K={len(topic_words)} · {start+1}–{start+len(page)}',fontsize=10)
            paths.extend(self._save_figure(fig,path,dpi=self.dpi));plt.close(fig)
        return paths

    def visualize_combined_wordcloud(
        self,
        topic_words: List[Tuple[int, List[Tuple[str, float]]]],
        num_words: int = 50,
        filename: str = None
    ) -> plt.Figure:
        """
        Generate a single combined wordcloud with all topic words.
        
        Args:
            topic_words: List of (topic_idx, [(word, prob), ...])
            num_words: Number of words per topic to include
            filename: Filename to save visualization
            
        Returns:
            Figure with combined wordcloud
        """
        if not WORDCLOUD_AVAILABLE:
            logger.warning("WordCloud package not available")
            return None
        
        # Combine all words from all topics
        combined_freq = {}
        num_topics = len(topic_words)
        colors = [matplotlib.colors.to_rgba(COLORS[i % len(COLORS)]) for i in range(num_topics)]
        word_colors = {}
        
        for i, (topic_idx, words) in enumerate(topic_words):
            for word, prob in words[:num_words]:
                if word not in combined_freq:
                    combined_freq[word] = prob
                    word_colors[word] = matplotlib.colors.to_rgba(COLORS[topic_idx % len(COLORS)])
                else:
                    # Keep the higher probability and its color
                    if prob > combined_freq[word]:
                        combined_freq[word] = prob
                        word_colors[word] = matplotlib.colors.to_rgba(COLORS[topic_idx % len(COLORS)])
        
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            if word in word_colors:
                c = word_colors[word]
                return f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})"
            return "rgb(100, 100, 100)"
        
        fig, ax = plt.subplots(figsize=(6.4, 4), facecolor='white')
        
        try:
            wc = self._make_wordcloud(combined_freq, 300, color_func=color_func)

            ax.imshow(wc, interpolation='bilinear')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
        
        ax.axis('off')
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def visualize_metrics(
        self,
        metrics: Dict,
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize evaluation metrics as a bar chart.
        
        Args:
            metrics: Dictionary of metric names and values
            filename: Filename to save visualization
            
        Returns:
            Figure with metrics bar chart
        """
        aliases = {'TD': ['TD', 'topic_diversity_td'], 'iRBO': ['iRBO', 'topic_diversity_irbo'],
                   'NPMI': ['NPMI', 'topic_coherence_npmi_avg'], 'C_V': ['C_V', 'topic_coherence_cv_avg'],
                   'UMass': ['UMass', 'topic_coherence_umass_avg'],
                   'Exclusivity': ['Exclusivity', 'topic_exclusivity_avg'], 'PPL': ['PPL', 'perplexity']}
        items = []
        for label, keys in aliases.items():
            value = next((metrics[key] for key in keys if isinstance(metrics.get(key), (int, float))
                          and np.isfinite(metrics[key])), None)
            if value is not None: items.append((label, value))
        if not items: return None
        fig, axes = plt.subplots(1, len(items), figsize=(7.2, 1.2), squeeze=False)
        for ax, (name, value) in zip(axes[0], items):
            ax.axis('off')
            ax.text(.5, .75, name, ha='center', weight='bold')
            ax.text(.5, .35, f'{value:.4g}', ha='center')
        return self._save_or_show(fig, filename)

    def visualize_topic_similarity(
        self,
        beta: np.ndarray,
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        metric: str = 'cosine',
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize topic similarity as a heatmap.
        
        Args:
            beta: Topic-word distribution matrix (K x V)
            topic_words: Optional list of topic words for labels
            metric: Similarity metric ('cosine', 'euclidean', 'correlation')
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        num_topics = beta.shape[0]
        
        # Compute similarity matrix
        if metric == 'cosine':
            sim_matrix = cosine_similarity(beta)
        elif metric == 'euclidean':
            # Convert distances to similarities
            dist_matrix = euclidean_distances(beta)
            sim_matrix = 1 / (1 + dist_matrix)
        elif metric == 'correlation':
            sim_matrix = np.corrcoef(beta)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Create labels if topic_words is provided
        if topic_words:
            labels = []
            for topic_idx, words in topic_words:
                top_words = [word for word, _ in words[:3]]
                label = f"{topic_idx}: {', '.join(top_words)}"
                labels.append(label)
            labels = labels[:num_topics]
        else:
            labels = [self._get_topic_label(i) for i in range(num_topics)]
        
        # Adjust figure size and annotation based on number of topics
        if num_topics > 20:
            # For many topics, use larger figure and no annotations
            base_size = max(14, num_topics * 0.5)
            fig, ax = plt.subplots(figsize=(base_size, base_size - 2))
            show_annot = False
            fontsize = 8
        else:
            fig, ax = plt.subplots(figsize=(5.4, 4.6))
            show_annot = True
            fontsize = 10
        
        sns.heatmap(
            sim_matrix,
            annot=show_annot,
            fmt='.2f' if show_annot else '',
            annot_kws={'size': 8} if show_annot else {},
            cmap='RdBu_r' if metric == 'correlation' else 'Blues',
            xticklabels=[self._get_topic_label(i, short=True) for i in range(num_topics)],
            yticklabels=[self._get_topic_label(i, short=True) for i in range(num_topics)],
            ax=ax,
            vmin=-1 if metric == 'correlation' else 0,
            vmax=1,
            linewidths=0.3 if num_topics > 20 else 0.5,
            square=True,
            cbar_kws={'shrink': 0.8, 'label': self._get_label('similarity')}
        )
        # Remove title as requested
        # ax.set_title(f"{self._get_label('topic_similarity_matrix')} ({metric.title()})", fontsize=16, fontweight='bold')
        
        # Rotate x-axis labels for readability
        ax.set_title(f'主题词相似度 · { {"cosine": "余弦", "jaccard": "杰卡德", "correlation": "相关系数"}.get(metric, metric)}')
        plt.xticks(rotation=0, fontsize=fontsize)
        plt.yticks(rotation=0, fontsize=fontsize)
        
        # Save or show
        return self._save_or_show(fig, filename)
    
    def visualize_document_topics(
        self,
        theta: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = 'umap',
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        max_docs: int = 10000,
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize document-topic distributions in 2D space.
        
        Args:
            theta: Document-topic distribution matrix (D x K)
            labels: Optional document labels for coloring
            method: Dimensionality reduction method ('umap', 'pca')
            topic_words: Optional list of topic words for annotation
            max_docs: Maximum number of documents to visualize
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        # Sample documents if too many
        n_docs = theta.shape[0]
        if n_docs > max_docs:
            indices = np.sort(np.random.default_rng(self.random_state).choice(n_docs, max_docs, replace=False))
            theta_sample = theta[indices]
        else:
            indices = np.arange(n_docs)
            theta_sample = theta
            max_docs = n_docs
        
        # Get dominant topic for each document
        dominant_topics = np.asarray(labels)[indices] if labels is not None else np.argmax(theta_sample, axis=1)
        num_topics = theta_sample.shape[1]
        
        # Apply dimensionality reduction
        if method == 'umap' and (len(theta_sample) < 5 or theta_sample.shape[1] < 2):
            method = 'pca'
        if method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=2,
                    random_state=self.random_state,
                    n_neighbors=min(30, len(theta_sample)-1),
                    n_jobs=1,
                    min_dist=0.3,
                    spread=1.0,
                    metric='cosine'
                )
            except ImportError:
                logger.warning("UMAP not available, falling back to PCA")
                reducer = PCA(n_components=2, random_state=self.random_state)
                method = 'pca'
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reduce dimensions
        if min(theta_sample.shape) < 2:
            theta_2d = np.column_stack((theta_sample[:, 0], np.zeros(len(theta_sample))))
        else:
            theta_2d = reducer.fit_transform(theta_sample)
        
        fig = draw_document_projection(theta_2d, dominant_topics, method=method.upper(),
                                       language=self.language, total_count=n_docs)
        return self._save_or_show(fig, filename)

    def visualize_training_history(
        self,
        history: Dict,
        filename: str = None
    ) -> plt.Figure:
        """
        Deprecated: Single training charts already exist in visualization_generator.
        Skipped to avoid duplication.
        """
        print("  [SKIP] training_history composite chart (single charts already generated)")
        return None
    
    def visualize_topic_embeddings(
        self,
        topic_embeddings: np.ndarray,
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        method: str = 'tsne',
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize topic embeddings in 2D space.
        
        Args:
            topic_embeddings: Topic embedding matrix (K x E)
            topic_words: Optional list of topic words for annotation
            method: Dimensionality reduction method ('tsne', 'pca')
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(
                n_components=2,
                random_state=self.random_state,
                init='pca',
                learning_rate='auto'
            )
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reduce dimensions
        embeddings_2d = reducer.fit_transform(topic_embeddings)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot topic embeddings
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.8,
            s=100,
            c=range(len(embeddings_2d)),
            cmap='tab20'
        )
        
        # Add annotations if topic_words is provided
        if topic_words:
            for i, (topic_idx, words) in enumerate(topic_words):
                if i >= len(embeddings_2d):
                    break
                
                # Get top words
                top_words = [word for word, _ in words[:2]]
                label = f"{topic_idx}: {', '.join(top_words)}"
                
                # Add annotation
                ax.annotate(
                    label,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=9,
                    alpha=0.8,
                    ha='center',
                    va='bottom',
                    xytext=(0, 5),
                    textcoords='offset points'
                )
        
        ax.set_xlabel(f'{method.upper()} Dimension 1')
        ax.set_ylabel(f'{method.upper()} Dimension 2')
        
        # Save or show
        return self._save_or_show(fig, filename)
    
    def visualize_topic_proportions(
        self,
        theta: np.ndarray,
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        top_k: int = None,
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize average topic proportions across documents.
        
        Args:
            theta: Document-topic distribution matrix (D x K)
            topic_words: Optional list of topic words for labels
            top_k: Number of top topics to show (None for all)
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        # Calculate average topic proportions
        topic_props = theta.mean(axis=0)
        num_topics = len(topic_props)
        
        if top_k is None:
            top_k = num_topics
        
        # Create figure
        horizontal = num_topics > 20
        fig, ax = plt.subplots(figsize=(8, max(4, num_topics * .24)) if horizontal else (12, 6), facecolor='white')
        
        # Use viridis colormap
        colors = [matplotlib.colors.to_rgba(COLORS[i % len(COLORS)]) for i in range(num_topics)]
        
        # Create vertical bar plot for all topics
        x_pos = np.arange(num_topics)
        bars = (ax.barh if horizontal else ax.bar)(x_pos, topic_props, color=colors, edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, topic_props):
            ax.annotate(f'{val:.3f}',
                       xy=(val, bar.get_y() + bar.get_height() / 2) if horizontal else (bar.get_x() + bar.get_width() / 2, val),
                       xytext=(4, 0) if horizontal else (0, 3), textcoords="offset points",
                       ha='left' if horizontal else 'center', va='center' if horizontal else 'bottom', fontsize=8, fontweight='bold')
        
        # Set labels
        labels = [self._get_topic_label(i, short=True) for i in range(num_topics)]
        if horizontal:
            ax.set_yticks(x_pos, labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel(self._get_label('average_proportion'))
            ax.set_ylabel(self._get_label('topic'))
            ax.set_xlim(0, max(topic_props) * 1.2)
        else:
            ax.set_xticks(x_pos, labels, fontsize=9)
            ax.set_xlabel(self._get_label('topic'))
            ax.set_ylabel(self._get_label('average_proportion'))
            ax.set_ylim(0, max(topic_props) * 1.15)
        ax.set_title('主题占比分布' if self.language == 'zh' else 'Topic proportion distribution')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save or show
        return self._save_or_show(fig, filename)
    
    def visualize_intertopic_distance(
        self,
        theta: np.ndarray,
        beta: np.ndarray,
        filename: str = None
    ) -> plt.Figure:
        """
        Create Intertopic Distance Map (left panel of pyLDAvis-style).
        
        Args:
            theta: Document-topic distribution matrix (D x K)
            beta: Topic-word distribution matrix (K x V)
            filename: Filename to save. If None, uses language-aware default.
            
        Returns:
            Figure
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        n_topics = beta.shape[0]
        topic_proportions = theta.mean(axis=0)
        
        if min(beta.shape) >= 2:
            topic_coords = PCA(n_components=2).fit_transform(beta)
        else:
            topic_coords = np.zeros((n_topics, 2))

        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        
        cmap = plt.cm.tab20
        colors = [COLORS[i % len(COLORS)] for i in range(n_topics)]
        sizes = topic_proportions * 2800
        sorted_indices = np.argsort(-sizes)
        
        for idx, i in enumerate(sorted_indices):
            z_order = 2 + (n_topics - idx)
            ax.scatter(topic_coords[i, 0], topic_coords[i, 1],
                       s=sizes[i], c=[colors[i]], alpha=0.75,
                       edgecolors='white', linewidths=.8, zorder=z_order)
        # Label packing changes annotations only, not PCA coordinates or bubble areas.
        xspan, yspan = np.maximum(np.ptp(topic_coords, axis=0), 1e-4)
        mid = np.median(topic_coords[:, 0])
        for side in [-1, 1]:
            ids = np.flatnonzero(topic_coords[:, 0] <= mid if side < 0 else topic_coords[:, 0] > mid)
            ids = ids[np.argsort(topic_coords[ids, 1])]
            label_y = topic_coords[ids, 1].copy()
            for j in range(1, len(ids)): label_y[j] = max(label_y[j], label_y[j-1]+yspan*.12)
            if len(ids): label_y -= (label_y.mean()-topic_coords[ids, 1].mean())
            for i, y in zip(ids, label_y):
                x = topic_coords[i,0]+side*.10*xspan
                # Degenerate PCA axes can have a span far below the label spacing.
                # Include annotation anchors in the view limits before tight export;
                # otherwise the labels produce an enormous off-canvas bounding box.
                ax.update_datalim([(x, y)])
                ax.annotate(f'T{i+1}', topic_coords[i], xytext=(x,y), color=colors[i], weight='bold',
                            ha='right' if side<0 else 'left', va='center',
                            arrowprops=dict(arrowstyle='-',color='#9BA3A8',lw=.65), annotation_clip=False)
        ax.margins(x=.3,y=.25)

        ax.set_xlabel('PC1', fontsize=14)
        ax.set_ylabel('PC2', fontsize=14)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        ax.set_title('主题词权重主成分分析；圆面积 ∝ 平均主题权重')

        plt.tight_layout()
        
        if filename is None:
            filename = '主题间距离图.png' if self.language == 'zh' else 'Intertopic Distance Map.png'
        return self._save_or_show(fig, filename)
    
    def visualize_topic_word_frequency(self, beta, topic_words, selected_topic=0, n_words=30, filename=None):
        """Only actual exported weights; no fabricated corpus counts."""
        words = topic_words[selected_topic][1][:n_words]
        fig, ax = plt.subplots(figsize=(5.2, max(3, len(words)*.19)))
        ax.barh(range(len(words)), [float(p) for w,p in words], color=COLORS[selected_topic % len(COLORS)], height=.55)
        for i, (_, value) in enumerate(words):
            ax.text(1.01, i, f'{value:.4g}', transform=ax.get_yaxis_transform(), va='center', fontsize=7)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
        ax.set_yticks(range(len(words)), [w for w,p in words]); ax.invert_yaxis()
        ax.set_xlabel('导出的主题词权重' if self.language == 'zh' else 'Exported topic-word weight')
        ax.set_title(f'主题 {selected_topic+1} · 显著词汇')
        fig.tight_layout()
        return self._save_or_show(fig, filename or ('最显著词汇.png' if self.language == 'zh' else 'Top Salient Terms.png'))
    
    def visualize_pyldavis_style(
        self,
        theta: np.ndarray,
        beta: np.ndarray,
        topic_words: List[Tuple[int, List[Tuple[str, float]]]],
        selected_topic: int = 0,
        n_words: int = 30,
        filename: str = None
    ) -> plt.Figure:
        """
        Deprecated combined view. Now calls split functions for individual charts.
        """
        self.visualize_intertopic_distance(theta, beta)
        self.visualize_topic_word_frequency(beta, topic_words, selected_topic, n_words)
        return None


def load_etm_results(results_dir: str, timestamp: str = None):
    """
    Load ETM results from files.
    
    Args:
        results_dir: Directory containing ETM results
        timestamp: Specific timestamp to load (None for latest)
        
    Returns:
        Dictionary with loaded results
    """
    # Find result files
    fixed_theta = os.path.join(results_dir, "theta.npy")
    fixed_beta = os.path.join(results_dir, "beta.npy")
    fixed_topic_words = os.path.join(results_dir, "topic_words.json")
    if timestamp:
        timestamped_theta = os.path.join(results_dir, f"theta_{timestamp}.npy")
        timestamped_beta = os.path.join(results_dir, f"beta_{timestamp}.npy")
        timestamped_topic_words = os.path.join(results_dir, f"topic_words_{timestamp}.json")
        # Current THETA runs atomically replace fixed filenames. Preserve support
        # for timestamped legacy results while preferring a complete matching
        # timestamp set when it exists.
        if all(os.path.exists(path) for path in (timestamped_theta, timestamped_beta, timestamped_topic_words)):
            theta_path = timestamped_theta
            beta_path = timestamped_beta
            topic_words_path = timestamped_topic_words
        else:
            theta_path = fixed_theta
            beta_path = fixed_beta
            topic_words_path = fixed_topic_words
        metrics_path = os.path.join(results_dir, f"metrics_{timestamp}.json")
    else:
        # Prefer current fixed files, then fall back to timestamped legacy files.
        theta_files = [Path(fixed_theta)] if os.path.exists(fixed_theta) else sorted(Path(results_dir).glob("theta_*.npy"), reverse=True)
        beta_files = [Path(fixed_beta)] if os.path.exists(fixed_beta) else sorted(Path(results_dir).glob("beta_*.npy"), reverse=True)
        topic_words_files = [Path(fixed_topic_words)] if os.path.exists(fixed_topic_words) else sorted(Path(results_dir).glob("topic_words_*.json"), reverse=True)
        metrics_files = sorted(Path(results_dir).glob("metrics_*.json"), reverse=True)
        
        if not theta_files or not beta_files or not topic_words_files:
            raise FileNotFoundError(f"Could not find ETM result files in {results_dir}")
        
        theta_path = str(theta_files[0])
        beta_path = str(beta_files[0])
        topic_words_path = str(topic_words_files[0])
        metrics_path = str(metrics_files[0]) if metrics_files else None
    
    # Load files
    theta = np.load(theta_path)
    beta = np.load(beta_path)
    
    with open(topic_words_path, 'r') as f:
        topic_words = json.load(f)
    
    # Convert topic_words format - handle different formats
    if isinstance(topic_words, dict):
        # Format: {topic_id: [[word, prob], ...]} or {topic_id: [(word, prob), ...]}
        converted = []
        for k, words in topic_words.items():
            word_list = []
            for item in words:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    word_list.append((item[0], float(item[1])))
                elif isinstance(item, str):
                    word_list.append((item, 1.0))
            converted.append((int(k), word_list))
        topic_words = converted
    elif isinstance(topic_words, list):
        # Format: [[topic_id, [[word, prob], ...]], ...]
        converted = []
        for item in topic_words:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                tid = int(item[0])
                words = item[1]
                word_list = []
                for w in words:
                    if isinstance(w, (list, tuple)) and len(w) >= 2:
                        word_list.append((w[0], float(w[1])))
                    elif isinstance(w, str):
                        word_list.append((w, 1.0))
                converted.append((tid, word_list))
        topic_words = converted
    
    # Load metrics if available
    metrics = None
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    return {
        'theta': theta,
        'beta': beta,
        'topic_words': topic_words,
        'metrics': metrics
    }


def visualize_etm_results(
    results_dir: str,
    output_dir: str = None,
    timestamp: str = None,
    show_wordcloud: bool = True
):
    """
    Visualize ETM results.
    
    Args:
        results_dir: Directory containing ETM results
        output_dir: Directory to save visualizations
        timestamp: Specific timestamp to load (None for latest)
        show_wordcloud: Whether to show word clouds
    """
    # Load results
    results = load_etm_results(results_dir, timestamp)
    
    # Create visualizer
    visualizer = TopicVisualizer(output_dir=output_dir)
    
    # Create visualizations
    logger.info("Generating topic word visualization...")
    visualizer.visualize_topic_words(
        results['topic_words'],
        num_topics=10,
        as_wordcloud=show_wordcloud and WORDCLOUD_AVAILABLE,
        filename="topic_words.png"
    )
    
    logger.info("Generating topic similarity visualization...")
    visualizer.visualize_topic_similarity(
        results['beta'],
        results['topic_words'],
        filename="topic_similarity.png"
    )
    
    logger.info("Generating topic proportions visualization...")
    visualizer.visualize_topic_proportions(
        results['theta'],
        results['topic_words'],
        filename="topic_proportions.png"
    )
    
    logger.info("Generating document-topic visualization...")
    visualizer.visualize_document_topics(
        results['theta'],
        method='tsne',
        filename="document_topics_tsne.png"
    )
    
    logger.info("Generating topic embeddings visualization...")
    topic_embeddings = results['beta'] @ results['beta'].T  # Approximate topic embeddings
    visualizer.visualize_topic_embeddings(
        topic_embeddings,
        results['topic_words'],
        filename="topic_embeddings.png"
    )
    
    logger.info("Visualizations complete!")


def generate_pyldavis_visualization(
    theta: np.ndarray,
    beta: np.ndarray,
    bow_matrix,
    vocab: List[str],
    output_path: str,
    mds: str = 'pcoa',
    sort_topics: bool = False,
    R: int = 30
) -> Optional[str]:
    """
    Generate interactive pyLDAvis HTML visualization.
    
    Args:
        theta: Document-topic distribution (N x K)
        beta: Topic-word distribution (K x V)
        bow_matrix: BOW matrix (N x V), can be sparse or dense
        vocab: Vocabulary list
        output_path: Path to save HTML file
        mds: Multidimensional scaling method ('tsne', 'mmds', 'pcoa')
        sort_topics: Whether to sort topics by prevalence
        R: Number of terms to display in barcharts
        
    Returns:
        Path to saved HTML file, or None if pyLDAvis not available
    """
    try:
        import pyLDAvis
    except ImportError:
        logger.warning("pyLDAvis not installed. Install with: pip install pyLDAvis")
        return None
    
    from scipy import sparse
    
    if bow_matrix is None:
        raise ValueError('pyLDAvis requires observed aligned BOW counts')
    theta = np.asarray(theta, dtype=float).copy()
    beta = np.asarray(beta, dtype=float).copy()
    if bow_matrix.shape != (len(theta), len(vocab)) or beta.shape != (theta.shape[1], len(vocab)):
        raise ValueError('pyLDAvis matrix axes are not aligned')
    if not np.isfinite(theta).all() or not np.isfinite(beta).all() or (theta < 0).any() or (beta < 0).any():
        raise ValueError('pyLDAvis requires finite nonnegative probabilities')
    if (theta.sum(axis=1) <= 0).any() or (beta.sum(axis=1) <= 0).any():
        raise ValueError('pyLDAvis cannot invent distributions for empty rows')
    theta /= theta.sum(axis=1, keepdims=True)
    beta_normalized = beta / beta.sum(axis=1, keepdims=True)
    doc_lengths = np.asarray(bow_matrix.sum(axis=1)).ravel()
    term_frequency = np.asarray(bow_matrix.sum(axis=0)).ravel()
    positive = doc_lengths > 0
    if not positive.any():
        raise ValueError('pyLDAvis requires at least one nonempty document')
    scope = {'inputDocuments': len(theta), 'displayedDocuments': int(positive.sum()),
             'excludedEmptyBowRows': np.flatnonzero(~positive).tolist(),
             'note': 'Interactive view excludes zero-token documents only; global figures retain all model rows. Original matrices unchanged.'}
    theta = theta[positive]
    doc_lengths = doc_lengths[positive]

    try:
        # Create pyLDAvis visualization data
        vis_data = pyLDAvis.prepare(
            topic_term_dists=beta_normalized,
            doc_topic_dists=theta,
            doc_lengths=doc_lengths,
            vocab=vocab,
            term_frequency=term_frequency,
            mds=mds,
            n_jobs=1,
            sort_topics=sort_topics,
            R=R
        )
        
        # Save to HTML
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        from pyLDAvis import urls
        page = pyLDAvis.prepared_data_to_html(vis_data, template_type='simple')
        for url, local in [(urls.D3_URL, urls.D3_LOCAL), (urls.LDAVIS_URL, urls.LDAVIS_LOCAL)]:
            script = Path(local).read_text(encoding='utf-8')
            page = page.replace(f'<script type="text/javascript" src="{url}"></script>', f'<script>{script}</script>')
        page = page.replace(f'<link rel="stylesheet" type="text/css" href="{urls.LDAVIS_CSS_URL}">',
                            '<style>' + Path(urls.LDAVIS_CSS_LOCAL).read_text() + '</style>')
        notice = f'<p>Documents: {scope["displayedDocuments"]:,}/{scope["inputDocuments"]:,}; empty BOW rows excluded only in this view. Topic IDs retain model order.</p>'
        Path(output_path).write_text('<!doctype html><meta charset="utf-8">'+notice+page, encoding='utf-8')
        Path(output_path).with_suffix('.scope.json').write_text(json.dumps(scope,indent=2),encoding='utf-8')
        logger.info(f"pyLDAvis visualization saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to generate pyLDAvis visualization: {e}")
        return None


def generate_pyldavis_notebook(
    theta: np.ndarray,
    beta: np.ndarray,
    bow_matrix,
    vocab: List[str],
    mds: str = 'pcoa'
):
    """
    Generate pyLDAvis visualization for Jupyter notebook display.
    
    Args:
        theta: Document-topic distribution (N x K)
        beta: Topic-word distribution (K x V)
        bow_matrix: BOW matrix (N x V)
        vocab: Vocabulary list
        mds: Multidimensional scaling method
        
    Returns:
        pyLDAvis prepared data object for notebook display
    """
    try:
        import pyLDAvis
        pyLDAvis.enable_notebook()
    except ImportError:
        logger.warning("pyLDAvis not installed")
        return None
    
    from scipy import sparse
    
    if sparse.issparse(bow_matrix):
        bow_dense = bow_matrix.toarray()
    else:
        bow_dense = np.asarray(bow_matrix)
    
    theta = np.asarray(theta, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    beta_normalized = beta / beta.sum(axis=1, keepdims=True)
    
    doc_lengths = bow_dense.sum(axis=1).astype(np.int64)
    term_frequency = bow_dense.sum(axis=0).astype(np.int64)
    
    nonzero_mask = term_frequency > 0
    if not nonzero_mask.all():
        term_frequency = term_frequency[nonzero_mask]
        beta_normalized = beta_normalized[:, nonzero_mask]
        vocab = [v for v, m in zip(vocab, nonzero_mask) if m]
    
    try:
        vis_data = pyLDAvis.prepare(
            topic_term_dists=beta_normalized,
            doc_topic_dists=theta,
            doc_lengths=doc_lengths,
            vocab=vocab,
            term_frequency=term_frequency,
            mds=mds,
            n_jobs=1,
            sort_topics=True
        )
        return vis_data
    except Exception as e:
        logger.error(f"Failed to prepare pyLDAvis data: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize ETM results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing ETM results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save visualizations")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Specific timestamp to load")
    parser.add_argument("--no_wordcloud", action="store_true",
                        help="Disable word cloud visualization")
    
    args = parser.parse_args()
    
    visualize_etm_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        timestamp=args.timestamp,
        show_wordcloud=not args.no_wordcloud
    )
