"""
Temporal Topic Analysis

Visualize how topics evolve over time when timestamps are available.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


class TemporalTopicAnalyzer:
    """
    Analyze and visualize topic evolution over time.
    """
    
    def __init__(
        self,
        theta: np.ndarray,
        timestamps: np.ndarray,
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        output_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
        dpi: int = 150
    ):
        """
        Initialize temporal analyzer.
        
        Args:
            theta: Document-topic distribution (N x K)
            timestamps: Array of timestamps for each document
            topic_words: Optional list of (topic_idx, [(word, prob), ...])
            output_dir: Directory to save visualizations
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.theta = theta
        self.timestamps = self._parse_timestamps(timestamps)
        self.topic_words = topic_words
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        self.num_docs, self.num_topics = theta.shape
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def _parse_timestamps(self, timestamps: np.ndarray) -> pd.DatetimeIndex:
        """Parse timestamps to datetime"""
        try:
            return pd.to_datetime(timestamps)
        except Exception as e:
            logger.warning(f"Could not parse timestamps: {e}")
            # Create synthetic timestamps
            return pd.date_range(start='2020-01-01', periods=len(timestamps), freq='D')
    
    def _get_topic_label(self, topic_idx: int, max_words: int = 3) -> str:
        """Get topic label from top words"""
        if self.topic_words:
            for idx, words in self.topic_words:
                if idx == topic_idx:
                    top_words = [w for w, _ in words[:max_words]]
                    return f"T{topic_idx}: {', '.join(top_words)}"
        return f"Topic {topic_idx}"
    
    def _save_or_show(self, fig, filename: Optional[str] = None):
        """Save figure or show it"""
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {filepath}")
            plt.close(fig)
            return filepath
        else:
            plt.show()
            return None
    
    def compute_temporal_topic_distribution(
        self,
        time_bins: int = 10,
        aggregation: str = 'mean'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Compute topic distribution over time bins.
        
        Args:
            time_bins: Number of time bins
            aggregation: Aggregation method ('mean', 'sum', 'count')
            
        Returns:
            (DataFrame with temporal distribution, list of time labels)
        """
        # Create time bins
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            **{f'topic_{k}': self.theta[:, k] for k in range(self.num_topics)}
        })
        
        # Bin by time
        df['time_bin'] = pd.cut(df['timestamp'], bins=time_bins, labels=False)
        
        # Aggregate
        topic_cols = [f'topic_{k}' for k in range(self.num_topics)]
        
        if aggregation == 'mean':
            temporal_dist = df.groupby('time_bin')[topic_cols].mean()
        elif aggregation == 'sum':
            temporal_dist = df.groupby('time_bin')[topic_cols].sum()
        elif aggregation == 'count':
            # Count documents where topic is dominant
            for k in range(self.num_topics):
                df[f'dominant_{k}'] = (df[topic_cols].idxmax(axis=1) == f'topic_{k}').astype(int)
            dominant_cols = [f'dominant_{k}' for k in range(self.num_topics)]
            temporal_dist = df.groupby('time_bin')[dominant_cols].sum()
            temporal_dist.columns = topic_cols
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Get time labels
        time_ranges = df.groupby('time_bin')['timestamp'].agg(['min', 'max'])
        time_labels = [
            f"{row['min'].strftime('%Y-%m-%d')} to {row['max'].strftime('%Y-%m-%d')}"
            for _, row in time_ranges.iterrows()
        ]
        
        return temporal_dist, time_labels
    
    def plot_topic_evolution(
        self,
        time_bins: int = 10,
        top_k_topics: int = 10,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot topic evolution over time as line chart.
        
        Args:
            time_bins: Number of time bins
            top_k_topics: Number of top topics to show
            filename: Output filename
            
        Returns:
            Figure
        """
        temporal_dist, time_labels = self.compute_temporal_topic_distribution(time_bins)
        
        # Select top topics by average proportion
        avg_props = temporal_dist.mean()
        top_topics = avg_props.nlargest(top_k_topics).index
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for col in top_topics:
            topic_idx = int(col.split('_')[1])
            label = self._get_topic_label(topic_idx)
            ax.plot(range(len(temporal_dist)), temporal_dist[col], marker='o', label=label)
        
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Average Topic Proportion')
        ax.set_title('Topic Evolution Over Time')
        ax.set_xticks(range(len(time_labels)))
        ax.set_xticklabels([l.split(' to ')[0] for l in time_labels], rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_topic_heatmap(
        self,
        time_bins: int = 10,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot topic distribution over time as heatmap.
        
        Args:
            time_bins: Number of time bins
            filename: Output filename
            
        Returns:
            Figure
        """
        temporal_dist, time_labels = self.compute_temporal_topic_distribution(time_bins)
        
        # Create topic labels
        topic_labels = [self._get_topic_label(k, max_words=2) for k in range(self.num_topics)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Transpose for better visualization (topics on y-axis, time on x-axis)
        data = temporal_dist.T.values
        
        sns.heatmap(
            data,
            ax=ax,
            cmap='YlOrRd',
            xticklabels=[l.split(' to ')[0] for l in time_labels],
            yticklabels=topic_labels,
            cbar_kws={'label': 'Topic Proportion'}
        )
        
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Topic')
        ax.set_title('Topic Distribution Over Time')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def plot_topic_stacked_area(
        self,
        time_bins: int = 10,
        top_k_topics: int = 10,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot topic evolution as stacked area chart.
        
        Args:
            time_bins: Number of time bins
            top_k_topics: Number of top topics to show
            filename: Output filename
            
        Returns:
            Figure
        """
        temporal_dist, time_labels = self.compute_temporal_topic_distribution(time_bins)
        
        # Select top topics
        avg_props = temporal_dist.mean()
        top_topics = avg_props.nlargest(top_k_topics).index.tolist()
        
        # Normalize to sum to 1
        data = temporal_dist[top_topics]
        data_norm = data.div(data.sum(axis=1), axis=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create stacked area
        labels = [self._get_topic_label(int(col.split('_')[1])) for col in top_topics]
        ax.stackplot(
            range(len(data_norm)),
            data_norm.T.values,
            labels=labels,
            alpha=0.8
        )
        
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Topic Proportion')
        ax.set_title('Topic Composition Over Time')
        ax.set_xticks(range(len(time_labels)))
        ax.set_xticklabels([l.split(' to ')[0] for l in time_labels], rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_topic_trends(
        self,
        time_bins: int = 10,
        top_k_rising: int = 5,
        top_k_falling: int = 5,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot rising and falling topic trends.
        
        Args:
            time_bins: Number of time bins
            top_k_rising: Number of top rising topics
            top_k_falling: Number of top falling topics
            filename: Output filename
            
        Returns:
            Figure
        """
        temporal_dist, time_labels = self.compute_temporal_topic_distribution(time_bins)
        
        # Calculate trend (slope) for each topic
        x = np.arange(len(temporal_dist))
        trends = {}
        
        for col in temporal_dist.columns:
            y = temporal_dist[col].values
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                trends[col] = slope
        
        # Sort by trend
        sorted_trends = sorted(trends.items(), key=lambda x: x[1], reverse=True)
        rising_topics = [t[0] for t in sorted_trends[:top_k_rising]]
        falling_topics = [t[0] for t in sorted_trends[-top_k_falling:]]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))
        
        # Rising topics
        for col in rising_topics:
            topic_idx = int(col.split('_')[1])
            label = self._get_topic_label(topic_idx)
            ax1.plot(range(len(temporal_dist)), temporal_dist[col], marker='o', label=label)
        
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Topic Proportion')
        ax1.set_title('Rising Topics')
        ax1.set_xticks(range(len(time_labels)))
        ax1.set_xticklabels([l.split(' to ')[0] for l in time_labels], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Falling topics
        for col in falling_topics:
            topic_idx = int(col.split('_')[1])
            label = self._get_topic_label(topic_idx)
            ax2.plot(range(len(temporal_dist)), temporal_dist[col], marker='o', label=label)
        
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Topic Proportion')
        ax2.set_title('Falling Topics')
        ax2.set_xticks(range(len(time_labels)))
        ax2.set_xticklabels([l.split(' to ')[0] for l in time_labels], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def compute_topic_burst_detection(
        self,
        time_bins: int = 20,
        threshold_std: float = 2.0
    ) -> Dict[int, List[Dict]]:
        """
        Detect topic bursts (sudden increases in topic proportion).
        
        Args:
            time_bins: Number of time bins
            threshold_std: Standard deviation threshold for burst detection
            
        Returns:
            Dictionary mapping topic_idx to list of burst events
        """
        temporal_dist, time_labels = self.compute_temporal_topic_distribution(time_bins)
        
        bursts = {}
        
        for col in temporal_dist.columns:
            topic_idx = int(col.split('_')[1])
            values = temporal_dist[col].values
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val > 0:
                threshold = mean_val + threshold_std * std_val
                burst_indices = np.where(values > threshold)[0]
                
                if len(burst_indices) > 0:
                    bursts[topic_idx] = [
                        {
                            'time_bin': int(idx),
                            'time_label': time_labels[idx] if idx < len(time_labels) else 'Unknown',
                            'value': float(values[idx]),
                            'z_score': float((values[idx] - mean_val) / std_val)
                        }
                        for idx in burst_indices
                    ]
        
        return bursts
    
    def generate_temporal_report(
        self,
        time_bins: int = 10,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive temporal analysis report.
        
        Args:
            time_bins: Number of time bins
            output_path: Path to save report
            
        Returns:
            Report dictionary
        """
        temporal_dist, time_labels = self.compute_temporal_topic_distribution(time_bins)
        
        # Calculate trends
        x = np.arange(len(temporal_dist))
        trends = {}
        for col in temporal_dist.columns:
            topic_idx = int(col.split('_')[1])
            y = temporal_dist[col].values
            if len(y) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                trends[topic_idx] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'direction': 'rising' if slope > 0 else 'falling',
                    'avg_proportion': float(np.mean(y)),
                    'max_proportion': float(np.max(y)),
                    'min_proportion': float(np.min(y))
                }
        
        # Detect bursts
        bursts = self.compute_topic_burst_detection(time_bins)
        
        report = {
            'num_documents': self.num_docs,
            'num_topics': self.num_topics,
            'time_range': {
                'start': str(self.timestamps.min()),
                'end': str(self.timestamps.max())
            },
            'time_bins': time_bins,
            'time_labels': time_labels,
            'topic_trends': trends,
            'topic_bursts': bursts,
            'temporal_distribution': temporal_dist.to_dict()
        }
        
        if output_path:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Temporal report saved to {output_path}")
        
        return report


def analyze_temporal_topics(
    theta: np.ndarray,
    timestamps: np.ndarray,
    topic_words: Optional[List] = None,
    output_dir: str = None,
    time_bins: int = 10
) -> Dict:
    """
    Convenience function to run full temporal analysis.
    
    Args:
        theta: Document-topic distribution
        timestamps: Document timestamps
        topic_words: Optional topic words
        output_dir: Output directory
        time_bins: Number of time bins
        
    Returns:
        Analysis report
    """
    analyzer = TemporalTopicAnalyzer(
        theta=theta,
        timestamps=timestamps,
        topic_words=topic_words,
        output_dir=output_dir
    )
    
    # Generate visualizations
    analyzer.plot_topic_evolution(time_bins, filename="topic_evolution.png")
    analyzer.plot_topic_heatmap(time_bins, filename="topic_heatmap.png")
    analyzer.plot_topic_stacked_area(time_bins, filename="topic_stacked_area.png")
    analyzer.plot_topic_trends(time_bins, filename="topic_trends.png")
    
    # Generate report
    report = analyzer.generate_temporal_report(
        time_bins,
        output_path=os.path.join(output_dir, "temporal_report.json") if output_dir else None
    )
    
    return report


if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    
    num_docs = 1000
    num_topics = 10
    
    # Generate random theta
    theta = np.random.dirichlet(np.ones(num_topics) * 0.5, size=num_docs)
    
    # Generate random timestamps over 2 years
    timestamps = pd.date_range(start='2022-01-01', periods=num_docs, freq='D').values
    np.random.shuffle(timestamps)
    
    # Generate topic words
    topic_words = [
        (k, [(f"word_{k}_{i}", 0.1 - i * 0.01) for i in range(10)])
        for k in range(num_topics)
    ]
    
    # Run analysis
    analyzer = TemporalTopicAnalyzer(
        theta=theta,
        timestamps=timestamps,
        topic_words=topic_words
    )
    
    report = analyzer.generate_temporal_report(time_bins=12)
    print(f"Generated temporal report with {len(report['topic_trends'])} topic trends")
