"""
Visualization utilities for ETM.

Includes:
- Topic word visualization (bar charts, word clouds)
- Topic similarity heatmap
- Document-topic distribution (t-SNE, PCA)
- Topic proportions
- Temporal topic analysis
- pyLDAvis interactive visualization
"""

from .topic_visualizer import (
    TopicVisualizer,
    load_etm_results,
    visualize_etm_results,
    generate_pyldavis_visualization,
    generate_pyldavis_notebook
)
from .temporal_analysis import (
    TemporalTopicAnalyzer,
    analyze_temporal_topics
)

__all__ = [
    'TopicVisualizer',
    'load_etm_results',
    'visualize_etm_results',
    'generate_pyldavis_visualization',
    'generate_pyldavis_notebook',
    'TemporalTopicAnalyzer',
    'analyze_temporal_topics'
]
