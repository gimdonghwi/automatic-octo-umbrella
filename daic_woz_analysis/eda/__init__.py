"""
DAIC-WOZ EDA Package

Production-ready exploratory data analysis infrastructure for
depression detection research using the DAIC-WOZ dataset.

Example usage:
    from eda import DAICDataLoader, StatisticalAnalyzer, ClinicalVisualizationSuite
    
    loader = DAICDataLoader.from_config('config/paths.yaml')
    train_df = loader.load_split('train')
    
    analyzer = StatisticalAnalyzer()
    results = analyzer.analyze_feature_dataframe(train_df, 'ttr')
    
    viz = ClinicalVisualizationSuite()
    viz.plot_distribution_comparison(train_df, 'ttr', stat_test_result=results)
"""

from .data_loader import DAICDataLoader, TranscriptProcessor
from .statistical_analyzer import StatisticalAnalyzer, FeatureImportanceRanker
from .feature_profiler import (
    LinguisticFeatureExtractor,
    AudioFeatureProfiler,
    QuestionTypeAnalyzer,
    ComprehensiveFeatureProfiler
)
from .visualizations import ClinicalVisualizationSuite

__version__ = '1.0.0'

__all__ = [
    'DAICDataLoader',
    'TranscriptProcessor',
    'StatisticalAnalyzer',
    'FeatureImportanceRanker',
    'LinguisticFeatureExtractor',
    'AudioFeatureProfiler',
    'QuestionTypeAnalyzer',
    'ComprehensiveFeatureProfiler',
    'ClinicalVisualizationSuite'
]
