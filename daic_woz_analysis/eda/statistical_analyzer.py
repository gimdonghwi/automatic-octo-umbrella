"""
Statistical Analysis Module for Depression Detection Research

Implements:
- Non-parametric tests (Mann-Whitney U)
- Effect size calculation (Cohen's d)
- Feature importance ranking
- Clinical significance assessment
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Clinical hypothesis testing for depression detection features.
    
    Usage:
        analyzer = StatisticalAnalyzer()
        result = analyzer.compare_groups(depressed_features, non_depressed_features)
        print(f"p-value: {result['p_value']}, Effect size: {result['cohens_d']}")
    """
    
    @staticmethod
    def mann_whitney_test(
        group1: np.ndarray, 
        group2: np.ndarray
    ) -> Dict[str, float]:
        """
        Non-parametric test for group differences.
        
        Preferred over t-test due to:
        - Small sample sizes (n<30 in some splits)
        - Non-normal distributions (typical in clinical data)
        
        Args:
            group1: Depressed group samples
            group2: Non-depressed group samples
            
        Returns:
            {
                'statistic': float,
                'p_value': float,
                'significant': bool (p < 0.05)
            }
        """
        statistic, p_value = stats.mannwhitneyu(
            group1, 
            group2, 
            alternative='two-sided'
        )
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate effect size (Cohen's d).
        
        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large
        
        Clinical relevance: Large effect sizes indicate clinically
        meaningful differences beyond statistical significance.
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return d
    
    @staticmethod
    def interpret_effect_size(d: float) -> str:
        """Interpret Cohen's d value"""
        abs_d = abs(d)
        
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def compare_groups(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray,
        feature_name: str = "feature"
    ) -> Dict:
        """
        Comprehensive group comparison with statistical and effect size analysis.
        
        Returns:
            {
                'feature': str,
                'n_depressed': int,
                'n_non_depressed': int,
                'mean_depressed': float,
                'mean_non_depressed': float,
                'median_depressed': float,
                'median_non_depressed': float,
                'std_depressed': float,
                'std_non_depressed': float,
                'mann_whitney_u': float,
                'p_value': float,
                'significant': bool,
                'cohens_d': float,
                'effect_size': str
            }
        """
        mw_result = self.mann_whitney_test(group1, group2)
        d = self.cohens_d(group1, group2)
        
        result = {
            'feature': feature_name,
            'n_depressed': len(group1),
            'n_non_depressed': len(group2),
            'mean_depressed': np.mean(group1),
            'mean_non_depressed': np.mean(group2),
            'median_depressed': np.median(group1),
            'median_non_depressed': np.median(group2),
            'std_depressed': np.std(group1, ddof=1),
            'std_non_depressed': np.std(group2, ddof=1),
            'mann_whitney_u': mw_result['statistic'],
            'p_value': mw_result['p_value'],
            'significant': mw_result['significant'],
            'cohens_d': d,
            'effect_size': self.interpret_effect_size(d)
        }
        
        return result
    
    def analyze_feature_dataframe(
        self, 
        df: pd.DataFrame, 
        feature_col: str, 
        label_col: str = 'PHQ8_Binary'
    ) -> Dict:
        """
        Analyze a feature from a DataFrame with depression labels.
        
        Args:
            df: DataFrame with features and labels
            feature_col: Name of feature column to analyze
            label_col: Name of binary label column (1=depressed, 0=non-depressed)
        """
        depressed = df[df[label_col] == 1][feature_col].dropna().values
        non_depressed = df[df[label_col] == 0][feature_col].dropna().values
        
        if len(depressed) == 0 or len(non_depressed) == 0:
            logger.warning(f"Empty group for feature: {feature_col}")
            return None
        
        return self.compare_groups(depressed, non_depressed, feature_col)
    
    def batch_analyze_features(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        label_col: str = 'PHQ8_Binary'
    ) -> pd.DataFrame:
        """
        Analyze multiple features and return results as DataFrame.
        
        Returns:
            DataFrame sorted by effect size (descending)
        """
        results = []
        
        for feature in feature_cols:
            result = self.analyze_feature_dataframe(df, feature, label_col)
            if result:
                results.append(result)
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            results_df = results_df.sort_values('cohens_d', key=abs, ascending=False)
        
        return results_df


class FeatureImportanceRanker:
    """
    Rank features by their discriminative power for depression detection.
    
    Combines:
    - Statistical significance (p-value)
    - Effect size (Cohen's d)
    - Predictive power (AUC-ROC)
    """
    
    def __init__(self, statistical_results: pd.DataFrame):
        self.results = statistical_results
    
    def get_top_features(self, n: int = 10, metric: str = 'cohens_d') -> pd.DataFrame:
        """
        Get top N features by a given metric.
        
        Args:
            n: Number of top features to return
            metric: 'cohens_d', 'p_value', or custom column
        """
        if metric == 'p_value':
            return self.results.nsmallest(n, metric)
        else:
            return self.results.nlargest(n, metric, keep='all')
    
    def filter_significant_features(self, alpha: float = 0.05) -> pd.DataFrame:
        """Get features with statistically significant differences"""
        return self.results[self.results['p_value'] < alpha]
    
    def get_clinical_relevant_features(self, min_effect_size: float = 0.5) -> pd.DataFrame:
        """
        Get features with clinically meaningful effect sizes.
        
        Args:
            min_effect_size: Minimum |Cohen's d| (default 0.5 = medium effect)
        """
        return self.results[self.results['cohens_d'].abs() >= min_effect_size]
