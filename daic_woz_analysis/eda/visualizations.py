"""
Visualization Module for DAIC-WOZ EDA

Publication-quality plots:
- Distribution comparisons (depressed vs non-depressed)
- Correlation matrices
- Statistical test visualizations
- Question type analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ClinicalVisualizationSuite:
    """
    Publication-ready visualization tools for depression detection research.
    
    Usage:
        viz = ClinicalVisualizationSuite()
        viz.plot_distribution_comparison(df, 'ttr', 'PHQ8_Binary')
        viz.save_figure('ttr_distribution.png')
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', figsize: Tuple = (12, 6)):
        plt.style.use(style)
        self.figsize = figsize
        self.color_depressed = '#E74C3C'
        self.color_non_depressed = '#3498DB'
        
        sns.set_palette([self.color_non_depressed, self.color_depressed])
    
    def plot_distribution_comparison(
        self,
        df: pd.DataFrame,
        feature: str,
        label_col: str = 'PHQ8_Binary',
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        stat_test_result: Optional[Dict] = None,
        bins: int = 30
    ) -> plt.Figure:
        """
        Side-by-side distribution comparison with statistical annotations.
        
        Args:
            df: DataFrame with features and labels
            feature: Feature column name
            label_col: Binary label column (0/1)
            stat_test_result: Optional dict from StatisticalAnalyzer
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        depressed = df[df[label_col] == 1][feature].dropna()
        non_depressed = df[df[label_col] == 0][feature].dropna()
        
        axes[0].hist(
            non_depressed, 
            bins=bins, 
            alpha=0.7, 
            color=self.color_non_depressed,
            edgecolor='black',
            label=f'Non-depressed (n={len(non_depressed)})'
        )
        axes[0].axvline(
            non_depressed.mean(), 
            color=self.color_non_depressed, 
            linestyle='--', 
            linewidth=2,
            label=f'Mean: {non_depressed.mean():.3f}'
        )
        axes[0].set_title('Non-depressed Group')
        axes[0].set_xlabel(xlabel or feature)
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].hist(
            depressed, 
            bins=bins, 
            alpha=0.7, 
            color=self.color_depressed,
            edgecolor='black',
            label=f'Depressed (n={len(depressed)})'
        )
        axes[1].axvline(
            depressed.mean(), 
            color=self.color_depressed, 
            linestyle='--', 
            linewidth=2,
            label=f'Mean: {depressed.mean():.3f}'
        )
        axes[1].set_title('Depressed Group')
        axes[1].set_xlabel(xlabel or feature)
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        if stat_test_result:
            stats_text = (
                f"Mann-Whitney U: p={stat_test_result['p_value']:.4f}\n"
                f"Cohen's d: {stat_test_result['cohens_d']:.3f} "
                f"({stat_test_result['effect_size']})"
            )
            fig.text(
                0.5, -0.05, stats_text, 
                ha='center', 
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        plt.tight_layout()
        return fig
    
    def plot_boxplot_comparison(
        self,
        df: pd.DataFrame,
        feature: str,
        label_col: str = 'PHQ8_Binary',
        title: Optional[str] = None,
        ylabel: Optional[str] = None
    ) -> plt.Figure:
        """
        Box plot comparison with individual data points overlay.
        
        Shows:
        - Median, quartiles, outliers
        - Individual participant values (jittered)
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        plot_df = df[[feature, label_col]].dropna()
        plot_df['Group'] = plot_df[label_col].map({0: 'Non-depressed', 1: 'Depressed'})
        
        sns.boxplot(
            data=plot_df,
            x='Group',
            y=feature,
            ax=ax,
            palette=[self.color_non_depressed, self.color_depressed],
            width=0.5
        )
        
        sns.stripplot(
            data=plot_df,
            x='Group',
            y=feature,
            ax=ax,
            color='black',
            alpha=0.3,
            size=3,
            jitter=True
        )
        
        ax.set_ylabel(ylabel or feature, fontsize=12)
        ax.set_xlabel('Depression Status', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        features: List[str],
        title: str = "Feature Correlation Matrix",
        annot: bool = True,
        cmap: str = 'coolwarm'
    ) -> plt.Figure:
        """
        Correlation matrix heatmap for feature relationships.
        
        Args:
            df: DataFrame with features
            features: List of feature column names
            annot: Show correlation coefficients
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_matrix = df[features].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=annot,
            fmt='.2f',
            cmap=cmap,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    def plot_question_type_distribution(
        self,
        df: pd.DataFrame,
        qtype_columns: List[str],
        label_col: str = 'PHQ8_Binary',
        title: str = "Question Type Distribution by Depression Status"
    ) -> plt.Figure:
        """
        Grouped bar chart comparing question type distributions.
        
        Args:
            df: DataFrame with question type ratio columns
            qtype_columns: List of column names (e.g., ['qtype_0_ratio', ...])
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        depressed_means = df[df[label_col] == 1][qtype_columns].mean()
        non_depressed_means = df[df[label_col] == 0][qtype_columns].mean()
        
        x = np.arange(len(qtype_columns))
        width = 0.35
        
        ax.bar(
            x - width/2, 
            non_depressed_means, 
            width, 
            label='Non-depressed',
            color=self.color_non_depressed,
            edgecolor='black'
        )
        ax.bar(
            x + width/2, 
            depressed_means, 
            width, 
            label='Depressed',
            color=self.color_depressed,
            edgecolor='black'
        )
        
        ax.set_xlabel('Question Type', fontsize=12)
        ax.set_ylabel('Average Ratio', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([col.replace('qtype_', 'Q').replace('_ratio', '') 
                            for col in qtype_columns], rotation=0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_effect_size_ranking(
        self,
        stat_results: pd.DataFrame,
        top_n: int = 15,
        title: str = "Feature Importance by Effect Size (Cohen's d)"
    ) -> plt.Figure:
        """
        Horizontal bar chart of top features by Cohen's d.
        
        Args:
            stat_results: DataFrame from StatisticalAnalyzer.batch_analyze_features()
            top_n: Number of top features to display
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = stat_results.nlargest(top_n, 'cohens_d', keep='all')
        
        colors = [self.color_depressed if d > 0 else self.color_non_depressed 
                  for d in top_features['cohens_d']]
        
        ax.barh(
            range(len(top_features)),
            top_features['cohens_d'],
            color=colors,
            edgecolor='black'
        )
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
        ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0.8, color='gray', linestyle=':', alpha=0.5, label='Large effect')
        ax.axvline(-0.8, color='gray', linestyle=':', alpha=0.5)
        
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_class_balance(
        self,
        df: pd.DataFrame,
        label_col: str = 'PHQ8_Binary',
        title: str = "Dataset Class Balance"
    ) -> plt.Figure:
        """
        Pie chart showing class distribution.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        counts = df[label_col].value_counts()
        labels = ['Non-depressed', 'Depressed']
        colors = [self.color_non_depressed, self.color_depressed]
        
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
        
        ax.set_title(
            f"{title}\n(n={len(df)})", 
            fontsize=14, 
            fontweight='bold',
            pad=20
        )
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300):
        """Save figure to file with high resolution"""
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure saved: {filepath}")
        plt.close(fig)
