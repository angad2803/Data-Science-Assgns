"""
Main Orchestration Script for TOPSIS-Based Model Selection

This script coordinates the entire pipeline:
1. Evaluates pretrained summarization models
2. Applies TOPSIS multi-criteria decision analysis
3. Generates visualizations and reports

Author: ML Research Assistant
Date: February 10, 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.evaluate_models import ModelEvaluator
from topsis.topsis import apply_topsis_to_dataframe


def setup_visualization_style():
    """Configure matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def create_ranking_visualization(df: pd.DataFrame, output_path: str):
    """
    Create a horizontal bar chart showing TOPSIS rankings.
    
    Args:
        df (pd.DataFrame): DataFrame with Model, TOPSIS Score, and Rank
        output_path (str): Path to save the figure
    """
    # Sort by rank
    df_sorted = df.sort_values('Rank', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))
    bars = ax.barh(
        df_sorted['Model'],
        df_sorted['TOPSIS Score'],
        color=colors,
        edgecolor='black',
        linewidth=1.2
    )
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, df_sorted['TOPSIS Score'])):
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{score:.4f}',
            ha='left',
            va='center',
            fontweight='bold',
            fontsize=10
        )
    
    # Add rank medals to top 3
    medals = {1: '🥇', 2: '🥈', 3: '🥉'}
    for i, row in df_sorted.iterrows():
        if row['Rank'] in medals:
            ax.text(
                0.01,
                row['Model'],
                medals[row['Rank']],
                ha='left',
                va='center',
                fontsize=16
            )
    
    # Customize plot
    ax.set_xlabel('TOPSIS Score', fontweight='bold', fontsize=12)
    ax.set_ylabel('Model', fontweight='bold', fontsize=12)
    ax.set_title(
        'TOPSIS-Based Model Ranking for Text Summarization',
        fontweight='bold',
        fontsize=14,
        pad=20
    )
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Invert y-axis so best model is on top
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Ranking visualization saved to: {output_path}")
    plt.close()


def create_metrics_comparison(df: pd.DataFrame, output_path: str):
    """
    Create a multi-panel visualization comparing all metrics.
    
    Args:
        df (pd.DataFrame): DataFrame with all metrics
        output_path (str): Path to save the figure
    """
    # Quality metrics
    quality_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']
    
    # Efficiency metrics
    efficiency_metrics = ['Latency (ms)', 'Size (M params)', 'Memory (MB)']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Comprehensive Model Comparison Across All Metrics',
        fontsize=16,
        fontweight='bold',
        y=1.00
    )
    
    # Plot 1: Quality Metrics (Grouped Bar Chart)
    ax1 = axes[0, 0]
    quality_data = df[['Model'] + quality_metrics].set_index('Model')
    quality_data.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Quality Metrics (Higher is Better)', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_xlabel('')
    ax1.legend(title='Metrics', loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Efficiency Metrics (Grouped Bar Chart)
    ax2 = axes[0, 1]
    efficiency_data = df[['Model'] + efficiency_metrics].set_index('Model')
    
    # Normalize efficiency metrics for visualization (0-1 scale)
    efficiency_normalized = efficiency_data.copy()
    for col in efficiency_metrics:
        max_val = efficiency_normalized[col].max()
        efficiency_normalized[col] = efficiency_normalized[col] / max_val
    
    efficiency_normalized.plot(kind='bar', ax=ax2, width=0.8, color=['coral', 'orange', 'gold'])
    ax2.set_title('Efficiency Metrics - Normalized (Lower is Better)', fontweight='bold')
    ax2.set_ylabel('Normalized Score', fontweight='bold')
    ax2.set_xlabel('')
    ax2.legend(title='Metrics', loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 3: TOPSIS Score Comparison
    ax3 = axes[1, 0]
    df_sorted = df.sort_values('TOPSIS Score', ascending=False)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))
    ax3.bar(df_sorted['Model'], df_sorted['TOPSIS Score'], color=colors, edgecolor='black')
    ax3.set_title('Final TOPSIS Scores', fontweight='bold')
    ax3.set_ylabel('TOPSIS Score', fontweight='bold')
    ax3.set_xlabel('Model', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 4: Radar Chart for Top Model
    ax4 = axes[1, 1]
    top_model = df.loc[df['TOPSIS Score'].idxmax()]
    
    # Prepare radar chart data
    categories = quality_metrics + ['Efficiency']
    
    # Normalize all metrics to 0-1 scale
    values = []
    for metric in quality_metrics:
        max_val = df[metric].max()
        normalized = top_model[metric] / max_val if max_val > 0 else 0
        values.append(normalized)
    
    # Efficiency (inverse normalized - higher is better after inversion)
    efficiency_score = 1 - (top_model['Latency (ms)'] / df['Latency (ms)'].max())
    values.append(efficiency_score)
    
    # Close the plot
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    ax4.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', label=top_model['Model'])
    ax4.fill(angles, values, alpha=0.25, color='#2E86AB')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, size=9)
    ax4.set_ylim(0, 1)
    ax4.set_title(f'Best Model: {top_model["Model"]}', fontweight='bold', pad=20)
    ax4.grid(True)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive metrics comparison saved to: {output_path}")
    plt.close()


def print_summary_report(df: pd.DataFrame):
    """
    Print a formatted summary report to console.
    
    Args:
        df (pd.DataFrame): DataFrame with all results
    """
    print("\n" + "="*80)
    print("TOPSIS-BASED MODEL SELECTION SUMMARY REPORT")
    print("="*80)
    
    # Best model
    best_model = df.iloc[0]
    
    print(f"\n🏆 RECOMMENDED MODEL: {best_model['Model']}")
    print(f"   TOPSIS Score: {best_model['TOPSIS Score']:.4f}")
    print(f"   Rank: {best_model['Rank']}")
    
    print("\n📊 PERFORMANCE BREAKDOWN:")
    print(f"   ROUGE-1:      {best_model['ROUGE-1']:.4f}")
    print(f"   ROUGE-2:      {best_model['ROUGE-2']:.4f}")
    print(f"   ROUGE-L:      {best_model['ROUGE-L']:.4f}")
    print(f"   BLEU:         {best_model['BLEU']:.4f}")
    print(f"   Latency:      {best_model['Latency (ms)']:.2f} ms")
    print(f"   Model Size:   {best_model['Size (M params)']:.1f} M parameters")
    print(f"   Memory:       {best_model['Memory (MB)']:.1f} MB")
    
    print("\n📈 COMPLETE RANKINGS:")
    print("-"*80)
    display_df = df[['Rank', 'Model', 'TOPSIS Score', 'ROUGE-1', 'BLEU', 'Latency (ms)']]
    print(display_df.to_string(index=False))
    print("-"*80)
    
    print("\n💡 KEY INSIGHTS:")
    
    # Find best in each category
    best_quality = df.loc[df['ROUGE-1'].idxmax()]['Model']
    best_speed = df.loc[df['Latency (ms)'].idxmin()]['Model']
    best_size = df.loc[df['Size (M params)'].idxmin()]['Model']
    
    print(f"   • Best Quality:     {best_quality}")
    print(f"   • Fastest Inference: {best_speed}")
    print(f"   • Smallest Model:   {best_size}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main execution pipeline."""
    print("\n" + "="*80)
    print("TOPSIS-BASED TEXT SUMMARIZATION MODEL SELECTION")
    print("="*80 + "\n")
    
    # Define paths
    dataset_path = project_root / 'dataset' / 'sample_texts.json'
    metrics_csv_path = project_root / 'results' / 'metrics.csv'
    ranked_csv_path = project_root / 'results' / 'ranked_models.csv'
    graph_path = project_root / 'results' / 'graph.png'
    comparison_path = project_root / 'results' / 'comparison.png'
    
    # Ensure results directory exists
    (project_root / 'results').mkdir(exist_ok=True)
    
    # Step 1: Evaluate models
    print("STEP 1: Evaluating Pretrained Models")
    print("-"*80)
    
    if not metrics_csv_path.exists():
        evaluator = ModelEvaluator(str(dataset_path))
        results_df = evaluator.evaluate_all_models()
        evaluator.save_results(str(metrics_csv_path))
    else:
        print(f"Loading existing metrics from: {metrics_csv_path}")
        results_df = pd.read_csv(metrics_csv_path)
        print("\nLoaded Results:")
        print(results_df.to_string(index=False))
    
    # Step 2: Apply TOPSIS
    print("\n" + "="*80)
    print("STEP 2: Applying TOPSIS Multi-Criteria Decision Analysis")
    print("-"*80 + "\n")
    
    # Define criteria
    criteria_columns = [
        'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU',
        'Latency (ms)', 'Size (M params)', 'Memory (MB)'
    ]
    
    # Define weights (sum must equal 1)
    weights = {
        'ROUGE-1': 0.20,        # Quality metrics: 65% total
        'ROUGE-2': 0.20,
        'ROUGE-L': 0.15,
        'BLEU': 0.10,
        'Latency (ms)': 0.20,   # Efficiency metrics: 35% total
        'Size (M params)': 0.10,
        'Memory (MB)': 0.05
    }
    
    # Define criteria types (+ for benefit, - for cost)
    criteria_types = {
        'ROUGE-1': '+',
        'ROUGE-2': '+',
        'ROUGE-L': '+',
        'BLEU': '+',
        'Latency (ms)': '-',
        'Size (M params)': '-',
        'Memory (MB)': '-'
    }
    
    print("Criteria Weights:")
    for criterion, weight in weights.items():
        criterion_type = "Benefit (↑)" if criteria_types[criterion] == '+' else "Cost (↓)"
        print(f"  {criterion:20s}: {weight:.2f} ({criterion_type})")
    
    # Apply TOPSIS
    ranked_df = apply_topsis_to_dataframe(
        results_df,
        criteria_columns,
        weights,
        criteria_types,
        alternative_column='Model'
    )
    
    # Save ranked results
    ranked_df.to_csv(ranked_csv_path, index=False)
    print(f"\n✓ Ranked results saved to: {ranked_csv_path}")
    
    # Step 3: Generate visualizations
    print("\n" + "="*80)
    print("STEP 3: Generating Visualizations")
    print("-"*80)
    
    setup_visualization_style()
    
    # Create ranking visualization
    create_ranking_visualization(ranked_df, str(graph_path))
    
    # Create comprehensive comparison
    create_metrics_comparison(ranked_df, str(comparison_path))
    
    # Step 4: Print summary report
    print_summary_report(ranked_df)
    
    print("✅ Pipeline completed successfully!")
    print(f"\nResults saved to: {project_root / 'results'}")
    print("\nGenerated files:")
    print(f"  • {metrics_csv_path.name}")
    print(f"  • {ranked_csv_path.name}")
    print(f"  • {graph_path.name}")
    print(f"  • {comparison_path.name}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
