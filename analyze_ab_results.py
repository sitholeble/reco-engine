"""
Analysis and visualization script for A/B testing results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ab_testing_service import ABTestingService
import json

def plot_experiment_metrics(service: ABTestingService, experiment_id: str, save_path: str = None):
    """
    Create visualization of experiment metrics.
    
    Args:
        service: ABTestingService instance
        experiment_id: Experiment identifier
        save_path: Optional path to save the plot
    """
    metrics = service.get_experiment_metrics(experiment_id)
    experiment = service.framework.experiments[experiment_id]
    
    # Prepare data for plotting
    variants = list(metrics.keys())
    ctr_values = [metrics[v]['click_through_rate'] for v in variants]
    conversion_values = [metrics[v]['conversion_rate'] for v in variants]
    engagement_values = [metrics[v]['engagement_score'] for v in variants]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # CTR comparison
    axes[0].bar(variants, ctr_values, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_title('Click-Through Rate by Variant')
    axes[0].set_ylabel('CTR')
    axes[0].set_ylim([0, max(ctr_values) * 1.2 if ctr_values else 1])
    for i, v in enumerate(ctr_values):
        axes[0].text(i, v + max(ctr_values) * 0.02, f'{v:.3f}', ha='center')
    
    # Conversion rate comparison
    axes[1].bar(variants, conversion_values, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1].set_title('Conversion Rate by Variant')
    axes[1].set_ylabel('Conversion Rate')
    axes[1].set_ylim([0, max(conversion_values) * 1.2 if conversion_values else 1])
    for i, v in enumerate(conversion_values):
        axes[1].text(i, v + max(conversion_values) * 0.02, f'{v:.3f}', ha='center')
    
    # Engagement score comparison
    axes[2].bar(variants, engagement_values, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[2].set_title('Engagement Score by Variant')
    axes[2].set_ylabel('Engagement Score')
    axes[2].set_ylim([0, max(engagement_values) * 1.2 if engagement_values else 1])
    for i, v in enumerate(engagement_values):
        axes[2].text(i, v + max(engagement_values) * 0.02, f'{v:.3f}', ha='center')
    
    plt.suptitle(f'Experiment: {experiment.name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def print_detailed_report(service: ABTestingService, experiment_id: str):
    """Print a detailed text report of experiment results."""
    metrics = service.get_experiment_metrics(experiment_id)
    comparison = service.get_experiment_results(experiment_id)
    experiment = service.framework.experiments[experiment_id]
    
    print("\n" + "="*70)
    print(f"EXPERIMENT REPORT: {experiment.name}")
    print("="*70)
    print(f"Experiment ID: {experiment_id}")
    print(f"Status: {experiment.status}")
    print(f"Start Date: {experiment.start_date}")
    print(f"End Date: {experiment.end_date}")
    print(f"Variants: {', '.join(experiment.variants)}")
    print("\n" + "-"*70)
    
    # Metrics by variant
    print("\nMETRICS BY VARIANT:")
    print("-"*70)
    for variant in experiment.variants:
        variant_metrics = metrics[variant]
        print(f"\n{variant.upper()}:")
        print(f"  Total Interactions: {variant_metrics.get('total_interactions', 0)}")
        print(f"  Impressions: {variant_metrics.get('impressions', 0)}")
        print(f"  Clicks: {variant_metrics.get('clicks', 0)}")
        print(f"  Bookings: {variant_metrics.get('bookings', 0)}")
        print(f"  Confirmed Bookings: {variant_metrics.get('confirmed_bookings', 0)}")
        print(f"  Click-Through Rate: {variant_metrics.get('click_through_rate', 0):.4f} ({variant_metrics.get('click_through_rate', 0)*100:.2f}%)")
        print(f"  Conversion Rate: {variant_metrics.get('conversion_rate', 0):.4f} ({variant_metrics.get('conversion_rate', 0)*100:.2f}%)")
        print(f"  Booking Rate: {variant_metrics.get('booking_rate', 0):.4f} ({variant_metrics.get('booking_rate', 0)*100:.2f}%)")
        print(f"  Engagement Score: {variant_metrics.get('engagement_score', 0):.4f}")
    
    # Comparison
    if comparison.get('comparison'):
        print("\n" + "-"*70)
        print("VARIANT COMPARISON:")
        print("-"*70)
        control = experiment.variants[0]
        for variant, comp_data in comparison['comparison'].items():
            print(f"\n{variant.upper()} vs {control.upper()}:")
            print(f"  CTR Improvement: {comp_data['ctr_improvement']:+.2f}%")
            print(f"  Conversion Improvement: {comp_data['conversion_improvement']:+.2f}%")
            print(f"  Engagement Improvement: {comp_data['engagement_improvement']:+.2f}%")
            
            # Determine winner
            if comp_data['engagement_improvement'] > 5:
                print(f"  → {variant} is performing significantly better!")
            elif comp_data['engagement_improvement'] < -5:
                print(f"  → {control} is performing significantly better!")
            else:
                print(f"  → Results are similar (within 5% difference)")
    
    print("\n" + "="*70)

def export_results_to_csv(service: ABTestingService, experiment_id: str, output_path: str):
    """Export experiment results to CSV."""
    metrics = service.get_experiment_metrics(experiment_id)
    
    # Create DataFrame
    rows = []
    for variant, variant_metrics in metrics.items():
        row = {'variant': variant}
        row.update(variant_metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")

def main():
    """Main function to analyze experiment results."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_ab_results.py <experiment_id> [--plot] [--export <path>]")
        print("\nExample:")
        print("  python analyze_ab_results.py exp_001")
        print("  python analyze_ab_results.py exp_001 --plot")
        print("  python analyze_ab_results.py exp_001 --export results.csv")
        return
    
    experiment_id = sys.argv[1]
    service = ABTestingService()
    
    if experiment_id not in service.framework.experiments:
        print(f"Experiment {experiment_id} not found!")
        print(f"Available experiments: {list(service.framework.experiments.keys())}")
        return
    
    # Print report
    print_detailed_report(service, experiment_id)
    
    # Optional: create plot
    if '--plot' in sys.argv:
        try:
            plot_experiment_metrics(service, experiment_id)
        except ImportError:
            print("\nWarning: matplotlib not available. Install with: pip install matplotlib")
    
    # Optional: export to CSV
    if '--export' in sys.argv:
        idx = sys.argv.index('--export')
        if idx + 1 < len(sys.argv):
            export_path = sys.argv[idx + 1]
            export_results_to_csv(service, experiment_id, export_path)
        else:
            export_results_to_csv(service, experiment_id, f"{experiment_id}_results.csv")

if __name__ == '__main__':
    main()

