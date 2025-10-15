import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def load_json_report(file_path: str) -> Dict[str, Any]:
    """Load JSON report file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not load {file_path}: {e}")
        return {}

def compare_model_performance(root: str):
    """Generate comprehensive comparison report"""
    print("=== GENERATING FINAL MODEL COMPARISON REPORT ===")
    
    # Load all reports
    reports = {}
    report_files = [
        "gnn_multitask_report.json",
        "gnn_shot_report.json", 
        "gnn_receiver_report.json",
        "tabular_report.json",
        "hyperparameter_search_report.json"
    ]
    
    for file_name in report_files:
        file_path = os.path.join(root, file_name)
        reports[file_name] = load_json_report(file_path)
    
    # Extract key metrics
    comparison_data = []
    
    # Tabular baseline
    if reports["tabular_report.json"]:
        tabular_shot = reports["tabular_report.json"].get("shot", {})
        tabular_receiver = reports["tabular_report.json"].get("receiver", {})
        comparison_data.append({
            "model": "Tabular Baseline",
            "shot_f1": tabular_shot.get("f1", 0),
            "shot_roc_auc": tabular_shot.get("roc_auc", 0),
            "shot_pr_auc": tabular_shot.get("pr_auc", 0),
            "receiver_top1": tabular_receiver.get("top1", 0),
            "receiver_top3": tabular_receiver.get("top3", 0),
            "receiver_top5": tabular_receiver.get("top5", 0),
            "receiver_map3": tabular_receiver.get("map3", 0),
            "receiver_ndcg3": tabular_receiver.get("ndcg3", 0)
        })
    
    # GNN Multitask
    if reports["gnn_multitask_report.json"]:
        gnn_shot = reports["gnn_multitask_report.json"].get("shot", {})
        gnn_receiver = reports["gnn_multitask_report.json"].get("receiver", {})
        comparison_data.append({
            "model": "GNN Multitask",
            "shot_f1": gnn_shot.get("f1", 0),
            "shot_roc_auc": gnn_shot.get("roc_auc", 0),
            "shot_pr_auc": gnn_shot.get("pr_auc", 0),
            "receiver_top1": gnn_receiver.get("top1", 0),
            "receiver_top3": gnn_receiver.get("top3", 0),
            "receiver_top5": gnn_receiver.get("top5", 0),
            "receiver_map3": gnn_receiver.get("map3", 0),
            "receiver_ndcg3": gnn_receiver.get("ndcg3", 0)
        })
    
    # GNN Single Task Shot
    if reports["gnn_shot_report.json"]:
        shot_metrics = reports["gnn_shot_report.json"].get("shot", {})
        # Find existing receiver metrics for this model combination
        receiver_metrics = {}
        if reports["gnn_receiver_report.json"]:
            receiver_metrics = reports["gnn_receiver_report.json"].get("receiver", {})
        comparison_data.append({
            "model": "GNN Single Task",
            "shot_f1": shot_metrics.get("f1", 0),
            "shot_roc_auc": shot_metrics.get("roc_auc", 0),
            "shot_pr_auc": shot_metrics.get("pr_auc", 0),
            "receiver_top1": receiver_metrics.get("top1", 0),
            "receiver_top3": receiver_metrics.get("top3", 0),
            "receiver_top5": receiver_metrics.get("top5", 0),
            "receiver_map3": receiver_metrics.get("map3", 0),
            "receiver_ndcg3": receiver_metrics.get("ndcg3", 0)
        })
    
    # Create comparison DataFrame
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print("\n=== MODEL PERFORMANCE COMPARISON ===")
        print(df.to_string(index=False))
        
        # Save to CSV
        df.to_csv(os.path.join(root, "model_comparison.csv"), index=False)
        
        # Create visualization
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Shot prediction metrics
        shot_metrics = ['shot_f1', 'shot_roc_auc', 'shot_pr_auc']
        x_pos = np.arange(len(df))
        width = 0.25
        
        for i, metric in enumerate(shot_metrics):
            ax = axes[0, i] if i < 3 else axes[1, 0]
            values = df[metric].values
            bars = ax.bar(x_pos, values, width, label=metric)
            ax.set_xlabel('Models')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'Shot Prediction - {metric.upper()}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df['model'], rotation=45, ha='right')
            ax.legend()
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Receiver prediction metrics
        receiver_metrics = ['receiver_top1', 'receiver_top3', 'receiver_map3']
        for i, metric in enumerate(receiver_metrics):
            ax = axes[1, i]
            values = df[metric].values
            bars = ax.bar(x_pos, values, width, label=metric)
            ax.set_xlabel('Models')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'Receiver Prediction - {metric.upper()}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df['model'], rotation=45, ha='right')
            ax.legend()
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(root, "model_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Determine best model
        best_model = determine_best_model(df)
        print(f"\n=== RECOMMENDATION ===")
        print(f"Based on evaluation metrics, the {best_model} model performs best overall.")
        
        # Generate detailed recommendation report
        recommendation = {
            "best_model": best_model,
            "performance_summary": df.to_dict('records'),
            "recommendation_details": generate_recommendation_details(df, best_model)
        }
        
        with open(os.path.join(root, "final_recommendation.json"), "w") as f:
            json.dump(recommendation, f, indent=2)
        
        print(f"\nDetailed recommendation saved to final_recommendation.json")
        return recommendation
    
    else:
        print("No model reports found for comparison")
        return None


def determine_best_model(df):
    """Determine the best performing model based on weighted metrics"""
    if df.empty:
        return "Unknown"
    
    # Weighted scoring: 40% shot F1, 30% shot ROC-AUC, 30% receiver Top-3
    df['combined_score'] = (
        0.4 * df['shot_f1'] + 
        0.3 * df['shot_roc_auc'] + 
        0.3 * df['receiver_top3']
    )
    
    best_idx = df['combined_score'].idxmax()
    return df.loc[best_idx, 'model']


def generate_recommendation_details(df, best_model):
    """Generate detailed recommendation based on model comparison"""
    details = {
        "best_model_name": best_model,
        "improvement_analysis": {}
    }
    
    if df.empty:
        return details
    
    best_row = df[df['model'] == best_model].iloc[0] if best_model in df['model'].values else df.iloc[0]
    
    # Compare with tabular baseline
    tabular_row = df[df['model'] == 'Tabular Baseline'].iloc[0] if 'Tabular Baseline' in df['model'].values else None
    
    if tabular_row is not None:
        shot_improvement = best_row['shot_f1'] - tabular_row['shot_f1']
        receiver_improvement = best_row['receiver_top3'] - tabular_row['receiver_top3']
        
        details["improvement_analysis"] = {
            "shot_f1_improvement": float(shot_improvement),
            "receiver_top3_improvement": float(receiver_improvement),
            "shot_relative_improvement": f"{(shot_improvement/tabular_row['shot_f1']*100):+.1f}%" if tabular_row['shot_f1'] > 0 else "N/A",
            "receiver_relative_improvement": f"{(receiver_improvement/tabular_row['receiver_top3']*100):+.1f}%" if tabular_row['receiver_top3'] > 0 else "N/A"
        }
        
        if shot_improvement > 0.02 or receiver_improvement > 0.02:
            details["recommendation"] = f"The {best_model} model shows significant improvement over the tabular baseline and is recommended for production use."
        elif shot_improvement > 0 or receiver_improvement > 0:
            details["recommendation"] = f"The {best_model} model shows modest improvement over the tabular baseline and could be considered for production use."
        else:
            details["recommendation"] = "The tabular baseline performs as well as or better than GNN models. Consider sticking with the simpler tabular approach."
    else:
        details["recommendation"] = f"The {best_model} model is recommended based on available metrics."
    
    return details


def generate_model_insights(root: str):
    """Generate insights about model performance and feature importance"""
    print("\n=== GENERATING MODEL INSIGHTS ===")
    
    # Load hyperparameter search results
    hyperparameter_report = load_json_report(os.path.join(root, "hyperparameter_search_report.json"))
    
    if hyperparameter_report and "all_results" in hyperparameter_report:
        results = hyperparameter_report["all_results"]
        
        # Analyze hyperparameter impact
        param_impact = {}
        metrics = ["shot_f1", "receiver_top3", "combined_score"]
        
        for metric in metrics:
            param_impact[metric] = {}
            
            # Group by each hyperparameter
            for param in ["hidden", "heads", "lr", "dropout", "radius"]:
                param_values = {}
                for result in results:
                    param_val = result["config"][param]
                    metric_val = result["metrics"][metric]
                    
                    if param_val not in param_values:
                        param_values[param_val] = []
                    param_values[param_val].append(metric_val)
                
                # Calculate average performance for each parameter value
                param_impact[metric][param] = {
                    str(k): {
                        "mean": float(np.mean(v)),
                        "std": float(np.std(v)),
                        "count": len(v)
                    } for k, v in param_values.items()
                }
        
        # Save insights
        with open(os.path.join(root, "hyperparameter_insights.json"), "w") as f:
            json.dump(param_impact, f, indent=2)
        
        print("Hyperparameter impact analysis saved to hyperparameter_insights.json")
        
        # Print key insights
        print("\n=== KEY HYPERPARAMETER INSIGHTS ===")
        for metric in metrics[:2]:  # Focus on key metrics
            print(f"\n{metric.upper()} performance by hyperparameter:")
            for param, values in param_impact[metric].items():
                print(f"  {param}:")
                for val, stats in values.items():
                    print(f"    {val}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")


def generate_final_report(root: str):
    """Generate the complete final report"""
    print("=== GENERATING COMPLETE FINAL REPORT ===")
    
    # Run model comparison
    comparison_result = compare_model_performance(root)
    
    # Generate insights
    generate_model_insights(root)
    
    # Create summary report
    summary = {
        "report_generated": pd.Timestamp.now().isoformat(),
        "model_comparison": comparison_result,
        "key_findings": [],
        "recommendations": []
    }
    
    if comparison_result:
        best_model = comparison_result["best_model_name"]
        summary["key_findings"].append(f"The {best_model} model achieved the best overall performance.")
        
        improvement = comparison_result.get("improvement_analysis", {})
        if improvement.get("shot_f1_improvement", 0) > 0:
            summary["key_findings"].append(
                f"Shot prediction F1 score improved by {improvement['shot_f1_improvement']:.3f} "
                f"({improvement['shot_relative_improvement']}) compared to tabular baseline."
            )
        
        if improvement.get("receiver_top3_improvement", 0) > 0:
            summary["key_findings"].append(
                f"Receiver prediction Top-3 accuracy improved by {improvement['receiver_top3_improvement']:.3f} "
                f"({improvement['receiver_relative_improvement']}) compared to tabular baseline."
            )
        
        summary["recommendations"].append(comparison_result.get("recommendation", "See detailed analysis for recommendations."))
    
    # Save summary
    with open(os.path.join(root, "final_summary_report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nComplete final report saved to {os.path.join(root, 'final_summary_report.json')}")
    
    return summary


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    generate_final_report(here)