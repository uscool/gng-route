import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add Capstone root to path
CAPSTONE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CAPSTONE_ROOT)

from uniroute.data import load_routerbench_data, get_embeddings, split_dataset, DEFAULT_LLM_NAMES
from gng_shared.main import GNGSharedRouterModel

def evaluate_gng_shared_router(router, test_embeddings, test_df, llm_names, lambda_grid=None):
    """
    Highly optimized evaluation for the GNG shared router.
    """
    n_samples = len(test_embeddings)
    n_models = len(llm_names)
    
    if lambda_grid is None:
        lambda_grid = [0.0] + list(np.logspace(-1, 6, 500))
        
    # Get mean accuracy and cost for each single LLM on the test set
    llm_stats = {}
    for model_name in llm_names:
        llm_stats[model_name] = {
            'accuracy': test_df[model_name].mean(),
            'cost': test_df[f"{model_name}|total_cost"].mean()
        }
        
    costs_list = [stats['cost'] for stats in llm_stats.values()]
    c_min = min(costs_list)
    c_max = max(costs_list)
    a_max = max(stats['accuracy'] for stats in llm_stats.values())
    
    # Precompute nearest node index for each test sample
    node_ids = list(router.topology.nodes.keys())
    node_weights = np.array([router.topology.nodes[nid]['weight'] for nid in node_ids])
    
    nearest_node_indices = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        x = test_embeddings[i]
        dists = np.sum((node_weights - x) ** 2, axis=1)
        nearest_node_indices[i] = np.argmin(dists)
        
    # Precompute node error matrix: shape (n_nodes, n_models)
    error_matrix = np.zeros((len(node_ids), n_models))
    for i, nid in enumerate(node_ids):
        for m_idx, name in enumerate(llm_names):
            error_matrix[i, m_idx] = router.node_error_rates[nid][name]
            
    # LLM costs and accuracy vectors
    llm_costs = np.array([llm_stats[name]['cost'] for name in llm_names])
    correctness_matrix = test_df[llm_names].values
    cost_matrix = test_df[[f"{name}|total_cost" for name in llm_names]].values
    
    # Precompute mapped error matrix for test set: shape (n_samples, n_models)
    test_errors = error_matrix[nearest_node_indices]
    
    # Sweep lambda
    raw_points = []
    for l_val in lambda_grid:
        scores = test_errors + l_val * llm_costs[np.newaxis, :]
        routed_indices = np.argmin(scores, axis=1)
        
        # Evaluate accuracy and cost
        acc = np.mean(correctness_matrix[np.arange(n_samples), routed_indices])
        cost = np.mean(cost_matrix[np.arange(n_samples), routed_indices])
        raw_points.append((cost, acc))
        
    # Normalize costs to Relative Cost in [0, 1]
    rel_points = []
    for cost, acc in raw_points:
        rel_cost = (cost - c_min) / (c_max - c_min) if c_max > c_min else 0.0
        rel_cost = np.clip(rel_cost, 0.0, 1.0)
        rel_points.append((rel_cost, acc))
        
    # Add boundary points
    cheapest_model_name = min(llm_names, key=lambda name: llm_stats[name]['cost'])
    a_cheapest = llm_stats[cheapest_model_name]['accuracy']
    expensive_model_name = max(llm_names, key=lambda name: llm_stats[name]['cost'])
    a_expensive = llm_stats[expensive_model_name]['accuracy']
    
    rel_points.append((0.0, a_cheapest))
    rel_points.append((1.0, a_expensive))
    
    # Sort points by relative cost
    rel_points = sorted(list(set(rel_points)), key=lambda p: p[0])
    
    relative_costs = np.array([p[0] for p in rel_points])
    accuracies = np.array([p[1] for p in rel_points])
    
    # Calculate Area (AUDC) using trapezoid integration
    area_100 = np.trapezoid(accuracies, relative_costs)
    
    # Calculate Area@50%
    a_50 = np.interp(0.5, relative_costs, accuracies)
    idx_50 = relative_costs <= 0.5
    rel_costs_50 = list(relative_costs[idx_50]) + [0.5]
    accuracies_50 = list(accuracies[idx_50]) + [a_50]
    
    points_50 = sorted(list(set(zip(rel_costs_50, accuracies_50))), key=lambda p: p[0])
    r_50 = np.array([p[0] for p in points_50])
    acc_50 = np.array([p[1] for p in points_50])
    area_50 = np.trapezoid(acc_50, r_50)
    
    # Calculate QNC
    valid_indices = np.where(accuracies >= a_max)[0]
    qnc = float(relative_costs[valid_indices[0]]) if len(valid_indices) > 0 else float('inf')
    
    return {
        'area': area_100,
        'area_50': area_50,
        'qnc': qnc,
        'relative_costs': relative_costs,
        'accuracies': accuracies,
        'c_min': c_min,
        'c_max': c_max,
        'a_max': a_max,
        'llm_stats': llm_stats
    }

def plot_gng_deferral_curve(metrics, output_path):
    plt.figure(figsize=(8, 6))
    
    # Plot deferral curve
    plt.plot(metrics['relative_costs'], metrics['accuracies'], label='GNG Shared Router', color='blue', linewidth=2)
    
    # Plot individual LLMs
    for name, stats in metrics['llm_stats'].items():
        short_name = name.split('/')[-1]
        rel_cost = (stats['cost'] - metrics['c_min']) / (metrics['c_max'] - metrics['c_min']) if metrics['c_max'] > metrics['c_min'] else 0.0
        plt.scatter(rel_cost, stats['accuracy'], color='red', s=40, zorder=5)
        plt.text(rel_cost + 0.01, stats['accuracy'], short_name, fontsize=8, alpha=0.8)
        
    # Label formatting
    plt.xlabel('Relative Cost (0.0 = Cheapest, 1.0 = Most Expensive)', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Deferral Curve - GNG Shared Router', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-0.05, 1.05)
    
    # Add text box for metrics
    qnc_str = f"{metrics['qnc']*100:.1f}%" if metrics['qnc'] != float('inf') else "N/A"
    textstr = '\n'.join((
        f"Area: {metrics['area']:.4f}",
        f"Area@50%: {metrics['area_50']:.4f}",
        f"QNC: {qnc_str}"
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Deferral curve plot saved to: {output_path}")

def main():
    print("=== GNG Shared Router Evaluation ===")
    os.makedirs("gng_shared/outputs", exist_ok=True)
    
    # 1. Load data & embeddings
    df = load_routerbench_data("routertrain.csv")
    embeddings = get_embeddings(df['prompt'].tolist(), model_name="BAAI/bge-small-en-v1.5", cache_path="prompt_embeddings_cache.pkl")
    
    # 2. Split dataset (70-30 train-test)
    splits = split_dataset(df, embeddings, train_ratio=0.7, val_ratio=0.0, test_ratio=0.3, seed=42)
    
    # 3. Train GNG Router
    router = GNGSharedRouterModel(llm_names=DEFAULT_LLM_NAMES, max_nodes=200)
    router.fit(splits['train_embeddings'], splits['train_df'], epochs=5)
    
    # Costs
    llm_costs = [splits['train_df'][f"{llm}|total_cost"].mean() for llm in DEFAULT_LLM_NAMES]
    
    # 4. Save model
    with open("gng_shared_model.pkl", "wb") as f:
        pickle.dump({
            'router': router,
            'llm_names': DEFAULT_LLM_NAMES,
            'llm_costs': llm_costs,
            'model_name': 'BAAI/bge-small-en-v1.5'
        }, f)
    print("Saved trained model to gng_shared_model.pkl")
    
    # 5. Evaluate
    metrics = evaluate_gng_shared_router(router, splits['test_embeddings'], splits['test_df'], DEFAULT_LLM_NAMES)
    
    # Print results
    print("\n" + "="*50)
    print("GNG SHARED ROUTER EVALUATION METRICS (70-30 split):")
    print(f"  Area:     {metrics['area']:.4f}")
    print(f"  Area@50%: {metrics['area_50']:.4f}")
    qnc_val = f"{metrics['qnc']*100:.2f}%" if metrics['qnc'] != float('inf') else "N/A"
    print(f"  QNC:      {qnc_val}")
    print("="*50 + "\n")
    
    # Save metrics JSON
    with open("gng_shared/outputs/metrics.json", "w") as f:
        json.dump({
            'area': float(metrics['area']),
            'area_50': float(metrics['area_50']),
            'qnc': float(metrics['qnc']) if metrics['qnc'] != float('inf') else None,
            'c_min': float(metrics['c_min']),
            'c_max': float(metrics['c_max']),
            'a_max': float(metrics['a_max'])
        }, f, indent=4)
        
    # Plot curve
    plot_gng_deferral_curve(metrics, "gng_shared/outputs/deferral_curve.png")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
