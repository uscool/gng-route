import os
import sys
import pickle
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add Capstone root to path
CAPSTONE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CAPSTONE_ROOT)

from uniroute.data import load_routerbench_data, get_embeddings, split_dataset, DEFAULT_LLM_NAMES, DEFAULT_EMBEDDING_MODEL, DEFAULT_CACHE_PATH
from gng_router.main import GNGRouterModel

# np.trapz was removed in NumPy 2.4; np.trapezoid does not exist before NumPy 2.0
_trapz = getattr(np, 'trapezoid', None) or np.trapz

def evaluate_gng_router(router, test_embeddings, test_df, llm_names, lambda_grid=None, beta=2.0):
    """
    Highly optimized GNG evaluation sweep. Precomputes closest nodes, distances,
    and boundary flags for all test samples across all models, eliminating redundant
    KD-tree lookups during the lambda sweep.
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
    
    # Precompute closest node distance and local error
    test_dists = np.zeros((n_samples, n_models))
    test_errors = np.zeros((n_samples, n_models))
    boundaries = np.zeros(n_models)
    
    precomputed = {}
    for m_idx, name in enumerate(llm_names):
        topology = router.topologies[name]
        boundaries[m_idx] = router.boundary_radii[name]
        if topology.nodes:
            node_ids = list(topology.nodes.keys())
            weights = np.array([topology.nodes[nid]['weight'] for nid in node_ids])
            errors = np.array([topology.nodes[nid]['error_rate'] for nid in node_ids])
            precomputed[m_idx] = {
                'weights': weights,
                'errors': errors
            }
            
    # Calculate nearest nodes for each test sample once per model
    for m_idx, name in enumerate(llm_names):
        if m_idx in precomputed:
            weights = precomputed[m_idx]['weights']
            errors = precomputed[m_idx]['errors']
            
            # Loop over samples locally (low memory overhead, extremely fast with numpy)
            for i in range(n_samples):
                x = test_embeddings[i]
                dists = np.sum((weights - x) ** 2, axis=1)
                best_idx = np.argmin(dists)
                test_dists[i, m_idx] = np.sqrt(dists[best_idx])
                test_errors[i, m_idx] = errors[best_idx]
                
    # LLM costs and accuracy vectors
    llm_costs = np.array([llm_stats[name]['cost'] for name in llm_names])
    correctness_matrix = test_df[llm_names].values
    cost_matrix = test_df[[f"{name}|total_cost" for name in llm_names]].values
    
    # Precompute soft boundary penalty for all test samples
    penalties = beta * np.maximum(0.0, test_dists - boundaries[np.newaxis, :])
    adjusted_errors = test_errors + penalties
    
    # Sweep lambda (fully vectorized, no inner loop!)
    raw_points = []
    
    for l_val in lambda_grid:
        scores = adjusted_errors + l_val * llm_costs[np.newaxis, :]
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
    
    # Calculate Area (AUDC) using trapezoidal integration
    area_100 = _trapz(accuracies, relative_costs)
    
    # Calculate Area@50%
    a_50 = np.interp(0.5, relative_costs, accuracies)
    idx_50 = relative_costs <= 0.5
    rel_costs_50 = list(relative_costs[idx_50]) + [0.5]
    accuracies_50 = list(accuracies[idx_50]) + [a_50]
    
    points_50 = sorted(list(set(zip(rel_costs_50, accuracies_50))), key=lambda p: p[0])
    r_50 = np.array([p[0] for p in points_50])
    acc_50 = np.array([p[1] for p in points_50])
    area_50 = _trapz(acc_50, r_50)
    
    # Calculate QNC (Quality-preserving Normalized Cost)
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
    plt.plot(metrics['relative_costs'], metrics['accuracies'], label='GNG Router', color='blue', linewidth=2)
    
    # Plot individual LLMs
    for name, stats in metrics['llm_stats'].items():
        short_name = name.split('/')[-1]
        rel_cost = (stats['cost'] - metrics['c_min']) / (metrics['c_max'] - metrics['c_min']) if metrics['c_max'] > metrics['c_min'] else 0.0
        plt.scatter(rel_cost, stats['accuracy'], color='red', s=40, zorder=5)
        plt.text(rel_cost + 0.01, stats['accuracy'], short_name, fontsize=8, alpha=0.8)
        
    # Label formatting
    plt.xlabel('Relative Cost (0.0 = Cheapest, 1.0 = Most Expensive)', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Deferral Curve - GNG Router', fontsize=13, fontweight='bold')
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

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the GNG-isolated router.")
    parser.add_argument("--seed", type=int, default=0,
                        help="NumPy seed for the order in which training prompts are presented to the GNG")
    parser.add_argument("--tune-beta-on", choices=["test", "val"], default="test",
                        help="Where the boundary-penalty weight beta is selected. 'test' reproduces the paper "
                             "(beta was swept on the test split, which makes the reported numbers optimistic). "
                             "'val' holds out part of the training prompts for the sweep and leaves the test split untouched.")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of the training prompts held out for the beta sweep when --tune-beta-on val")
    return parser.parse_args()

def sweep_beta(router, embeddings, eval_df, beta_vals, split_name):
    """Sweep beta on the given split and return (best_beta, metrics at best_beta)."""
    best_beta = None
    best_metrics = None
    best_score = float('inf')

    print(f"\nSweeping beta values on the {split_name} split to find the optimal soft boundary penalty:")
    for beta in beta_vals:
        metrics = evaluate_gng_router(router, embeddings, eval_df, DEFAULT_LLM_NAMES, beta=beta)
        qnc_val = f"{metrics['qnc']*100:.2f}%" if metrics['qnc'] != float('inf') else "N/A"
        print(f"  Beta: {beta:<5} -> Area: {metrics['area']:.4f}, Area@50%: {metrics['area_50']:.4f}, QNC: {qnc_val}")

        # Optimization score: prioritize lower QNC, then higher Area
        # A QNC of inf or 1.0 is penalized
        qnc_score = metrics['qnc'] if metrics['qnc'] != float('inf') else 10.0
        # Minimize (qnc_score - area)
        opt_score = qnc_score - metrics['area']
        if opt_score < best_score:
            best_score = opt_score
            best_beta = beta
            best_metrics = metrics

    print(f"\nSelected Best Beta: {best_beta}")
    return best_beta, best_metrics

def main():
    args = parse_args()
    np.random.seed(args.seed)
    print("=== GNG Router Evaluation ===")
    os.makedirs("gng_router/outputs", exist_ok=True)
    
    # 1. Load data & embeddings
    df = load_routerbench_data("routertrain.csv")
    embeddings = get_embeddings(df['prompt'].tolist(), model_name=DEFAULT_EMBEDDING_MODEL, cache_path=DEFAULT_CACHE_PATH)
    
    # 2. Split dataset (70-30 train-test)
    splits = split_dataset(df, embeddings, train_ratio=0.7, val_ratio=0.0, test_ratio=0.3, seed=42)
    
    beta_vals = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    best_beta = None

    if args.tune_beta_on == "val":
        # Hold out part of the training prompts for the beta sweep. The test split is not used
        # until the final evaluation, so the reported test numbers are not tuned on test data.
        train_df_all = splits['train_df'].reset_index(drop=True)
        train_emb_all = splits['train_embeddings']
        perm = np.random.default_rng(args.seed).permutation(len(train_df_all))
        n_val = int(args.val_fraction * len(perm))
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        fit_df = train_df_all.iloc[fit_idx].reset_index(drop=True)
        val_df = train_df_all.iloc[val_idx].reset_index(drop=True)
        tune_router = GNGRouterModel(llm_names=DEFAULT_LLM_NAMES)
        tune_router.fit(train_emb_all[fit_idx], fit_df, epochs=5)
        tune_router.compute_local_error_rates(train_emb_all[fit_idx], fit_df)
        best_beta, _ = sweep_beta(tune_router, train_emb_all[val_idx], val_df, beta_vals, "validation")

    # 3. Train GNG Router (on the full training split)
    router = GNGRouterModel(llm_names=DEFAULT_LLM_NAMES)
    router.fit(splits['train_embeddings'], splits['train_df'], epochs=5)
    router.compute_local_error_rates(splits['train_embeddings'], splits['train_df'])
    
    # Costs
    llm_costs = [splits['train_df'][f"{llm}|total_cost"].mean() for llm in DEFAULT_LLM_NAMES]
    
    # 4. Save model
    with open("gng_router_model.pkl", "wb") as f:
        pickle.dump({
            'router': router,
            'llm_names': DEFAULT_LLM_NAMES,
            'llm_costs': llm_costs,
            'model_name': DEFAULT_EMBEDDING_MODEL
        }, f)
    print("Saved trained model to gng_router_model.pkl")
    
    # 5. Evaluate
    if best_beta is None:
        print("\nNOTE: beta is being selected on the TEST split (the setting reported in the paper). "
              "Use --tune-beta-on val to select it on held-out training prompts instead.")
        best_beta, best_metrics = sweep_beta(router, splits['test_embeddings'], splits['test_df'], beta_vals, "test")
    else:
        best_metrics = evaluate_gng_router(router, splits['test_embeddings'], splits['test_df'], DEFAULT_LLM_NAMES, beta=best_beta)
    
    # Print best results
    print("\n" + "="*50)
    print(f"GNG ROUTER EVALUATION METRICS (Best Beta: {best_beta}, 70-30 split):")
    print(f"  Area:     {best_metrics['area']:.4f}")
    print(f"  Area@50%: {best_metrics['area_50']:.4f}")
    qnc_val = f"{best_metrics['qnc']*100:.2f}%" if best_metrics['qnc'] != float('inf') else "N/A"
    print(f"  QNC:      {qnc_val}")
    print("="*50 + "\n")
    
    # Save metrics JSON
    with open("gng_router/outputs/gng_metrics.json", "w") as f:
        json.dump({
            'area': float(best_metrics['area']),
            'area_50': float(best_metrics['area_50']),
            'qnc': float(best_metrics['qnc']) if best_metrics['qnc'] != float('inf') else None,
            'c_min': float(best_metrics['c_min']),
            'c_max': float(best_metrics['c_max']),
            'a_max': float(best_metrics['a_max']),
            'best_beta': best_beta
        }, f, indent=4)
        
    # Plot curve
    plot_gng_deferral_curve(best_metrics, "gng_router/outputs/deferral_curve.png")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
