import numpy as np
import matplotlib.pyplot as plt

def evaluate_policy(routed_indices, test_df, llm_names):
    """
    Computes the average accuracy and average cost of the routed decisions over the test set.
    """
    n_samples = len(test_df)
    correctness = np.zeros(n_samples)
    costs = np.zeros(n_samples)
    
    for idx, model_idx in enumerate(routed_indices):
        model_name = llm_names[model_idx]
        correctness[idx] = test_df[model_name].iloc[idx]
        costs[idx] = test_df[f"{model_name}|total_cost"].iloc[idx]
        
    avg_accuracy = np.mean(correctness)
    avg_cost = np.mean(costs)
    
    return avg_accuracy, avg_cost

def compute_deferral_curve_and_metrics(router, test_embeddings, test_df, llm_names, lambda_grid=None):
    """
    Sweeps lambda, computes accuracy and cost, normalizes cost, and calculates
    Area, Area@50%, and QNC.
    """
    if lambda_grid is None:
        # A dense log-spaced grid covering all trade-off ranges, plus 0.0
        lambda_grid = [0.0] + list(np.logspace(-1, 6, 500))
        
    # Get mean accuracy and cost for each single LLM on the test set
    llm_stats = {}
    for model_name in llm_names:
        llm_stats[model_name] = {
            'accuracy': test_df[model_name].mean(),
            'cost': test_df[f"{model_name}|total_cost"].mean()
        }
        
    # Find min and max cost among the LLM pool on the test set
    costs_list = [stats['cost'] for stats in llm_stats.values()]
    c_min = min(costs_list)
    c_max = max(costs_list)
    
    # Accuracy of the most accurate LLM in the pool
    accuracies_list = [stats['accuracy'] for stats in llm_stats.values()]
    a_max = max(accuracies_list)
    
    # 1. Sweep lambda to collect cost and accuracy points
    raw_points = []
    # We also retrieve the cost of each model (using mean cost from test set for routing decision C_m)
    # Note: the routing rule is argmin_m (E_{m,k} + lambda * C_m). We use the mean cost on the test set as C_m.
    llm_costs_for_routing = [llm_stats[model_name]['cost'] for model_name in llm_names]
    
    for l_val in lambda_grid:
        decisions = router.route(test_embeddings, l_val, llm_costs_for_routing)
        acc, cost = evaluate_policy(decisions, test_df, llm_names)
        raw_points.append((cost, acc))
        
    # 2. Normalize costs to Relative Cost in [0, 1]
    rel_points = []
    for cost, acc in raw_points:
        rel_cost = (cost - c_min) / (c_max - c_min) if c_max > c_min else 0.0
        # Clip relative cost to [0, 1] to avoid float precision issues
        rel_cost = np.clip(rel_cost, 0.0, 1.0)
        rel_points.append((rel_cost, acc))
        
    # Add boundary points for the extreme limits:
    # Cheapest model (relative cost 0.0)
    cheapest_model_name = min(llm_names, key=lambda name: llm_stats[name]['cost'])
    a_cheapest = llm_stats[cheapest_model_name]['accuracy']
    
    # Most expensive model (relative cost 1.0)
    expensive_model_name = max(llm_names, key=lambda name: llm_stats[name]['cost'])
    a_expensive = llm_stats[expensive_model_name]['accuracy']
    
    rel_points.append((0.0, a_cheapest))
    rel_points.append((1.0, a_expensive))
    
    # Sort points by relative cost
    rel_points = sorted(list(set(rel_points)), key=lambda p: p[0])
    
    relative_costs = np.array([p[0] for p in rel_points])
    accuracies = np.array([p[1] for p in rel_points])
    
    # 3. Calculate Area under the deferral curve (AUDC) using trapezoidal rule
    area_100 = np.trapz(accuracies, relative_costs)
    
    # 4. Calculate Area@50% (AUDC up to relative cost 0.5)
    # Find accuracy at exactly relative cost 0.5 via linear interpolation
    a_50 = np.interp(0.5, relative_costs, accuracies)
    
    # Filter points up to 0.5 and append the interpolated point at 0.5
    idx_50 = relative_costs <= 0.5
    rel_costs_50 = list(relative_costs[idx_50]) + [0.5]
    accuracies_50 = list(accuracies[idx_50]) + [a_50]
    
    # Sort and integrate
    points_50 = sorted(list(set(zip(rel_costs_50, accuracies_50))), key=lambda p: p[0])
    r_50 = np.array([p[0] for p in points_50])
    acc_50 = np.array([p[1] for p in points_50])
    area_50 = np.trapz(acc_50, r_50)
    
    # 5. Calculate Quality-Neutral Cost (QNC)
    # Minimum relative cost to achieve accuracy >= a_max
    eligible_costs = [r for r, acc in zip(relative_costs, accuracies) if acc >= a_max]
    if len(eligible_costs) > 0:
        qnc = min(eligible_costs)
    else:
        qnc = float('inf')
        
    return {
        'relative_costs': relative_costs,
        'accuracies': accuracies,
        'area': area_100,
        'area_50': area_50,
        'qnc': qnc,
        'llm_stats': llm_stats,
        'c_min': c_min,
        'c_max': c_max,
        'a_max': a_max
    }

def plot_deferral_curve(metrics, output_path):
    """
    Plots the deferral curve and saves the figure.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot router curve
    plt.plot(metrics['relative_costs'], metrics['accuracies'], label='UniRoute (K-Means)', color='blue', linewidth=2)
    
    # Plot individual LLMs
    for name, stats in metrics['llm_stats'].items():
        # Shorten model names for display
        short_name = name.split('/')[-1]
        rel_cost = (stats['cost'] - metrics['c_min']) / (metrics['c_max'] - metrics['c_min'])
        plt.scatter(rel_cost, stats['accuracy'], color='red', s=40, zorder=5)
        plt.text(rel_cost + 0.01, stats['accuracy'], short_name, fontsize=8, alpha=0.8)
        
    # Label formatting
    plt.xlabel('Relative Cost (0.0 = Cheapest, 1.0 = Most Expensive)', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Deferral Curve - UniRoute (K-Means)', fontsize=13, fontweight='bold')
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
