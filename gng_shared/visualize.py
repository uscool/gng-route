import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

# Add Capstone root to path
CAPSTONE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CAPSTONE_ROOT)

from uniroute.data import load_router_pickle  # noqa: E402

def main():
    print("=== GNG Shared Router Topology Visualizer ===")
    
    ap = argparse.ArgumentParser(description="Draw the GNG-shared topology figures.")
    ap.add_argument("--model", default="gng_shared_model.pkl",
                    help="router pickle to draw (the headline model is written by: python gng_shared/run_eval.py --seed 0)")
    ap.add_argument("--out-dir", default="gng_shared/outputs", help="folder for the figures (e.g. figures/)")
    args = ap.parse_args()
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: Trained model '{model_path}' not found. Please run run_eval.py first.")
        sys.exit(1)
        
    print(f"Loading GNG shared model from {model_path}...")
    saved_data = load_router_pickle(model_path)
        
    router = saved_data['router']
    llm_names = saved_data['llm_names']
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Build NetworkX graph
    node_ids = list(router.topology.nodes.keys())
    G = nx.Graph()
    for nid in node_ids:
        G.add_node(nid)
    for edge, age in router.topology.edges.items():
        n1, n2 = edge
        G.add_edge(n1, n2, age=age)
        
    print("Computing NetworkX force-directed layout...")
    pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes)), seed=42)
    
    # Multi-panel figure (4x3 grid)
    n_models = len(llm_names)
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.flatten()
    
    print("Generating visualizations for LLM local error rates on GNG...")
    for idx, name in enumerate(llm_names):
        ax = axes[idx]
        short_name = name.split('/')[-1]
        
        # Get error rates of this model on all nodes in order of node_ids
        node_colors = [router.node_error_rates[nid][name] for nid in node_ids]
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray')
        
        # Draw nodes colored by local error
        nodes_draw = nx.draw_networkx_nodes(
            G, pos, ax=ax, 
            node_color=node_colors, 
            cmap='RdYlGn_r', 
            node_size=60,
            vmin=0.0, vmax=1.0
        )
        
        ax.set_title(f"{short_name}", fontsize=11, fontweight='bold')
        ax.axis('off')
        
    # Clear the last empty panel
    if len(axes) > n_models:
        for extra_idx in range(n_models, len(axes)):
            axes[extra_idx].axis('off')
            
    # Add a global colorbar for local error rates
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Local LLM Node Error Rate (Green is better/competent)', fontsize=12, fontweight='bold')
    
    plt.suptitle("GNG Shared Topology: Competency Mapping (Spring Layout)", fontsize=18, fontweight='bold', y=0.96)
    plt.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.1)
    
    output_path = os.path.join(args.out_dir, "topologies_visual.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Topological visualization saved to: {output_path}")
    print("Visualizations complete!")

if __name__ == "__main__":
    main()
