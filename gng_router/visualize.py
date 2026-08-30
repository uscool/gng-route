import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

# Add Capstone root to path
CAPSTONE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CAPSTONE_ROOT)

from uniroute.data import load_routerbench_data, get_embeddings, DEFAULT_LLM_NAMES

def main():
    print("=== GNG Router Topology Visualizer ===")
    
    # 1. Load trained GNG router model
    model_path = "gng_router_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Trained model '{model_path}' not found. Please run run_eval.py first to train the model.")
        sys.exit(1)
        
    print(f"Loading GNG router model from {model_path}...")
    with open(model_path, "rb") as f:
        saved_data = pickle.load(f)
        
    router = saved_data['router']
    llm_names = saved_data['llm_names']
    
    # Create outputs directory
    os.makedirs("gng_router/outputs", exist_ok=True)
    
    n_models = len(llm_names)
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.flatten()
    
    print("Generating visualizations for GNG topologies...")
    for idx, name in enumerate(llm_names):
        ax = axes[idx]
        topology = router.topologies[name]
        short_name = name.split('/')[-1]
        
        if not topology.nodes:
            ax.text(0.5, 0.5, "Empty Topology", ha='center', va='center')
            ax.set_title(short_name)
            continue
            
        node_ids = list(topology.nodes.keys())
        weights = np.array([topology.nodes[nid]['weight'] for nid in node_ids])
        errors = np.array([topology.nodes[nid]['error_rate'] for nid in node_ids])
        
        # 1. Build NetworkX graph
        G = nx.Graph()
        for nid in node_ids:
            G.add_node(nid, error_rate=topology.nodes[nid]['error_rate'])
            
        for edge, age in topology.edges.items():
            n1, n2 = edge
            G.add_edge(n1, n2, age=age)
            
        # 2. Compute Layout (Force-directed spring layout)
        # Use spring layout for clean topological visualization
        pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes)), seed=42)
        
        # 3. Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray')
        
        # 4. Draw nodes colored by local error rate (Red = High Error, Green = Low Error)
        node_colors = errors
        nodes_draw = nx.draw_networkx_nodes(
            G, pos, ax=ax, 
            node_color=node_colors, 
            cmap='RdYlGn_r', 
            node_size=60,
            vmin=0.0, vmax=1.0
        )
        
        # Count connected components (disjoint topologies)
        n_components = nx.number_connected_components(G)
        
        ax.set_title(f"{short_name}\n({len(node_ids)} nodes, {n_components} components)", fontsize=11, fontweight='bold')
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
    cbar.set_label('Local Node Error Rate (Lower is Better / Green represents competence)', fontsize=12, fontweight='bold')
    
    plt.suptitle("GNG Competency Graph Visualizations (Disjoint Spring Layouts)", fontsize=18, fontweight='bold', y=0.96)
    plt.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.1)
    
    output_path = "gng_router/outputs/gng_topologies_visual.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Topological visualization saved to: {output_path}")
    
    # 2. Generate a PCA Spatial Projection for comparison
    print("Generating PCA spatial projection plots...")
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.flatten()
    
    for idx, name in enumerate(llm_names):
        ax = axes[idx]
        topology = router.topologies[name]
        short_name = name.split('/')[-1]
        
        if not topology.nodes:
            ax.axis('off')
            continue
            
        node_ids = list(topology.nodes.keys())
        weights = np.array([topology.nodes[nid]['weight'] for nid in node_ids])
        errors = np.array([topology.nodes[nid]['error_rate'] for nid in node_ids])
        
        # Fit 2D PCA on the node weights
        pca = PCA(n_components=2)
        if len(node_ids) >= 2:
            coords = pca.fit_transform(weights)
        else:
            coords = np.zeros((len(node_ids), 2))
            
        # Draw edges in projected space
        for edge, age in topology.edges.items():
            n1, n2 = edge
            i1 = node_ids.index(n1)
            i2 = node_ids.index(n2)
            ax.plot([coords[i1, 0], coords[i2, 0]], [coords[i1, 1], coords[i2, 1]], color='gray', alpha=0.3)
            
        # Draw nodes
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], 
            c=errors, cmap='RdYlGn_r', 
            s=50, edgecolor='black', linewidths=0.5,
            vmin=0.0, vmax=1.0
        )
        
        # Draw boundary circle (90th percentile distance from training)
        boundary_r = router.boundary_radii[name]
        # Draw a dotted circle at the center of the nodes to represent the competence radius boundary
        center = np.mean(coords, axis=0)
        circle = plt.Circle(center, boundary_r, fill=False, color='blue', linestyle='--', alpha=0.5, label='Boundary')
        ax.add_patch(circle)
        
        ax.set_title(f"{short_name} (PCA Projection)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        
    if len(axes) > n_models:
        for extra_idx in range(n_models, len(axes)):
            axes[extra_idx].axis('off')
            
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Local Node Error Rate', fontsize=12, fontweight='bold')
    
    plt.suptitle("GNG Competency Spatial Projection (2D PCA)", fontsize=18, fontweight='bold', y=0.96)
    plt.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.1)
    
    pca_output_path = "gng_router/outputs/gng_topologies_pca.png"
    plt.savefig(pca_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA spatial visualization saved to: {pca_output_path}")
    print("Visualizations complete!")

if __name__ == "__main__":
    main()
