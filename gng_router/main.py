import os
import sys
import pickle
import json
import argparse
import numpy as np
import pandas as pd

# Add Capstone root to path
CAPSTONE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CAPSTONE_ROOT)

from uniroute.data import load_routerbench_data, get_embeddings, split_dataset, DEFAULT_LLM_NAMES, DEFAULT_EMBEDDING_MODEL, DEFAULT_CACHE_PATH
from uniroute.eval import compute_deferral_curve_and_metrics, plot_deferral_curve

class GNGTopology:
    """
    Implements a single Growing Neural Gas (GNG) topology representing the competence region of an LLM.
    """
    def __init__(self, embedding_dim=768, max_nodes=200, eps_b=0.05, eps_n=0.005, 
                 a_max=50, alpha_err=0.5, beta_err=0.99, time_decay_rate=0.0,  # kappa = 0, identical to the previous behaviour of this class (weight was always 1)
                 node_lifespan=20000, growth_dist_threshold=3.0, max_edge_distance=0.5):
        self.embedding_dim = embedding_dim
        self.max_nodes = max_nodes
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.a_max = a_max
        self.alpha_err = alpha_err
        self.beta_err = beta_err
        self.time_decay_rate = time_decay_rate
        self.node_lifespan = node_lifespan
        self.growth_dist_threshold = growth_dist_threshold
        self.max_edge_distance = max_edge_distance
        
        # State variables
        self.nodes = {}       # node_id -> {'weight': np.array, 'error': float, 't_last': int, 'error_rate': float, 'neighbors': set}
        self.edges = {}       # (node_id1, node_id2) -> age (with node_id1 < node_id2)
        self.next_node_id = 0
        self.total_dist_error = 0.0
        self.time_step = 0
        
    def add_node(self, weight, error=0.0, t_last=0):
        node_id = self.next_node_id
        self.next_node_id += 1
        self.nodes[node_id] = {
            'weight': np.array(weight, dtype=float),
            'error': float(error),
            't_last': int(t_last),
            'error_rate': 0.0,
            'neighbors': set()
        }
        return node_id
        
    def remove_node(self, node_id):
        if node_id in self.nodes:
            # Remove node from its neighbors' sets
            for neighbor in list(self.nodes[node_id]['neighbors']):
                if neighbor in self.nodes:
                    self.nodes[neighbor]['neighbors'].discard(node_id)
            del self.nodes[node_id]
        # Remove any edges connected to this node
        edges_to_remove = [edge for edge in self.edges if node_id in edge]
        for edge in edges_to_remove:
            del self.edges[edge]
            
    def add_edge(self, n1, n2):
        edge = (min(n1, n2), max(n1, n2))
        self.edges[edge] = 0
        if n1 in self.nodes:
            self.nodes[n1]['neighbors'].add(n2)
        if n2 in self.nodes:
            self.nodes[n2]['neighbors'].add(n1)
        
    def remove_edge(self, n1, n2):
        edge = (min(n1, n2), max(n1, n2))
        if edge in self.edges:
            del self.edges[edge]
        if n1 in self.nodes and n2 in self.nodes[n1]['neighbors']:
            self.nodes[n1]['neighbors'].remove(n2)
        if n2 in self.nodes and n1 in self.nodes[n2]['neighbors']:
            self.nodes[n2]['neighbors'].remove(n1)
            
    def get_neighbors(self, node_id):
        if node_id in self.nodes:
            return list(self.nodes[node_id]['neighbors'])
        return []
        
    def fit_step(self, x, t):
        """
        Processes a single input embedding signal x at virtual timestamp t.
        """
        self.time_step = max(self.time_step, t)
        
        if len(self.nodes) < 2:
            self.add_node(x, error=0.0, t_last=t)
            return
            
        # 1. Find the two nearest nodes
        node_ids = list(self.nodes.keys())
        weights = np.array([self.nodes[nid]['weight'] for nid in node_ids])
        
        # Compute squared Euclidean distances
        dists = np.sum((weights - x) ** 2, axis=1)
        sorted_indices = np.argsort(dists)
        
        s1_id = node_ids[sorted_indices[0]]
        s2_id = node_ids[sorted_indices[1]]
        
        d1 = dists[sorted_indices[0]]
        
        # 2. Temporal Point Decay weight
        # Points processed earlier have less weight on topology growth
        time_diff = max(0, self.time_step - t)
        w_t = np.exp(-self.time_decay_rate * time_diff)
        
        # 3. Update errors and activation time
        # Growth rate depends on node distances
        self.nodes[s1_id]['error'] += d1 * w_t
        self.nodes[s1_id]['t_last'] = t
        self.total_dist_error += np.sqrt(d1) * w_t
        
        # 4. Update winner and neighbors positions
        self.nodes[s1_id]['weight'] += self.eps_b * w_t * (x - self.nodes[s1_id]['weight'])
        neighbors = self.get_neighbors(s1_id)
        for n_id in neighbors:
            self.nodes[n_id]['weight'] += self.eps_n * w_t * (x - self.nodes[n_id]['weight'])
            
        # 5. Age all edges connected to the winner
        for n_id in list(self.nodes[s1_id]['neighbors']):
            edge = (min(s1_id, n_id), max(s1_id, n_id))
            if edge in self.edges:
                self.edges[edge] += 1
                
        # 6. Create/refresh edge between s1 and s2 (if within max_edge_distance or topology is small)
        dist_s1_s2 = np.sqrt(np.sum((self.nodes[s1_id]['weight'] - self.nodes[s2_id]['weight']) ** 2))
        if self.max_edge_distance is None or len(self.nodes) < 20 or dist_s1_s2 <= self.max_edge_distance:
            self.add_edge(s1_id, s2_id)
        
        # 7. Prune old edges
        for edge, age in list(self.edges.items()):
            if age > self.a_max:
                n1, n2 = edge
                self.remove_edge(n1, n2)
                
        # 8. Prune isolated nodes
        for nid in list(self.nodes.keys()):
            if len(self.nodes[nid]['neighbors']) == 0:
                self.remove_node(nid)
                
        # 9. Distance-based growth node insertion
        # If accumulated distance error exceeds threshold, insert a new node
        if self.total_dist_error >= self.growth_dist_threshold and len(self.nodes) < self.max_nodes:
            self.total_dist_error = 0.0
            
            # Find node u with max error
            max_err_val = -1.0
            u_id = None
            for nid, nd in self.nodes.items():
                if nd['error'] > max_err_val:
                    max_err_val = nd['error']
                    u_id = nid
                    
            if u_id is not None:
                # Find neighbor v of u with max error
                u_neighbors = self.get_neighbors(u_id)
                if u_neighbors:
                    max_neighbor_err = -1.0
                    v_id = None
                    for nid in u_neighbors:
                        if self.nodes[nid]['error'] > max_neighbor_err:
                            max_neighbor_err = self.nodes[nid]['error']
                            v_id = nid
                            
                    if v_id is not None:
                        # Insert node r halfway
                        w_u = self.nodes[u_id]['weight']
                        w_v = self.nodes[v_id]['weight']
                        w_r = 0.5 * (w_u + w_v)
                        
                        r_id = self.add_node(w_r, error=self.nodes[u_id]['error'], t_last=t)
                        
                        # Remove edge (u, v) and insert edges (u, r) and (r, v)
                        self.remove_edge(u_id, v_id)
                        self.add_edge(u_id, r_id)
                        self.add_edge(r_id, v_id)
                        
                        # Also link r to any other existing close nodes (neighborhood connectivity)
                        for nid in list(self.nodes.keys()):
                            if nid != r_id and nid != u_id and nid != v_id:
                                dist_r_nid = np.sqrt(np.sum((w_r - self.nodes[nid]['weight']) ** 2))
                                if self.max_edge_distance is None or dist_r_nid <= self.max_edge_distance:
                                    self.add_edge(r_id, nid)
                        
                        # Decay errors
                        self.nodes[u_id]['error'] *= self.alpha_err
                        self.nodes[v_id]['error'] *= self.alpha_err
                        self.nodes[r_id]['error'] = self.nodes[u_id]['error']
                        
        # 10. Global error decay
        for nid in self.nodes.keys():
            self.nodes[nid]['error'] *= self.beta_err

    def prune_stale_nodes(self, current_time):
        """
        Removes nodes that haven't been winner/active for node_lifespan time steps.
        """
        for nid in list(self.nodes.keys()):
            if current_time - self.nodes[nid]['t_last'] > self.node_lifespan:
                self.remove_node(nid)


class GNGRouterModel:
    """
    Unified router system holding isolated GNGTopologies for each LLM candidate.
    """
    def __init__(self, llm_names=None, max_edge_distance=0.5):
        self.llm_names = llm_names if llm_names else DEFAULT_LLM_NAMES
        self.topologies = {name: GNGTopology(max_edge_distance=max_edge_distance) for name in self.llm_names}
        self.boundary_radii = {name: 0.5 for name in self.llm_names} # Fallback radius
        
    def fit(self, train_embeddings, train_df, epochs=5):
        """
        Fits separate isolated GNG topologies using the training samples.
        For model m, we train only on the subset of prompts where model m is correct.
        """
        print("Training isolated GNG topologies...")
        for name in self.llm_names:
            print(f"  Growing GNG topology for {name}...")
            # Filter rows where this model is correct (correctness == 1)
            correct_mask = train_df[name].values == 1.0
            correct_indices = np.where(correct_mask)[0]
            
            if len(correct_indices) < 2:
                print(f"    Warning: Too few correct prompts ({len(correct_indices)}) to grow topology for {name}.")
                # Initialize with whatever we have
                for idx in correct_indices:
                    self.topologies[name].add_node(train_embeddings[idx], error=0.0, t_last=idx)
                continue
                
            # Perform multiple passes (epochs) over the correct prompts
            t_counter = 0
            for epoch in range(epochs):
                # Shuffle correct indices per epoch for GNG stability
                shuffled_indices = np.random.permutation(correct_indices)
                for idx in shuffled_indices:
                    x = train_embeddings[idx]
                    self.topologies[name].fit_step(x, t_counter)
                    t_counter += 1
            
            # Prune stale nodes at the end of training
            self.topologies[name].prune_stale_nodes(t_counter)
            
            # Compute boundary radius (e.g. 90th percentile of distance from correct prompts to their nearest node)
            correct_embs = train_embeddings[correct_indices]
            dists = []
            topology = self.topologies[name]
            if len(topology.nodes) > 0:
                node_weights = np.array([nd['weight'] for nd in topology.nodes.values()])
                for x in correct_embs:
                    d = np.min(np.sum((node_weights - x) ** 2, axis=1))
                    dists.append(np.sqrt(d))
                self.boundary_radii[name] = float(np.percentile(dists, 90)) if dists else 0.5
            else:
                self.boundary_radii[name] = 0.5
                
            print(f"    GNG for {name} grown: {len(topology.nodes)} nodes, {len(topology.edges)} edges. Boundary radius: {self.boundary_radii[name]:.4f}")
            
    def compute_local_error_rates(self, train_embeddings, train_df):
        """
        Maps all training prompts to their nearest nodes in each model's topology,
        and computes the local accuracy/error rates of the nodes.
        """
        print("Computing local error rates for GNG nodes...")
        for name in self.llm_names:
            topology = self.topologies[name]
            if not topology.nodes:
                continue
                
            node_ids = list(topology.nodes.keys())
            node_weights = np.array([topology.nodes[nid]['weight'] for nid in node_ids])
            
            # Correctness labels (0 or 1)
            y = train_df[name].values
            
            # Track incorrect outputs per node
            node_errors = {nid: [] for nid in node_ids}
            
            for idx, x in enumerate(train_embeddings):
                # Find nearest node
                dists = np.sum((node_weights - x) ** 2, axis=1)
                best_idx = np.argmin(dists)
                best_nid = node_ids[best_idx]
                
                # Error is 1 - correctness
                node_errors[best_nid].append(1.0 - y[idx])
                
            # Assign computed average error rate to each node (fallback to overall mean error)
            overall_mean_error = 1.0 - train_df[name].mean()
            for nid in node_ids:
                if node_errors[nid]:
                    topology.nodes[nid]['error_rate'] = float(np.mean(node_errors[nid]))
                else:
                    topology.nodes[nid]['error_rate'] = float(overall_mean_error)

    def route(self, embeddings, lambda_val, llm_costs, beta=2.0):
        """
        Routes the test prompts to the best LLM using soft boundary penalties and UniRoute cost resolution.
        """
        batch_size = len(embeddings)
        routed_decisions = []
        
        # Precompute models and their average costs
        llm_to_idx = {name: i for i, name in enumerate(self.llm_names)}
        
        # Precompute node weights, error rates, and boundaries for speed
        precomputed = {}
        for name in self.llm_names:
            topology = self.topologies[name]
            if topology.nodes:
                node_ids = list(topology.nodes.keys())
                weights = np.array([topology.nodes[nid]['weight'] for nid in node_ids])
                errors = np.array([topology.nodes[nid]['error_rate'] for nid in node_ids])
                precomputed[name] = {
                    'weights': weights,
                    'errors': errors,
                    'boundary': self.boundary_radii[name]
                }
                
        for idx in range(batch_size):
            x = embeddings[idx]
            scores = []
            
            for name, data in precomputed.items():
                weights = data['weights']
                errors = data['errors']
                boundary = data['boundary']
                
                dists = np.sum((weights - x) ** 2, axis=1)
                best_idx = np.argmin(dists)
                d = np.sqrt(dists[best_idx])
                
                # Soft boundary penalty: beta * max(0, distance - boundary)
                penalty = beta * max(0.0, d - boundary)
                err = errors[best_idx]
                cost = llm_costs[llm_to_idx[name]]
                
                # score = error + penalty + lambda * cost
                score = err + penalty + lambda_val * cost
                scores.append((score, llm_to_idx[name]))
                
            best_model_idx = min(scores, key=lambda pair: pair[0])[1]
            routed_decisions.append(best_model_idx)
            
        return np.array(routed_decisions)


def main():
    parser = argparse.ArgumentParser(description="GNG Router CLI Utility")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "refresh", "reset", "remove"],
                        help="Action to perform: train, refresh, reset, or remove")
    parser.add_argument("--data_path", type=str, default="routertrain.csv", help="Path to training CSV data")
    parser.add_argument("--model_path", type=str, default="gng_router_model.pkl", help="Path to GNG pickle model")
    parser.add_argument("--remove_model", type=str, default=None, help="Name of the model to remove (for remove mode)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of GNG training epochs")
    args = parser.parse_args()
    
    if args.mode == "reset":
        if os.path.exists(args.model_path):
            os.remove(args.model_path)
            print(f"Successfully deleted and reset model at {args.model_path}")
        else:
            print("No model file exists to reset.")
        return
        
    # Load or initialize model
    model = None
    if os.path.exists(args.model_path) and args.mode != "reset":
        print(f"Loading existing GNG Router Model from {args.model_path}...")
        try:
            with open(args.model_path, 'rb') as f:
                saved_data = pickle.load(f)
                model = saved_data['router']
        except Exception as e:
            print(f"Failed to load model: {e}. Reinitializing from scratch.")
            
    # Load dataset & embeddings if training or refreshing
    if args.mode in ["train", "refresh"]:
        df = load_routerbench_data(args.data_path)
        embeddings = get_embeddings(df['prompt'].tolist(), model_name=DEFAULT_EMBEDDING_MODEL, cache_path=DEFAULT_CACHE_PATH)
        
        # Calculate mean LLM costs
        llm_costs = [df[f"{llm}|total_cost"].mean() for llm in DEFAULT_LLM_NAMES]
        
        if model is None:
            print("Initializing new GNG Router Model...")
            model = GNGRouterModel(llm_names=DEFAULT_LLM_NAMES)
            
    if args.mode == "train":
        # Grow topologies
        model.fit(embeddings, df, epochs=args.epochs)
        
        # Recalculate local error rates
        model.compute_local_error_rates(embeddings, df)
        
        # Save model
        with open(args.model_path, 'wb') as f:
            pickle.dump({
                'router': model,
                'llm_names': model.llm_names,
                'llm_costs': [df[f"{name}|total_cost"].mean() for name in model.llm_names],
                'model_name': DEFAULT_EMBEDDING_MODEL
            }, f)
        print(f"Saved trained GNG router model to {args.model_path}")
        
    elif args.mode == "refresh":
        if model is None:
            print("Error: No GNG model exists to refresh. Run train first.")
            return
            
        print("Refreshing GNG topologies (applying decay, pruning stale nodes, and updating error rates)...")
        current_time = len(embeddings)
        
        for name in model.llm_names:
            # Apply time decay and prune stale nodes
            model.topologies[name].prune_stale_nodes(current_time)
            
        # Re-compute local error rates on latest data
        model.compute_local_error_rates(embeddings, df)
        
        # Save model
        with open(args.model_path, 'wb') as f:
            pickle.dump({
                'router': model,
                'llm_names': model.llm_names,
                'llm_costs': [df[f"{name}|total_cost"].mean() for name in model.llm_names],
                'model_name': DEFAULT_EMBEDDING_MODEL
            }, f)
        print(f"Successfully refreshed and saved GNG router model to {args.model_path}")
        
    elif args.mode == "remove":
        if model is None:
            print("Error: No GNG model exists to remove from. Run train first.")
            return
        if not args.remove_model:
            print("Error: Specify model to remove with --remove_model")
            return
        if args.remove_model not in model.llm_names:
            print(f"Error: Model '{args.remove_model}' not found in the router's candidate pool.")
            return
            
        print(f"Removing candidate LLM '{args.remove_model}' from GNG Router...")
        # Delete from topologies list and candidate names
        if args.remove_model in model.topologies:
            del model.topologies[args.remove_model]
        model.llm_names = [name for name in model.llm_names if name != args.remove_model]
        
        # Save model
        # Recalculate costs for remaining models
        df = load_routerbench_data(args.data_path)
        with open(args.model_path, 'wb') as f:
            pickle.dump({
                'router': model,
                'llm_names': model.llm_names,
                'llm_costs': [df[f"{name}|total_cost"].mean() for name in model.llm_names],
                'model_name': DEFAULT_EMBEDDING_MODEL
            }, f)
        print(f"Successfully removed '{args.remove_model}' and saved updated model to {args.model_path}")

if __name__ == "__main__":
    main()
