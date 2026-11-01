import os
import sys
import pickle
import numpy as np

# Add Capstone root to path
CAPSTONE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CAPSTONE_ROOT)

from uniroute.data import load_routerbench_data, get_embeddings, split_dataset, DEFAULT_LLM_NAMES

class GNGTopology:
    """
    Implements a single Growing Neural Gas (GNG) topology representing the prompt distribution.
    """
    def __init__(self, embedding_dim=384, max_nodes=200, eps_b=0.05, eps_n=0.005, 
                 a_max=50, alpha_err=0.5, beta_err=0.99, time_decay_rate=0.0,  # kappa = 0: constant update weights (standard GNG), as reported in the paper
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
        
        self.nodes = {}  # {node_id: {'weight': ndarray, 'error': float, 't_last': int, 'neighbors': set}}
        self.edges = {}  # {(node_id_1, node_id_2): age}
        self.total_dist_error = 0.0
        
    def add_node(self, weight, error=0.0, t_last=0):
        node_id = len(self.nodes)
        while node_id in self.nodes:
            node_id += 1
        self.nodes[node_id] = {
            'weight': np.array(weight, dtype=np.float32),
            'error': float(error),
            't_last': int(t_last),
            'neighbors': set()
        }
        return node_id
        
    def remove_node(self, node_id):
        if node_id in self.nodes:
            for neighbor in list(self.nodes[node_id]['neighbors']):
                if neighbor in self.nodes:
                    self.nodes[neighbor]['neighbors'].discard(node_id)
            del self.nodes[node_id]
            
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
            return self.nodes[node_id]['neighbors']
        return set()
        
    def fit_step(self, x, t):
        # Temporal decay weight
        w_t = np.exp(-self.time_decay_rate * t)
        
        if len(self.nodes) < 2:
            self.add_node(x, error=0.0, t_last=t)
            return
            
        node_ids = list(self.nodes.keys())
        node_weights = np.array([self.nodes[nid]['weight'] for nid in node_ids])
        
        # 1. Find the two nearest nodes s1 and s2
        dists = np.sum((node_weights - x) ** 2, axis=1)
        nearest_indices = np.argpartition(dists, min(1, len(dists)-1))[:2]
        
        # Sort them to get s1 and s2 exactly
        if dists[nearest_indices[0]] > dists[nearest_indices[1]]:
            nearest_indices = nearest_indices[::-1]
            
        s1_id = node_ids[nearest_indices[0]]
        s2_id = node_ids[nearest_indices[1]]
        
        d1 = dists[nearest_indices[0]]
        
        # 3. Update errors and activation time
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
        for nid in list(self.nodes.keys()):
            if current_time - self.nodes[nid]['t_last'] > self.node_lifespan:
                self.remove_node(nid)


class GNGSharedRouterModel:
    """
    Unified router system using a single shared GNGTopology.
    The nodes map prompt space, and each node stores local error rates for all LLMs.
    """
    def __init__(self, llm_names=None, max_nodes=200):
        self.llm_names = llm_names if llm_names else DEFAULT_LLM_NAMES
        self.topology = GNGTopology(max_nodes=max_nodes)
        
        # Local error rates map: {node_id: {model_name: error_rate}}
        self.node_error_rates = {}
        
    def fit(self, train_embeddings, train_df, epochs=5):
        """
        Fits a single GNG topology representing the unsupervised prompt space distribution,
        then calculates the supervised model-specific error rates on each node.
        """
        print("Training shared GNG topology on all prompts...")
        n_samples = len(train_embeddings)
        
        # Train topology
        t_counter = 0
        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(n_samples)
            for idx in shuffled_indices:
                self.topology.fit_step(train_embeddings[idx], t_counter)
                t_counter += 1
                
        self.topology.prune_stale_nodes(t_counter)
        print(f"GNG Topology grown: {len(self.topology.nodes)} nodes, {len(self.topology.edges)} edges.")
        
        # Calculate local error rates for each node
        node_ids = list(self.topology.nodes.keys())
        node_weights = np.array([self.topology.nodes[nid]['weight'] for nid in node_ids])
        
        # Initialize error accumulators
        node_errors = {nid: {name: [] for name in self.llm_names} for nid in node_ids}
        
        # Map all training prompts to GNG nodes
        for idx in range(n_samples):
            x = train_embeddings[idx]
            dists = np.sum((node_weights - x) ** 2, axis=1)
            best_idx = np.argmin(dists)
            best_nid = node_ids[best_idx]
            
            for name in self.llm_names:
                correctness = train_df[name].values[idx]
                node_errors[best_nid][name].append(1.0 - correctness)
                
        # Assign mean error rate to each node (fallback to overall mean error)
        self.node_error_rates = {}
        for nid in node_ids:
            self.node_error_rates[nid] = {}
            for name in self.llm_names:
                if node_errors[nid][name]:
                    self.node_error_rates[nid][name] = float(np.mean(node_errors[nid][name]))
                else:
                    self.node_error_rates[nid][name] = float(1.0 - train_df[name].mean())
                    
        print("Completed competence mapping.")
        
    def route(self, embeddings, lambda_val, llm_costs):
        """
        Vectorized routing of test queries using the shared GNG node error rates.
        """
        n_samples = len(embeddings)
        node_ids = list(self.topology.nodes.keys())
        node_weights = np.array([self.topology.nodes[nid]['weight'] for nid in node_ids])
        
        # Precompute error matrix: shape (n_nodes, n_models)
        error_matrix = np.zeros((len(node_ids), len(self.llm_names)))
        for i, nid in enumerate(node_ids):
            for m_idx, name in enumerate(self.llm_names):
                error_matrix[i, m_idx] = self.node_error_rates[nid][name]
                
        llm_costs = np.array(llm_costs)
        routed_indices = np.zeros(n_samples, dtype=int)
        
        # For each sample, find nearest node and look up the best model index
        for idx in range(n_samples):
            x = embeddings[idx]
            dists = np.sum((node_weights - x) ** 2, axis=1)
            best_node_idx = np.argmin(dists)
            
            # score = local_error + lambda * cost
            scores = error_matrix[best_node_idx] + lambda_val * llm_costs
            routed_indices[idx] = np.argmin(scores)
            
        return routed_indices
