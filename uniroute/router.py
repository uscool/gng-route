import numpy as np
from sklearn.cluster import KMeans

class UniRouteKMeans:
    """
    Implements UniRoute (K-Means) routing algorithm from arXiv:2502.08773.
    """
    def __init__(self, K=10, random_state=42):
        self.K = K
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=K, random_state=random_state, n_init='auto')
        self.centroids = None
        self.val_errors = None  # Shape: (n_llms, K)
        
    def fit(self, train_embeddings):
        """
        Clusters the training-prompt embeddings to find K centroids.
        """
        print(f"Fitting K-Means with K={self.K} on training embeddings of shape {train_embeddings.shape}...")
        self.kmeans.fit(train_embeddings)
        self.centroids = self.kmeans.cluster_centers_
        return self
        
    def get_cluster_assignments(self, embeddings):
        """
        Assigns each prompt embedding to its nearest centroid.
        """
        if self.centroids is None:
            raise ValueError("Router has not been fitted yet. Call fit() first.")
        return self.kmeans.predict(embeddings)
        
    def compute_validation_errors(self, val_embeddings, val_df, llm_names):
        """
        Assigns validation prompts to their nearest centroids and computes the
        validation error rate E_{m,k} for each LLM and cluster.
        
        Args:
            val_embeddings: numpy array of validation prompt embeddings.
            val_df: pandas DataFrame containing correctness columns for validation set.
            llm_names: list of LLM column names.
        """
        val_clusters = self.get_cluster_assignments(val_embeddings)
        n_llms = len(llm_names)
        
        # Initialize validation error matrix E_{m,k}
        self.val_errors = np.zeros((n_llms, self.K))
        
        # Overall mean validation error as fallback for empty clusters
        overall_mean_errors = np.zeros(n_llms)
        for m_idx, llm in enumerate(llm_names):
            # correctness is 1.0 or 0.0, error is 1.0 - correctness
            overall_mean_errors[m_idx] = 1.0 - val_df[llm].mean()
            
        empty_clusters = 0
        for k in range(self.K):
            # Find validation prompts in cluster k
            in_cluster = (val_clusters == k)
            n_in_cluster = np.sum(in_cluster)
            
            if n_in_cluster > 0:
                cluster_df = val_df[in_cluster]
                for m_idx, llm in enumerate(llm_names):
                    # Compute mean error rate in this cluster
                    self.val_errors[m_idx, k] = 1.0 - cluster_df[llm].mean()
            else:
                # Fallback if no validation prompts fall in this cluster
                empty_clusters += 1
                for m_idx in range(n_llms):
                    self.val_errors[m_idx, k] = overall_mean_errors[m_idx]
        
        print(f"[Fallback Log] Number of clusters requiring fallback: {empty_clusters} / {self.K}")
        return self.val_errors
        
    def route(self, test_embeddings, lambda_val, llm_costs):
        """
        Routes test prompts to the best LLM using the routing rule:
        argmin_m (E_{m,k} + lambda * C_m)
        
        Args:
            test_embeddings: numpy array of test prompt embeddings.
            lambda_val: float hyperparameter trading off cost vs quality.
            llm_costs: list/numpy array of cost for each LLM.
            
        Returns:
            numpy array of indices representing the routed LLM for each test prompt.
        """
        if self.val_errors is None:
            raise ValueError("Validation error rates E_{m,k} have not been computed yet. Call compute_validation_errors() first.")
            
        test_clusters = self.get_cluster_assignments(test_embeddings)
        
        routed_llm_indices = []
        n_llms = len(llm_costs)
        
        # Precompute the routing decision for each cluster k to avoid repeating it for each prompt
        cluster_decisions = []
        for k in range(self.K):
            cluster_errors = self.val_errors[:, k]  # Shape: (n_llms,)
            objective_values = cluster_errors + lambda_val * np.array(llm_costs)
            # Find the best LLM index
            best_idx = np.argmin(objective_values)
            cluster_decisions.append(best_idx)
            
        # Assign routed LLM index based on the cluster of each test prompt
        for cluster_id in test_clusters:
            routed_llm_indices.append(cluster_decisions[cluster_id])
            
        return np.array(routed_llm_indices)
