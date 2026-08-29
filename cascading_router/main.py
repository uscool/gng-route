import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVR

# Add Capstone root to path
CAPSTONE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CAPSTONE_ROOT)

from uniroute.data import load_routerbench_data, get_embeddings, split_dataset, DEFAULT_LLM_NAMES
from uniroute.eval import compute_deferral_curve_and_metrics, plot_deferral_curve

class CascadingRouterModel:
    def __init__(self):
        self.predictors = {}
        self.sorted_llms = []
        
    def fit(self, train_embeddings, train_df, llm_names, llm_costs):
        print("Training Cascading Router Predictors...")
        self.sorted_llms = [x for _, x in sorted(zip(llm_costs, llm_names))]
        print(f"  Models sorted by cost: {self.sorted_llms}")
        
        for model_name in llm_names:
            print(f"  Fitting predictor for {model_name}...")
            svr = LinearSVR(C=0.1, max_iter=1000, random_state=1234)
            y = train_df[model_name].values
            svr.fit(train_embeddings, y)
            self.predictors[model_name] = svr
            
    def route(self, embeddings, lambda_val, llm_costs):
        # We map lambda_val directly to the threshold parameter.
        # High lambda = cost first (threshold goes low to accept cheap models early)
        # Low lambda = accuracy first (threshold goes high to reject weak answers and check next models)
        threshold = 1.0 / (1.0 + 0.1 * lambda_val)
        
        # Predict accuracy for all sorted models
        pred_accuracies = {}
        for model_name in self.sorted_llms:
            pred_accuracies[model_name] = self.predictors[model_name].predict(embeddings)
            
        batch_size = len(embeddings)
        decisions = []
        
        # Map sorted LLM names to their original indexes in DEFAULT_LLM_NAMES
        llm_to_idx = {name: i for i, name in enumerate(DEFAULT_LLM_NAMES)}
        
        for idx in range(batch_size):
            routed_model = self.sorted_llms[-1] # Fallback to most expensive
            for model_name in self.sorted_llms[:-1]:
                if pred_accuracies[model_name][idx] >= threshold:
                    routed_model = model_name
                    break
            decisions.append(llm_to_idx[routed_model])
            
        return np.array(decisions)

def main():
    print("=== Cascading Router (RouterBench Baseline) ===")
    os.makedirs("cascading_router/outputs", exist_ok=True)
    
    # 1. Load data & embeddings
    df = load_routerbench_data("routertrain.csv")
    embeddings = get_embeddings(df['prompt'].tolist(), cache_path="prompt_embeddings_cache_base.pkl")
    
    # 2. Split dataset (70-30 train-test)
    splits = split_dataset(df, embeddings, train_ratio=0.7, val_ratio=0.0, test_ratio=0.3, seed=42)
    
    # Costs
    llm_costs = [splits['train_df'][f"{llm}|total_cost"].mean() for llm in DEFAULT_LLM_NAMES]
    
    # 3. Train Cascading Router
    router = CascadingRouterModel()
    router.fit(splits['train_embeddings'], splits['train_df'], DEFAULT_LLM_NAMES, llm_costs)
    
    # 4. Save model
    with open("cascading_router_model.pkl", "wb") as f:
        pickle.dump({
            'router': router,
            'llm_names': DEFAULT_LLM_NAMES,
            'llm_costs': llm_costs,
            'model_name': 'BAAI/bge-base-en-v1.5'
        }, f)
    print("Saved trained model to cascading_router_model.pkl")
    
    # 5. Evaluate
    metrics = compute_deferral_curve_and_metrics(router, splits['test_embeddings'], splits['test_df'], DEFAULT_LLM_NAMES)
    
    # Print results
    print("\n" + "="*50)
    print("CASCADING ROUTER EVALUATION METRICS (70-30 split):")
    print(f"  Area:     {metrics['area']:.4f}")
    print(f"  Area@50%: {metrics['area_50']:.4f}")
    qnc_val = f"{metrics['qnc']*100:.2f}%" if metrics['qnc'] != float('inf') else "N/A"
    print(f"  QNC:      {qnc_val}")
    print("="*50 + "\n")
    
    # Save metrics JSON
    with open("cascading_router/outputs/metrics.json", "w") as f:
        json.dump({
            'area': float(metrics['area']),
            'area_50': float(metrics['area_50']),
            'qnc': float(metrics['qnc']) if metrics['qnc'] != float('inf') else None,
            'c_min': float(metrics['c_min']),
            'c_max': float(metrics['c_max']),
            'a_max': float(metrics['a_max'])
        }, f, indent=4)
        
    # Plot curve
    plot_deferral_curve(metrics, "cascading_router/outputs/deferral_curve.png")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
