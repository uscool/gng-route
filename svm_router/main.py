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

from uniroute.data import load_routerbench_data, get_embeddings, split_dataset, DEFAULT_LLM_NAMES, DEFAULT_EMBEDDING_MODEL, DEFAULT_CACHE_PATH
from uniroute.eval import compute_deferral_curve_and_metrics, plot_deferral_curve

class SVMRouterModel:
    def __init__(self, C=0.1):
        self.C = C
        self.svrs = {}
        
    def fit(self, train_embeddings, train_df, llm_names):
        print("Training SVM Regressors (one per model)...")
        for model_name in llm_names:
            print(f"  Fitting SVM for {model_name}...")
            svr = LinearSVR(
                C=self.C,
                max_iter=1000,
                random_state=1234
            )
            y = train_df[model_name].values
            svr.fit(train_embeddings, y)
            self.svrs[model_name] = svr
            
    def route(self, embeddings, lambda_val, llm_costs):
        # Predict accuracy for each model
        pred_accuracies = []
        for model_name in DEFAULT_LLM_NAMES:
            pred = self.svrs[model_name].predict(embeddings)
            pred_accuracies.append(pred)
        pred_accuracies = np.vstack(pred_accuracies).T # shape: (batch_size, 11)
        
        # Calculate score: performance - lambda * cost
        llm_costs_arr = np.array(llm_costs) # shape: (11,)
        scores = pred_accuracies - lambda_val * llm_costs_arr # shape: (batch_size, 11)
        
        # Return routed model index
        return np.argmax(scores, axis=1)

def main():
    print("=== SVM Router (RouterBench Baseline) ===")
    os.makedirs("svm_router/outputs", exist_ok=True)
    
    # 1. Load data & embeddings
    df = load_routerbench_data("routertrain.csv")
    embeddings = get_embeddings(df['prompt'].tolist(), model_name=DEFAULT_EMBEDDING_MODEL, cache_path=DEFAULT_CACHE_PATH)
    
    # 2. Split dataset (70-30 train-test)
    splits = split_dataset(df, embeddings, train_ratio=0.7, val_ratio=0.0, test_ratio=0.3, seed=42)
    
    # 3. Train SVM Router
    router = SVMRouterModel()
    router.fit(splits['train_embeddings'], splits['train_df'], DEFAULT_LLM_NAMES)
    
    # Costs
    llm_costs = [splits['train_df'][f"{llm}|total_cost"].mean() for llm in DEFAULT_LLM_NAMES]
    
    # 4. Save model
    with open("svm_router_model.pkl", "wb") as f:
        pickle.dump({
            'router': router,
            'llm_names': DEFAULT_LLM_NAMES,
            'llm_costs': llm_costs,
            'model_name': DEFAULT_EMBEDDING_MODEL
        }, f)
    print("Saved trained model to svm_router_model.pkl")
    
    # 5. Evaluate
    metrics = compute_deferral_curve_and_metrics(router, splits['test_embeddings'], splits['test_df'], DEFAULT_LLM_NAMES)
    
    # Print results
    print("\n" + "="*50)
    print("SVM ROUTER EVALUATION METRICS (70-30 split):")
    print(f"  Area:     {metrics['area']:.4f}")
    print(f"  Area@50%: {metrics['area_50']:.4f}")
    qnc_val = f"{metrics['qnc']*100:.2f}%" if metrics['qnc'] != float('inf') else "N/A"
    print(f"  QNC:      {qnc_val}")
    print("="*50 + "\n")
    
    # Save metrics JSON
    with open("svm_router/outputs/metrics.json", "w") as f:
        json.dump({
            'area': float(metrics['area']),
            'area_50': float(metrics['area_50']),
            'qnc': float(metrics['qnc']) if metrics['qnc'] != float('inf') else None,
            'c_min': float(metrics['c_min']),
            'c_max': float(metrics['c_max']),
            'a_max': float(metrics['a_max'])
        }, f, indent=4)
        
    # Plot curve
    plot_deferral_curve(metrics, "svm_router/outputs/deferral_curve.png", router_name="SVM")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
