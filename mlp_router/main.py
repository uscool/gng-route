import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

# Add Capstone root to path
CAPSTONE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CAPSTONE_ROOT)

from uniroute.data import load_routerbench_data, get_embeddings, split_dataset, DEFAULT_LLM_NAMES, DEFAULT_EMBEDDING_MODEL, DEFAULT_CACHE_PATH
from uniroute.eval import compute_deferral_curve_and_metrics, plot_deferral_curve

class MLPRouterModel:
    def __init__(self, hidden_layer_sizes=(100, 100, 100), activation="relu", lr=0.001):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.lr = lr
        self.mlps = {}
        
    def fit(self, train_embeddings, train_df, llm_names):
        print("Training MLP Regressors (one per model)...")
        for model_name in llm_names:
            print(f"  Fitting MLP for {model_name}...")
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                learning_rate_init=self.lr,
                max_iter=100,
                random_state=1234,
                verbose=False
            )
            # Train on the target accuracy values
            y = train_df[model_name].values
            mlp.fit(train_embeddings, y)
            self.mlps[model_name] = mlp
            
    def route(self, embeddings, lambda_val, llm_costs):
        # Predict accuracy for each model
        pred_accuracies = []
        for model_name in DEFAULT_LLM_NAMES:
            pred = self.mlps[model_name].predict(embeddings)
            pred_accuracies.append(pred)
        pred_accuracies = np.vstack(pred_accuracies).T # shape: (batch_size, 11)
        
        # Calculate score: performance - lambda * cost
        llm_costs_arr = np.array(llm_costs) # shape: (11,)
        scores = pred_accuracies - lambda_val * llm_costs_arr # shape: (batch_size, 11)
        
        # Return routed model index
        return np.argmax(scores, axis=1)

def main():
    print("=== MLP Router (RouterBench Baseline) ===")
    os.makedirs("mlp_router/outputs", exist_ok=True)
    
    # 1. Load data & embeddings
    df = load_routerbench_data("routertrain.csv")
    embeddings = get_embeddings(df['prompt'].tolist(), model_name=DEFAULT_EMBEDDING_MODEL, cache_path=DEFAULT_CACHE_PATH)
    
    # 2. Split dataset (70-30 train-test)
    splits = split_dataset(df, embeddings, train_ratio=0.7, val_ratio=0.0, test_ratio=0.3, seed=42)
    
    # 3. Train MLP Router
    router = MLPRouterModel()
    router.fit(splits['train_embeddings'], splits['train_df'], DEFAULT_LLM_NAMES)
    
    # Costs
    llm_costs = [splits['train_df'][f"{llm}|total_cost"].mean() for llm in DEFAULT_LLM_NAMES]
    
    # 4. Save model
    with open("mlp_router_model.pkl", "wb") as f:
        pickle.dump({
            'router': router,
            'llm_names': DEFAULT_LLM_NAMES,
            'llm_costs': llm_costs,
            'model_name': DEFAULT_EMBEDDING_MODEL
        }, f)
    print("Saved trained model to mlp_router_model.pkl")
    
    # 5. Evaluate
    metrics = compute_deferral_curve_and_metrics(router, splits['test_embeddings'], splits['test_df'], DEFAULT_LLM_NAMES)
    
    # Print results
    print("\n" + "="*50)
    print("MLP ROUTER EVALUATION METRICS (70-30 split):")
    print(f"  Area:     {metrics['area']:.4f}")
    print(f"  Area@50%: {metrics['area_50']:.4f}")
    qnc_val = f"{metrics['qnc']*100:.2f}%" if metrics['qnc'] != float('inf') else "N/A"
    print(f"  QNC:      {qnc_val}")
    print("="*50 + "\n")
    
    # Save metrics JSON
    with open("mlp_router/outputs/metrics.json", "w") as f:
        json.dump({
            'area': float(metrics['area']),
            'area_50': float(metrics['area_50']),
            'qnc': float(metrics['qnc']) if metrics['qnc'] != float('inf') else None,
            'c_min': float(metrics['c_min']),
            'c_max': float(metrics['c_max']),
            'a_max': float(metrics['a_max'])
        }, f, indent=4)
        
    # Plot curve
    plot_deferral_curve(metrics, "mlp_router/outputs/deferral_curve.png")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
