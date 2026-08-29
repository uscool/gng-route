import os
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from uniroute.data import load_routerbench_data, get_embeddings, split_dataset, DEFAULT_LLM_NAMES
from uniroute.router import UniRouteKMeans
from uniroute.eval import compute_deferral_curve_and_metrics, plot_deferral_curve

def main():
    parser = argparse.ArgumentParser(description="UniRoute (K-Means) Model Routing Baseline")
    
    # Dataset and model configuration
    parser.add_argument("--data_path", type=str, default="routertrain.csv", help="Path to RouterBench CSV")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-base-en-v1.5", help="Hugging Face embedding model")
    parser.add_argument("--cache_path", type=str, default="prompt_embeddings_cache.pkl", help="Path to cache prompt embeddings")
    parser.add_argument("--seed", type=type(42), default=42, help="Random seed for reproducibility")
    parser.add_argument("--K", type=type(10), default=10, help="Number of clusters for K-Means")
    parser.add_argument("--splits", type=str, default="0.7,0.0,0.3", help="Split ratios for Train, Val, Test (comma separated)")
    parser.add_argument("--output_dir", type=str, default="uniroute/outputs", help="Directory to save output metrics and files")
    
    # Model saving/loading
    parser.add_argument("--save_model", type=str, default="uniroute_model.pkl", help="Path to save the trained model pickle")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load a pre-trained model pickle")
    
    # Custom prediction mode
    parser.add_argument("--predict_prompt", type=str, default=None, help="A custom prompt to predict routing for")
    parser.add_argument("--lambda_val", type=float, default=1.0, help="Lambda value to use in custom prediction mode")
    
    # Configurable report lambdas
    parser.add_argument("--lambdas", type=str, default="0.0,0.1,0.5,1.0,5.0,10.0,50.0,100.0,500.0,1000.0,5000.0,10000.0", 
                        help="List of lambda values to evaluate and record in routing_results.csv (comma separated)")

    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Mode 1: Predict routing for a single custom prompt
    if args.predict_prompt is not None:
        print("=== Custom Prompt Prediction Mode ===")
        router = None
        llm_names = DEFAULT_LLM_NAMES
        llm_costs = []
        model_name = args.embedding_model
        
        # Load from saved model if specified and exists
        model_loaded = False
        load_path = args.load_model if args.load_model else args.save_model
        if load_path and os.path.exists(load_path):
            print(f"Loading pre-trained router model from: {load_path}")
            try:
                with open(load_path, 'rb') as f:
                    model_data = pickle.load(f)
                router = model_data['router']
                llm_names = model_data['llm_names']
                llm_costs = model_data['llm_costs']
                model_name = model_data['model_name']
                model_loaded = True
            except Exception as e:
                print(f"Failed to load model from {load_path}: {e}. Falling back to training on the fly.")
                
        if not model_loaded:
            print("Training router on the fly...")
            df = load_routerbench_data(args.data_path)
            embeddings = get_embeddings(df['prompt'].tolist(), model_name=args.embedding_model, cache_path=args.cache_path)
            
            # Split train, val, test
            train_r, val_r, test_r = map(float, args.splits.split(","))
            splits = split_dataset(df, embeddings, train_ratio=train_r, val_ratio=val_r, test_ratio=test_r, seed=args.seed)
            
            # Fit and compute errors
            router = UniRouteKMeans(K=args.K, random_state=args.seed)
            router.fit(splits['train_embeddings'])
            llm_names = DEFAULT_LLM_NAMES
            
            val_emb = splits['val_embeddings'] if len(splits['val_df']) > 0 else splits['train_embeddings']
            val_df = splits['val_df'] if len(splits['val_df']) > 0 else splits['train_df']
            
            llm_costs = [val_df[f"{llm}|total_cost"].mean() for llm in llm_names]
            router.compute_validation_errors(val_emb, val_df, llm_names)
            
            # Save the trained model
            if args.save_model:
                print(f"Saving trained model to: {args.save_model}")
                with open(args.save_model, 'wb') as f:
                    pickle.dump({
                        'router': router,
                        'llm_names': llm_names,
                        'llm_costs': llm_costs,
                        'model_name': model_name
                    }, f)
                    
        # Encode the custom prompt
        print(f"Encoding custom prompt: '{args.predict_prompt}' using {model_name}...")
        model = SentenceTransformer(model_name)
        prompt_emb = model.encode([args.predict_prompt], convert_to_numpy=True)
        
        # Determine closest centroid and route
        cluster_assignment = router.get_cluster_assignments(prompt_emb)[0]
        routed_idx = router.route(prompt_emb, args.lambda_val, llm_costs)[0]
        routed_model = llm_names[routed_idx]
        
        print("\n" + "="*50)
        print(f"ROUTING RESULT:")
        print(f"  Prompt:           '{args.predict_prompt}'")
        print(f"  Assigned Cluster: {cluster_assignment} / {args.K}")
        print(f"  Lambda Value:     {args.lambda_val}")
        print(f"  Routed Model:     {routed_model}")
        print(f"  Estimated Cost:   {llm_costs[routed_idx]:.6f}")
        print("="*50)
        return
        
    # Mode 2: Run standard evaluation pipeline
    print("=== Standard Evaluation Pipeline ===")
    
    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    df = load_routerbench_data(args.data_path)
    
    # 2. Generate Embeddings
    print("Generating/loading prompt embeddings...")
    embeddings = get_embeddings(df['prompt'].tolist(), model_name=args.embedding_model, cache_path=args.cache_path)
    
    # 3. Split Dataset
    train_r, val_r, test_r = map(float, args.splits.split(","))
    splits = split_dataset(df, embeddings, train_ratio=train_r, val_ratio=val_r, test_ratio=test_r, seed=args.seed)
    
    # 4. Fit Router on Train set
    router = UniRouteKMeans(K=args.K, random_state=args.seed)
    router.fit(splits['train_embeddings'])
    
    # 5. Compute Validation Error Rates (E_{m,k})
    print("Computing validation error rates E_{m,k}...")
    val_emb = splits['val_embeddings'] if len(splits['val_df']) > 0 else splits['train_embeddings']
    val_df = splits['val_df'] if len(splits['val_df']) > 0 else splits['train_df']
    router.compute_validation_errors(val_emb, val_df, DEFAULT_LLM_NAMES)
    
    # Mean costs of LLMs on validation split for routing
    llm_costs = [val_df[f"{llm}|total_cost"].mean() for llm in DEFAULT_LLM_NAMES]
    
    # Save the model
    if args.save_model:
        print(f"Saving trained model to: {args.save_model}")
        with open(args.save_model, 'wb') as f:
            pickle.dump({
                'router': router,
                'llm_names': DEFAULT_LLM_NAMES,
                'llm_costs': llm_costs,
                'model_name': args.embedding_model
            }, f)
            
    # 6. Sweep lambda and calculate metrics
    print("Sweeping lambda to compute deferral curve and metrics...")
    metrics = compute_deferral_curve_and_metrics(router, splits['test_embeddings'], splits['test_df'], DEFAULT_LLM_NAMES)
    
    # Log Results
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS:")
    print(f"  Area:     {metrics['area']:.4f}")
    print(f"  Area@50%: {metrics['area_50']:.4f}")
    qnc_val = f"{metrics['qnc']*100:.2f}%" if metrics['qnc'] != float('inf') else "N/A"
    print(f"  QNC:      {qnc_val}")
    print("="*50 + "\n")
    
    # 7. Save JSON report (metrics.json)
    metrics_report_path = os.path.join(args.output_dir, "metrics.json")
    report_data = {
        'area': float(metrics['area']),
        'area_50': float(metrics['area_50']),
        'qnc': float(metrics['qnc']) if metrics['qnc'] != float('inf') else None,
        'c_min': float(metrics['c_min']),
        'c_max': float(metrics['c_max']),
        'a_max': float(metrics['a_max']),
        'llm_stats': {name: {'accuracy': float(stats['accuracy']), 'cost': float(stats['cost'])} for name, stats in metrics['llm_stats'].items()}
    }
    with open(metrics_report_path, 'w') as f:
        json.dump(report_data, f, indent=4)
    print(f"Metrics JSON report saved to: {metrics_report_path}")
    
    # 8. Save config.json
    config_data = {
        "data_path": args.data_path,
        "embedding_model": args.embedding_model,
        "K": args.K,
        "seed": args.seed,
        "splits": args.splits,
        "lambdas": [float(l) for l in args.lambdas.split(",")],
        "output_dir": args.output_dir
    }
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(config_data, f, indent=4)
    print(f"Config JSON saved to: {os.path.join(args.output_dir, 'config.json')}")

    # 9. Save centroids.npy
    np.save(os.path.join(args.output_dir, "centroids.npy"), router.centroids)
    print(f"Centroids saved to: {os.path.join(args.output_dir, 'centroids.npy')}")

    # 10. Save cluster_assignments.csv
    train_clusters = router.get_cluster_assignments(splits['train_embeddings'])
    test_clusters = router.get_cluster_assignments(splits['test_embeddings'])
    
    train_assign = pd.DataFrame({
        'sample_id': splits['train_df']['sample_id'],
        'split': 'train',
        'cluster_id': train_clusters
    })
    test_assign = pd.DataFrame({
        'sample_id': splits['test_df']['sample_id'],
        'split': 'test',
        'cluster_id': test_clusters
    })
    
    assign_list = [train_assign]
    if len(splits['val_df']) > 0:
        val_clusters = router.get_cluster_assignments(splits['val_embeddings'])
        val_assign = pd.DataFrame({
            'sample_id': splits['val_df']['sample_id'],
            'split': 'val',
            'cluster_id': val_clusters
        })
        assign_list.append(val_assign)
    assign_list.append(test_assign)
    
    all_assign = pd.concat(assign_list, ignore_index=True)
    all_assign.to_csv(os.path.join(args.output_dir, "cluster_assignments.csv"), index=False)
    print(f"Cluster assignments CSV saved to: {os.path.join(args.output_dir, 'cluster_assignments.csv')}")

    # 11. Save llm_cluster_errors.csv
    errors_df = pd.DataFrame(
        router.val_errors,
        index=DEFAULT_LLM_NAMES,
        columns=[f"cluster_{k}" for k in range(args.K)]
    )
    errors_df.to_csv(os.path.join(args.output_dir, "llm_cluster_errors.csv"))
    print(f"LLM cluster errors CSV saved to: {os.path.join(args.output_dir, 'llm_cluster_errors.csv')}")

    # 12. Save routing_results.csv
    test_embeddings = splits['test_embeddings']
    test_df = splits['test_df']
    
    routing_results_rows = []
    lambdas_list = [float(l) for l in args.lambdas.split(",")]
    
    for l_val in lambdas_list:
        decisions = router.route(test_embeddings, l_val, llm_costs)
        for idx, model_idx in enumerate(decisions):
            model_name = DEFAULT_LLM_NAMES[model_idx]
            cost = test_df[f"{model_name}|total_cost"].iloc[idx]
            correctness = int(test_df[model_name].iloc[idx])
            
            routing_results_rows.append({
                'sample_id': test_df['sample_id'].iloc[idx],
                'cluster_id': test_clusters[idx],
                'selected_llm': model_name,
                'lambda': l_val,
                'selected_llm_cost': cost,
                'correct': correctness
            })
            
    routing_results_df = pd.DataFrame(routing_results_rows)
    routing_results_df.to_csv(os.path.join(args.output_dir, "routing_results.csv"), index=False)
    print(f"Routing results CSV saved to: {os.path.join(args.output_dir, 'routing_results.csv')}")

    # 13. Plot Deferral Curve
    plot_path = os.path.join(args.output_dir, "deferral_curve.png")
    plot_deferral_curve(metrics, plot_path)
    print("Evaluation pipeline run successfully!")

if __name__ == "__main__":
    main()
