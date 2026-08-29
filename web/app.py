import os
import sys
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer

# Add the Capstone root directory to sys.path so pickle can import uniroute.router
CAPSTONE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CAPSTONE_ROOT)

app = Flask(__name__, static_folder='static')

# Load the UniRoute model relative to Capstone root
MODEL_PATH = os.path.join(CAPSTONE_ROOT, "uniroute_model.pkl")
model_data = None
sentence_model = None

# Cache loaded models
loaded_routers = {}

def load_models():
    global sentence_model
    # Load sentence transformer using the base model
    model_name = "BAAI/bge-base-en-v1.5"
    print(f"Loading SentenceTransformer model {model_name}...")
    sentence_model = SentenceTransformer(model_name)
    print("SentenceTransformer loaded successfully!")

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__':
            if name == 'KNNRouterModel':
                from knn_router.main import KNNRouterModel
                return KNNRouterModel
            elif name == 'MLPRouterModel':
                from mlp_router.main import MLPRouterModel
                return MLPRouterModel
            elif name == 'SVMRouterModel':
                from svm_router.main import SVMRouterModel
                return SVMRouterModel
            elif name == 'CascadingRouterModel':
                from cascading_router.main import CascadingRouterModel
                return CascadingRouterModel
        return super().find_class(module, name)

def get_router(router_type):
    if router_type in loaded_routers:
        return loaded_routers[router_type]
        
    if router_type == "uniroute":
        model_file = "uniroute_model.pkl"
    else:
        model_file = f"{router_type}_router_model.pkl"
        
    model_path = os.path.join(CAPSTONE_ROOT, model_file)
    if os.path.exists(model_path):
        print(f"Loading {router_type} model from {model_path}...")
        with open(model_path, 'rb') as f:
            data = CustomUnpickler(f).load()
        loaded_routers[router_type] = data
        return data
    return None

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/route', methods=['POST'])
def api_route():
    if sentence_model is None:
        load_models()
        
    data = request.json or {}
    prompt = data.get('prompt', '').strip()
    lambda_val = float(data.get('lambda_val', 1.0))
    router_type = data.get('router_type', 'uniroute')
    
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400
        
    router_data = get_router(router_type)
    if router_data is None:
        return jsonify({"error": f"Model for router type '{router_type}' is not found. Please train it first."}), 400
        
    router = router_data['router']
    llm_names = router_data['llm_names']
    llm_costs = router_data['llm_costs']
    
    # 1. Embed the prompt
    prompt_emb = sentence_model.encode([prompt], convert_to_numpy=True)
    
    # 2. Run inference depending on the router type
    cluster_id = -1
    accuracies = []
    
    if router_type == 'uniroute':
        cluster_id = int(router.get_cluster_assignments(prompt_emb)[0])
        errors = router.val_errors[:, cluster_id]
        scores = errors + lambda_val * np.array(llm_costs)
        selected_idx = int(np.argmin(scores))
        accuracies = np.clip(1.0 - errors, 0.0, 1.0)
        # Invert score displaying logic for UniRoute so lower is better
        score_display = list(scores)
    elif router_type == 'knn':
        _, indices = router.knn.kneighbors(prompt_emb)
        accuracies = np.clip(router.train_y[indices].mean(axis=1)[0], 0.0, 1.0)
        scores = accuracies - lambda_val * np.array(llm_costs)
        selected_idx = int(np.argmax(scores))
        score_display = list(scores)
    elif router_type == 'mlp':
        for name in llm_names:
            pred = router.mlps[name].predict(prompt_emb)[0]
            accuracies.append(pred)
        accuracies = np.clip(np.array(accuracies), 0.0, 1.0)
        scores = accuracies - lambda_val * np.array(llm_costs)
        selected_idx = int(np.argmax(scores))
        score_display = list(scores)
    elif router_type == 'svm':
        for name in llm_names:
            pred = router.svrs[name].predict(prompt_emb)[0]
            accuracies.append(pred)
        accuracies = np.clip(np.array(accuracies), 0.0, 1.0)
        scores = accuracies - lambda_val * np.array(llm_costs)
        selected_idx = int(np.argmax(scores))
        score_display = list(scores)
    elif router_type == 'cascading':
        threshold = 1.0 / (1.0 + 0.1 * lambda_val)
        accuracies_dict = {}
        for name in llm_names:
            pred = router.predictors[name].predict(prompt_emb)[0]
            accuracies_dict[name] = pred
            
        selected_llm = router.sorted_llms[-1] # Fallback
        for name in router.sorted_llms[:-1]:
            if accuracies_dict[name] >= threshold:
                selected_llm = name
                break
        selected_idx = llm_names.index(selected_llm)
        accuracies = np.clip(np.array([accuracies_dict[name] for name in llm_names]), 0.0, 1.0)
        # Cascading scores reflect distance to threshold
        scores = accuracies - threshold
        score_display = list(scores)
        
    selected_llm = llm_names[selected_idx]
    
    # Construct breakdown
    llm_breakdown = []
    for idx, name in enumerate(llm_names):
        llm_breakdown.append({
            'name': name,
            'short_name': name.split('/')[-1],
            'accuracy': float(accuracies[idx]),
            'cost': float(llm_costs[idx]),
            'score': float(score_display[idx]),
            'is_selected': (idx == selected_idx)
        })
        
    # Sort breakdown (best score first)
    if router_type == 'uniroute':
        # For UniRoute, lower score is better
        llm_breakdown = sorted(llm_breakdown, key=lambda x: x['score'])
    else:
        # For others, higher score is better
        llm_breakdown = sorted(llm_breakdown, key=lambda x: x['score'], reverse=True)
        
    response = {
        'prompt': prompt,
        'cluster_id': cluster_id,
        'lambda_val': lambda_val,
        'selected_llm': selected_llm,
        'selected_llm_short': selected_llm.split('/')[-1],
        'selected_llm_cost': float(llm_costs[selected_idx]),
        'selected_llm_accuracy': float(accuracies[selected_idx]),
        'llm_breakdown': llm_breakdown
    }
    
    return jsonify(response)

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
