import os
import ast
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_LLM_NAMES = [
    'WizardLM/WizardLM-13B-V1.2',
    'claude-instant-v1',
    'claude-v1',
    'claude-v2',
    'gpt-3.5-turbo-1106',
    'gpt-4-1106-preview',
    'meta/code-llama-instruct-34b-chat',
    'meta/llama-2-70b-chat',
    'mistralai/mistral-7b-chat',
    'mistralai/mixtral-8x7b-chat',
    'zero-one-ai/Yi-34B-Chat'
]

def preprocess_prompt(prompt_str):
    """
    Cleans and preprocesses the prompt string. Handles cases where the prompt is stored
    as a string representation of a list of strings (e.g., system prompt and user prompt).
    """
    if not isinstance(prompt_str, str):
        return ""
    prompt_str = prompt_str.strip()
    if prompt_str.startswith('[') and prompt_str.endswith(']'):
        try:
            val = ast.literal_eval(prompt_str)
            if isinstance(val, list):
                return "\n".join(str(item) for item in val)
        except Exception:
            pass
    return prompt_str

def load_routerbench_data(data_path, llm_names=None):
    """
    Loads data from the RouterBench CSV.
    Extracts correctness columns and cost columns for each LLM.
    """
    if llm_names is None:
        llm_names = DEFAULT_LLM_NAMES
        
    df = pd.read_csv(data_path)
    
    # Verify columns exist
    for llm in llm_names:
        if llm not in df.columns:
            raise ValueError(f"Required LLM accuracy column '{llm}' not found in dataset.")
        cost_col = f"{llm}|total_cost"
        if cost_col not in df.columns:
            raise ValueError(f"Required LLM cost column '{cost_col}' not found in dataset.")
            
    return df

def get_embeddings(prompts, model_name="BAAI/bge-base-en-v1.5", cache_path=None, batch_size=32):
    """
    Encodes the given prompts using a SentenceTransformer model.
    Saves/loads embeddings from cache_path to avoid redundant computation.
    """
    # Preprocess all prompts
    preprocessed_prompts = [preprocess_prompt(p) for p in prompts]
    
    # Try loading from cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
            # Verify if cached prompts match
            if cached_data.get('model_name') == model_name and len(cached_data.get('embeddings', [])) == len(prompts):
                return cached_data['embeddings']
            else:
                print("Cache mismatch (model name or data length). Recomputing embeddings...")
                
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Encoding {len(preprocessed_prompts)} prompts (this may take a few minutes)...")
    embeddings = model.encode(
        preprocessed_prompts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    if cache_path:
        print(f"Saving embeddings to cache: {cache_path}")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'model_name': model_name,
                'embeddings': embeddings
            }, f)
            
    return embeddings

def split_dataset(df, embeddings, train_ratio=0.6, val_ratio=0.1, test_ratio=0.3, seed=42):
    """
    Splits the dataset and embeddings into train, validation, and test splits deterministically.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    
    n_samples = len(df)
    indices = np.arange(n_samples)
    
    # Deterministic shuffle
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    splits = {
        'train_df': df.iloc[train_idx].reset_index(drop=True),
        'val_df': df.iloc[val_idx].reset_index(drop=True),
        'test_df': df.iloc[test_idx].reset_index(drop=True),
        'train_embeddings': embeddings[train_idx],
        'val_embeddings': embeddings[val_idx],
        'test_embeddings': embeddings[test_idx]
    }
    
    print(f"Dataset split completed (Seed: {seed}):")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    
    return splits
