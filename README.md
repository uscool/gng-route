Code for the paper **"Explainable Region-Based Routing Across Heterogeneous Language Models with
Growing Neural Gas"** 

A router picks one LLM per prompt from a pool, trading answer quality against cost. This repo grows a
**Growing Neural Gas (GNG)** over prompt embeddings and stores, in every node, the error rate of each
candidate model on the training prompts that node wins. A routing decision is then explained by one
region of embedding space, and a single scalar (lambda) sets the cost-quality trade-off. Everything is
evaluated offline on [RouterBench](https://arxiv.org/abs/2403.12031) (11 LLMs, 35,189 prompts), so no
model is ever called.

## Results (test split, n = 10,557, all routers on the same 384-d embeddings)

| Router | Area | Area50 | QNC |
|---|---|---|---|
| GNG-shared (ours) | 0.7853 | 0.3808 | 0.7214 |
| UniRoute | 0.7818 | 0.3773 | 0.7491 |
| KNN | 0.7785 | 0.3760 | 1.0000 |
| GNG-isolated (ours) | 0.7775 | 0.3776 | 1.0000 |
| SVM | 0.7503 | 0.3605 | 1.0000 |
| MLP | 0.6815 | 0.3155 | 1.0000 |

Area and Area50 are areas under the accuracy-vs-normalized-cost deferral curve (higher is better).
QNC is the lowest normalized cost at which a router reaches the accuracy of the best single model,
GPT-4 (lower is better; 1.0 means it never does below maximum cost).

In paired bootstrap comparisons GNG-shared is ahead of UniRoute by 0.0035 in area (95% CI 0.0015 to
0.0056). That margin is small and comes from one split of one benchmark, so we describe the router as
competitive with the strongest baseline whose decisions can be inspected. See the paper for the full
discussion and limitations.

## Repository layout

```
cascading_router/   FrugalGPT-style cascade baseline (not reported in the paper)
gng_router/         GNG-isolated: one topology per model (main.py, run_eval.py, visualize.py)
gng_shared/         GNG-shared: one topology, statistics for all models
                    (main.py, run_eval.py, visualize.py, reproduce_paper.py)
knn_router/         KNN baseline
svm_router/         SVM baseline (one LinearSVR per model)
mlp_router/         MLP baseline (one MLP regressor per model)
uniroute/           UniRoute baseline (k-means clusters) plus shared data loading and evaluation
web/                Flask demo
scratchpad/         exploratory notebooks
outputs/, papers/, routerbench-github/   saved outputs, reference papers, RouterBench material
routertrain.csv     RouterBench prompts, per-model scores and costs
```

## Setup

Python 3 with: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`.
`sentence-transformers` is needed only to generate embeddings, and `flask` only for `web/`.
No GPU is required. The numbers in the paper were checked with numpy 2.4, pandas 3.0,
scikit-learn 1.8 and scipy 1.17.

Embeddings: every router uses **`BAAI/bge-small-en-v1.5`** (384-d, unit-normalized). The first run encodes
the prompts and caches them in `prompt_embeddings_cache.pkl` (git-ignored, about 54 MB). Keep that file so
later runs, and the paper's numbers, use identical embeddings.

## Running the routers

Run everything from the repository root. All scripts use a 70/30 random split with seed 42.

```bash
python gng_shared/run_eval.py --seed 0        # GNG-shared (the paper's headline run uses seed 0)
python gng_router/run_eval.py --seed 0        # GNG-isolated; sweeps beta on the TEST split (paper setting)
python gng_router/run_eval.py --seed 0 --tune-beta-on val   # same, but beta chosen on held-out training prompts
python knn_router/main.py
python svm_router/main.py
python mlp_router/main.py
PYTHONPATH=. python uniroute/main.py          # on Windows PowerShell: $env:PYTHONPATH="."; python uniroute/main.py
```

Each script prints Area, Area@50% and QNC and writes its model and metrics under its own `outputs/` folder.
The KNN, SVM and MLP scripts evaluate by recomputing predictions for each of 501 lambda values, so they can
take a very long time; `reproduce_paper.py` below is much faster.

## Reproducing the paper's tables

```bash
python gng_shared/reproduce_paper.py --isolated-model gng_router_model.pkl --decayed-model gng_shared_model.pkl
```

This trains GNG-shared (seeds 0 to 4), KNN, SVM, MLP and UniRoute with the repo's own classes, loads the two
stored GNG models, and prints and saves Table 4 (point estimates), Table 5 (1,000-resample bootstrap) and the
paired differences, marking each Table 4 row MATCH or CHECK against the paper. It takes roughly 35 to 40 minutes
on one CPU core; `--skip-mlp --seeds 0` gives a quick check. The report is written to
`gng_shared/outputs/reproduction/`.

Notes on exact reproduction:

- **GNG-isolated** and the **decayed-updates row** of the paper come from stored models
  (`gng_router_model.pkl`, an unseeded run, and `gng_shared_model.pkl`, a run with decayed updates).
  They reproduce through the two `--...-model` flags. A fresh `gng_router/run_eval.py` run is seeded and
  may land close to, but not exactly on, the paper's GNG-isolated row.
- `time_decay_rate` defaults to `0.0` (constant update weights, kappa = 0) for both variants.
  The first GNG-shared run used `0.0005` with a global step counter, which froze the topology after about
  9,000 of 123,160 steps (paper, Section 5).
- Older files in the `*/outputs/` folders may hold earlier numbers (baselines on bge-base, GNG-shared with
  decayed updates) until you re-run the scripts.

## How it works

1. **Grow a topology.** GNG over the training embeddings: nearest-node winner by squared Euclidean distance,
   threshold-triggered node insertion, edges capped by a maximum length, edge aging and node-lifespan pruning.
   The topology is unsupervised.
2. **Attach statistics.** Every training prompt is assigned to its nearest node, and each node stores the
   mean error of every model on its prompts (global error if the node is empty).
3. **Route.** `r(x) = argmin_k  err[s, k] + lambda * cost[k]`, where `s` is the nearest node of the prompt.
   Sweeping lambda traces the deferral curve. GNG-isolated instead grows one topology per model on the prompts
   that model answers correctly and adds a soft boundary penalty beta.
4. **Explain.** The explanation of a decision is the winning node, its neighbors, the per-model error rates
   and the penalized scores that decided the arg-min.

Pool changes (adding or removing a model) are a design extension described in the paper; they are not evaluated.

## Limitations

One benchmark and one random split; bootstrap intervals cover test-prompt sampling, not split variation.
beta for GNG-isolated was selected on the test split in the reported run. The routing results use a single
embedding model. Generalization to unseen task types, new LLMs, and price changes was not evaluated.
RouterBench is dominated by a few sources (hellaswag 27.8%, grade-school-math 21.2%). UniRoute was run
with K = 10 clusters while GNG-shared has 107 nodes, so the comparison does not separate the effect of the
number of regions from the effect of the GNG topology.

Update the entry with the final proceedings details when they are available. The data is RouterBench:
Hu et al., "RouterBench: A benchmark for multi-LLM routing system", arXiv:2403.12031.
