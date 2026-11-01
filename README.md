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
gng_shared/         GNG-shared (headline model): one topology, statistics for all models
                    main.py (model), run_eval.py (train + evaluate), visualize.py (topology figure),
                    reproduce_paper.py (all paper tables in one run)
gng_router/         GNG-isolated: one topology per model (main.py, run_eval.py, visualize.py)
uniroute/           UniRoute baseline (k-means clusters), plus the shared code every router uses:
                    data.py (data loading, embedding model and cache) and eval.py (deferral-curve metrics)
knn_router/         KNN baseline
svm_router/         SVM baseline (one LinearSVR per model)
mlp_router/         MLP baseline (one MLP regressor per model)
cascading_router/   FrugalGPT-style cascade baseline (not reported in the paper)
paper_models/       the paper's two stored GNG models (downloaded from the release; see its README)
web/                Flask demo that routes a typed prompt and explains the decision
scratchpad/         exploratory notebooks from development (not needed for any result)
routerbench-github/ material copied from the RouterBench authors' repository (see the RouterBench paper)
routertrain.csv     RouterBench prompts, per-model scores and costs
```

## Setup

Python 3 with: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib` (plus `networkx` for the topology
figures). `sentence-transformers` is needed only to compute embeddings, and `flask` only for `web/`.
No GPU is required. The numbers in the paper were checked with numpy 2.4, pandas 3.0,
scikit-learn 1.8 and scipy 1.17.

Embeddings: every router uses **`BAAI/bge-small-en-v1.5`** (384-d, unit-normalized), set once as
`DEFAULT_EMBEDDING_MODEL` in `uniroute/data.py`. Embeddings are cached in `prompt_embeddings_cache.pkl`
in the repository root (`DEFAULT_CACHE_PATH`, git-ignored, about 54 MB). If a script asks for a different
model or dataset than the cache holds, it stops with an error instead of overwriting the cache.

## Getting the paper's embeddings and models

The [GitHub release](https://github.com/uscool/gng-route/releases) has three files:

| File | Put it in | Used for |
|---|---|---|
| `prompt_embeddings_cache.pkl` | repository root | the exact embeddings behind every number in the paper |
| `gng_isolated.pkl` | `paper_models/` | the GNG-isolated row (an unseeded run) |
| `gng_shared_decayed.pkl` | `paper_models/` | the GNG-shared decayed-updates row (dagger) |

With these, every number is reproduced exactly, without downloading the embedding model. Without the cache,
the first run computes the embeddings itself (this needs `sentence-transformers`).

## Reproducing the paper's results

```bash
python gng_shared/reproduce_paper.py
```

This trains GNG-shared (seeds 0 to 4), KNN, SVM, MLP and UniRoute with the repository's own classes, loads
the two stored models from `paper_models/`, and writes the paper's tables to
`gng_shared/outputs/reproduction/`:

| Paper result | In the report |
|---|---|
| Table 4, point estimates for every router | "Table 4", each row marked MATCH or CHECK against the paper |
| Table 5, 1,000-resample bootstrap | "Table 5" |
| Paired differences against GNG-shared | "Paired differences" |
| Seed study (GNG-shared, seeds 0 to 4) | "GNG-shared over 5 training seeds" |

It takes roughly 35 to 40 minutes on one CPU core; `--skip-mlp --seeds 0` gives a quick check.
The report (`reproduction_report.md`, `reproduction_results.json`) is the one output kept in git.

Routing costs: as in the paper, the cost `C_m` in the routing rule is each model's mean cost on the test split.
Add `--routing-costs train` to use training-split means instead, so that no test-split statistic enters the
routing decision. `compute_deferral_curve_and_metrics` in `uniroute/eval.py` takes the same choice through its
`routing_costs` argument.

Notes on exact reproduction:

- `time_decay_rate` defaults to `0.0` (constant update weights, kappa = 0) for both GNG variants.
  The first GNG-shared run used `0.0005` with a global step counter, which froze the topology after about
  9,000 of 123,160 steps (paper, Section 5); that run is the stored `gng_shared_decayed.pkl`.
- A fresh `gng_router/run_eval.py` run is seeded and may land close to, but not exactly on, the paper's
  GNG-isolated row; the stored `gng_isolated.pkl` reproduces it exactly.

## Producing the paper's figures

```bash
python gng_shared/run_eval.py --seed 0 --out-dir figures            # headline GNG-shared: model + deferral curve
python gng_shared/visualize.py --out-dir figures                    # its topology (spring layout)
python gng_router/visualize.py --model paper_models/gng_isolated.pkl --out-dir figures   # GNG-isolated topologies
```

`gng_shared/run_eval.py --seed 0` trains the same model as the headline row of the report and saves it as
`gng_shared_model.pkl` in the repository root, which `gng_shared/visualize.py` reads by default.

## Running a single router

Run from the repository root. All scripts use a 70/30 random split with seed 42.

```bash
python gng_shared/run_eval.py --seed 0                      # GNG-shared
python gng_router/run_eval.py --seed 0                      # GNG-isolated; beta swept on the TEST split (paper setting)
python gng_router/run_eval.py --seed 0 --tune-beta-on val   # same, with beta chosen on held-out training prompts
python uniroute/main.py
python knn_router/main.py
python svm_router/main.py
python mlp_router/main.py
```

Each script prints Area, Area@50% and QNC, saves its trained model in the repository root and its metrics and
deferral curve in its own `outputs/` folder (git-ignored). The KNN, SVM and MLP scripts recompute predictions
for each of 501 lambda values, so they are slow; `reproduce_paper.py` evaluates the same routers much faster.

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

## Citation

Citation details will be added when the proceedings are published.

The data is RouterBench: Hu et al., "RouterBench: A benchmark for multi-LLM routing system", arXiv:2403.12031.
