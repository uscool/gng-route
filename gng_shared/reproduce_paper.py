"""
Reproduce the numbers reported in the paper (Tables 4 and 5 and the seed study).

Run from the repository root:

    python gng_shared/reproduce_paper.py

Requirements: numpy, pandas, scikit-learn, scipy, matplotlib (sentence-transformers is only needed to
regenerate embeddings; flask only for web/). Needs routertrain.csv and prompt_embeddings_cache.pkl in the
repository root (or pass --csv / --cache), and the two stored paper models in paper_models/
(download all three from the repository's GitHub release; see paper_models/README.md).
Takes roughly 35 minutes on one CPU core with the MLP baseline and five GNG seeds; use --skip-mlp
and fewer --seeds for a quick check. The report is written to
gng_shared/outputs/reproduction/.

Everything uses the repository's own router classes. The deferral-curve metrics are computed
in a vectorised way (one score matrix per router, then argmax(score - lambda * cost) for every
lambda); this gives the same Area / Area50 / QNC as uniroute/eval.py but is fast enough to
bootstrap. Setup, as in the paper:

  * RouterBench prompts in routertrain.csv, 35,189 prompts, 70/30 random split with seed 42
  * BAAI/bge-small-en-v1.5 embeddings (384-d) for every router, read from prompt_embeddings_cache.pkl
  * GNG-shared with constant update weights (kappa = 0, the default in gng_shared/main.py)
  * lambda grid: 0 and 500 log-spaced values in [1e-1, 1e6]; routing costs are the test-split mean costs
    (--routing-costs train uses training-split means instead, so no test statistic enters routing;
    the paper's numbers were produced with the default)

Stored models (the paper's exact runs; no script in this repository writes to paper_models/):
  --isolated-model   paper_models/gng_isolated.pkl        -> GNG-isolated row (an unseeded run, beta = 2.0)
  --decayed-model    paper_models/gng_shared_decayed.pkl  -> the dagger row (first GNG-shared run, decayed updates)
If a stored model is missing, its row is skipped and the script says so.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from uniroute.data import (load_routerbench_data, get_embeddings, split_dataset, DEFAULT_LLM_NAMES,
                           DEFAULT_EMBEDDING_MODEL, DEFAULT_CACHE_PATH, load_router_pickle)
from uniroute.router import UniRouteKMeans
from knn_router.main import KNNRouterModel
from svm_router.main import SVMRouterModel
from mlp_router.main import MLPRouterModel
from gng_shared.main import GNGSharedRouterModel

L = DEFAULT_LLM_NAMES
LAMBDAS = np.array([0.0] + list(np.logspace(-1, 6, 500)))
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Numbers printed in the paper (test split, seed 42). Used only for the PASS/CHECK column.
PAPER = {
    "GNG-shared": (0.7853, 0.3808, 0.7214),
    "UniRoute": (0.7818, 0.3773, 0.7491),
    "KNN": (0.7785, 0.3760, 1.0),
    "GNG-isolated": (0.7775, 0.3776, 1.0),
    "SVM": (0.7503, 0.3605, 1.0),
    "MLP": (0.6815, 0.3155, 1.0),
    "GNG-shared (decayed updates)": (0.7831, 0.3789, 0.7426),
}
PAPER_BOOT = {  # mean area, share of resamples reaching a_max
    "GNG-shared": (0.7852, 0.99), "UniRoute": (0.7817, 0.92), "KNN": (0.7783, 0.10),
    "GNG-isolated": (0.7774, 0.0), "SVM": (0.7502, 0.0), "MLP": (0.6812, 0.0),
    "GNG-shared (decayed updates)": (0.7829, 0.79),
}
PAPER_SEEDS = {"area_mean": 0.7856, "area_sd": 0.0010}


def decisions(scores, cost):
    """scores: (n, K), higher is better. Returns (n, len(LAMBDAS)) chosen model indices."""
    return np.stack([np.argmax(scores - lam * cost, axis=1) for lam in LAMBDAS], axis=1)


def chosen_arrays(D, Y, C):
    rows = np.arange(len(Y))[:, None]
    return Y[rows, D], C[rows, D]


def summarize(acc_chosen, cost_chosen, Y, C):
    acc, cst = acc_chosen.mean(0), cost_chosen.mean(0)
    macc, mcost = Y.mean(0), C.mean(0)
    cmin, cmax, amax = mcost.min(), mcost.max(), macc.max()
    rel = np.clip((cst - cmin) / (cmax - cmin), 0, 1)
    pts = set(zip(rel.tolist(), acc.tolist()))
    pts.add((0.0, float(macc[mcost.argmin()])))
    pts.add((1.0, float(macc[mcost.argmax()])))
    pts = sorted(pts)
    r = np.array([p[0] for p in pts])
    a = np.array([p[1] for p in pts])
    area = _trapz(a, r)
    a50 = np.interp(0.5, r, a)
    keep = r <= 0.5
    r5, a5 = np.append(r[keep], 0.5), np.append(a[keep], a50)
    order = np.argsort(r5, kind="stable")
    area50 = _trapz(a5[order], r5[order])
    reach = r[a >= amax]
    qnc = float(reach.min()) if len(reach) else 1.0
    return float(area), float(area50), qnc


def gng_shared_scores(router, X):
    ids = list(router.topology.nodes.keys())
    W = np.array([router.topology.nodes[i]["weight"] for i in ids])
    EM = np.array([[router.node_error_rates[i][m] for m in L] for i in ids])
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ W.T + (W ** 2).sum(1)[None, :]
    return -EM[d2.argmin(1)], len(ids), len(router.topology.edges)


def gng_isolated_scores(router, X, beta=2.0):
    S = np.zeros((len(X), len(L)))
    for k, m in enumerate(L):
        topo = router.topologies[m]
        ids = list(topo.nodes.keys())
        W = np.array([topo.nodes[i]["weight"] for i in ids])
        err = np.array([topo.nodes[i]["error_rate"] for i in ids])
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ W.T + (W ** 2).sum(1)[None, :]
        j = d2.argmin(1)
        dist = np.sqrt(np.maximum(d2[np.arange(len(X)), j], 0))
        S[:, k] = -(err[j] + beta * np.maximum(0, dist - router.boundary_radii[m]))
    return S


def status(name, got):
    exp = PAPER.get(name)
    if exp is None:
        return ""
    ok = abs(got[0] - exp[0]) <= 0.0006 and abs(got[1] - exp[1]) <= 0.0006 and abs(got[2] - exp[2]) <= 0.01
    return "MATCH" if ok else "CHECK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "routertrain.csv"))
    ap.add_argument("--cache", default=os.path.join(ROOT, DEFAULT_CACHE_PATH))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="GNG-shared training seeds; the first is the headline run")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--skip-mlp", action="store_true", help="skip the MLP baseline (about 10 minutes on one CPU core)")
    ap.add_argument("--isolated-model", default=os.path.join(ROOT, "paper_models", "gng_isolated.pkl"),
                    help="stored GNG-isolated model of the paper (beta = 2.0)")
    ap.add_argument("--decayed-model", default=os.path.join(ROOT, "paper_models", "gng_shared_decayed.pkl"),
                    help="stored GNG-shared model of the paper trained with decayed updates (dagger row)")
    ap.add_argument("--routing-costs", choices=["test", "train"], default="test",
                    help="split whose mean per-model costs enter the routing rule ('test' reproduces the paper)")
    ap.add_argument("--out", default=os.path.join(ROOT, "gng_shared", "outputs", "reproduction"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_routerbench_data(args.csv)
    emb = get_embeddings(df["prompt"].tolist(), model_name=DEFAULT_EMBEDDING_MODEL, cache_path=args.cache)
    sp = split_dataset(df, emb, train_ratio=0.7, val_ratio=0.0, test_ratio=0.3, seed=42)
    Xtr, Xte, tr, te = sp["train_embeddings"], sp["test_embeddings"], sp["train_df"], sp["test_df"]
    Y = te[L].values.astype(float)
    C = te[[f"{m}|total_cost" for m in L]].values
    cost = C.mean(0) if args.routing_costs == "test" else tr[[f"{m}|total_cost" for m in L]].values.mean(0)
    print(f"prompts={len(df)}  train={len(tr)}  test={len(te)}  embedding dim={Xtr.shape[1]}  routing costs={args.routing_costs}")

    scores, info = {}, {}
    t0 = time.time()

    # --- GNG-shared (headline run = first seed; the others give the seed study)
    seed_rows = []
    for s in args.seeds:
        np.random.seed(s)
        r = GNGSharedRouterModel(llm_names=L, max_nodes=200)
        r.fit(Xtr, tr, epochs=5)
        S, n_nodes, n_edges = gng_shared_scores(r, Xte)
        D = decisions(S, cost)
        m = summarize(*chosen_arrays(D, Y, C), Y, C)
        seed_rows.append({"seed": s, "nodes": n_nodes, "edges": n_edges, "area": m[0], "area50": m[1], "qnc": m[2]})
        print(f"GNG-shared seed {s}: nodes={n_nodes} edges={n_edges} area={m[0]:.4f} area50={m[1]:.4f} qnc={m[2]:.4f}  [{time.time()-t0:.0f}s]")
        if s == args.seeds[0]:
            scores["GNG-shared"] = S
            info["GNG-shared"] = {"nodes": n_nodes, "edges": n_edges, "seed": s}

    # --- baselines (repo classes)
    knn = KNNRouterModel(50, "cosine"); knn.fit(Xtr, tr, L)
    scores["KNN"] = knn.train_y[knn.knn.kneighbors(Xte)[1]].mean(1)
    svm = SVMRouterModel(); svm.fit(Xtr, tr, L)
    scores["SVM"] = np.vstack([svm.svrs[m].predict(Xte) for m in L]).T
    uni = UniRouteKMeans(K=10, random_state=42); uni.fit(Xtr); uni.compute_validation_errors(Xtr, tr, L)
    scores["UniRoute"] = -uni.val_errors[:, uni.get_cluster_assignments(Xte)].T
    if not args.skip_mlp:
        mlp = MLPRouterModel(); mlp.fit(Xtr, tr, L)
        scores["MLP"] = np.vstack([mlp.mlps[m].predict(Xte) for m in L]).T
    print(f"baselines done [{time.time()-t0:.0f}s]")

    # --- optional pickled models
    if args.isolated_model and os.path.exists(args.isolated_model):
        iso = load_router_pickle(args.isolated_model)["router"]
        scores["GNG-isolated"] = gng_isolated_scores(iso, Xte, beta=2.0)
    else:
        print(f"NOTE: {args.isolated_model} not found; the GNG-isolated row is skipped.")
    if args.decayed_model and os.path.exists(args.decayed_model):
        old = load_router_pickle(args.decayed_model)["router"]
        scores["GNG-shared (decayed updates)"] = gng_shared_scores(old, Xte)[0]
    else:
        print(f"NOTE: {args.decayed_model} not found; the decayed-updates row is skipped.")

    # --- point estimates (Table 4)
    order = ["GNG-shared", "UniRoute", "KNN", "GNG-isolated", "SVM", "MLP", "GNG-shared (decayed updates)"]
    names = [n for n in order if n in scores]
    arrays, table4 = {}, {}
    for n in names:
        D = decisions(scores[n], cost)
        arrays[n] = chosen_arrays(D, Y, C)
        table4[n] = summarize(*arrays[n], Y, C)

    # --- bootstrap over test prompts (Table 5); all routers share the same resamples
    rng = np.random.default_rng(0)
    nte = len(Y)
    boot = {n: np.zeros((args.bootstrap, 3)) for n in names}
    for b in range(args.bootstrap):
        idx = rng.integers(0, nte, nte)
        for n in names:
            boot[n][b] = summarize(arrays[n][0][idx], arrays[n][1][idx], Y[idx], C[idx])

    # --- report
    lines = []
    lines.append("## Table 4: routing performance (test split, n = %d)" % nte)
    lines.append("| Router | Area | Area50 | QNC | vs paper |")
    lines.append("|---|---|---|---|---|")
    for n in names:
        a, a5, q = table4[n]
        lines.append(f"| {n} | {a:.4f} | {a5:.4f} | {q:.4f} | {status(n, table4[n])} |")
    lines.append("")
    lines.append(f"## Table 5: bootstrap over {args.bootstrap} resamples")
    lines.append("| Router | Area (mean [95% CI]) | Reaches a_max | paper area / reach |")
    lines.append("|---|---|---|---|")
    for n in names:
        q = np.percentile(boot[n][:, 0], [2.5, 97.5])
        reach = float((boot[n][:, 2] < 0.9999).mean())
        pa = PAPER_BOOT.get(n)
        ptxt = f"{pa[0]:.4f} / {pa[1]*100:.0f}%" if pa else ""
        lines.append(f"| {n} | {boot[n][:,0].mean():.4f} [{q[0]:.4f}, {q[1]:.4f}] | {reach*100:.0f}% | {ptxt} |")
    lines.append("")
    lines.append("## Paired differences, GNG-shared minus router (area)")
    lines.append("| Router | mean diff | 95% CI |")
    lines.append("|---|---|---|")
    for n in names[1:]:
        d = boot["GNG-shared"][:, 0] - boot[n][:, 0]
        q = np.percentile(d, [2.5, 97.5])
        lines.append(f"| {n} | {d.mean():+.4f} | [{q[0]:+.4f}, {q[1]:+.4f}] |")
    both = (boot["GNG-shared"][:, 2] < 0.9999) & (boot["UniRoute"][:, 2] < 0.9999)
    if both.any():
        diff = (boot["GNG-shared"][both, 2] - boot["UniRoute"][both, 2]).mean()
        lines.append(f"\nQNC difference vs UniRoute when both reach a_max: {diff:+.3f} (n = {int(both.sum())} resamples)")
    lines.append("")
    areas = np.array([r["area"] for r in seed_rows])
    qn = np.array([r["qnc"] for r in seed_rows])
    lines.append(f"## GNG-shared over {len(seed_rows)} training seeds")
    lines.append("| seed | nodes | edges | area | area50 | qnc |")
    lines.append("|---|---|---|---|---|---|")
    for r in seed_rows:
        lines.append(f"| {r['seed']} | {r['nodes']} | {r['edges']} | {r['area']:.4f} | {r['area50']:.4f} | {r['qnc']:.4f} |")
    if len(areas) > 1:
        lines.append(f"\narea mean {areas.mean():.4f}, SD {areas.std(ddof=1):.4f} (paper: {PAPER_SEEDS['area_mean']:.4f}, {PAPER_SEEDS['area_sd']:.4f}); "
                     f"QNC range {qn.min():.3f} to {qn.max():.3f}")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.out, "reproduction_report.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "reproduction_results.json"), "w") as f:
        json.dump({"table4": table4, "seeds": seed_rows,
                   "bootstrap": {n: {"area_mean": float(boot[n][:, 0].mean()),
                                     "area_ci": [float(x) for x in np.percentile(boot[n][:, 0], [2.5, 97.5])],
                                     "reach_amax": float((boot[n][:, 2] < 0.9999).mean())} for n in names}}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
