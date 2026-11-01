# Stored paper models

This folder holds the exact models behind two rows of the paper. No script in this repository writes
here, so re-running training cannot overwrite them.

| File | Paper row | Why it is stored |
|---|---|---|
| `gng_isolated.pkl` | GNG-isolated | It came from an unseeded run, so retraining lands close to, but not exactly on, the paper's numbers. |
| `gng_shared_decayed.pkl` | GNG-shared (decayed updates), the dagger row | It was trained with the earlier decayed-update default, which the code no longer uses. |

Download both from the repository's
[GitHub release](https://github.com/uscool/gng-route/releases), together with
`prompt_embeddings_cache.pkl`, which goes in the repository root.
`gng_shared/reproduce_paper.py` uses all three by default.

The `.pkl` files are git-ignored; only this README is tracked.
