"""Shared evaluation machinery for the rebuttal experiments.

This package is a **cross-phase invariant** (experiment_protocol_phased.md
§1.1): copy it verbatim to the joint-limit branch, or cherry-pick the commit.
Divergent metric code is what makes the two branches' tables un-mergeable.

Modules
-------
metrics       metric definitions and thresholds, fixed once (§1)
stats         Wilson CIs, Spearman, AUROC, bootstrap -- pure stdlib (§2)
filters       deployment-time action filters (E1-A/B now, J1 clamp in Phase 2)
qsafe         the learned joint-space action-value Q_safe and its targets
eval_common   the evaluation loop, termination definition, episode records
seeds         the shared 1,000-episode eval seed list
cli           argument plumbing around IsaacGym's and Dreamer's arg parsers

`metrics`, `stats` and `seeds` import nothing beyond the standard library, so
the analysis and its tests run without a GPU, torch, or IsaacGym.
"""
