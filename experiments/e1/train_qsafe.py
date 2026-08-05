"""Step 2 of E1: train Q_safe offline and run its validation battery.

experiment_protocol_phased.md §2.  Needs no simulator -- pure offline
regression on the buffers `collect_qsafe_data.py` wrote, so it runs on any box
with a GPU and torch.

Validation battery (reported in App. J):
  (i)   Spearman rho between Q_hat(o_t, a_t) and realised R_safe(t), held out
  (ii)  AUROC for collision-within-20-steps
  (iii) calibration curve
  (iv)  branched-rollout spot check  -- NOT implemented here; it needs the
        simulator-state-reset harness that E6 builds.  The report says so
        explicitly rather than leaving it silently absent.

Plus the §5 sanity check the analysis protocol asks for: does Q_hat beat a
trivial distance heuristic (`min_cbf_h` alone) at the same AUROC task?  If it
does not, the critic adds nothing and E1's filter should be reported using the
heuristic instead.

Interpretation rule, fixed in advance (§2): E1/E2/E3 conclusions stand only if
rho >= ~0.6 and AUROC >= ~0.8.  Below that, the finding is not "projection
fails" but "learned joint-space barriers of usable quality are hard to obtain"
-- a different sentence in analysis_protocol.md §7, still supportive.  This
script prints which of the two applies; it does not decide it for you.

Usage
-----
    python experiments/e1/train_qsafe.py \
        --buffers logs/e1/qsafe_data/pi_nom logs/e1/qsafe_data/pi_rs \
        --out logs/e1/qsafe --epochs 30
"""

import argparse
import gc
import inspect
import json
import os

CURDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
os.sys.path.insert(0, ROOT)

import torch

from safeloco_eval import metrics as M
from safeloco_eval import stats as ST
from safeloco_eval.qsafe import (SafetyActionValue, compute_safety_returns,
                                 collision_within_horizon,
                                 events_within_horizon)

RHO_THRESHOLD = 0.6      # §2 interpretation rule
AUROC_THRESHOLD = 0.8
COLLISION_HORIZON = 20


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--buffers", nargs="+", required=True,
                   help="paths written by collect_qsafe_data.py, without .pt")
    p.add_argument("--out", type=str, default="logs/e1/qsafe")
    p.add_argument("--alpha", type=float, default=0.95,
                   help="safety-return discount. Three sources disagree: the "
                        "shipped rollout_storage.py body uses 0.7, its own "
                        "docstring says 0.95, Table 4 says 0.80 (addendum §2). "
                        "0.95 chosen; recorded in the report.")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--ensemble", type=int, default=1,
                   help="number of heads; 1 = the primary single head, 3 = the "
                        "min-aggregated control reported at one epsilon")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_val_rows", type=int, default=200000)
    return p.parse_args()


# ---------------------------------------------------------------------------

def load_buffers(paths, alpha):
    """Concatenate buffers along the env axis after computing per-buffer
    targets, episode ids and a validity mask."""
    parts, metas = [], []
    for path in paths:
        buf = torch.load(path + ".pt", map_location="cpu")
        with open(path + ".meta.json") as f:
            metas.append(json.load(f))
        sv, done = buf["safety_value"], buf["done"]
        buf["safety_return"] = compute_safety_returns(sv, done, alpha)

        # Episode id per (t, env); a step belongs to the episode that ends at
        # the next `done`.  Steps after the last `done` in an env form a
        # truncated episode whose returns are bootstrapped from zero, so they
        # are dropped rather than trained on.
        T = done.shape[0]
        ep = torch.cumsum(done.float(), dim=0) - done.float()
        buf["episode_id"] = ep.long()
        has_future_done = torch.flip(
            torch.cummax(torch.flip(done.float(), [0]), dim=0).values, [0])
        buf["valid"] = has_future_done > 0
        parts.append(buf)
        print("[qsafe] {}: T={} B={} valid={:.1%}".format(
            os.path.basename(path), T, done.shape[1],
            float(buf["valid"].float().mean())))
    return parts, metas


def flatten(parts, keys, ep_offset_step=100000):
    """Flatten [T, B, ...] -> [N, ...] with globally unique episode ids."""
    out = {k: [] for k in keys}
    out["episode_key"] = []
    offset = 0
    for pi, buf in enumerate(parts):
        T, B = buf["done"].shape
        for k in keys:
            if k not in buf:
                if pi > 0 and out[k]:
                    raise ValueError(
                        "buffer {} is missing '{}' but an earlier buffer has "
                        "it -- collect all buffers with the same "
                        "--store_grid".format(pi, k))
                continue
            v = buf[k]
            out[k].append(v.reshape(T * B, *v.shape[2:]))
        key = (buf["episode_id"] * B + torch.arange(B).unsqueeze(0)
               + offset).reshape(T * B)
        out["episode_key"].append(key)
        offset += (int(buf["episode_id"].max()) + 2) * B
    return {k: torch.cat(v, 0) for k, v in out.items() if v}


def split_by_episode(episode_key, val_frac, seed):
    """Train/val split **by episode**, per the protocol."""
    uniq = torch.unique(episode_key)
    g = torch.Generator().manual_seed(seed)
    perm = uniq[torch.randperm(len(uniq), generator=g)]
    n_val = max(1, int(val_frac * len(uniq)))
    val_ids = perm[:n_val]
    is_val = torch.isin(episode_key, val_ids)
    return ~is_val, is_val


# ---------------------------------------------------------------------------

def train_one(data, tr_idx, meta, args, seed):
    torch.manual_seed(seed)
    dev = args.device
    grid_enabled = "grid" in data
    model = SafetyActionValue(
        num_critic_obs=meta["num_critic_obs"],
        num_actions=data["action"].shape[1],
        wm_feature_dim=meta["wm_feature_dim"],
        grid_enabled=grid_enabled,
        grid_H=(meta["grid_shape"][0] if grid_enabled else 60),
        grid_W=(meta["grid_shape"][1] if grid_enabled else 30),
    ).to(dev)

    # Normaliser fitted on the train split only.
    sub = tr_idx[torch.randperm(len(tr_idx))[:50000]]
    model.fit_normalizer(data["critic_obs"][sub].float().to(dev),
                         data["wm_feature"][sub].float().to(dev))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    n = len(tr_idx)
    steps_per_epoch = max(1, n // args.batch)
    g = torch.Generator().manual_seed(seed + 1)

    for ep in range(args.epochs):
        perm = tr_idx[torch.randperm(n, generator=g)]
        tot = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            co = data["critic_obs"][idx].float().to(dev)
            wf = data["wm_feature"][idx].float().to(dev)
            ac = data["action"][idx].float().to(dev)
            tg = data["safety_return"][idx].float().to(dev)
            gr = data["grid"][idx].float().to(dev) if grid_enabled else None

            feat = model.state_features(co, wf, gr)
            q = model.q_from_features(feat, ac)
            v = model.v_from_features(feat)
            loss = (q - tg).pow(2).mean() + (v - tg).pow(2).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        print("[qsafe] seed {} epoch {}/{} loss {:.5f}".format(
            seed, ep + 1, args.epochs, tot / steps_per_epoch))

    return model


@torch.no_grad()
def predict(model, data, idx, args, chunk=8192):
    grid_enabled = "grid" in data
    qs, vs = [], []
    for s in range(0, len(idx), chunk):
        sl = idx[s:s + chunk]
        co = data["critic_obs"][sl].float().to(args.device)
        wf = data["wm_feature"][sl].float().to(args.device)
        ac = data["action"][sl].float().to(args.device)
        gr = data["grid"][sl].float().to(args.device) if grid_enabled else None
        feat = model.state_features(co, wf, gr)
        qs.append(model.q_from_features(feat, ac).cpu())
        vs.append(model.v_from_features(feat).cpu())
    return torch.cat(qs), torch.cat(vs)


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    parts, metas = load_buffers(args.buffers, args.alpha)
    meta = metas[0]

    # Collision-within-H labels, computed per buffer so they never cross an
    # episode (or a buffer) boundary.  Label from the geometric collision flag
    # when the buffer carries one, so the battery validates Q-hat against the
    # same definition the E1 tables report.
    label_source = ("geometric_link_cylinder"
                    if all("collision" in b and bool(b["collision"].any())
                           for b in parts) else "min_cbf_h_proxy")
    if label_source == "geometric_link_cylinder":
        labels = torch.cat([
            events_within_horizon(b["collision"], b["done"],
                                  COLLISION_HORIZON).reshape(-1)
            for b in parts], 0)
    else:
        print("[qsafe] WARNING: buffers carry no geometric collision flag (or "
              "it never fired); falling back to the min_cbf_h < {} proxy for "
              "the AUROC labels. Re-collect with the current "
              "collect_qsafe_data.py to label against the reported collision "
              "metric.".format(M.COLLISION_H_THRESH))
        labels = torch.cat([
            collision_within_horizon(b["min_cbf_h"], b["done"],
                                     COLLISION_HORIZON,
                                     M.COLLISION_H_THRESH).reshape(-1)
            for b in parts], 0)

    keys = ["critic_obs", "wm_feature", "action", "safety_return",
            "safety_value", "min_cbf_h", "collision", "done", "valid", "grid"]
    data = flatten(parts, keys)
    data["label20"] = labels
    # `flatten` copies; release the source buffers before training so peak RAM
    # is one copy of the dataset rather than two.
    del parts
    gc.collect()

    valid = data["valid"]
    all_idx = valid.nonzero(as_tuple=False).flatten()
    tr_mask, va_mask = split_by_episode(data["episode_key"][all_idx],
                                        args.val_frac, args.seed)
    tr_idx, va_idx = all_idx[tr_mask], all_idx[va_mask]
    print("[qsafe] train rows {}  val rows {}".format(len(tr_idx), len(va_idx)))

    if len(all_idx) == 0 or len(tr_idx) == 0 or len(va_idx) == 0:
        max_ep = max((m.get("max_episode_length") or 0) for m in metas)
        steps = max((m.get("steps") or 0) for m in metas)
        raise SystemExit(
            "\nNo usable rows: {} of {} collected steps are valid.\n\n"
            "A step only becomes usable once its episode has *finished*: the "
            "safety-return recursion runs backwards from the terminal step, "
            "so a trailing episode that never ends has no target and is "
            "dropped. If the collection window is shorter than an episode, "
            "nothing finishes and everything is dropped.\n\n"
            "This buffer collected {} steps against episodes of up to {} "
            "steps.\n\nFix: re-run collect_qsafe_data.py with a larger "
            "--collect_steps (comfortably more than {}), and make sure "
            "--stagger_episodes is on (the default) so episodes finish "
            "throughout the window rather than all at the end.".format(
                int(valid.sum()), int(valid.numel()), steps, max_ep, max_ep))

    models = [train_one(data, tr_idx, meta, args, args.seed + i)
              for i in range(args.ensemble)]
    model = models[0]

    # delta = 0.05 * IQR of Q_hat over the training set (protocol §3).
    q_tr, _ = predict(model, data, tr_idx[torch.randperm(len(tr_idx))[:100000]],
                      args)
    qs = q_tr.sort().values
    iqr = float(qs[int(0.75 * len(qs))] - qs[int(0.25 * len(qs))])
    model.q_iqr.fill_(iqr)

    # ---------------- validation battery -------------------------------
    if len(va_idx) > args.max_val_rows:
        va_idx = va_idx[torch.randperm(len(va_idx))[:args.max_val_rows]]
    q_va, v_va = predict(model, data, va_idx, args)
    tgt = data["safety_return"][va_idx].float()
    lab = data["label20"][va_idx].bool().tolist()
    h_va = data["min_cbf_h"][va_idx].float()

    rho = ST.spearman_rho(q_va.tolist(), tgt.tolist())
    rho_v = ST.spearman_rho(v_va.tolist(), tgt.tolist())
    auc_q = ST.auroc((-q_va).tolist(), lab)
    auc_v = ST.auroc((-v_va).tolist(), lab)
    auc_h = ST.auroc((-h_va).tolist(), lab)      # distance-heuristic baseline
    calib = ST.calibration_curve(q_va.tolist(), tgt.tolist(), n_bins=10)
    mse = float((q_va - tgt).pow(2).mean())

    passed = (rho >= RHO_THRESHOLD) and (auc_q >= AUROC_THRESHOLD)
    beats_heuristic = auc_q > auc_h

    report = {
        "alpha": args.alpha,
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "ensemble": args.ensemble, "seed": args.seed,
        "buffers": args.buffers, "buffer_meta": metas,
        "n_train_rows": int(len(tr_idx)), "n_val_rows": int(len(va_idx)),
        "auroc_label_source": label_source,
        "delta_iqr": iqr, "delta": 0.05 * iqr,
        "battery": {
            "spearman_rho_q": rho,
            "spearman_rho_v": rho_v,
            "auroc_q_collision20": auc_q,
            "auroc_v_collision20": auc_v,
            "auroc_min_cbf_h_baseline": auc_h,
            "val_mse": mse,
            "positive_rate_collision20": float(sum(lab)) / max(1, len(lab)),
            "calibration": [{"n": n, "pred": p, "actual": a}
                            for n, p, a in calib],
        },
        "thresholds": {"rho": RHO_THRESHOLD, "auroc": AUROC_THRESHOLD},
        "verdict": {
            "critic_informative": bool(passed),
            "beats_distance_heuristic": bool(beats_heuristic),
            "e1_interpretation": (
                "standard" if passed else
                "Q-hat failed its validation thresholds. E1's verdict is NOT "
                "'projection fails' but 'usable learned joint-space barriers "
                "are hard to obtain' -- analysis_protocol.md §7, row 'Q-hat "
                "failed validation'."),
            "filter_target_note": (
                "Q-hat beats the distance heuristic; use Q-hat." if
                beats_heuristic else
                "Q-hat does NOT beat predicting collision from min_cbf_h "
                "alone (analysis_protocol.md §5). The critic adds nothing and "
                "E1's filter should be reported using the heuristic instead -- "
                "simpler and more defensible."),
        },
        "not_run": ["branched-rollout spot check (battery item iv): needs the "
                    "simulator state-reset harness built for E6"],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    model.save(args.out + ".pt", meta={"report": report,
                                       "action_scale": meta["action_scale"],
                                       "clip_actions": meta["clip_actions"]})
    if args.ensemble > 1:
        for i, m in enumerate(models[1:], start=1):
            m.q_iqr.fill_(iqr)
            m.save("{}.member{}.pt".format(args.out, i))
    with open(args.out + ".report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Q_safe validation battery " + "=" * 45)
    print("  Spearman rho (Q, R_safe)        {:.3f}   (threshold {:.2f})".format(
        rho, RHO_THRESHOLD))
    print("  Spearman rho (V, R_safe)        {:.3f}".format(rho_v))
    print("  AUROC collision-within-20 (Q)   {:.3f}   (threshold {:.2f})".format(
        auc_q, AUROC_THRESHOLD))
    print("  AUROC collision-within-20 (V)   {:.3f}".format(auc_v))
    print("  AUROC baseline (min_cbf_h)      {:.3f}".format(auc_h))
    print("  val MSE                         {:.5f}".format(mse))
    print("  delta = 0.05 * IQR(Q)           {:.5f}".format(0.05 * iqr))
    print("  critic informative?             {}".format(
        "YES" if passed else "NO"))
    print("  beats distance heuristic?       {}".format(
        "YES" if beats_heuristic else "NO"))
    if not passed:
        print("\n  " + report["verdict"]["e1_interpretation"])
    if not beats_heuristic:
        print("\n  " + report["verdict"]["filter_target_note"])
    print("=" * 74)
    print("wrote {}.pt and {}.report.json".format(args.out, args.out))


if __name__ == "__main__":
    main()
