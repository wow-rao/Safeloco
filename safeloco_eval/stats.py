"""Statistics for the rebuttal experiments (analysis_protocol.md §2).

Pure stdlib on purpose: this module is imported by the analysis scripts *and*
by the offline test suite, and it must run anywhere (including a machine with
no IsaacGym / torch installed).  Everything here operates on plain lists of
floats.

Copy this file verbatim to the joint-limit branch (§1.1 invariant).
"""

import math

Z95 = 1.959963984540054  # two-sided normal quantile at 95%


# ---------------------------------------------------------------- rates ----

def wilson_ci(k, n, z=Z95):
    """Wilson score interval for a binomial rate.

    Preferred over the normal approximation at rates near 0 (analysis_protocol
    §2).  Returns (point, lo, hi) as fractions in [0, 1].
    """
    if n <= 0:
        return (float("nan"), 0.0, 1.0)
    if k > n:
        raise ValueError(
            "wilson_ci: successes exceed trials (k={}, n={}). This is an "
            "accounting bug in whatever produced the counts, not a rate."
            .format(k, n))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def ci_overlap(ci_a, ci_b):
    """True when two (lo, hi) intervals overlap.

    The E1 verdict rules are phrased as "CIs do not overlap", so this is the
    single place that decision is made.
    """
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    return not (hi_a < lo_b or hi_b < lo_a)


# ------------------------------------------------------- per-episode means --

def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs, ddof=1):
    xs = list(xs)
    n = len(xs)
    if n <= ddof:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def sem(xs):
    """Standard error of the mean.  §2: with N=1000 this is what decides
    whether a difference in per-episode means is real."""
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    return std(xs) / math.sqrt(len(xs))


def mean_ci(xs, z=Z95):
    """(mean, lo, hi) normal-approximation CI on a per-episode mean."""
    m, s = mean(xs), sem(xs)
    if math.isnan(s):
        return (m, float("nan"), float("nan"))
    return (m, m - z * s, m + z * s)


# ------------------------------------------------------------- rankings ----

def _rankdata(xs):
    """Average ranks, ties shared (needed for a correct Spearman under the
    clamp(max=0.5) saturation in the safety returns)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson_r(x, y):
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman_rho(x, y):
    """Rank correlation.  Primary Q-hat quality metric: the codebase clamps
    safety values at 0.5, which compresses the top of the range and makes raw
    MSE misleading (addendum §3.2)."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    return pearson_r(_rankdata(list(x)), _rankdata(list(y)))


def auroc(scores, labels):
    """Area under the ROC curve via the rank (Mann-Whitney) identity.

    `labels` is 1 for the positive class.  For the Q-hat battery the positive
    class is "collision within 20 steps", and a *lower* Q-hat should predict
    it, so callers pass negated scores.
    """
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    ranks = _rankdata(list(scores))
    sum_pos = sum(r for r, l in zip(ranks, labels) if l)
    n_p, n_n = len(pos), len(neg)
    return (sum_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def calibration_curve(pred, actual, n_bins=10):
    """Equal-count bins over `pred`; returns [(n, mean_pred, mean_actual)]."""
    if not pred:
        return []
    order = sorted(range(len(pred)), key=lambda i: pred[i])
    out, per = [], max(1, len(order) // n_bins)
    for b in range(0, len(order), per):
        idx = order[b:b + per]
        if not idx:
            continue
        out.append((len(idx),
                    mean([pred[i] for i in idx]),
                    mean([actual[i] for i in idx])))
    return out


# ------------------------------------------------------------ bootstrap ----

def bootstrap_ci(xs, statistic=mean, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap.  Used for paired differences (E6) and anywhere a
    closed-form interval is not available."""
    xs = list(xs)
    if len(xs) < 2:
        return (statistic(xs) if xs else float("nan"), float("nan"), float("nan"))
    rng = _LCG(seed)
    n = len(xs)
    stats = []
    for _ in range(n_boot):
        stats.append(statistic([xs[rng.randbelow(n)] for _ in range(n)]))
    stats.sort()
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (statistic(xs), lo, hi)


class _LCG:
    """Deterministic PRNG so bootstrap CIs are reproducible without numpy."""

    def __init__(self, seed=0):
        self.state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def randbelow(self, n):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (self.state >> 33) % n


# --------------------------------------------------------------- format ----

def fmt_rate(point, lo, hi, pct=True, digits=1):
    """`12.3 [10.9, 13.8]` — the table format used in every R-table."""
    if math.isnan(point):
        return "--"
    s = 100.0 if pct else 1.0
    return "{p:.{d}f} [{l:.{d}f}, {h:.{d}f}]".format(
        p=point * s, l=lo * s, h=hi * s, d=digits)


def fmt_mean(m, s, digits=1):
    """`33.9 ± 4.2` — the paper's per-episode convention."""
    if math.isnan(m):
        return "--"
    if math.isnan(s):
        return "{m:.{d}f}".format(m=m, d=digits)
    return "{m:.{d}f} ± {s:.{d}f}".format(m=m, s=s, d=digits)
