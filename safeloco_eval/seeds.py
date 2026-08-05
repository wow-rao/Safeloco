"""The shared evaluation seed list (experiment_protocol_phased.md §1.1).

`eval_seeds.json` is a checked-in file of 1,000 entries, identical on both
branches, consumed by the eval script.  Divergent seeds make the two branches'
tables un-mergeable, so this file is an invariant: regenerate it only if both
branches are regenerated together, and record the hash in the run manifest.

How the seeds are actually used.  IsaacGym has no per-episode seed; episode
randomness (spawn pose, commands, domain randomisation draws) comes off the
global RNG at reset.  The eval loop therefore runs episodes in **chunks** of
`num_envs`: before each chunk it re-seeds globally with the chunk's seed and
does a full `env.reset()`, then records **only the first episode each env
completes**.  Because the RNG state at the reset is identical across methods,
every method sees exactly the same 1,000 initial conditions, and because only
the first episode per env per chunk is kept, no drift from earlier filtered
steps contaminates later episodes.
"""

import hashlib
import json
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SEED_FILE = os.path.join(_HERE, "eval_seeds.json")

N_EVAL_EPISODES = 1000
GENERATOR_SEED = 20250801  # frozen; do not change without regenerating both branches


def generate(n=N_EVAL_EPISODES, generator_seed=GENERATOR_SEED):
    rng = random.Random(generator_seed)
    return [rng.randrange(1, 2 ** 31 - 1) for _ in range(n)]


def load(path=None):
    with open(path or DEFAULT_SEED_FILE) as f:
        blob = json.load(f)
    return blob["seeds"], blob


def file_hash(path=None):
    with open(path or DEFAULT_SEED_FILE, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def chunk_plan(seeds, num_envs, n_episodes):
    """[(chunk_idx, chunk_seed, episodes_to_keep)] covering `n_episodes`.

    The last chunk keeps only as many episodes as are still needed, so the
    episode count is exactly `n_episodes` regardless of `num_envs`.
    """
    plan, done, c = [], 0, 0
    while done < n_episodes:
        keep = min(num_envs, n_episodes - done)
        plan.append((c, seeds[done % len(seeds)], keep))
        done += keep
        c += 1
    return plan


def write(path=None, n=N_EVAL_EPISODES):
    path = path or DEFAULT_SEED_FILE
    blob = {
        "n": n,
        "generator_seed": GENERATOR_SEED,
        "note": ("Shared eval seed list, experiment_protocol_phased.md §1.1. "
                 "Identical on the collision and joint-limit branches. "
                 "Consumed as per-chunk global seeds; see seeds.py."),
        "seeds": generate(n),
    }
    with open(path, "w") as f:
        json.dump(blob, f, indent=1)
    return path


if __name__ == "__main__":
    p = write()
    print("wrote {} ({} seeds, sha256[:16]={})".format(
        p, N_EVAL_EPISODES, file_hash(p)))
