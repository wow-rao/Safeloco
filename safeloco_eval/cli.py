"""Argument plumbing.

Two consumers parse `sys.argv` behind our back and both hard-fail on unknown
flags:

* `legged_gym.utils.helpers.get_args` -> `gymutil.parse_arguments(...)`
* `WMPRunner._build_world_model`, which builds its own `ArgumentParser` over
  every key in `dreamer/configs.yaml` and calls `parse_args()`

So an experiment script cannot simply add `--epsilon` to the command line.
`parse_and_strip` parses the experiment's own flags first and removes them from
`sys.argv`, leaving exactly the arguments those two parsers expect.

Import this before `isaacgym` -- it only needs the stdlib.
"""

import sys


def parse_and_strip(parser):
    """Consume this script's flags and hide them from downstream parsers."""
    ns, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return ns


def add_common_eval_args(parser):
    parser.add_argument("--out_dir", type=str, default="logs/e1",
                        help="directory for CSV / manifest output")
    parser.add_argument("--run_id", type=str, default=None,
                        help="row label; defaults to an auto-generated tag")
    parser.add_argument("--n_episodes", type=int, default=1000,
                        help="episodes per configuration (protocol: 1000)")
    parser.add_argument("--eval_envs", type=int, default=250,
                        help="parallel envs; n_episodes should be a multiple")
    parser.add_argument("--eval_terrain", type=str, default="corridor")
    parser.add_argument("--dr", type=str, default="off", choices=["off", "on"],
                        help="eval-time domain randomisation (protocol §1.1: "
                             "decide once, record it)")
    parser.add_argument("--eval_seed_file", type=str, default=None)
    return parser
