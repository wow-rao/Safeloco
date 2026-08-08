"""Offline tests for the step-trap study -- terrain, l-value, filter closure.

The experiment's claim has three legs, and each is checked here as code, with
no GPU and no isaacgym:

  1. GEOMETRY (numpy): the pen is a *closed ring* in the 2-D projection --
     for the geometric body margin, and even for a zero-radius body -- so
     "the body-level controller is stuck" is a property of the world, not of
     any particular filter.  The analytic wall volumes partition (no
     overlap), match the rasterised heightfield exactly, and leave the
     runway free.

  2. THE l-VALUE (torch): the volumetric margin admits exactly what the
     planar one cannot express -- the same XY location flips from unsafe to
     safe when the foot is lifted above the sill plus its margin -- while
     the 0.4 m walls stay un-crossable through the same formula.  The rate
     term penalises approach, and the analytic gradient matches finite
     differences.

  3. FILTER CLOSURE (torch): the actual E4-Y unicycle CBF filter, granted
     full steering authority and perfect command tracking (kinematic
     rollout, the filter's best case), escapes the pen from no heading.

Plus the analyzer end-to-end on synthetic CSVs: the verdict separates,
refuses to separate, and flags a leaking geometry, each from the right data.

Run:  python tests/test_etrap_offline.py
"""

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print("  [PASS] %s" % name)
    else:
        print("  [FAIL] %s %s" % (name, detail))
        FAILED.append(name)


def load_by_path(name, relpath):
    """Import a module by file path, dodging package __init__ chains that
    would pull in isaacgym."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    import numpy as np
    st = load_by_path("step_trap", "legged_gym/utils/step_trap.py")
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

try:
    import torch
    bs = load_by_path("box_sdf", "legged_gym/utils/box_sdf.py")
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# The margin the E4/E4-Y geometric barrier demands of the body: collision
# tolerance + body extent (command_filters.BODY_COLLISION_TOL +
# DEFAULT_BODY_EXTENT).  Restated here so this file needs no torch to run
# its geometry half; the equality is asserted in the filter section.
GEOMETRIC_BODY_MARGIN = 0.23


# ---------------------------------------------------------------- geometry --

def test_geometry():
    for diff in (0.0, 0.5, 1.0):
        info = st.build_step_trap(diff, vertical_scale=0.005, cell_length=8.0)
        g = info["gate"]
        expect = 0.08 + diff * 0.08
        check("difficulty %.1f -> sill %.0f cm" % (diff, 100 * expect),
              abs(g["height"] - expect) < 1e-6, "got %r" % g["height"])

    info = st.build_step_trap(0.5, vertical_scale=0.005, cell_length=8.0)

    check("ring closed for the geometric body margin",
          st.ring_is_closed(info["positions"], info["radius"],
                            GEOMETRIC_BODY_MARGIN, 1.7, 0.2))
    check("ring closed even for a ZERO-margin body -- no filter admits an exit",
          st.ring_is_closed(info["positions"], info["radius"], 0.0, 1.7, 0.2))

    # negative control: the closure check must be able to fail
    holed = np.array([p for p in info["positions"] if not
                      (abs(p[0] - 1.8) < 0.6 and abs(p[1]) < 0.6)],
                     dtype=np.float32)
    check("a ring with a 1 m hole is reported open",
          not st.ring_is_closed(holed, info["radius"], 0.0, 1.7, 0.2))

    own = info["own_boxes"]
    check("four own walls; three borrowed from the next cell",
          own.shape == (4, 6) and info["boxes"].shape == (7, 6),
          "own %r boxes %r" % (own.shape, info["boxes"].shape))
    check("borrowed volumes sit one cell (+8 m) downstream",
          bool((info["boxes"][4:, 0] > 4.0).all()))
    no_next = st.build_step_trap(0.5, vertical_scale=0.005, cell_length=None)
    check("last row borrows nothing", no_next["boxes"].shape == (4, 6))

    # partition: pairwise XY overlap of own boxes must be empty (touching is
    # fine).  If the gate box reached into a tall corner, the l-value would
    # call space safe where the mesh is 0.4 m.
    def xy_overlap(a, b):
        ox = min(a[0] + a[3], b[0] + b[3]) - max(a[0] - a[3], b[0] - b[3])
        oy = min(a[1] + a[4], b[1] + b[4]) - max(a[1] - a[4], b[1] - b[4])
        return max(ox, 0.0) * max(oy, 0.0)

    worst = max(xy_overlap(own[i], own[j])
                for i in range(4) for j in range(i + 1, 4))
    # touching faces in float32 leave ~1e-8 m^2 of numeric "overlap";
    # anything at the mm^2 scale would be a real modelling error
    check("wall volumes partition (no XY overlap)", worst < 1e-6,
          "worst overlap area %g" % worst)

    gate = own[0]
    talls = own[1:]
    check("the gate is the low box; the other three are tall",
          abs((gate[2] + gate[5]) - info["gate"]["height"]) < 1e-6
          and all(abs((b[2] + b[5]) - 0.4) < 1e-6 for b in talls))

    # mesh <-> volumes: rasterise and probe cell centres
    hf = np.zeros((80, 80), dtype=np.int16)
    st.rasterize_boxes(hf, own, 0.1, 0.005, 8.0, 8.0)

    def h_at(x, y):
        return hf[int(round((x + 4.0) / 0.1)), int(round((y + 4.0) / 0.1))] \
            * 0.005

    probes = [
        ((0.0, 0.0), 0.0, "pen centre flat"),
        ((1.8, 0.0), info["gate"]["height"], "gate mid = sill height"),
        ((-1.8, 0.0), 0.4, "back wall tall"),
        ((0.0, 1.8), 0.4, "left wall tall"),
        ((0.0, -1.8), 0.4, "right wall tall"),
        ((1.8, 1.8), 0.4, "corner belongs to the tall side wall"),
        ((1.8, -1.8), 0.4, "corner belongs to the tall side wall (right)"),
        ((2.5, 0.0), 0.0, "runway free just past the gate"),
        ((3.6, 0.0), 0.0, "runway free to the cell edge"),
    ]
    for (x, y), expect, name in probes:
        check(name, abs(h_at(x, y) - expect) < 1e-9,
              "h(%.1f, %.1f) = %r, want %r" % (x, y, h_at(x, y), expect))

    # dense agreement: probe every cell CENTRE -- the mesh must be solid
    # exactly where some own volume covers it (half-cell slack at the faces
    # for the rasteriser's rounding)
    mism = 0
    for i in range(80):
        for j in range(80):
            x = -4.0 + 0.1 * i + 0.05
            y = -4.0 + 0.1 * j + 0.05
            solid = hf[i, j] > 0
            inside = any(abs(x - b[0]) < b[3] - 0.06
                         and abs(y - b[1]) < b[4] - 0.06 for b in own)
            covered = any(abs(x - b[0]) < b[3] + 0.06
                          and abs(y - b[1]) < b[4] + 0.06 for b in own)
            if solid and not covered:
                mism += 1          # mesh solid where no volume is near
            if inside and not solid:
                mism += 1          # volume interior missing from the mesh
    check("heightfield and analytic volumes agree everywhere", mism == 0,
          "%d mismatching probes" % mism)

    check("spawn noise (+-1 m) stays >= 0.35 m clear of the walls",
          1.7 - 1.0 >= 0.35)


# ---------------------------------------------------------------- l-value --

def test_lvalue():
    info = st.build_step_trap(0.5, vertical_scale=0.005, cell_length=8.0)
    sill = info["gate"]["height"]                       # 0.12 m
    boxes = torch.tensor(info["boxes"]).unsqueeze(0)    # [1, 7, 6]
    mask = torch.ones(1, boxes.shape[1], dtype=torch.bool)

    def l_of(p, margin, v=(0, 0, 0)):
        pts = torch.tensor([[list(p)]], dtype=torch.float)
        vel = torch.tensor([[list(v)]], dtype=torch.float)
        m = torch.tensor([margin])
        sv, lmin = bs.box_lvalue(pts, vel, m, boxes, mask)
        return float(sv.item()), float(lmin.item())

    FOOT, BASE = 0.03, 0.10

    _, l = l_of((0.0, 0.0, 0.02), FOOT)
    check("foot at pen centre: safe", l > 0.5, "l=%r" % l)

    _, l = l_of((1.68, 0.0, 0.05), FOOT)
    check("foot 2 cm from the gate face: unsafe (inside the margin)",
          l < 0, "l=%r" % l)
    _, l = l_of((1.60, 0.0, 0.05), FOOT)
    check("foot 10 cm from the gate face: still safe", l > 0, "l=%r" % l)

    # THE property the planar margin cannot express: same XY, lifted -> safe
    _, l_low = l_of((1.8, 0.0, 0.05), FOOT)
    _, l_up = l_of((1.8, 0.0, sill + FOOT + 0.05), FOOT)
    check("same XY over the sill: unsafe low, SAFE lifted",
          l_low < 0 < l_up, "low=%r lifted=%r" % (l_low, l_up))

    _, l = l_of((1.8, 0.0, sill + 0.01), FOOT)
    check("grazing the sill top inside the margin still counts unsafe",
          l < 0, "l=%r" % l)

    _, l = l_of((1.8, 0.0, 0.30), BASE)
    check("base can pass the sill at standing height", l > 0, "l=%r" % l)
    _, l = l_of((0.0, 1.8, 0.30), BASE)
    check("base over a tall wall: unsafe (cannot body-cross)", l < 0,
          "l=%r" % l)
    _, l = l_of((0.0, 1.8, 0.45), BASE)
    check("even hovering at 0.45 m the tall wall violates the base margin",
          l < 0, "l=%r" % l)

    sv_to, _ = l_of((1.5, 0.0, 0.05), FOOT, v=(0.6, 0, 0))
    sv_away, _ = l_of((1.5, 0.0, 0.05), FOOT, v=(-0.6, 0, 0))
    check("approach velocity contracts the margin (Eq. 17's rate term)",
          sv_to < sv_away, "to=%r away=%r" % (sv_to, sv_away))

    # analytic gradient vs finite differences.  Two care points: (i) the SDF
    # is piecewise smooth -- central differences straddle its kinks
    # (face-plane extensions, tied nearest faces, the |.| axes), so compare
    # only at points a clear 2 mm from every non-smooth set, and check the
    # mask is not doing the work; (ii) float32 roundoff on O(1) values,
    # divided by the 2e-4 step, is ~5e-3 of pure noise -- so difference in
    # float64, where the same noise is ~1e-9.
    torch.manual_seed(0)
    P = torch.rand(1, 300, 3, dtype=torch.double)
    P[..., 0] = P[..., 0] * 8 - 4
    P[..., 1] = P[..., 1] * 8 - 4
    P[..., 2] = P[..., 2] * 0.6
    boxes_d = boxes.double()
    p_ = P.unsqueeze(-2)
    c_ = boxes_d[..., :3].unsqueeze(-3)
    e_ = boxes_d[..., 3:].unsqueeze(-3)
    q_ = (p_ - c_).abs() - e_
    top2 = q_.topk(2, dim=-1).values
    smooth = (q_.abs() > 2e-3).all(-1) \
        & ((top2[..., 0] - top2[..., 1]).abs() > 2e-3) \
        & ((p_ - c_).abs() > 2e-3).all(-1)
    g = bs.box_sdf_gradient(P, boxes_d)
    eps, worst = 1e-4, 0.0
    for d in range(3):
        dP = torch.zeros_like(P)
        dP[..., d] = eps
        fd = (bs.box_sdf(P + dP, boxes_d)
              - bs.box_sdf(P - dP, boxes_d)) / (2 * eps)
        worst = max(worst, float(((fd - g[..., d]).abs() * smooth).max()))
    kept = int(smooth.all(-1).sum())
    check("SDF gradient matches finite differences away from kinks",
          worst < 1e-3 and kept > 200,
          "worst err %r over %d/300 fully-smooth samples" % (worst, kept))

    # masking and the obstacle-free convention
    pts = torch.zeros(2, 1, 3)
    pts[:, 0, 2] = 0.02
    vel = torch.zeros_like(pts)
    m = torch.tensor([0.03])
    b2 = boxes.repeat(2, 1, 1)
    mk = torch.zeros(2, boxes.shape[1], dtype=torch.bool)
    mk[0] = True                                 # env 1 has no live boxes
    sv, lmin = bs.box_lvalue(pts, vel, m, b2, mk)
    check("padded envs come back safe (clamped), like the cylinder path",
          abs(float(sv[1]) - bs.CLAMP_MAX) < 1e-9 and float(lmin[1]) > 1e5,
          "sv=%r l=%r" % (sv.tolist(), lmin.tolist()))
    check("live env is unaffected by its padded sibling",
          abs(float(sv[0]) - min(bs.CLAMP_MAX, 1.7 - 0.03)) < 1e-4
          or float(sv[0]) == bs.CLAMP_MAX)

    # volumetric contact: over the sill with clearance is NOT contact
    body = torch.tensor([[[1.8, 0.0, sill + 0.05],     # swinging over
                          [0.5, 0.0, 0.02]]])          # planted inside
    hit = bs.any_point_in_boxes(body, boxes, mask, tol=0.03)
    check("passing over the sill with 5 cm clearance is not wall contact",
          not bool(hit[0]))
    body2 = torch.tensor([[[1.8, 0.0, sill - 0.02],    # inside the sill
                           [0.5, 0.0, 0.02]]])
    check("a link inside the sill volume IS wall contact",
          bool(bs.any_point_in_boxes(body2, boxes, mask, tol=0.03)[0]))


# ---------------------------------------------------- body-filter closure --

def test_filter_closure():
    from safeloco_eval.command_filters import (
        UnicycleCBFCommandFilter, BODY_COLLISION_TOL, DEFAULT_BODY_EXTENT)

    check("restated geometric body margin matches command_filters",
          abs((BODY_COLLISION_TOL + DEFAULT_BODY_EXTENT)
              - GEOMETRIC_BODY_MARGIN) < 1e-12)

    info = st.build_step_trap(0.5, vertical_scale=0.005, cell_length=8.0)
    ring = torch.tensor(info["positions"])              # [N, 3], local

    class FakeEnv(object):
        def __init__(self):
            n = ring.shape[0]
            self.root_states = torch.zeros(1, 13)
            self.root_states[0, 6] = 1.0
            self.obstacle_positions = ring.unsqueeze(0).clone()
            self.obstacle_radii = torch.full((1,), float(info["radius"]))
            self.obstacle_mask = torch.ones(1, n, dtype=torch.bool)
            self.has_obstacles = torch.ones(1, dtype=torch.bool)
            self.commands = torch.tensor([[0.6, 0.0, 0.0, 0.0]])

        def set_pose(self, x, y, th):
            self.root_states[0, 0] = x
            self.root_states[0, 1] = y
            self.root_states[0, 5] = math.sin(th / 2.0)
            self.root_states[0, 6] = math.cos(th / 2.0)

    def rollout(filt, x0, y0, th0, steps=1000, dt=0.02):
        """Kinematic unicycle under the filtered twist -- perfect command
        tracking, i.e. the filter's best case.  Returns peak |x|,|y| and the
        final filtered forward speed."""
        env = FakeEnv()
        x, y, th = x0, y0, th0
        max_x = abs(x)
        max_y = abs(y)
        v_last = 0.0
        for _ in range(steps):
            env.set_pose(x, y, th)
            out, _ = filt.apply(env.commands, env)
            v, w = float(out[0, 0]), float(out[0, 2])
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
            th += w * dt
            max_x = max(max_x, abs(x))
            max_y = max(max_y, abs(y))
            v_last = v
        return max_x, max_y, v_last

    steer = UnicycleCBFCommandFilter(
        alpha=2.0, lookahead=0.35, w_v=4.0, w_omega=1.0,
        vx_range=(0.0, 0.8), omega_range=(-1.0, 1.0))

    wall = 1.7   # the walls' inner faces; reaching them = collision course

    mx, my, v_end = rollout(steer, 0.0, 0.0, 0.0)
    check("full steering authority: parked short of the gate, never out",
          mx < wall and my < wall and v_end < 0.05,
          "max|x|=%.2f max|y|=%.2f v_end=%.3f" % (mx, my, v_end))

    escaped = []
    for k in range(16):
        th = 2.0 * math.pi * k / 16.0
        mx, my, _ = rollout(steer, 0.0, 0.0, th, steps=750)
        if mx >= wall or my >= wall:
            escaped.append(round(th, 2))
    check("no heading escapes the pen (16-way sweep)", not escaped,
          "escaped at %r" % escaped)

    # worst-case spawn: the +-1 m noise puts the robot near a corner
    mx, my, _ = rollout(steer, 0.9, 0.9, 0.25 * math.pi, steps=750)
    check("corner spawn, aimed at the corner: still contained",
          mx < wall and my < wall, "max|x|=%.2f max|y|=%.2f" % (mx, my))

    brake = UnicycleCBFCommandFilter(
        alpha=2.0, lookahead=0.35, w_v=4.0, w_omega=1.0,
        vx_range=(0.0, 0.8), omega_range=(0.0, 0.0))
    mx, my, v_end = rollout(brake, 0.0, 0.0, 0.0)
    check("braking-only arm parks identically",
          mx < wall and v_end < 0.05,
          "max|x|=%.2f v_end=%.3f" % (mx, v_end))

    # and the crossing the filter forbids is real: the same command held for
    # the same duration walks straight out through the sill's location
    x = 0.0
    for _ in range(1000):
        x += 0.6 * 0.02
    check("unfiltered, the commanded walk would be well past the gate",
          x > info["gate"]["x_outer"] + 0.25, "x=%r" % x)


# ---------------------------------------------------------------- analyzer --

sys.path.insert(0, os.path.join(ROOT, "experiments", "etrap"))
import analyze_etrap as A  # noqa: E402


def synth_rows(tmp, ours_exits, body_exits=0, n=40):
    d = os.path.join(tmp, "sweep_%d_%d" % (ours_exits, body_exits))
    os.makedirs(d, exist_ok=True)
    fields = ["run_id", "method", "variant", "policy_tag", "controller",
              "epsilon_or_alpha", "seed", "episode_idx", "chunk_idx",
              "env_idx", "terrain_level", "gate_height", "n_steps", "exited",
              "exit_step", "progress_x", "progress_beyond_gate", "parked",
              "mean_fwd_vel", "vel_err", "mean_abs_yaw_rate",
              "wall_contact_steps", "contacted_wall", "gate_foot_apex",
              "interior_foot_apex", "min_l", "mean_l", "trigger_steps",
              "activation_steps", "infeasible_steps", "sum_dnorm_v",
              "sum_dnorm_omega", "term_timeout", "term_fall_flag",
              "term_vel_violate", "term_contact", "term_proj_grav_z",
              "term_base_z_rel", "fell"]

    def write(run_id, tag, controller, k_exit):
        with open(os.path.join(d, run_id + ".csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for i in range(n):
                exited = i < k_exit
                level = i % 10
                w.writerow({
                    "run_id": run_id, "method": "ETRAP", "variant": controller,
                    "policy_tag": tag, "controller": controller,
                    "epsilon_or_alpha": 2.0 if controller != "none" else "",
                    "seed": 1, "episode_idx": i, "chunk_idx": 0, "env_idx": i,
                    "terrain_level": level,
                    "gate_height": 0.08 + 0.008 * level,
                    "n_steps": 1000, "exited": int(exited),
                    "exit_step": 300 if exited else -1,
                    "progress_x": 4.0 if exited else 1.2,
                    "progress_beyond_gate": 2.0 if exited else 0.0,
                    "parked": int(not exited and controller != "none"),
                    "mean_fwd_vel": 0.5 if exited else 0.08,
                    "vel_err": 0.1, "mean_abs_yaw_rate": 0.2,
                    "wall_contact_steps": 0 if exited else 4,
                    "contacted_wall": 0 if exited else 1,
                    "gate_foot_apex": 0.21 if exited else 0.0,
                    "interior_foot_apex": 0.09,
                    "min_l": 0.02 if exited else -0.01, "mean_l": 0.8,
                    "trigger_steps": 0, "activation_steps": 0,
                    "infeasible_steps": 0, "sum_dnorm_v": 0.0,
                    "sum_dnorm_omega": 0.0,
                    "term_timeout": 1, "term_fall_flag": 0,
                    "term_vel_violate": 0, "term_contact": 0,
                    "term_proj_grav_z": -0.98, "term_base_z_rel": 0.3,
                    "fell": 0,
                })

    write("pi_nom_body_a2_L0.35_w1", "pi_nom", "body", body_exits)
    write("pi_nom_body_noyaw_a2_L0.35_w1", "pi_nom", "body_noyaw", 0)
    write("pi_nom_unfiltered", "pi_nom", "none", 2)
    write("pi_trap_ours_unfiltered", "pi_trap_ours", "none", ours_exits)
    write("pi_trap_ours_body_a2_L0.35_w1", "pi_trap_ours", "body", 0)
    write("pi_trap_nom_unfiltered", "pi_trap_nom", "none", 10)
    return d


def test_analyzer(tmp):
    d = synth_rows(tmp, ours_exits=30)
    rows = [(rid, A.summarize(recs)) for rid, recs in A.load_rows(d)]
    verdict, detail = A.apply_gate(rows)
    check("body rows parked + ours exits -> separated",
          verdict == "separated", "got %r (%s)" % (verdict, detail))
    check("unfiltered rows sort before controller rows",
          rows[0][1]["controller"] == "none")

    d2 = synth_rows(tmp, ours_exits=0)
    rows2 = [(rid, A.summarize(recs)) for rid, recs in A.load_rows(d2)]
    v2, _ = A.apply_gate(rows2)
    check("ours failing to exit -> not_separated (reported plainly)",
          v2 == "not_separated", "got %r" % v2)

    d3 = synth_rows(tmp, ours_exits=30, body_exits=1)
    rows3 = [(rid, A.summarize(recs)) for rid, recs in A.load_rows(d3)]
    v3, det3 = A.apply_gate(rows3)
    check("a single body-arm exit -> geometry_violated, and it is named",
          v3 == "geometry_violated" and "pi_nom_body" in det3,
          "got %r (%s)" % (v3, det3))

    out = os.path.join(tmp, "R_trap")
    r = subprocess.run(
        [sys.executable,
         os.path.join(ROOT, "experiments", "etrap", "analyze_etrap.py"),
         "--dir", d, "--out", out],
        capture_output=True, text=True)
    check("analyze_etrap.py exits cleanly", r.returncode == 0,
          (r.stderr or "")[-400:])
    check("writes the markdown table", os.path.exists(out + ".md"))
    check("writes the json blob", os.path.exists(out + ".json"))
    if os.path.exists(out + ".json"):
        blob = json.load(open(out + ".json"))
        check("json carries the verdict", blob["verdict"] == "separated")
        check("json carries the sill-height curve",
              "pi_trap_ours_unfiltered" in blob["exit_by_gate_height"])
    if os.path.exists(out + ".md"):
        md = open(out + ".md").read()
        check("the mechanism column is in the table", "apex" in md)


# ------------------------------------------------------------ wiring text --

def test_wiring_text():
    """Cheap textual invariants on files that cannot be imported without
    isaacgym: the terrain slot, the config blocks, the eval proportions."""
    terrain_src = open(os.path.join(
        ROOT, "legged_gym", "utils", "terrain.py")).read()
    check("terrain.py has the slot-12 branch",
          "proportions[11]" in terrain_src
          and "_make_step_trap_terrain" in terrain_src)

    cfg_src = open(os.path.join(
        ROOT, "legged_gym", "envs", "go1", "go1_amp_config.py")).read()
    check("go1_amp carries the step_trap geometry block",
          "class step_trap" in cfg_src)
    check("go1_amp carries the lvalue block", "class lvalue" in cfg_src)
    check("go1_amp shipped proportions are untouched (pi_nom reproduces)",
          "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8]"
          in cfg_src)

    robot_src = open(os.path.join(
        ROOT, "legged_gym", "envs", "base", "legged_robot.py")).read()
    check("legged_robot builds box tensors", "box_obstacles" in robot_src
          and "has_box_obstacles" in robot_src)
    check("box envs use ONLY the volumetric l-value",
          "cyl_envs = self.has_obstacles & ~has_box" in robot_src)

    if HAVE_TORCH:
        from safeloco_eval import eval_common as EC
        props = EC.TERRAIN_PROPORTIONS.get("step_trap")
        check("eval_common knows the step_trap terrain",
              props is not None and len(props) == 13 and props[12] == 1.0
              and sum(props[:12]) == 0,
              "got %r" % (props,))
    else:
        print("  SKIP eval_common proportions -- torch not available")


def main():
    tmp = tempfile.mkdtemp(prefix="etrap_offline_")
    try:
        if HAVE_NUMPY:
            print("[geometry: the pen is closed in 2-D]")
            test_geometry()
        else:
            print("SKIP geometry tests -- numpy not available")

        if HAVE_NUMPY and HAVE_TORCH:
            print("\n[the volumetric l-value]")
            test_lvalue()
            print("\n[body-level filter closure]")
            test_filter_closure()
        else:
            print("SKIP l-value / filter tests -- torch not available")

        print("\n[analyzer]")
        test_analyzer(tmp)

        print("\n[wiring]")
        test_wiring_text()
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 70)
    if FAILED:
        print("FAILED (%d): %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
