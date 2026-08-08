"""Geometry for the *step-trap* terrain (terrain slot 12) -- pure numpy.

The experiment this terrain exists for
--------------------------------------
A command-space (body-level) safety controller reasons about obstacles as
regions of the *plane*: every obstacle is a 2-D footprint the body must not
enter, whatever its height.  A joint-space method with a 3-D safety value can
distinguish a 0.4 m wall from a 0.1 m sill, because the failure set it
penalises is the obstacle's occupied *volume* -- the space above a low sill is
safe, and a swing foot may pass through it.

The step trap is the geometry that turns that difference into a number.  The
robot spawns inside a rectangular pen:

           tall (0.4 m)
        +----------------+
        |                |
  tall  |    x  spawn    |  tall          ----> +x  (the commanded direction)
        |                |
        +----------------+
             GATE: a low sill, 0.08-0.16 m by difficulty

Three walls are 0.4 m -- un-steppable, and taller than the base, so there is
no way over them for anything.  The fourth ("the gate") is a sill low enough
to step across with a deliberately high swing, and too long to walk around:
its footprint meets the tall side walls at both ends.

Projected to the plane, the pen is a *closed* ring of obstacle: the 2-D free
space inside is disconnected from the outside.  Consequently a body-level
controller -- braking-only or with full steering authority; this is a property
of the world, not of any particular filter -- has no safe path out, and its
best safe answer is to park.  The 3-D l-value (see
`legged_gym/utils/box_sdf.py`) keeps the volume above the sill in the safe
set, so a policy trained under the gradient-projection method can lift its
legs over the gate and keep the 0.6 m/s task alive.

Numbers the terrain is designed to produce: exit rate and distance beyond the
gate (0 for every body-level controller, by construction), and swing-foot
apex over the sill (the mechanism evidence: "lifts its legs higher than
usual").

Two representations of the same walls, published together
---------------------------------------------------------
* ``boxes`` -- axis-aligned volumes ``(cx, cy, cz, hx, hy, hz)`` (centre +
  half-extents), the exact solids rasterised into the heightfield.  This is
  what the custom l-value consumes; l == signed distance to these volumes,
  so "above the sill" is safe by construction.
* ``positions`` / ``radius`` -- the wall centrelines sampled as points, the
  format every existing 2-D consumer (`_build_obstacle_tensor`, the E4/E4-Y
  command filters, `link_obstacle_collision`) already reads.  The sampling is
  dense enough that the union of the point footprints covers the wall
  footprints with no gap (asserted in tests), so to a planar controller the
  pen is exactly the closed ring it physically is.

Everything here is numpy-only and free of isaacgym imports on purpose: the
offline tests (tests/test_etrap_offline.py) load this file directly and prove
the pen is closed in 2-D without a simulator.

Coordinates: local, origin at the terrain-cell centre (the same frame the
corridor obstacle info uses); the caller adds the env origin.
"""

import numpy as np

# Defaults, overridable from `cfg.terrain.step_trap` (go1_amp_config.py).
# The pen interior must contain the spawn randomisation (+-1 m around the
# origin, applied twice: actor creation and reset) plus body clearance.
INNER_HALF = 1.7          # pen interior half-extent [m]
WALL_THICKNESS = 0.2      # every wall, along its normal [m]
TALL_WALL_HEIGHT = 0.4    # un-steppable; also >= base height, so no body pass
GATE_HEIGHT_MIN = 0.08    # difficulty 0 -- near the Go1's natural swing apex
GATE_HEIGHT_MAX = 0.16    # difficulty 1 -- roughly twice the natural apex
RING_SPACING = 0.24       # centreline sample spacing for the 2-D point ring
RING_RADIUS = 0.16        # >= sqrt((spacing/2)^2 + (thickness/2)^2): no gap


def gate_height(difficulty, gate_min=GATE_HEIGHT_MIN, gate_max=GATE_HEIGHT_MAX,
                vertical_scale=None):
    """Sill height at a difficulty in [0, 1], optionally snapped to the
    heightfield's vertical quantum so the analytic boxes match the mesh
    exactly."""
    h = gate_min + float(np.clip(difficulty, 0.0, 1.0)) * (gate_max - gate_min)
    if vertical_scale:
        h = round(h / vertical_scale) * vertical_scale
    return h


def _wall_boxes(inner_half, thickness, tall_height, gate_h):
    """The four walls as a *partition* (no overlaps).

    The corners belong to the side walls.  This matters: if the gate box
    reached into a corner where the mesh is tall, the l-value would call the
    space above the sill safe where the physical wall is 0.4 m -- the safe set
    must never exceed the mesh.  With a partition, boxes == mesh exactly.
    """
    a = inner_half
    t = thickness
    o = a + t                       # outer half-extent
    boxes = [
        # gate (front, +x): the low sill, spanning only the interior width
        (a + t / 2.0, 0.0, gate_h / 2.0, t / 2.0, a, gate_h / 2.0),
        # back (-x): tall, interior width
        (-(a + t / 2.0), 0.0, tall_height / 2.0, t / 2.0, a, tall_height / 2.0),
        # left (+y) and right (-y): tall, full length -- they own the corners
        (0.0, a + t / 2.0, tall_height / 2.0, o, t / 2.0, tall_height / 2.0),
        (0.0, -(a + t / 2.0), tall_height / 2.0, o, t / 2.0, tall_height / 2.0),
    ]
    return np.array(boxes, dtype=np.float32)


def _ring_points(inner_half, thickness, spacing):
    """Sample the closed rectangular centreline of the pen as 2-D points.

    One loop through the wall midlines, corners included, so the planar
    representation is the closed ring the pen physically is.
    """
    c = inner_half + thickness / 2.0          # centreline half-extent
    corners = np.array([
        [c, -c], [c, c], [-c, c], [-c, -c],
    ], dtype=np.float64)
    pts = []
    for k in range(4):
        p0, p1 = corners[k], corners[(k + 1) % 4]
        seg = np.linalg.norm(p1 - p0)
        n = max(int(np.ceil(seg / spacing)), 1)
        for s in range(n):                     # p1 itself starts the next edge
            q = p0 + (p1 - p0) * (s / float(n))
            pts.append([q[0], q[1], 0.0])
    return np.array(pts, dtype=np.float32)


def build_step_trap(difficulty,
                    inner_half=INNER_HALF,
                    wall_thickness=WALL_THICKNESS,
                    tall_wall_height=TALL_WALL_HEIGHT,
                    gate_height_min=GATE_HEIGHT_MIN,
                    gate_height_max=GATE_HEIGHT_MAX,
                    ring_spacing=RING_SPACING,
                    ring_radius=RING_RADIUS,
                    vertical_scale=None,
                    cell_length=None,
                    include_next_cell_walls=True):
    """Build one cell's step trap. Returns the obstacle-info dict.

    ``cell_length`` (terrain rows advance along +x by this much): when given
    and ``include_next_cell_walls``, the *next* cell's back and side walls are
    appended to ``boxes`` -- they bound this cell's runway, and a robot that
    exits the pen can physically reach them well inside one episode, so the
    l-value has to know about them.  Their mesh belongs to the next cell; only
    the volumes are borrowed.  The 2-D ring stays local: the pen is closed
    with or without them.
    """
    gate_h = gate_height(difficulty, gate_height_min, gate_height_max,
                         vertical_scale)
    boxes = _wall_boxes(inner_half, wall_thickness, tall_wall_height, gate_h)
    own_boxes = boxes.copy()

    if include_next_cell_walls and cell_length:
        nxt = _wall_boxes(inner_half, wall_thickness, tall_wall_height, gate_h)
        shifted = nxt[1:].copy()               # back + both sides, not the gate
        shifted[:, 0] += float(cell_length)
        boxes = np.concatenate([boxes, shifted], axis=0)

    positions = _ring_points(inner_half, wall_thickness, ring_spacing)
    return {
        "positions": positions,               # 2-D body-controller view
        "radius": float(ring_radius),
        "boxes": boxes.astype(np.float32),    # 3-D l-value view
        "kind": "step_trap",
        "gate": {
            "height": float(gate_h),
            "x_inner": float(inner_half),
            "x_outer": float(inner_half + wall_thickness),
            "y_half": float(inner_half),
        },
        # own_boxes = exactly what this cell rasterises (no borrowed volumes)
        "own_boxes": own_boxes.astype(np.float32),
        "tall_wall_height": float(tall_wall_height),
    }


def rasterize_boxes(height_field_raw, boxes, horizontal_scale, vertical_scale,
                    length_m, width_m):
    """Stamp axis-aligned boxes into a cell-local heightfield, in place.

    ``height_field_raw`` is the SubTerrain array: axis 0 is x (row r maps to
    local x = r * horizontal_scale - length_m / 2), axis 1 is y likewise.
    Heights combine by max, so overlapping stamps keep the taller solid.
    Returns the array.
    """
    n_rows, n_cols = height_field_raw.shape
    for (cx, cy, cz, hx, hy, hz) in np.asarray(boxes, dtype=np.float64):
        x0 = (cx - hx + length_m / 2.0) / horizontal_scale
        x1 = (cx + hx + length_m / 2.0) / horizontal_scale
        y0 = (cy - hy + width_m / 2.0) / horizontal_scale
        y1 = (cy + hy + width_m / 2.0) / horizontal_scale
        r0 = max(int(round(x0)), 0)
        r1 = min(int(round(x1)), n_rows)
        c0 = max(int(round(y0)), 0)
        c1 = min(int(round(y1)), n_cols)
        if r1 <= r0 or c1 <= c0:
            continue                          # entirely outside this cell
        top = int(round((cz + hz) / vertical_scale))
        region = height_field_raw[r0:r1, c0:c1]
        np.maximum(region, top, out=region)
    return height_field_raw


# ------------------------------------------------------------------ checks --

def ring_is_closed(positions, radius, extra_margin, inner_half,
                   wall_thickness, n_rays=720):
    """True iff no straight escape path exists for a body of ``extra_margin``.

    Casts rays from the pen centre outward and asks whether each one passes
    within ``radius + extra_margin`` of some ring point before clearing the
    wall band.  This is the geometric fact the whole experiment rests on --
    the 2-D free space is disconnected -- so it is checked as code, offline,
    rather than asserted in prose.
    """
    pts = np.asarray(positions, dtype=np.float64)[:, :2]
    reach = radius + extra_margin
    far = inner_half + wall_thickness + reach + 0.5
    for k in range(n_rays):
        th = 2.0 * np.pi * k / n_rays
        d = np.array([np.cos(th), np.sin(th)])
        # distance from each ring point to the ray {t*d : t >= 0}
        t = pts @ d                            # projection parameter
        t = np.clip(t, 0.0, far)
        closest = np.linalg.norm(pts - t[:, None] * d[None, :], axis=1)
        if closest.min() > reach:
            return False
    return True
