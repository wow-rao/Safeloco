# Persistent Ego-Centric Occupancy Grid

## Implementation Plan

---

## 1. Grid specification

```
Grid frame: robot base frame, aligned with heading
Forward:    +x, 2.0m
Backward:   -x, 1.0m
Left:       +y, 0.75m
Right:      -y, 0.75m
Resolution: 0.05m per cell
Dimensions: 60 cells (x) × 30 cells (y)
Total:      1800 cells
Values:     float in [0, 1], 1.0 = occupied
```

---

## 2. Config

### Files: `legged_gym/envs/base/legged_robot_config.py`

Add inside `LeggedRobotCfg`:

```python
class occupancy_grid:
    enabled = True
    resolution = 0.05          # meters per cell
    x_forward = 2.0            # meters ahead of robot
    x_backward = 1.0           # meters behind robot
    y_left = 0.75              # meters to the left
    y_right = 0.75             # meters to the right
    height_min = 0.05          # ignore ground hits below this (meters)
    height_max = 0.6           # ignore points above this (meters)
    depth_max = 1.5            # ignore depth readings beyond this (meters)
    grid_H = 60                # cells along x: (x_forward + x_backward) / resolution
    grid_W = 30                # cells along y: (y_left + y_right) / resolution
    latent_dim = 32            # output dim of grid CNN encoder
```

---

## 3. Environment: grid lifecycle

### Files: `legged_gym/envs/base/legged_robot.py`

### 3.1 Initialization

In `__init__`, after `self.init_done = True`:

```python
if self.cfg.occupancy_grid.enabled:
    self._init_occupancy_grid()
```

```python
def _init_occupancy_grid(self):
    cfg = self.cfg.occupancy_grid
    self.grid = torch.zeros(
        self.num_envs, cfg.grid_H, cfg.grid_W,
        device=self.device, dtype=torch.float32
    )
    # Store previous base pose for ego-motion computation
    self.prev_base_pos = self.root_states[:, :3].clone()
    self.prev_base_yaw = self._get_base_yaw()

    # Precompute camera ray directions (fixed, computed once)
    # These go from pixel (u, v) to a unit ray in camera frame
    H_img, W_img = self.cfg.depth.resized  # (87, 58)
    hfov = self.cfg.depth.horizontal_fov * (torch.pi / 180.0)
    vfov = hfov * (H_img / W_img)  # approximate

    # Pixel grid
    u = torch.arange(W_img, device=self.device, dtype=torch.float32)
    v = torch.arange(H_img, device=self.device, dtype=torch.float32)
    uu, vv = torch.meshgrid(u, v, indexing='xy')  # (H_img, W_img)

    # Normalized image coordinates centered at principal point
    fx = W_img / (2.0 * torch.tan(hfov / 2.0))
    fy = H_img / (2.0 * torch.tan(vfov / 2.0))
    cx, cy = W_img / 2.0, H_img / 2.0

    # Ray directions in camera frame (z forward, x right, y down)
    ray_x = (uu - cx) / fx
    ray_y = (vv - cy) / fy
    ray_z = torch.ones_like(ray_x)
    # Stack and normalize
    self.cam_rays = torch.stack([ray_x, ray_y, ray_z], dim=-1)  # (H, W, 3)
    self.cam_rays = self.cam_rays / (
        self.cam_rays.norm(dim=-1, keepdim=True) + 1e-8
    )
    self.cam_rays = self.cam_rays.reshape(-1, 3)  # (H*W, 3)

    # Camera mount transform (camera frame → base frame)
    # Camera position in base frame
    self.cam_pos_base = torch.tensor(
        self.cfg.depth.position, device=self.device, dtype=torch.float32
    )
    # Camera pitch (average of angle range)
    cam_pitch = torch.tensor(
        sum(self.cfg.depth.angle) / 2.0 * (torch.pi / 180.0),
        device=self.device
    )
    # Rotation matrix: camera frame → base frame
    # Camera convention: z forward, x right, y down
    # Base convention: x forward, y left, z up
    # This depends on your specific camera mounting — adjust accordingly
    cp, sp = torch.cos(cam_pitch), torch.sin(cam_pitch)
    self.cam_R_base = torch.tensor([
        [0,  0,  1],   # cam z (forward) → base x
        [-1, 0,  0],   # cam x (right) → base -y
        [0,  1,  0],   # cam y (down) → base z (adjusted by pitch)
    ], device=self.device, dtype=torch.float32)
    # Apply pitch rotation around base y-axis
    pitch_R = torch.tensor([
        [cp,  0, sp],
        [0,   1, 0],
        [-sp, 0, cp],
    ], device=self.device, dtype=torch.float32)
    self.cam_R_base = pitch_R @ self.cam_R_base
```

**Important note on the camera-to-base transform**: the rotation matrix above is a template. You need to verify this against your actual WARP camera sensor convention. The key is: given a point in camera frame `p_cam`, the point in base frame is `cam_R_base @ p_cam + cam_pos_base`.

### 3.2 Yaw extraction helper

```python
def _get_base_yaw(self):
    """Extract yaw angle from base quaternion. Returns (num_envs,)."""
    quat = self.root_states[:, 3:7]  # (E, 4) — x, y, z, w
    # Yaw from quaternion: atan2(2(wz + xy), 1 - 2(y² + z²))
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw
```

### 3.3 Grid update — called inside `update_depth_buffer`

After the depth image is obtained, call `self._update_occupancy_grid(depth_image)`.
Modify `update_depth_buffer` to:

```python
def update_depth_buffer(self):
    if not self.cfg.depth.use_camera:
        return

    depth_active_mask = torch.ones(
        self.num_envs, dtype=torch.uint8, device=self.device
    )
    depth_image = self.warp_camera_sensor.update(
        depth_active_mask, self.base_pos, self.base_quat
    )  # (num_envs, 1, H, W)

    init_flag = (self.episode_length_buf <= 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    init_stack = depth_image.expand(-1, self.cfg.depth.buffer_len, -1, -1)
    self.depth_buffer = torch.where(
        init_flag, init_stack,
        torch.cat([self.depth_buffer[:, 1:], depth_image], dim=1)
    )

    # --- NEW: update occupancy grid ---
    if self.cfg.occupancy_grid.enabled:
        self._update_occupancy_grid(depth_image[:, 0])  # (E, H, W)
```

### 3.4 Core grid update method

This is the main logic. Three phases: shift old grid by ego-motion, clear cells within current FOV, write new detections.

```python
def _update_occupancy_grid(self, depth_image):
    """
    Update the persistent ego-centric occupancy grid.

    Args:
        depth_image: (num_envs, H_img, W_img) raw depth values
    """
    cfg = self.cfg.occupancy_grid
    E = self.num_envs

    # ───────────────────────────────────────
    # Phase 1: Shift old grid by ego-motion
    # ───────────────────────────────────────
    cur_pos = self.root_states[:, :3]  # (E, 3) global
    cur_yaw = self._get_base_yaw()     # (E,)

    # Compute displacement in previous base frame
    delta_global = cur_pos[:, :2] - self.prev_base_pos[:, :2]  # (E, 2)
    delta_yaw = cur_yaw - self.prev_base_yaw                    # (E,)

    # Rotate global delta into previous base frame
    cos_prev = torch.cos(self.prev_base_yaw)
    sin_prev = torch.sin(self.prev_base_yaw)
    delta_local_x = cos_prev * delta_global[:, 0] + sin_prev * delta_global[:, 1]
    delta_local_y = -sin_prev * delta_global[:, 0] + cos_prev * delta_global[:, 1]

    # Convert displacement from meters to grid cells
    dx_cells = delta_local_x / cfg.resolution  # (E,)
    dy_cells = delta_local_y / cfg.resolution  # (E,)

    # Build affine transform for grid_sample
    # grid_sample expects a flow field mapping output pixels to input pixels
    # We want to shift the old grid content by (-dx, -dy) and rotate by (-delta_yaw)
    cos_dy = torch.cos(-delta_yaw)
    sin_dy = torch.sin(-delta_yaw)

    # Normalize shifts to [-1, 1] range for grid_sample
    # grid_sample uses coordinates in [-1, 1] where -1 is left/top, +1 is right/bottom
    norm_dx = -2.0 * dx_cells / cfg.grid_H
    norm_dy = -2.0 * dy_cells / cfg.grid_W

    # Build 2x3 affine matrix per environment
    theta = torch.zeros(E, 2, 3, device=self.device)
    theta[:, 0, 0] = cos_dy
    theta[:, 0, 1] = -sin_dy
    theta[:, 0, 2] = norm_dx
    theta[:, 1, 0] = sin_dy
    theta[:, 1, 1] = cos_dy
    theta[:, 1, 2] = norm_dy

    # Apply affine transform to shift the old grid
    # grid_sample needs (N, C, H, W) input
    grid_input = self.grid.unsqueeze(1)  # (E, 1, grid_H, grid_W)
    flow = torch.nn.functional.affine_grid(
        theta, grid_input.size(), align_corners=False
    )
    self.grid = torch.nn.functional.grid_sample(
        grid_input, flow, mode='nearest',
        padding_mode='zeros', align_corners=False
    ).squeeze(1)  # (E, grid_H, grid_W)

    # Update stored pose for next iteration
    self.prev_base_pos = cur_pos.clone()
    self.prev_base_yaw = cur_yaw.clone()

    # ───────────────────────────────────────
    # Phase 2: Back-project depth to base frame
    # ───────────────────────────────────────
    H_img, W_img = self.cfg.depth.resized
    depth_flat = depth_image.reshape(E, -1)  # (E, H*W)

    # Scale rays by depth to get 3D points in camera frame
    # cam_rays: (H*W, 3), depth_flat: (E, H*W)
    points_cam = self.cam_rays.unsqueeze(0) * depth_flat.unsqueeze(-1)  # (E, H*W, 3)

    # Transform to base frame
    # cam_R_base: (3, 3), cam_pos_base: (3,)
    points_base = torch.einsum(
        'ij,enj->eni', self.cam_R_base, points_cam
    ) + self.cam_pos_base  # (E, H*W, 3)

    # ───────────────────────────────────────
    # Phase 3: Filter and write to grid
    # ───────────────────────────────────────
    px = points_base[:, :, 0]  # (E, H*W) forward
    py = points_base[:, :, 1]  # (E, H*W) left
    pz = points_base[:, :, 2]  # (E, H*W) up

    # Valid point mask
    valid = (
        (depth_flat > 0.01) &                     # not invalid depth
        (depth_flat < cfg.depth_max) &             # not too far
        (pz > cfg.height_min) &                    # above ground
        (pz < cfg.height_max) &                    # below robot top
        (px > -cfg.x_backward) &                   # within grid bounds
        (px < cfg.x_forward) &
        (py > -cfg.y_right) &
        (py < cfg.y_left)
    )  # (E, H*W)

    # Convert to grid indices
    # Grid origin: x=-x_backward is row 0, y=-y_right is col 0
    grid_row = ((px + cfg.x_backward) / cfg.resolution).long()  # (E, H*W)
    grid_col = ((py + cfg.y_right) / cfg.resolution).long()     # (E, H*W)

    # Clamp to grid bounds (safety)
    grid_row = grid_row.clamp(0, cfg.grid_H - 1)
    grid_col = grid_col.clamp(0, cfg.grid_W - 1)

    # ───────────────────────────────────────
    # Phase 3a: Clear cells within current FOV
    # ───────────────────────────────────────
    # Compute the FOV footprint on the grid: all cells that the camera
    # can see (even if no obstacle there). These should be cleared.
    # We approximate this by projecting the camera frustum onto the
    # ground plane and clearing all grid cells within it.
    #
    # Simple approximation: clear cells where we have any valid depth
    # reading OR where depth is at max range (meaning clear space).
    # More precisely: for each column of the depth image, the camera
    # sees a specific angular slice. Cells in that slice up to the
    # measured depth should be cleared.
    #
    # Pragmatic approach: clear all cells that any ray passes through.
    # This is expensive to do per-ray. Instead, use a coarser approach:

    # For each environment, build a mask of cells that fall within the
    # camera frustum by checking angular bounds.
    # Grid cell centers in base frame:
    row_coords = torch.arange(
        cfg.grid_H, device=self.device, dtype=torch.float32
    ) * cfg.resolution - cfg.x_backward + cfg.resolution / 2  # (grid_H,)
    col_coords = torch.arange(
        cfg.grid_W, device=self.device, dtype=torch.float32
    ) * cfg.resolution - cfg.y_right + cfg.resolution / 2     # (grid_W,)

    cell_x, cell_y = torch.meshgrid(row_coords, col_coords, indexing='ij')
    # (grid_H, grid_W)

    # Angle of each cell relative to forward (+x)
    cell_angle = torch.atan2(cell_y, cell_x)  # (grid_H, grid_W)
    cell_dist = torch.sqrt(cell_x**2 + cell_y**2)

    half_fov = (self.cfg.depth.horizontal_fov / 2.0) * (torch.pi / 180.0)
    fov_mask = (
        (cell_angle.abs() < half_fov) &
        (cell_x > 0) &                            # only forward
        (cell_dist < cfg.depth_max)
    )  # (grid_H, grid_W)

    # Clear FOV cells across all environments
    self.grid[:, fov_mask] = 0.0

    # ───────────────────────────────────────
    # Phase 3b: Write occupied cells
    # ───────────────────────────────────────
    # Scatter detections into the grid
    # Use a batch index for scatter
    batch_idx = torch.arange(
        E, device=self.device
    ).unsqueeze(1).expand_as(grid_row)  # (E, H*W)

    # Only write valid points
    b_valid = batch_idx[valid]
    r_valid = grid_row[valid]
    c_valid = grid_col[valid]

    self.grid[b_valid, r_valid, c_valid] = 1.0
```

### 3.5 Reset

In `reset_idx`, zero the grid for reset environments:

```python
def reset_idx(self, env_ids):
    # ... existing reset code ...

    if self.cfg.occupancy_grid.enabled and len(env_ids) > 0:
        self.grid[env_ids] = 0.0
        self.prev_base_pos[env_ids] = self.root_states[env_ids, :3]
        self.prev_base_yaw[env_ids] = self._get_base_yaw()[env_ids]
```

### 3.6 Pass grid in extras

In `step`, alongside depth and other extras:

```python
if self.cfg.occupancy_grid.enabled:
    self.extras["occupancy_grid"] = self.grid.clone()  # (E, grid_H, grid_W)
else:
    self.extras["occupancy_grid"] = None
```

---

## 4. Actor-Critic: grid encoder

### Files: `rsl_rl/modules/actor_critic_wmp.py`

### 4.1 Grid CNN encoder

Add a small CNN that takes the `(grid_H, grid_W)` grid and outputs a
`grid_latent_dim`-dimensional vector. Add to `__init__`:

```python
# ── Grid CNN encoder (shared between actor and critic) ──
if grid_enabled:
    self.grid_enabled = True
    self.grid_H = grid_H        # 60
    self.grid_W = grid_W        # 30
    grid_latent_dim = grid_latent_dim  # 32

    self.grid_encoder_actor = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # (16, 30, 15)
        nn.ELU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 15, 8)
        nn.ELU(),
        nn.Flatten(),                                            # 32*15*8 = 3840
        nn.Linear(32 * 15 * 8, 128),
        nn.ELU(),
        nn.Linear(128, grid_latent_dim),
    )

    self.grid_encoder_critic = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
        nn.ELU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ELU(),
        nn.Flatten(),
        nn.Linear(32 * 15 * 8, 128),
        nn.ELU(),
        nn.Linear(128, grid_latent_dim),
    )
else:
    self.grid_enabled = False
    grid_latent_dim = 0
```

**Important**: Verify the intermediate dimensions. With input `(1, 60, 30)`:
- Conv1: stride 2 → `(16, 30, 15)`
- Conv2: stride 2 → `(32, 15, 8)`
- Flatten: `32 * 15 * 8 = 3840`

If your grid dimensions differ, adjust accordingly.

### 4.2 Modify input dimensions

The actor and critic input dimensions grow by `grid_latent_dim`:

```python
mlp_input_dim_a = latent_dim + 3 + wm_latent_dim + grid_latent_dim
mlp_input_dim_c = num_critic_obs + wm_latent_dim + grid_latent_dim
```

### 4.3 Modify `act`

```python
def act(self, observations, history, wm_feature, grid=None, **kwargs):
    latent_vector = self.history_encoder(history)
    command = observations[:, self.privileged_dim + 6:self.privileged_dim + 9]
    wm_latent_vector = self.wm_feature_encoder(wm_feature)

    if self.grid_enabled and grid is not None:
        grid_input = grid.unsqueeze(1)  # (B, 1, H, W)
        grid_latent = self.grid_encoder_actor(grid_input)
        concat_observations = torch.cat(
            (latent_vector, command, wm_latent_vector, grid_latent), dim=-1
        )
    else:
        concat_observations = torch.cat(
            (latent_vector, command, wm_latent_vector), dim=-1
        )

    self.update_distribution(concat_observations)
    return self.distribution.sample()
```

### 4.4 Modify `act_inference`

Same pattern — accept `grid`, encode, concatenate.

### 4.5 Modify `evaluate`

```python
def evaluate(self, critic_observations, wm_feature, grid=None, **kwargs):
    wm_latent_vector = self.critic_wm_feature_encoder(wm_feature)

    if self.grid_enabled and grid is not None:
        grid_input = grid.unsqueeze(1)
        grid_latent = self.grid_encoder_critic(grid_input)
        concat_observations = torch.cat(
            (critic_observations, wm_latent_vector, grid_latent), dim=-1
        )
    else:
        concat_observations = torch.cat(
            (critic_observations, wm_latent_vector), dim=-1
        )

    value = self.critic(concat_observations)
    return value
```

### 4.6 Modify `get_export_actor`

Add `grid_encoder_actor` to the exported model so deployment has access.

---

## 5. Rollout storage

### Files: `rsl_rl/storage/rollout_storage.py`

### 5.1 Add grid buffer to `__init__`

```python
def __init__(self, ..., grid_shape=(60, 30)):
    # ... existing code ...

    self.grids = torch.zeros(
        num_transitions_per_env, num_envs,
        *grid_shape, device=self.device
    )
```

### 5.2 Add to `Transition`

```python
self.grid = torch.empty(0)
```

### 5.3 Add to `add_transitions`

```python
self.grids[self.step].copy_(transition.grid)
```

### 5.4 Modify `mini_batch_generator`

```python
grids = self.grids.flatten(0, 1)  # (B, grid_H, grid_W)

# Inside loop:
grid_batch = grids[batch_idx]

# Add to yield
yield obs_batch, ..., grid_batch
```

---

## 6. PPO

### Files: `rsl_rl/algorithms/ppo.py`

### 6.1 Modify `act`

```python
def act(self, obs, critic_obs, history, wm_feature, grid=None):
    self.transition.grid = grid.detach() if grid is not None else None
    # ... pass grid through to actor_critic.act ...
    self.transition.actions = self.actor_critic.act(
        obs.detach(), history, wm_feature, grid=grid
    ).detach()
    self.transition.values = self.actor_critic.evaluate(
        critic_obs.detach(), wm_feature, grid=grid
    ).detach()
    # ... rest unchanged ...
```

### 6.2 Modify `update`

Unpack `grid_batch` from the generator and pass to `act` and `evaluate`:

```python
self.actor_critic.act(
    aug_obs_batch, history_batch, wm_feature_batch, grid=grid_batch, ...
)
value_batch = self.actor_critic.evaluate(
    aug_critic_obs_batch, wm_feature_batch, grid=grid_batch, ...
)
```

### 6.3 Modify `compute_returns`

Pass grid to the final value evaluation:

```python
def compute_returns(self, last_critic_obs, wm_feature, grid=None):
    last_values = self.actor_critic.evaluate(
        last_critic_obs.detach(), wm_feature, grid=grid
    ).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)
```

---

## 7. Runner

### Files: `rsl_rl/runners/wmp_runner.py`

### 7.1 Construction

Pass grid config to actor-critic:

```python
actor_critic = ActorCriticWMP(
    ...,
    grid_enabled=self.env.cfg.occupancy_grid.enabled,
    grid_H=self.env.cfg.occupancy_grid.grid_H,
    grid_W=self.env.cfg.occupancy_grid.grid_W,
    grid_latent_dim=self.env.cfg.occupancy_grid.latent_dim,
)
```

Pass grid shape to storage:

```python
self.alg.init_storage(
    ...,
    grid_shape=(
        self.env.cfg.occupancy_grid.grid_H,
        self.env.cfg.occupancy_grid.grid_W,
    ),
)
```

### 7.2 Rollout loop

Initialize grid tensor before the loop:

```python
occ_grid = torch.zeros(
    self.env.num_envs,
    self.env.cfg.occupancy_grid.grid_H,
    self.env.cfg.occupancy_grid.grid_W,
    device=self.device,
)
```

Inside the rollout loop, after `env.step`:

```python
# Get updated grid from environment
new_grid = infos.get("occupancy_grid", None)
if new_grid is not None:
    occ_grid = new_grid.to(self.device)

# Pass to act
actions, critic_values = self.alg.act(
    obs, critic_obs, history, wm_feature, grid=occ_grid
)
```

### 7.3 After rollout

```python
self.alg.compute_returns(critic_obs, wm_feature, grid=occ_grid)
```

---

## 8. Dimensions summary

```
Existing actor input:  latent_dim(32) + cmd(3) + wm_latent(16) = 51
Grid addition:         grid_latent_dim(32)
New actor input:       51 + 32 = 83

Existing critic input: num_critic_obs(235) + wm_latent(16) = 251
Grid addition:         grid_latent_dim(32)
New critic input:      251 + 32 = 283
```

---

## 9. File change summary

| File | Changes |
|---|---|
| `legged_robot_config.py` | Add `occupancy_grid` config class |
| `legged_robot.py` | `_init_occupancy_grid`, `_update_occupancy_grid`, `_get_base_yaw`, modify `update_depth_buffer`, `reset_idx`, `step` |
| `actor_critic_wmp.py` | Add `grid_encoder_actor`, `grid_encoder_critic`, modify `act`, `act_inference`, `evaluate`, `get_export_actor`, input dims |
| `rollout_storage.py` | Add `grids` buffer, modify `Transition`, `add_transitions`, `mini_batch_generator` |
| `ppo.py` | Pass `grid` through `act`, `update`, `compute_returns` |
| `wmp_runner.py` | Pass grid config to actor-critic, grid shape to storage, wire grid through rollout loop |

---

## 10. Debugging checklist

1. **Verify camera-to-base transform**: Render the grid as a top-down image.
   Place a single obstacle in front of the robot. The occupied cells should
   appear at the correct relative position. If they're mirrored or rotated,
   the `cam_R_base` matrix is wrong.

2. **Verify ego-motion shift**: Have the robot walk forward with no obstacles.
   Place a single detection manually. After 10 steps, the detection should
   have shifted backward in the grid by the distance the robot walked.
   If it shifts sideways or disappears, the affine transform is wrong.

3. **Verify FOV clearing**: Walk toward an obstacle, then turn 90°. The
   obstacle should persist in the grid (it's outside FOV). Now turn back
   to face it. The grid should re-detect it at the same position. If the
   obstacle disappears when you turn, the FOV mask is too wide.

4. **Verify CNN dimensions**: Before wiring into training, run a standalone
   test: `grid_encoder_actor(torch.randn(4, 1, 60, 30))` should output
   `(4, 32)`. If it crashes, the intermediate conv dimensions are wrong.

5. **Verify grid reset**: On episode reset, the grid should be all zeros.
   If stale detections from the previous episode leak through, the
   `reset_idx` modification is missing.

---

## 11. Performance notes

- The grid update runs at 10 Hz (same as depth camera), not 50 Hz.
  The ego-motion shift and scatter are cheap — the `grid_sample` call
  is the most expensive part, but it's operating on a tiny 60×30 grid.

- The CNN encoder runs at 50 Hz (every actor forward pass) on a 60×30
  input. Two conv layers on a 1800-element input is negligible compared
  to the depth CNN in the world model.

- Memory: 1800 floats × num_envs × num_transitions_per_env for the
  rollout storage. With 4096 envs and 24 transitions, that's
  4096 × 24 × 1800 × 4 bytes ≈ 700 MB. If this is too large, quantize
  the grid to uint8 (0 or 255) and convert to float only when feeding
  to the CNN.
