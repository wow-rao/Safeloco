# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticWMP
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.algorithms.cone_constraint import (
    extract_flat_grads, write_flat_grads, damped_null_space_project,
    DEFAULT_D_SAFE, DEFAULT_D_DANGER
)


class PPO:
    actor_critic: ActorCriticWMP

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 vel_predict_coef=1.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 min_std=None,
                 safety_coef=0.5,
                 safety_value_loss_coef=1.0,
                 d_safe=DEFAULT_D_SAFE,
                 d_danger=DEFAULT_D_DANGER,
                 safety_return_alpha=0.7,
                 train_safety_critic_when_off=False,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.vel_predict_coef = vel_predict_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Safety parameters
        self.safety_coef = safety_coef
        self.safety_value_loss_coef = safety_value_loss_coef
        self.d_safe = d_safe
        self.d_danger = d_danger
        self.safety_return_alpha = safety_return_alpha
        # With safety_coef == 0 the safety branch is skipped entirely; this flag
        # still fits the reachability critic (a parameter-disjoint head) on the
        # rollouts so V_safe can be used diagnostically without influencing the
        # policy update.
        self.train_safety_critic_when_off = train_safety_critic_when_off

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, history_dim, wm_feature_dim, grid_shape=None):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, history_dim=history_dim,
                                      wm_feature_dim = wm_feature_dim, grid_shape=grid_shape, device = self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, history, wm_feature, safety_value=None, grid=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.history = history
        self.transition.wm_feature = wm_feature.detach()
        self.transition.grid = grid.detach() if grid is not None else None
        aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()
        self.transition.actions = self.actor_critic.act(aug_obs, history, wm_feature, grid=grid).detach()
        self.transition.values = self.actor_critic.evaluate(aug_critic_obs, wm_feature, grid=grid).detach()
        self.transition.safety_critic_value = self.actor_critic.evaluate_safety(
            aug_critic_obs, wm_feature, grid=grid
        ).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        if safety_value is not None:
            self.transition.safety_value = safety_value.detach()
        else:
            self.transition.safety_value = torch.zeros(obs.shape[0], device=self.device)
        return self.transition.actions, self.transition.values

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        not_done_idxs = (dones == False).nonzero().squeeze()

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, wm_feature, grid=None):
        aug_last_critic_obs = last_critic_obs.detach()
        last_values = self.actor_critic.evaluate(aug_last_critic_obs, wm_feature, grid=grid).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        last_safety_values = self.actor_critic.evaluate_safety(
            aug_last_critic_obs, wm_feature, grid=grid
        ).detach()
        self.storage.compute_safety_returns(last_safety_values, alpha=self.safety_return_alpha)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_safety_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for sample in generator:
            (obs_batch, critic_obs_batch, actions_batch, history_batch,
             wm_feature_batch, target_values_batch, advantages_batch,
             returns_batch, old_actions_log_prob_batch, old_mu_batch,
             old_sigma_batch, hid_states_batch, masks_batch,
             safety_advantages_batch, safety_returns_batch, grid_batch) = sample

            aug_obs_batch = obs_batch.detach()
            aug_critic_obs_batch = critic_obs_batch.detach()

            # ── PASS 1: Task loss → g_task ──
            self.actor_critic.act(aug_obs_batch, history_batch, wm_feature_batch,
                                  grid=grid_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, wm_feature_batch,
                                                     grid=grid_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Adaptive LR
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                    torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                    2.0 * torch.square(sigma_batch)) - 0.5, dim=-1)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Task surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Task value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Velocity prediction loss
            predicted_linear_vel = self.actor_critic.get_linear_vel(aug_obs_batch, history_batch)
            target_linear_vel = aug_critic_obs_batch[:,
                                self.actor_critic.privileged_dim - 3: self.actor_critic.privileged_dim]
            vel_predict_loss = (predicted_linear_vel - target_linear_vel).pow(2).mean()

            task_loss = (
                surrogate_loss
                + self.vel_predict_coef * vel_predict_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            self.optimizer.zero_grad()
            task_loss.backward()

            if self.safety_coef > 0:
                g_task = extract_flat_grads(self.actor_critic)

                # ── PASS 2: Safety loss → g_safety ──
                self.actor_critic.act(aug_obs_batch, history_batch, wm_feature_batch,
                                      grid=grid_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch_s = self.actor_critic.get_actions_log_prob(actions_batch)
                safety_value_batch = self.actor_critic.evaluate_safety(
                    aug_critic_obs_batch, wm_feature_batch, grid=grid_batch,
                    masks=masks_batch, hidden_states=hid_states_batch[1])

                ratio_s = torch.exp(
                    actions_log_prob_batch_s - torch.squeeze(old_actions_log_prob_batch))
                safety_surrogate = -torch.squeeze(safety_advantages_batch) * ratio_s
                safety_surrogate_clipped = -torch.squeeze(safety_advantages_batch) * torch.clamp(
                    ratio_s, 1.0 - self.clip_param, 1.0 + self.clip_param)
                safety_surrogate_loss = torch.max(
                    safety_surrogate, safety_surrogate_clipped).mean()

                safety_value_loss = (safety_returns_batch - safety_value_batch).pow(2).mean()
                safety_loss = safety_surrogate_loss + self.safety_value_loss_coef * safety_value_loss

                self.optimizer.zero_grad()
                safety_loss.backward()
                g_safety = extract_flat_grads(self.actor_critic)

                # ── COMBINE: damped null-space projection ──
                with torch.no_grad():
                    batch_safety_values = safety_value_batch.detach().squeeze()
                g_safety_proj = damped_null_space_project(
                    g_task, g_safety, batch_safety_values,
                    d_safe=self.d_safe, d_danger=self.d_danger)
                g_final = g_task + self.safety_coef * g_safety_proj

                g_final_norm = torch.norm(g_final)
                if g_final_norm > self.max_grad_norm:
                    g_final = g_final * (self.max_grad_norm / g_final_norm)

                write_flat_grads(self.actor_critic, g_final)
            elif self.train_safety_critic_when_off:
                # Critic-only regression: the safety head shares no parameters
                # with the actor, so accumulating this loss on top of the task
                # gradients leaves the policy update untouched.
                safety_value_batch = self.actor_critic.evaluate_safety(
                    aug_critic_obs_batch, wm_feature_batch, grid=grid_batch,
                    masks=masks_batch, hidden_states=hid_states_batch[1])
                safety_value_loss = (safety_returns_batch - safety_value_batch).pow(2).mean()
                safety_loss = self.safety_value_loss_coef * safety_value_loss
                safety_loss.backward()

            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            if self.safety_coef > 0 or self.train_safety_critic_when_off:
                mean_safety_loss += safety_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_safety_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_safety_loss
