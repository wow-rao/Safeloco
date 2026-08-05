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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class ActorCriticWMP(nn.Module):
    is_recurrent = False

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 encoder_hidden_dims=[256, 128],
                 wm_encoder_hidden_dims = [64, 32],
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 fixed_std=False,
                 latent_dim = 32,
                 height_dim=187,
                 privileged_dim=3 + 24,
                 history_dim = 42*5,
                 wm_feature_dim = 1536,
                 wm_latent_dim=16,
                 grid_enabled=False,
                 grid_H=60,
                 grid_W=30,
                 grid_latent_dim=32,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCriticWMP, self).__init__()

        activation = get_activation(activation)
        critic_activation = get_activation('tanh')

        self.latent_dim = latent_dim
        self.height_dim = height_dim
        self.privileged_dim = privileged_dim
        self.grid_enabled = grid_enabled

        if grid_enabled:
            self.grid_H = grid_H
            self.grid_W = grid_W
            self.grid_latent_dim = grid_latent_dim

            def _conv_out(size, kernel, stride, padding):
                return (size + 2 * padding - kernel) // stride + 1

            h1 = _conv_out(grid_H, 5, 2, 2)
            w1 = _conv_out(grid_W, 5, 2, 2)
            h2 = _conv_out(h1, 3, 2, 1)
            w2 = _conv_out(w1, 3, 2, 1)
            grid_flat_dim = 32 * h2 * w2

            self.grid_encoder_actor = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.ELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Flatten(),
                nn.Linear(grid_flat_dim, 128),
                nn.ELU(),
                nn.Linear(128, grid_latent_dim),
            )

            self.grid_encoder_critic = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.ELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Flatten(),
                nn.Linear(grid_flat_dim, 128),
                nn.ELU(),
                nn.Linear(128, grid_latent_dim),
            )
        else:
            grid_latent_dim = 0

        mlp_input_dim_a = latent_dim + 3 + wm_latent_dim + grid_latent_dim
        mlp_input_dim_c = num_critic_obs + wm_latent_dim + grid_latent_dim

        # History Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(history_dim, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], latent_dim))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*encoder_layers)

        # World Model Feature Encoder
        wm_encoder_layers = []
        wm_encoder_layers.append(nn.Linear(wm_feature_dim, wm_encoder_hidden_dims[0]))
        wm_encoder_layers.append(activation)
        for l in range(len(wm_encoder_hidden_dims)):
            if l == len(wm_encoder_hidden_dims) - 1:
                wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_latent_dim))
            else:
                wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                wm_encoder_layers.append(activation)
        self.wm_feature_encoder = nn.Sequential(*wm_encoder_layers)

        # Critic World Model Feature Encoder
        critic_wm_encoder_layers = []
        critic_wm_encoder_layers.append(nn.Linear(wm_feature_dim, wm_encoder_hidden_dims[0]))
        critic_wm_encoder_layers.append(critic_activation)
        for l in range(len(wm_encoder_hidden_dims)):
            if l == len(wm_encoder_hidden_dims) - 1:
                critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_latent_dim))
            else:
                critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                critic_wm_encoder_layers.append(critic_activation)
        self.critic_wm_feature_encoder = nn.Sequential(*critic_wm_encoder_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(critic_activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(critic_activation)

        self.critic = nn.Sequential(*critic_layers)

        # Safety Critic (same architecture as task critic)
        safety_critic_layers = []
        safety_critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        safety_critic_layers.append(critic_activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                safety_critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                safety_critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                safety_critic_layers.append(critic_activation)
        self.safety_critic = nn.Sequential(*safety_critic_layers)

        safety_critic_wm_encoder_layers = []
        safety_critic_wm_encoder_layers.append(nn.Linear(wm_feature_dim, wm_encoder_hidden_dims[0]))
        safety_critic_wm_encoder_layers.append(critic_activation)
        for l in range(len(wm_encoder_hidden_dims)):
            if l == len(wm_encoder_hidden_dims) - 1:
                safety_critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_latent_dim))
            else:
                safety_critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                safety_critic_wm_encoder_layers.append(critic_activation)
        self.safety_critic_wm_feature_encoder = nn.Sequential(*safety_critic_wm_encoder_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Safety Critic MLP: {self.safety_critic}")

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)

        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        # self.distribution = Normal(mean, mean * 0. + torch.sigmoid(std))
        self.distribution = Normal(mean, mean * 0. + std.clamp(max=20.0))

    def _encode_grid_actor(self, grid):
        if self.grid_enabled and grid is not None:
            return self.grid_encoder_actor(grid.unsqueeze(1))
        return None

    def _encode_grid_critic(self, grid):
        if self.grid_enabled and grid is not None:
            return self.grid_encoder_critic(grid.unsqueeze(1))
        return None

    def act(self, observations, history, wm_feature, grid=None, **kwargs):
        latent_vector = self.history_encoder(history)
        command = observations[:, self.privileged_dim + 6:self.privileged_dim + 9]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        parts = [latent_vector, command, wm_latent_vector]
        grid_latent = self._encode_grid_actor(grid)
        if grid_latent is not None:
            parts.append(grid_latent)
        concat_observations = torch.concat(parts, dim=-1)
        self.update_distribution(concat_observations)
        return self.distribution.sample()

    def get_latent_vector(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        return latent_vector

    def get_linear_vel(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        linear_vel = latent_vector[:,-3:]
        return linear_vel

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, wm_feature, grid=None):
        latent_vector = self.history_encoder(history)
        command = observations[:, self.privileged_dim + 6:self.privileged_dim + 9]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        parts = [latent_vector, command, wm_latent_vector]
        grid_latent = self._encode_grid_actor(grid)
        if grid_latent is not None:
            parts.append(grid_latent)
        concat_observations = torch.concat(parts, dim=-1)
        actions_mean = self.actor(concat_observations)
        return actions_mean

    def evaluate(self, critic_observations, wm_feature, grid=None, **kwargs):
        wm_latent_vector = self.critic_wm_feature_encoder(wm_feature)
        parts = [critic_observations, wm_latent_vector]
        grid_latent = self._encode_grid_critic(grid)
        if grid_latent is not None:
            parts.append(grid_latent)
        concat_observations = torch.concat(parts, dim=-1)
        value = self.critic(concat_observations)
        return value

    def evaluate_safety(self, critic_observations, wm_feature, grid=None, **kwargs):
        wm_latent_vector = self.safety_critic_wm_feature_encoder(wm_feature)
        parts = [critic_observations, wm_latent_vector]
        grid_latent = self._encode_grid_critic(grid)
        if grid_latent is not None:
            parts.append(grid_latent)
        concat_observations = torch.concat(parts, dim=-1)
        safety_value = self.safety_critic(concat_observations)
        return safety_value.clamp(max=0.5)

    def get_export_actor(self):
        """ Make an exportable actor model for inference in jax"""
        grid_enabled = self.grid_enabled
        grid_encoder = self.grid_encoder_actor if grid_enabled else None

        class JaxExportActor(nn.Module):
            def __init__(self, history_encoder, wm_feature_encoder, actor, privileged_dim, grid_enc, use_grid):
                super(JaxExportActor, self).__init__()
                self.history_encoder = history_encoder
                self.wm_feature_encoder = wm_feature_encoder
                self.actor = actor
                self.privileged_dim = privileged_dim
                self.use_grid = use_grid
                if use_grid:
                    self.grid_encoder = grid_enc

            def forward(self, cmd, history, wm_feature, grid=None):
                latent_vector = self.history_encoder(history)
                wm_latent_vector = self.wm_feature_encoder(wm_feature)
                parts = [latent_vector, cmd, wm_latent_vector]
                if self.use_grid and grid is not None:
                    parts.append(self.grid_encoder(grid.unsqueeze(1)))
                concat_observations = torch.concat(parts, dim=-1)
                actions_mean = self.actor(concat_observations)
                return actions_mean

        return JaxExportActor(self.history_encoder, self.wm_feature_encoder, self.actor,
                              self.privileged_dim, grid_encoder, grid_enabled)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None