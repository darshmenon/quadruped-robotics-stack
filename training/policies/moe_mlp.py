"""Mixture-of-experts actor trunk for the asymmetric PPO policy (enable with GO2_MOE=<n_experts>).

The actor trunk is K parallel expert MLPs blended by a small softmax gate that reads the same
features (incl. the height scan), so e.g. one expert can specialise on flat ground and another on
steps. Output size is unchanged, so the action head and critic are untouched. Inspired by the
2026 MoE-locomotion papers (CTS-MoE, sparse-MoE parkour); the gate-usage log below is our own
addition so we can see whether experts actually specialise.
"""
from __future__ import annotations

import torch as th
from torch import nn

from .asymmetric_mlp import AsymmetricActorCriticPolicy, AsymmetricMlpExtractor


class MoETrunk(nn.Module):
    def __init__(self, in_dim, layers, activation_fn, n_experts):
        super().__init__()
        self.experts = nn.ModuleList(
            AsymmetricMlpExtractor._build_stack(in_dim, layers, activation_fn) for _ in range(n_experts))
        self.gate = nn.Linear(in_dim, n_experts)
        self.last_gate = None  # (batch, K) gate weights of the latest forward, for logging

    def forward(self, x):
        w = th.softmax(self.gate(x), dim=-1)
        self.last_gate = w.detach()
        return sum(w[:, k:k + 1] * e(x) for k, e in enumerate(self.experts))


class MoEMlpExtractor(AsymmetricMlpExtractor):
    def __init__(self, *args, n_experts=2, **kwargs):
        super().__init__(*args, **kwargs)
        pi_dim = args[0] if args else kwargs["pi_feature_dim"]
        net_arch = kwargs.get("net_arch", args[2] if len(args) > 2 else None)
        layers = net_arch.get("pi", []) if isinstance(net_arch, dict) else net_arch
        act = kwargs.get("activation_fn", args[3] if len(args) > 3 else nn.ReLU)
        self.policy_net = MoETrunk(pi_dim, layers, act, n_experts).to(next(self.value_net.parameters()).device)


class MoEActorCriticPolicy(AsymmetricActorCriticPolicy):
    def __init__(self, *args, n_experts: int = 2, **kwargs):
        self.n_experts = n_experts  # set before super().__init__, which calls _build_mlp_extractor
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(n_experts=self.n_experts)  # so PPO.load rebuilds the same number of experts
        return data

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MoEMlpExtractor(
            pi_feature_dim=self.features_dim,
            vf_feature_dim=self.vf_features_extractor.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            n_experts=self.n_experts,
        )
