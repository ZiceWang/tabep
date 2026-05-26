from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _call_readout(readout: nn.Module, state: Tensor, x: Tensor) -> Tensor:
    try:
        return readout(state, x)
    except TypeError:
        return readout(state)


class DeepEnergyModel(nn.Module):
    """Layered continuous-state model trained with centered Equilibrium Propagation."""

    def __init__(
        self,
        layer_sizes: list[int],
        *,
        rho: str = "hardtanh",
        fhn_delta: float = 0.75,
        fhn_epsilon: float = 0.85,
        fhn_alpha: float = 1.08,
        fhn_beta: float = 0.0,
        weight_scale: float = 0.014,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include at least input and output dimensions")

        self.layer_sizes = layer_sizes
        self.rho_name = rho
        self.fhn_delta = fhn_delta
        self.fhn_epsilon = fhn_epsilon
        self.fhn_alpha = fhn_alpha
        self.fhn_beta = fhn_beta

        self.weights = nn.ParameterList(
            [
                nn.Parameter(weight_scale * torch.randn(in_dim, out_dim))
                for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])
            ]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for dim in layer_sizes[1:]]
        )

    @property
    def num_weight_layers(self) -> int:
        return len(self.weights)

    def rho(self, x: Tensor) -> Tensor:
        if self.rho_name == "sigmoid":
            return torch.sigmoid(x)
        if self.rho_name == "tanh":
            return torch.tanh(x)
        if self.rho_name == "hardtanh":
            return torch.clamp(x, 0.0, 1.0)
        raise ValueError(f"unknown rho: {self.rho_name}")

    def init_state(self, x: Tensor) -> list[Tensor]:
        states = [x]
        for dim in self.layer_sizes[1:]:
            states.append(torch.zeros(x.shape[0], dim, device=x.device, dtype=x.dtype))
        return states

    def step(self, states: list[Tensor], y: Tensor | None, beta_nudge: float, dt: float) -> list[Tensor]:
        x = states[0]
        old = states
        rhos = [self.rho(s) for s in old]
        new_states = [x]

        for idx in range(1, len(old)):
            total = self.biases[idx - 1].unsqueeze(0).expand_as(old[idx])
            total = total + rhos[idx - 1] @ self.weights[idx - 1]
            if idx < len(old) - 1:
                total = total + rhos[idx + 1] @ self.weights[idx].T

            # FHN-inspired activator reaction term u - u^3 with damping controlled by alpha.
            reaction = self.fhn_epsilon * (old[idx] - old[idx].pow(3) - self.fhn_alpha * old[idx] - self.fhn_beta)
            update = -old[idx] + self.fhn_delta * total + reaction
            if idx == len(old) - 1 and y is not None and beta_nudge != 0.0:
                update = update + beta_nudge * (F.one_hot(y, old[idx].shape[1]).to(old[idx].dtype) - old[idx])
            new_states.append(old[idx] + dt * update)

        return new_states

    def run_dynamics(
        self,
        x: Tensor,
        *,
        y: Tensor | None = None,
        beta_nudge: float = 0.0,
        steps: int = 55,
        dt: float = 0.1,
        states: list[Tensor] | None = None,
    ) -> list[Tensor]:
        if states is None:
            states = self.init_state(x)
        for _ in range(steps):
            states = self.step(states, y, beta_nudge, dt)
        return states

    @torch.no_grad()
    def predict(self, x: Tensor, *, steps: int = 55, dt: float = 0.1) -> Tensor:
        states = self.run_dynamics(x, steps=steps, dt=dt)
        return states[-1]

    def supervised_dynamics_loss(
        self,
        x: Tensor,
        y: Tensor,
        *,
        steps: int = 55,
        dt: float = 0.1,
        target_beta: float = 0.0,
        readout: nn.Linear | None = None,
        class_weights: Tensor | None = None,
        hidden_state_l2: float = 0.0,
    ) -> Tensor:
        states = self.run_dynamics(x, y=y if target_beta != 0.0 else None, beta_nudge=target_beta, steps=steps, dt=dt)
        logits = _call_readout(readout, states[-1], x) if readout is not None else states[-1]
        loss = F.cross_entropy(logits, y, weight=class_weights)
        if hidden_state_l2 != 0.0:
            penalty = x.new_zeros(())
            for state in states[1:]:
                penalty = penalty + state.square().mean()
            loss = loss + hidden_state_l2 * penalty
        return loss

    def trajectory_guided_loss(
        self,
        x: Tensor,
        y: Tensor,
        *,
        steps: int = 55,
        dt: float = 0.1,
        readout: nn.Module | None = None,
        class_weights: Tensor | None = None,
        hidden_state_l2: float = 0.0,
        consistency_weight: float = 0.1,
        margin_weight: float = 0.05,
        target_beta: float = 0.0,
    ) -> Tensor:
        """Supervise the whole relaxation trajectory, not only its final state.

        This keeps the score-focused GD path but adds a dynamics-specific inductive
        bias: intermediate states should become predictive and successive logits
        should agree as the system relaxes.
        """
        states = self.init_state(x)
        logits_over_time: list[Tensor] = []
        for _ in range(steps):
            states = self.step(states, y=y if target_beta != 0.0 else None, beta_nudge=target_beta, dt=dt)
            logits_over_time.append(_call_readout(readout, states[-1], x) if readout is not None else states[-1])

        total = x.new_zeros(())
        normalizer = 0.0
        for idx, logits in enumerate(logits_over_time, start=1):
            weight = idx / steps
            total = total + weight * F.cross_entropy(logits, y, weight=class_weights)
            normalizer += weight
        loss = total / normalizer

        if consistency_weight != 0.0 and len(logits_over_time) > 1:
            final_probs = logits_over_time[-1].detach().softmax(dim=1)
            consistency = x.new_zeros(())
            for logits in logits_over_time[:-1]:
                consistency = consistency + F.kl_div(logits.log_softmax(dim=1), final_probs, reduction="batchmean")
            loss = loss + consistency_weight * consistency / (len(logits_over_time) - 1)

        if margin_weight != 0.0:
            logits = logits_over_time[-1]
            target_logits = logits.gather(1, y[:, None]).squeeze(1)
            masked = logits.masked_fill(F.one_hot(y, logits.shape[1]).bool(), -torch.inf)
            competitor_logits = masked.max(dim=1).values
            loss = loss + margin_weight * F.softplus(competitor_logits - target_logits).mean()

        if hidden_state_l2 != 0.0:
            penalty = x.new_zeros(())
            for state in states[1:]:
                penalty = penalty + state.square().mean()
            loss = loss + hidden_state_l2 * penalty
        return loss

    def guided_eqprop_loss(
        self,
        x: Tensor,
        y: Tensor,
        *,
        beta_nudge: float = 0.9,
        free_steps: int = 55,
        nudge_steps: int = 14,
        dt: float = 0.1,
        guidance_weight: float = 1.0,
        guidance_steps: int | None = None,
        guidance_beta: float = 0.0,
        readout: nn.Linear | None = None,
        class_weights: Tensor | None = None,
        hidden_state_l2: float = 1e-4,
    ) -> Tensor:
        ep_loss = self.centered_eqprop_loss(
            x,
            y,
            beta_nudge=beta_nudge,
            free_steps=free_steps,
            nudge_steps=nudge_steps,
            dt=dt,
        )
        if guidance_weight == 0.0:
            return ep_loss
        gd_loss = self.supervised_dynamics_loss(
            x,
            y,
            steps=guidance_steps or free_steps,
            dt=dt,
            target_beta=guidance_beta,
            readout=readout,
            class_weights=class_weights,
            hidden_state_l2=hidden_state_l2,
        )
        return ep_loss + guidance_weight * gd_loss

    def centered_eqprop_loss(
        self,
        x: Tensor,
        y: Tensor,
        *,
        beta_nudge: float = 0.9,
        free_steps: int = 55,
        nudge_steps: int = 14,
        dt: float = 0.1,
    ) -> Tensor:
        free = self.run_dynamics(x, steps=free_steps, dt=dt)
        positive = self.run_dynamics(
            x,
            y=y,
            beta_nudge=beta_nudge,
            steps=nudge_steps,
            dt=dt,
            states=[s.detach() for s in free],
        )
        negative = self.run_dynamics(
            x,
            y=y,
            beta_nudge=-beta_nudge,
            steps=nudge_steps,
            dt=dt,
            states=[s.detach() for s in free],
        )

        objective = x.new_zeros(())
        scale = 2.0 * beta_nudge * x.shape[0]
        for idx, weight in enumerate(self.weights):
            pos_pre = self.rho(positive[idx])
            pos_post = self.rho(positive[idx + 1])
            neg_pre = self.rho(negative[idx])
            neg_post = self.rho(negative[idx + 1])
            pos_energy = (pos_pre @ weight * pos_post).sum()
            neg_energy = (neg_pre @ weight * neg_post).sum()
            objective = objective - (pos_energy - neg_energy) / scale

            pos_bias = (self.biases[idx] * pos_post).sum()
            neg_bias = (self.biases[idx] * neg_post).sum()
            objective = objective - (pos_bias - neg_bias) / scale

        return objective
