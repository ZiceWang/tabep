from __future__ import annotations

from functools import lru_cache

import torch

from .tilelang_utils import install_tilelang_stderr_filter, silence_stderr_fd

install_tilelang_stderr_filter()

try:
    with silence_stderr_fd():
        import tilelang
        import tilelang.language as T
except Exception:  # pragma: no cover - TileLang is optional at import time.
    tilelang = None
    T = None


def _hardtanh(x):
    return T.min(T.max(x, 0.0), 1.0)


@lru_cache(maxsize=128)
def _middle_step_kernel(batch: int, prev_dim: int, curr_dim: int, next_dim: int, block_b: int = 8, block_c: int = 16):
    if tilelang is None or T is None:
        raise RuntimeError("TileLang is not available.")

    @tilelang.jit(target="cuda")
    def middle_step():
        @T.prim_func
        def kernel(
            Prev: T.Tensor((batch, prev_dim), "float32"),  # type: ignore[valid-type]
            Curr: T.Tensor((batch, curr_dim), "float32"),  # type: ignore[valid-type]
            Next: T.Tensor((batch, next_dim), "float32"),  # type: ignore[valid-type]
            WPrev: T.Tensor((prev_dim, curr_dim), "float32"),  # type: ignore[valid-type]
            WNext: T.Tensor((curr_dim, next_dim), "float32"),  # type: ignore[valid-type]
            Bias: T.Tensor((curr_dim,), "float32"),  # type: ignore[valid-type]
            Out: T.Tensor((batch, curr_dim), "float32"),  # type: ignore[valid-type]
            dt: T.float32,
            fhn_delta: T.float32,
            fhn_epsilon: T.float32,
            fhn_alpha: T.float32,
            fhn_beta: T.float32,
        ):
            with T.Kernel(T.ceildiv(curr_dim, block_c), T.ceildiv(batch, block_b), threads=(block_b, block_c)) as (bx, by):
                for tb, tc in T.Parallel(block_b, block_c):
                    b = by * block_b + tb
                    c = bx * block_c + tc
                    total = T.alloc_local((), "float32")
                    total[()] = Bias[c]
                    for p in T.serial(prev_dim):
                        total[()] += _hardtanh(Prev[b, p]) * WPrev[p, c]
                    for n in T.serial(next_dim):
                        total[()] += _hardtanh(Next[b, n]) * WNext[c, n]
                    u = Curr[b, c]
                    reaction = fhn_epsilon * (u - u * u * u - fhn_alpha * u - fhn_beta)
                    update = -u + fhn_delta * total[()] + reaction
                    Out[b, c] = u + dt * update

        return kernel

    return middle_step()


@lru_cache(maxsize=128)
def _output_step_kernel(batch: int, prev_dim: int, out_dim: int, block_b: int = 8, block_c: int = 16):
    if tilelang is None or T is None:
        raise RuntimeError("TileLang is not available.")

    @tilelang.jit(target="cuda")
    def output_step():
        @T.prim_func
        def kernel(
            Prev: T.Tensor((batch, prev_dim), "float32"),  # type: ignore[valid-type]
            Curr: T.Tensor((batch, out_dim), "float32"),  # type: ignore[valid-type]
            WPrev: T.Tensor((prev_dim, out_dim), "float32"),  # type: ignore[valid-type]
            Bias: T.Tensor((out_dim,), "float32"),  # type: ignore[valid-type]
            Target: T.Tensor((batch,), "int64"),  # type: ignore[valid-type]
            Out: T.Tensor((batch, out_dim), "float32"),  # type: ignore[valid-type]
            dt: T.float32,
            beta_nudge: T.float32,
            fhn_delta: T.float32,
            fhn_epsilon: T.float32,
            fhn_alpha: T.float32,
            fhn_beta: T.float32,
            has_target: T.int32,
        ):
            with T.Kernel(T.ceildiv(out_dim, block_c), T.ceildiv(batch, block_b), threads=(block_b, block_c)) as (bx, by):
                for tb, tc in T.Parallel(block_b, block_c):
                    b = by * block_b + tb
                    c = bx * block_c + tc
                    total = T.alloc_local((), "float32")
                    total[()] = Bias[c]
                    for p in T.serial(prev_dim):
                        total[()] += _hardtanh(Prev[b, p]) * WPrev[p, c]
                    u = Curr[b, c]
                    reaction = fhn_epsilon * (u - u * u * u - fhn_alpha * u - fhn_beta)
                    update = T.alloc_local((), "float32")
                    update[()] = -u + fhn_delta * total[()] + reaction
                    if has_target != 0 and beta_nudge != 0.0:
                        target_value = T.if_then_else(Target[b] == c, 1.0, 0.0)
                        update[()] += beta_nudge * (target_value - u)
                    Out[b, c] = u + dt * update[()]

        return kernel

    return output_step()


@torch.no_grad()
def run_tilelang_dynamics(
    model,
    x: torch.Tensor,
    *,
    y: torch.Tensor | None = None,
    beta_nudge: float = 0.0,
    steps: int = 55,
    dt: float = 0.1,
    states: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    if not x.is_cuda:
        return model.run_dynamics(x, y=y, beta_nudge=beta_nudge, steps=steps, dt=dt, states=states)
    if x.dtype != torch.float32 or model.rho_name != "hardtanh":
        return model.run_dynamics(x, y=y, beta_nudge=beta_nudge, steps=steps, dt=dt, states=states)

    if states is None:
        states = model.init_state(x)
    else:
        states = [state.detach() for state in states]

    target = y if y is not None else torch.empty((x.shape[0],), device=x.device, dtype=torch.int64)
    target = target.to(device=x.device, dtype=torch.int64).contiguous()
    has_target = 1 if y is not None else 0
    batch = int(x.shape[0])

    for _ in range(steps):
        old = states
        new_states = [old[0]]
        for idx in range(1, len(old)):
            out = torch.empty_like(old[idx])
            if idx < len(old) - 1:
                _middle_step_kernel(batch, old[idx - 1].shape[1], old[idx].shape[1], old[idx + 1].shape[1])(
                    old[idx - 1].contiguous(),
                    old[idx].contiguous(),
                    old[idx + 1].contiguous(),
                    model.weights[idx - 1].detach().contiguous(),
                    model.weights[idx].detach().contiguous(),
                    model.biases[idx - 1].detach().contiguous(),
                    out,
                    float(dt),
                    float(model.fhn_delta),
                    float(model.fhn_epsilon),
                    float(model.fhn_alpha),
                    float(model.fhn_beta),
                )
            else:
                _output_step_kernel(batch, old[idx - 1].shape[1], old[idx].shape[1])(
                    old[idx - 1].contiguous(),
                    old[idx].contiguous(),
                    model.weights[idx - 1].detach().contiguous(),
                    model.biases[idx - 1].detach().contiguous(),
                    target,
                    out,
                    float(dt),
                    float(beta_nudge),
                    float(model.fhn_delta),
                    float(model.fhn_epsilon),
                    float(model.fhn_alpha),
                    float(model.fhn_beta),
                    has_target,
                )
            new_states.append(out)
        states = new_states
    return states
