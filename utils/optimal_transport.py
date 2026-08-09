"""Minibatch cosine optimal transport for rectified-flow training.

Reorders the noise batch so each clean sample is paired with its most-similar
noise vector (an OT coupling under cosine distance). This straightens the
flow-matching trajectories within a minibatch and tends to improve convergence.

Ported from the Anima pipeline; kept dependency-light (torch + scipy, with an
optional torch_linear_assignment CUDA fast path) so any model can reuse it.
Only meaningful when batch size > 1.
"""

import torch


def cosine_optimal_transport(X: torch.Tensor, Y: torch.Tensor, backend: str = "auto"):
    """Optimal assignment between rows of X and Y under cosine distance.

    Returns (cost_matrix, (row_indices, col_indices)); apply ``Y[col_indices]``
    to reorder Y into the OT-matched order for X.
    """
    X_norm = X / torch.norm(X, dim=1, keepdim=True)
    Y_norm = Y / torch.norm(Y, dim=1, keepdim=True)
    cost = -torch.mm(X_norm, Y_norm.t())

    if backend == "cuda":
        return _cuda_assignment(cost)
    if backend == "scipy":
        return _scipy_assignment(cost)
    try:
        return _cuda_assignment(cost)
    except (ImportError, RuntimeError):
        return _scipy_assignment(cost)


def _cuda_assignment(cost: torch.Tensor):
    from torch_linear_assignment import assignment_to_indices, batch_linear_assignment
    assignment = batch_linear_assignment(cost.unsqueeze(0))
    row_idx, col_idx = assignment_to_indices(assignment)
    return cost, (row_idx, col_idx)


def _scipy_assignment(cost: torch.Tensor):
    from scipy.optimize import linear_sum_assignment
    cost_np = cost.to(torch.float32).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    row = torch.from_numpy(row_ind).to(cost.device, torch.long)
    col = torch.from_numpy(col_ind).to(cost.device, torch.long)
    return cost, (row, col)


def ot_reorder_noise(latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Return ``noise`` reordered so each latent pairs with its OT-matched noise.

    ``latents`` / ``noise`` are [B, ...] tensors flattened per-sample for the
    cosine assignment. No-op guard for B <= 1 should be done by the caller.
    """
    bs = latents.shape[0]
    with torch.no_grad():
        _, (_, col) = cosine_optimal_transport(latents.reshape(bs, -1), noise.reshape(bs, -1))
        return noise[col.squeeze(0)]
