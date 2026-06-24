"""Custom lume-torch model utilities for the beam-evolution surrogate.

This module defines the custom transforms needed for loading/evaluating
the model via lume-torch.  Unlike the 571 covariance model, there is no
M-normalization or Cholesky decomposition — just z-score denormalization.
"""

import torch
import torch.nn as nn
from botorch.models.transforms.input import AffineInputTransform


class OutputDenormTransform(nn.Module):
    """Output transformer: z-score normalized predictions -> physical units.

    pred_raw = pred_norm * y_std + y_mean
    """

    def __init__(self, y_mean: torch.Tensor, y_std: torch.Tensor):
        super().__init__()
        self.register_buffer("y_mean", y_mean.to(torch.float32))
        self.register_buffer("y_std", y_std.to(torch.float32))

    def forward(self, pred_norm: torch.Tensor) -> torch.Tensor:
        return pred_norm * self.y_std + self.y_mean


class PVToSimWithS(nn.Module):
    """Input transform: (19 machine-PV values, s) -> (19 sim params, s).

    Applies the affine PV-to-sim mapping on the first 19 channels and
    passes `s` (channel 19) through unchanged.
    """

    def __init__(self, pv_to_sim_transform: AffineInputTransform):
        super().__init__()
        self.pv_to_sim = pv_to_sim_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 20) — last column is s
        x_pv = x[:, :19]
        s = x[:, 19:]
        x_sim = self.pv_to_sim(x_pv)
        return torch.cat([x_sim, s], dim=-1)
