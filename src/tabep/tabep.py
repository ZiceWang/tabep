from __future__ import annotations

from .model import DeepEnergyModel
from .module import LitEP


class TabEnergyModel(DeepEnergyModel):
    """TabEP backbone for mixed tabular classification.

    The current design follows the common TabM/FT-Transformer-style tabular recipe at
    the data boundary: numeric features are normalized, categorical features are
    embedded via one-hot expansion, and a compact MLP-like interaction stack is trained
    as an energy model with equilibrium propagation.
    """


class LitTabEP(LitEP):
    """Lightning module alias for tabular equilibrium propagation experiments."""
