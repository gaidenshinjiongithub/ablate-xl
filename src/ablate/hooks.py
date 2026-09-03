"""Runtime residual-stream ablation via forward hooks.

For an orthonormal basis ``Q`` ``(k, D)`` (or a single unit direction ``(D,)``)
and residual activation ``h``:

    h' = h - alpha * (h Qᵀ) Q

applied to the *output* of each selected decoder block, at every token position,
during the forward pass. The checkpoint is never modified, so this is the fast
path for iteration and hyperparameter search. ``k == 1`` is single-direction
ablation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from . import utils
from .config import AblationConfig
from .model import LM
from .subspace import as_basis, orthonormalize, project_subspace


def project_out(h: torch.Tensor, basis: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Project ``basis`` (a ``(D,)`` direction or ``(k, D)`` orthonormal basis)
    out of ``h``. Kept as the public name for backward compatibility."""
    return project_subspace(h, basis, alpha)


class AblationHooks:
    """Context manager / handle that installs residual-stream ablation hooks.

    ``basis`` may be a single direction ``(D,)`` or an orthonormal basis
    ``(k, D)``; it is orthonormalized defensively on entry.

    Usage::

        with AblationHooks(lm, basis, config):
            text = lm.generate([...])
    """

    def __init__(self, lm: LM, basis: torch.Tensor, config: AblationConfig):
        self.lm = lm
        self.basis = orthonormalize(as_basis(basis))
        self.config = config
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._basis_cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _basis_for(self, hidden: torch.Tensor) -> torch.Tensor:
        key = (hidden.device, hidden.dtype)
        if key not in self._basis_cache:
            self._basis_cache[key] = self.basis.to(
                dtype=hidden.dtype,
                device=hidden.device,
            )
        return self._basis_cache[key]

    def _make_hook(self, alpha: float):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                h = project_subspace(output[0], self._basis_for(output[0]), alpha)
                return (h,) + tuple(output[1:])
            return project_subspace(output, self._basis_for(output), alpha)

        return hook

    def __enter__(self) -> "AblationHooks":
        layers = utils.get_decoder_layers(self.lm.model)
        rng = self.config.layer_range(len(layers))
        hook = self._make_hook(self.config.alpha)
        for i in rng:
            self._handles.append(layers[i].register_forward_hook(hook))
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._basis_cache = {}


def ablated(lm: LM, basis: torch.Tensor, config: AblationConfig) -> AblationHooks:
    """Convenience factory mirroring ``AblationHooks(...)``."""
    return AblationHooks(lm, basis, config)
