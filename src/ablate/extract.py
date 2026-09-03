"""Activation extraction and refusal-direction computation.

The primary method is **difference-of-means** (Arditi et al., 2024, "Refusal in
LLMs is mediated by a single direction"): for each layer, the candidate direction
is ``mean(harmful_activations) - mean(harmless_activations)`` at the last prompt
token. Empirically this isolates the *causal* refusal component better than a
linear probe, which tends to latch onto spurious separating features.

PCA and mean-difference-on-PCA variants are provided for comparison.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from tqdm.auto import tqdm

from . import utils
from .model import LM


def _hidden_tensor(output) -> torch.Tensor:
    """Get the residual tensor from a decoder block's common output forms."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(hidden):
        return hidden
    raise TypeError(f"Could not find a hidden-state tensor in {type(output).__name__}")


class _ActivationCapture:
    """Capture only pooled residuals and offload them to CPU layer by layer."""

    def __init__(self, layers, positions: int):
        self.layers = layers
        self.positions = max(1, positions)
        self.values: List[Optional[torch.Tensor]] = [None] * (len(layers) + 1)
        self.handles = []

    def _store(self, index: int, hidden: torch.Tensor) -> None:
        p = min(self.positions, hidden.shape[-2])
        self.values[index] = hidden[..., -p:, :].mean(dim=-2).float().cpu()

    def _capture_input(self, module, inputs) -> None:
        if not inputs:
            raise RuntimeError("Decoder layer received no positional hidden state")
        self._store(0, _hidden_tensor(inputs[0]))

    def _capture_output(self, index: int):
        def hook(module, inputs, output) -> None:
            self._store(index, _hidden_tensor(output))

        return hook

    def __enter__(self) -> "_ActivationCapture":
        self.handles.append(self.layers[0].register_forward_pre_hook(self._capture_input))
        for index, layer in enumerate(self.layers, start=1):
            self.handles.append(layer.register_forward_hook(self._capture_output(index)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def stacked(self) -> torch.Tensor:
        missing = [i for i, value in enumerate(self.values) if value is None]
        if missing:
            raise RuntimeError(f"Activation hooks did not run for residual indices: {missing}")
        return torch.stack(self.values, dim=0)  # type: ignore[arg-type]


@torch.no_grad()
def collect_activations(
    lm: LM,
    prompts: List[str],
    positions: int = 1,
    batch_size: int = 8,
    system: Optional[str] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Return per-layer residual-stream activations averaged over the last
    ``positions`` tokens of each prompt.

    Output shape: ``(n_layers + 1, n_prompts, hidden_size)`` where index 0 is the
    first decoder layer's input residual and index ``i`` is the output of decoder
    layer ``i-1``.
    """
    if not prompts:
        raise ValueError("prompts must contain at least one string")

    layers = utils.get_decoder_layers(lm.model)
    if not layers:
        raise ValueError("model has no decoder layers")
    backbone = utils.get_decoder_backbone(lm.model)
    all_layers: List[torch.Tensor] = []  # per batch: (L+1, B, D)
    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="extract")

    for start in iterator:
        chunk = prompts[start:start + batch_size]
        enc = lm.tokenize(chunk, system=system)
        # Hooks reduce each layer from (B, S, D) to (B, D) and move it to CPU
        # immediately. Calling the backbone skips the memory-heavy LM head.
        with _ActivationCapture(layers, positions) as capture:
            out = backbone(**enc, use_cache=False, output_hidden_states=False)
        all_layers.append(capture.stacked())
        del out, enc

    return torch.cat(all_layers, dim=1)  # (L+1, N, D)


def _normalize(v: torch.Tensor) -> torch.Tensor:
    n = v.norm()
    return v / n if n > 0 else v


def diff_of_means(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """Per-layer difference of means, unit-normalized.

    Inputs are ``(L+1, N, D)``; output is ``(L+1, D)`` unit vectors.
    """
    d = harmful.mean(dim=1) - harmless.mean(dim=1)  # (L+1, D)
    return torch.stack([_normalize(d[i]) for i in range(d.shape[0])], dim=0)


def pca_direction(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """Per-layer top principal component of the pooled, mean-centered
    activations, sign-aligned to point from harmless -> harmful."""
    L = harmful.shape[0]
    dirs = []
    for i in range(L):
        X = torch.cat([harmful[i], harmless[i]], dim=0)  # (N, D)
        X = X - X.mean(dim=0, keepdim=True)
        # top right-singular vector
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        pc = Vh[0]
        mean_diff = harmful[i].mean(0) - harmless[i].mean(0)
        if torch.dot(pc, mean_diff) < 0:
            pc = -pc
        dirs.append(_normalize(pc))
    return torch.stack(dirs, dim=0)


def probe_direction(harmful: torch.Tensor, harmless: torch.Tensor, epochs: int = 200, lr: float = 0.05) -> torch.Tensor:
    """Per-layer logistic-regression probe weight vector (baseline).

    Included for comparison. In practice diff-of-means usually ablates better;
    the probe direction here overfits separating features on small samples.
    """
    L, _, D = harmful.shape
    dirs = []
    for i in range(L):
        X = torch.cat([harmful[i], harmless[i]], dim=0)
        y = torch.cat([torch.ones(harmful.shape[1]), torch.zeros(harmless.shape[1])])
        X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
        w = torch.zeros(D, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            logits = X @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
        dirs.append(_normalize(w.detach()))
    return torch.stack(dirs, dim=0)


def extract_directions(
    lm: LM,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    method: str = "diff_of_means",
    positions: int = 1,
    batch_size: int = 8,
    show_progress: bool = True,
) -> torch.Tensor:
    """High-level entry point: returns ``(n_layers + 1, hidden_size)`` unit
    directions, one candidate per layer."""
    h_act = collect_activations(lm, harmful_prompts, positions, batch_size, show_progress=show_progress)
    s_act = collect_activations(lm, harmless_prompts, positions, batch_size, show_progress=show_progress)
    if method == "diff_of_means":
        return diff_of_means(h_act, s_act)
    if method == "pca":
        return pca_direction(h_act, s_act)
    if method == "probe":
        return probe_direction(h_act, s_act)
    raise ValueError(f"unknown method: {method}")
