"""Weight orthogonalization ("baking" a direction or subspace into the model).

Runtime hooks are ideal for search, but for a shippable model we fold the
projection directly into every weight matrix that writes to the residual stream,
so the model can *never* express the subspace ``span(Q)`` and no hook is needed:

    W_out  <-  (I - QᵀQ) W_out            (attention / MLP output projections)
    E      <-  E (I - QᵀQ)                 (token embeddings, per row)

``Q`` is an orthonormal basis ``(k, D)``; ``k == 1`` is single-direction baking.
Supports both ``nn.Linear`` (Llama/Mistral/Qwen/SmolLM/Pythia) and GPT-2's
``Conv1D`` (weight stored transposed).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import utils
from .subspace import as_basis, orthonormalize


def _bake_plan(model: nn.Module, hidden_size: int):
    """Validate the whole model before mutating any weight tensor."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        placements = {str(value) for value in device_map.values()}
        if "disk" in placements or len(placements) > 1:
            raise RuntimeError(
                "Weight baking is not supported for sharded or disk-offloaded "
                "models. Keep the checkpoint unchanged and use AblationHooks."
            )

    if (
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
        or getattr(model, "hf_quantizer", None) is not None
    ):
        raise RuntimeError(
            "Weight baking is not supported for quantized models. "
            "Keep the checkpoint unchanged and use AblationHooks."
        )

    embedding = utils.get_embedding(model)
    if embedding.weight.shape[-1] != hidden_size:
        raise ValueError("Embedding width does not match the ablation basis")

    writers = []
    for layer in utils.get_decoder_layers(model):
        layer_writers = utils.get_residual_writers(layer)
        for writer in layer_writers:
            if not hasattr(writer, "weight"):
                raise TypeError(
                    f"Unsupported residual writer {type(writer).__name__}: no weight tensor"
                )
            output_size = writer.weight.shape[1] if utils.is_conv1d(writer) else writer.weight.shape[0]
            if output_size != hidden_size:
                raise ValueError(
                    f"Residual writer {type(writer).__name__} outputs {output_size} values; "
                    f"expected hidden size {hidden_size}"
                )
            writers.append(writer)
    return embedding, writers


@torch.no_grad()
def _orthogonalize_module(module: nn.Module, Q: torch.Tensor) -> None:
    """Remove ``span(Q)`` from a residual-writing module's output."""
    W = module.weight.data
    Qd = Q.to(dtype=W.dtype, device=W.device)  # (k, D)

    if utils.is_conv1d(module):
        # GPT-2 Conv1D: y = x @ W, W is (in, out=D). Output dim is dim 1.
        # W <- W (I - QᵀQ) = W - (W Qᵀ) Q
        module.weight.data = W - (W @ Qd.T) @ Qd
    else:
        # nn.Linear: y = W x (+ b), W is (out=D, in). Output dim is dim 0.
        # W <- (I - QᵀQ) W = W - Qᵀ (Q W)
        module.weight.data = W - Qd.T @ (Qd @ W)

    bias = getattr(module, "bias", None)
    if bias is not None:
        b = bias.data
        bq = Qd.to(b.dtype)
        bias.data = b - bq.T @ (bq @ b)


@torch.no_grad()
def _orthogonalize_embedding(emb: nn.Embedding, Q: torch.Tensor) -> None:
    W = emb.weight.data  # (vocab, D): each row is a residual vector
    Qd = Q.to(dtype=W.dtype, device=W.device)
    emb.weight.data = W - (W @ Qd.T) @ Qd


@torch.no_grad()
def bake_subspace(model: nn.Module, basis: torch.Tensor, include_embedding: bool = True) -> None:
    """Permanently orthogonalize ``model``'s weights against ``span(basis)``.

    Mutates ``model`` in place. ``basis`` may be a ``(D,)`` direction or a
    ``(k, D)`` basis; it is orthonormalized defensively.
    """
    Q = orthonormalize(as_basis(basis))
    if Q.shape[0] == 0:
        return
    embedding, writers = _bake_plan(model, Q.shape[1])
    if include_embedding:
        _orthogonalize_embedding(embedding, Q)
    for writer in writers:
        _orthogonalize_module(writer, Q)


# Backward-compatible single-direction alias.
def bake_direction(model: nn.Module, direction: torch.Tensor, include_embedding: bool = True) -> None:
    bake_subspace(model, direction, include_embedding)
