"""Unit tests for sharded loading and memory-bounded activation capture."""
from types import SimpleNamespace

import torch
import torch.nn as nn

from ablate.extract import collect_activations
from ablate.model import LM
from ablate.weights import bake_subspace


class _Tokenizer:
    chat_template = None
    pad_token_id = 0

    def __call__(self, texts, return_tensors, padding, add_special_tokens):
        lengths = [max(1, len(text.split())) for text in texts]
        width = max(lengths)
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, length in enumerate(lengths):
            ids[row, -length:] = torch.arange(1, length + 1)
            mask[row, -length:] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def batch_decode(self, rows, skip_special_tokens=True):
        return [str(int(row[-1])) for row in rows]


class _TupleBlock(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, hidden_states):
        return (hidden_states + self.offset,)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([_TupleBlock(1.0), _TupleBlock(2.0)])

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(last_hidden_state=hidden)


class _CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.config = SimpleNamespace(hidden_size=4)
        self.generated_batch_sizes = []

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, **kwargs):  # pragma: no cover - extraction must bypass it
        raise AssertionError("collect_activations should call the decoder backbone")

    def generate(self, input_ids, **kwargs):
        self.generated_batch_sizes.append(input_ids.shape[0])
        continuation = torch.full((input_ids.shape[0], 1), 7, dtype=input_ids.dtype)
        return torch.cat([input_ids, continuation], dim=1)


def test_collect_activations_uses_hooks_and_pools_on_cpu():
    model = _CausalLM()
    lm = LM(model, _Tokenizer(), torch.device("cpu"), torch.float32, "tiny")
    prompts = ["one two three", "one two"]

    actual = collect_activations(lm, prompts, positions=2, batch_size=2)

    enc = lm.tokenize(prompts)
    hidden = model.model.embed_tokens(enc["input_ids"])
    expected = [hidden[:, -2:].mean(dim=1)]
    for layer in model.model.layers:
        hidden = layer(hidden)[0]
        expected.append(hidden[:, -2:].mean(dim=1))
    expected = torch.stack(expected)

    assert actual.device.type == "cpu"
    assert actual.shape == (3, 2, 4)
    assert torch.allclose(actual, expected)
    assert not model.model.layers[0]._forward_pre_hooks
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_collect_activations_rejects_empty_prompt_list():
    model = _CausalLM()
    lm = LM(model, _Tokenizer(), torch.device("cpu"), torch.float32, "tiny")

    try:
        collect_activations(lm, [])
    except ValueError as exc:
        assert "prompts" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("empty prompt collection should fail clearly")


def test_generation_respects_batch_size():
    model = _CausalLM()
    lm = LM(model, _Tokenizer(), torch.device("cpu"), torch.float32, "tiny")

    outputs = lm.generate(["one", "two", "three"], batch_size=2)

    assert outputs == ["7", "7", "7"]
    assert model.generated_batch_sizes == [2, 1]


def test_device_map_loading_does_not_move_dispatched_model(monkeypatch):
    from ablate import model as model_module

    recorded = {}

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        padding_side = "right"

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return FakeTokenizer()

    class FakeModel(_CausalLM):
        def __init__(self):
            super().__init__()
            self.to_calls = []

        def to(self, device):
            self.to_calls.append(device)
            return self

    loaded = FakeModel()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            recorded.update(kwargs)
            return loaded

    monkeypatch.setattr(model_module, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(model_module, "AutoModelForCausalLM", FakeAutoModel)

    lm = LM.load(
        "huge/model",
        device_map="auto",
        offload_folder="offload",
        max_memory={0: "70GiB"},
    )

    assert recorded["device_map"] == "auto"
    assert recorded["low_cpu_mem_usage"] is True
    assert recorded["offload_folder"] == "offload"
    assert recorded["max_memory"] == {0: "70GiB"}
    assert loaded.to_calls == []
    assert lm.device == loaded.get_input_embeddings().weight.device


def test_baking_rejects_sharded_model_before_mutation():
    model = _CausalLM()
    model.hf_device_map = {"model.layers.0": 0, "model.layers.1": 1}
    original = model.get_input_embeddings().weight.detach().clone()

    try:
        bake_subspace(model, torch.ones(4))
    except RuntimeError as exc:
        assert "sharded" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("sharded baking should fail")

    assert torch.equal(model.get_input_embeddings().weight, original)
