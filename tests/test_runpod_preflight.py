"""Tests for the fail-fast RunPod hardware gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_preflight_module():
    path = Path(__file__).parents[1] / "runpod" / "preflight.py"
    spec = importlib.util.spec_from_file_location("ablate_runpod_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_accepts_sufficient_single_host(monkeypatch, tmp_path, capsys):
    preflight = _load_preflight_module()
    gib = 1024 ** 3
    monkeypatch.delenv("NUM_NODES", raising=False)
    monkeypatch.setattr(preflight.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        preflight.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name=f"Fake GPU {index}", total_memory=80 * gib),
    )

    status = preflight.main(
        ["--workspace", str(tmp_path), "--min-total-vram-gib", "150"]
    )

    assert status == 0
    assert "Aggregate VRAM: 160.0 GiB" in capsys.readouterr().out


def test_preflight_rejects_multi_node_launch(monkeypatch, tmp_path, capsys):
    preflight = _load_preflight_module()
    monkeypatch.setenv("NUM_NODES", "2")
    monkeypatch.setattr(preflight.torch.cuda, "device_count", lambda: 0)

    status = preflight.main(["--workspace", str(tmp_path)])

    assert status == 2
    assert "single-host" in capsys.readouterr().err
