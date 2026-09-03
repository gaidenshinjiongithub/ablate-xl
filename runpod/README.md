# RunPod frontier-model quickstart

This workflow is for checkpoints that need every GPU in one large RunPod Pod.
It runs Ablate XL directly from this GitHub checkout; PyPI publication is not
required.

## Supported topology

Use **one multi-GPU Pod**. Ablate XL currently uses Transformers
`device_map="auto"`, which can place layers across all GPUs visible to one
Python process. It cannot span host boundaries. `preflight.py` rejects
`NUM_NODES > 1` so an Instant Cluster cannot accidentally download and load a
complete model once per node.

The first practical profile is `moonshotai/Kimi-K3` on an 8x B300-class host.
Kimi K3's multimodal wrapper is supported by routing activation capture to its
nested text decoder (`language_model.model.layers`). The Qwen3.8 FP8 profile is
included as a guarded target, but its memory floor is intentionally higher than
a normal 8x B300 Pod.

The NVIDIA Qwen3.8 NVFP4 checkpoint is **not** used here: its published runtime
path is vLLM/SGLang, while Ablate XL currently needs direct PyTorch decoder
modules for residual hooks.

## 1. Create the Pod

In RunPod:

1. Deploy one Pod with 8x B300 GPUs (or another single host with at least the
   profile's aggregate VRAM).
2. Attach a persistent network volume mounted at `/workspace`. For Kimi K3,
   budget at least 1.7 TiB free before starting.
3. Use a recent RunPod PyTorch image with CUDA support for the selected GPU.
4. Put `HF_TOKEN` in the Pod environment if the checkpoint requires it. Never
   commit the token to this repository.

## 2. Clone and install from GitHub

Run this in the Pod terminal:

```bash
cd /workspace
git clone https://github.com/gaidenshinjiongithub/ablate-xl.git
cd ablate-xl
bash runpod/setup.sh
```

`setup.sh` installs the checkout in editable mode plus Accelerate,
compressed-tensors support, fast Hugging Face downloads, and FlashAttention.
To use an image that already contains a compatible FlashAttention build, run
`INSTALL_FLASH_ATTN=0 bash runpod/setup.sh`.

## 3. Preflight before spending download time

The extraction launcher always checks visible CUDA devices, aggregate VRAM,
free `/workspace` storage, and the single-host topology before downloading the
checkpoint:

```bash
bash runpod/extract.sh runpod/profiles/kimi-k3.env
```

If it passes, the model loads with `device_map="auto"`, activation extraction
runs at batch size 1, and the direction tensor is saved to:

```text
/workspace/ablate-output/kimi-k3-directions.pt
```

For a different open-weight Transformers checkpoint, set the variables
directly:

```bash
MODEL_ID="org/frontier-model" \
MIN_TOTAL_VRAM_GIB=1800 \
MIN_DISK_GIB=1800 \
TRUST_REMOTE_CODE=1 \
bash runpod/extract.sh
```

## Qwen3.8 status

You can inspect the guarded Qwen profile without starting a download:

```bash
bash runpod/extract.sh runpod/profiles/qwen38-fp8.env
```

On an ordinary 8x B300 Pod, preflight should stop on the VRAM floor. Do not
lower that floor merely to make the check green. True multi-node Qwen3.8
support requires a distributed expert/tensor-parallel backend that preserves
decoder-layer hook access; that is separate from this single-host RunPod
integration.

## What this does and does not validate

The local test suite validates the Kimi wrapper adapter, bounded activation
capture, sharded loading behavior, and the preflight logic. A successful
full-checkpoint RunPod job is still required before claiming Kimi K3 or Qwen3.8
research results. Save the preflight output, package versions, model revision,
and resulting tensor shape with the experiment record.
