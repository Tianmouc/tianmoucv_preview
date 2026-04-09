Demo_slim — Independent UNet demo (no diffusers)
===============================================
credit to Yapeng Meng

Overview
--------
This folder contains a self-contained minimal implementation to run the
UNet-style demo from the repository without depending on `diffusers` or the
`CBRDM` package at runtime.

Contents
--------
- `networks.py` — Standalone network implementations adapted from `CBRDM`.
  Includes `UpsampleTSDConv`, the UNet encoder blocks, `ThickConvGRU`,
  `TmcRGBTSDFusionBranch_Dev` and `BiIterativeTSDInterpolatorFull` used by the
  original demo.
- `run_demo.py` — Runner that replicates the data-loading and preprocessing
  flow from `demo_use/HSDeblurUNet_demo.py`, runs a forward pass through the
  local model and saves visualization outputs (JPEG and MP4).

Key points
----------
- This package intentionally avoids importing `CBRDM` or `diffusers`.
- If you have trained checkpoints, provide a plain PyTorch `state_dict`
  (`.pth` or `.pt`) that matches the standalone model names expected by
  `networks.py` and pass it via `--weights` to `run_demo.py`.
- `run_demo.py` still uses the project's data reader (`TianmoucDataReader`) and
  visualization helpers from `demo_use/demo_utils.py` — these are not modified.

How to run
----------
Example (single frame pair inference):

```bash
python demo_slim/run_demo.py \
  --data_root /path/to/dataroot \
  --sample indoor_1 \
  --pidx 50 \
  --device cuda:0 \
  --weights /path/to/state_dict.pth \
  --out_dir ./demo_slim_output
```

- `--data_root` should point to the folder that contains the sample subfolders
  used by `TianmoucDataReader`.
- `--sample` is the sample folder name (e.g. `indoor_1`).
- `--pidx` is the frame index to process (the script reads `pidx` and `pidx+1`).
- `--weights` is optional; if omitted the model runs with random init.

Weights format notes
--------------------
- Prefer a `state_dict` that directly maps to the model defined in
  `demo_slim/networks.py` (i.e., the top-level keys correspond to layer
  names in the BiIterativeTSDInterpolatorFull instance).
- If your current checkpoints are stored in the project's original format
  (diffusers-style or wrapped), export a plain state_dict before using here.

If you want help
---------------
- I can add a small loader that maps your existing diffusers-style checkpoint
  to this standalone model automatically — upload one example checkpoint and
  I’ll implement a robust loader.
Slim demo package
=================

This folder contains a minimal shim of the parts of `diffusers` used by the
original project so the demo script can be run without installing the full
`diffusers` package.

Usage:

1. Optionally copy any pretrained weights you want into a folder and pass the
   path to `--pretrained` when running `slim_demo_HSDeblurUNet.py`.

2. Run the slim demo:

   python demo_slim/slim_demo_HSDeblurUNet.py --device cuda:0 --pretrained /path/to/checkpoint

Notes:
- This shim provides minimal compatibility only. If your original trained
  checkpoints rely on diffusers-specific serialization, you'll need to export a
  plain `state_dict` for the submodules you want to load into the slim demo.
