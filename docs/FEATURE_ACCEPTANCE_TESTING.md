# Feature Acceptance and Testing

## Purpose

This document defines the mandatory acceptance protocol for every new TianMouCV feature. A feature must complete this protocol before it can be described as accepted or ready to merge. Import success, a synthetic tensor test, or a CPU-only result is not sufficient. Acceptance must cover repository integrity, installation, public APIs, real Tianmouc data, CUDA execution, user-facing examples, and the artifacts produced by the feature.

The official `fishe8.7z` capture referenced by the project README is the default real-data fixture. A task-specific real capture must also be used when the effect under test is not observable in `fishe8`, such as strong motion blur, severe noise, high dynamic range, or camera streaming behavior.

## Non-Negotiable Rules

- Run acceptance from the exact commit and environment that will be reported.
- Use real Tianmouc data for functional acceptance. Random tensors are permitted only for fast shape, dtype, device, and compatibility smoke tests.
- Run at least one complete real-data inference path on CUDA. A CUDA-capable feature cannot be accepted from CPU-only evidence.
- Exercise the public `tianmoucv` import path. Importing a private training module is not a package acceptance test.
- Execute notebooks from top to bottom in a fresh kernel. Existing cell outputs are not evidence.
- Verify generated files by reading or decoding them, not only by checking that a path exists.
- Do not add a dependency when the same result can be obtained with the current TianMouCV requirements or the Python standard library.
- Record every skipped or blocked gate. A blocked gate is not a pass.

## 1. Repository and Merge Integrity

Before testing code, verify that the repository represents one coherent merge result.

```bash
git status
git diff --name-only --diff-filter=U
git diff --check
git diff --cached --check
rg -n '^(<<<<<<<|=======|>>>>>>>)' . --glob '!*.git/**'
git ls-files --others --exclude-standard
```

Required results:

- No unmerged paths are reported.
- No conflict markers remain in source, notebooks, documentation, or configuration files.
- Both staged and unstaged diffs pass whitespace checks.
- Every untracked file is intentional.
- Deleted, renamed, and independently added files are reviewed explicitly. In particular, a modify/delete conflict must not be resolved only by running `git add -A` without deciding which public artifact should remain.

During a merge, inspect both the staged and unstaged views. A clean working-tree diff does not prove that the index has no unresolved stages.

## 2. Environment and Installation

Use a supported environment and install the working tree through the same package boundary used by users.

```bash
python -m pip install --no-deps -e .
python -m pip show tianmoucv
python -m pip check
```

Then verify the loaded source and version:

```python
from importlib.metadata import version
from pathlib import Path
import inspect
import tianmoucv

print(version("tianmoucv"))
print(Path(inspect.getfile(tianmoucv)).resolve())
```

Required results:

- The imported package resolves to the intended working tree.
- Package metadata and the documented release version agree.
- Dependency problems are either fixed or documented as unrelated environment issues.
- Installation branches controlled by environment variables are tested in both the set and unset cases.
- Installation and test commands do not silently compile against a different checkout.

## 3. Static Source Validation

At minimum, parse every changed Python file and every changed notebook code cell. Notebook validation must understand IPython magics instead of deleting arbitrary lines and accidentally changing the code structure.

```python
import ast
import json
from pathlib import Path
from IPython.core.inputtransformer2 import TransformerManager

repository = Path.cwd()

for path in repository.joinpath("tianmoucv").rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

transformer = TransformerManager()
for path in repository.joinpath("tianmoucv_example").rglob("*.ipynb"):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.strip():
            compile(
                transformer.transform_cell(source),
                f"{path}:cell-{index}",
                "exec",
            )
```

Syntax warnings should be reviewed, even when they do not fail the gate. Generated caches and build output should be redirected outside the source tree during validation.

## 4. Official Real-Data Fixture

Use the project utility to obtain the official capture referenced by the README:

```python
from tianmoucv.data import TianmoucDataReader
from tianmoucv.utils import ensure_tutorial_dataset

fixture = ensure_tutorial_dataset()
dataset = TianmoucDataReader(
    str(fixture.sample_dir),
    N=1,
    print_info=False,
)

assert len(dataset) > 0
sample = dataset[0]
assert "F0" in sample
assert "rawDiff" in sample or "tsdiff" in sample
assert "meta" in sample
print(fixture.sample_dir)
print(len(dataset))
print({key: getattr(value, "shape", None) for key, value in sample.items()})
```

The acceptance record must include:

- The resolved data path and capture name.
- The number of reader samples.
- The tested sample indices.
- Input keys, shapes, dtypes, ranges, and relevant metadata.
- Reader options that affect correction, upsampling, camera selection, or temporal length.

Do not replace this gate with a previously exported NumPy array unless the change specifically concerns that export format.

## 5. Algorithm Acceptance

For every changed algorithm:

1. Load the public class or function from `tianmoucv`.
2. Resolve the default weight through the documented download and cache path, or provide a recorded local checkpoint.
3. Verify strict state-dictionary loading when the architecture requires an exact match.
4. Run at least one synthetic smoke test for shape and compatibility.
5. Run end-to-end inference on real Tianmouc data with CUDA.
6. Check output shape, dtype, device, finite values, and expected range.
7. Inspect a visual result from a capture where the target effect is visible.
8. Exercise retained legacy class names, constructor arguments, input keys, and output keys.

For stateful or recurrent models, test both a reset sequence and at least two consecutive samples. Confirm that state is reset when batch or spatial compatibility changes.

### Mandatory CUDA Gate

Run the feature in the intended CUDA-enabled acceptance environment and record the exact device:

```python
import sys
import torch

assert torch.cuda.is_available(), "CUDA is required for feature acceptance"
device = torch.device("cuda:0")
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(device))

torch.cuda.reset_peak_memory_stats(device)
# Construct the public feature on device and run real-data inference here.
torch.cuda.synchronize(device)
print("peak CUDA memory:", torch.cuda.max_memory_allocated(device))
```

The CUDA gate must verify:

- Model parameters and required tensors are on the intended CUDA device.
- The complete public preprocessing, inference, and postprocessing path runs without a device mismatch.
- Output tensors are finite and have the documented shape, dtype, device, and range.
- At least two consecutive inputs run successfully for stateful or streaming features.
- Peak CUDA memory is recorded for memory-sensitive features.
- The generated user-facing artifact comes from the CUDA result, not from a separate CPU fallback.

An inherently CPU-only or native-hardware feature may mark CUDA computation as not applicable, but the reason must be explicit in the acceptance report. This exception must not be used for a feature that exposes a CUDA execution path.

For exposure-dependent deblurring, parse exposure from the capture metadata rather than hard-coding it. Record the selected cone index, exposure time, measured AOP interval, and resulting blur length. Use a real high-motion or high-exposure capture when `fishe8` does not make the deblurring effect visible.

For denoising, save the denoised TSD itself in addition to a visualization. Compare raw and denoised TD, SDX, and SDY using identical scaling so that visual differences are meaningful.

## 6. Notebook Acceptance

Each changed demo notebook must:

- Use a registered kernel that points to the intended environment.
- Import the installed TianMouCV public API.
- Validate the input directory before model construction.
- Expose data path, output path, sample range, device, and checkpoint as editable parameters.
- Run all cells in order with cleared outputs and no hidden state from an earlier session.
- Produce the documented outputs in a temporary or explicitly selected output directory.
- Avoid imports from adjacent development scripts that are not part of the installed package.

For GPU notebooks, record the Python executable, PyTorch version, CUDA availability, and GPU name in the acceptance report.

The notebook acceptance run itself must select the registered CUDA environment and execute the feature on CUDA. A notebook that silently falls back to CPU does not pass.

## 7. Artifact and Video Validation

A generated artifact passes only if it can be consumed again.

For arrays and images:

- Reload each representative output.
- Verify shape, dtype, finite values, and metadata.
- Confirm that RGB/BGR conversion and signed-signal scaling are consistent.

For MP4 files:

```bash
ffprobe -v error \
  -show_entries stream=codec_name,pix_fmt,width,height,nb_frames \
  -of default=noprint_wrappers=1 output.mp4

ffmpeg -v error -i output.mp4 -f null -
```

Required results:

- The writer opens successfully and the file is non-empty.
- The reported dimensions match the generated frames.
- The pixel format is broadly playable, preferably `yuv420p`.
- The expected frame count and duration are present.
- A full decode completes without errors.
- At least the first, middle, and final frames are inspected visually.

## 8. Native Camera Changes

Camera changes require both Python-side and native-side acceptance.

Build outside the source tree:

```bash
build_directory="$(mktemp -d /tmp/tianmoucv-camera-build.XXXXXX)"
cmake \
  -S tianmoucv/camera \
  -B "$build_directory" \
  -DSDK_PATH=/path/to/tianmouc_sdk \
  -DTIANMOUCV_BUILD_EXAMPLE=OFF
cmake --build "$build_directory" --parallel
```

Required results:

- CMake finds the OpenCV C++ development package, USB dependencies, and Tianmouc SDK.
- The shared library builds with the declared C++ standard.
- Exported native function signatures match the `ctypes` declarations.
- Open, configure, start, read, timeout, preview, and idempotent close paths are tested.
- Zero-copy buffers are copied before being retained by asynchronous work.
- A physical camera test verifies both ROD and CONE frames when hardware-facing code changes.

If native build dependencies or camera hardware are unavailable, report the gate as blocked and do not describe static Python tests as a substitute.

## 9. Dependency Review

For every import added to an inference path:

- Confirm that it is already in TianMouCV requirements, belongs to the Python standard library, or is genuinely unavoidable.
- Prefer optional imports for optional features.
- Do not force training-only packages into inference.
- Test import behavior in an environment containing only the declared runtime requirements.
- Review lower and upper version bounds for binary ABI compatibility.

## 10. Acceptance Report

Every merge-ready change should provide a compact report containing:

```text
Commit or merge head:
Branch:
Python executable and version:
TianMouCV version:
PyTorch and CUDA versions:
GPU or CPU:
CUDA device and runtime:
Peak CUDA memory:
Real capture and sample indices:
Checkpoint path or URL and checksum:
Commands executed:
Passed gates:
Blocked or skipped gates and reasons:
Output artifact paths:
Known limitations:
```

## Merge-Ready Checklist

- [ ] No unresolved Git stages or conflict markers remain.
- [ ] The final staged diff has been reviewed.
- [ ] No unintended files were added, removed, or renamed.
- [ ] Package installation and public imports pass.
- [ ] Changed Python files and notebooks pass static validation.
- [ ] The official real-data reader test passes.
- [ ] Every changed algorithm passes real-data inference.
- [ ] At least one complete real-data run uses CUDA through the public feature API.
- [ ] Legacy compatibility required by the release is exercised.
- [ ] Notebook execution passes in a fresh registered kernel.
- [ ] Arrays, images, and videos are reopened or fully decoded.
- [ ] Native camera code builds and runs on hardware when applicable.
- [ ] New dependencies are justified and declared.
- [ ] Blocked gates and known limitations are documented.
- [ ] The acceptance report contains reproducible commands and artifact paths.
