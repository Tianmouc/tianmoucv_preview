# Tianmouc camera wrapper

`TMstreamingDataReader` decodes native SDK frames into NumPy arrays.

```python
from tianmoucv.camera.sdk_utils import TMstreamingDataReader

with TMstreamingDataReader(device_id=0, latest=True) as camera:
    frame = camera(copy=False)
    if frame is not None:
        is_rod, timestamp, data = frame
```

## Performance modes

- `latest=False` (default) consumes every queued SDK frame and is appropriate
  for capture workflows that cannot intentionally discard data.
- `latest=True` discards frames that are already stale when a read starts. Use
  it for preview and real-time inference where bounded latency is more important.
- `copy=False` returns a zero-copy NumPy view. The view is invalidated by the
  next frame of the same modality or by closing the reader.
- `copy=True` returns independent arrays. Use it before retaining a frame or
  passing one to an asynchronous worker.

`open_camera.py` defaults to the low-overhead preview used by the native GUI:

- AOP is rate-limited to 20 FPS and reconstructed by the SDK's fast C++ path.
- COP displays the SDK's decoded 8-bit BGR image without Python HDR fusion.
- Frame reads are zero-copy; only optional asynchronous processing owns copies.
- OpenCV event handling is bounded to 60 Hz rather than running for every ROD.

Press `q` or Escape to stop it. Useful modes are:

```bash
# Recommended on Jetson Nano
python tianmoucv_example/camera/open_camera.py

# Cheapest view of the three signed ROD channels
python tianmoucv_example/camera/open_camera.py --aop-mode raw

# Optional high-quality processing (more CPU intensive)
python tianmoucv_example/camera/open_camera.py --aop-mode poisson --iterations 5
python tianmoucv_example/camera/open_camera.py --cop-mode hdr --cop-fps 10 --iterations 5
```

Pass `--lossless` to disable stale-frame dropping. This is intended for capture,
not preview, and can increase latency when processing cannot keep up.

## Build

The CMake configuration defaults to a Release build. Generated files belong in
the ignored `build/` directory or an out-of-tree build directory.
