"""Low-overhead ctypes interface for the Tianmouc streaming SDK."""

import ctypes
import os
from queue import Empty, Full, Queue

import numpy as np


ROD_W = 160
ROD_H = 160
CONE_W = 320
CONE_H = 320
ROD_PREVIEW_SHAPE = (ROD_H, ROD_W * 2)
CONE_PREVIEW_SHAPE = (CONE_H, CONE_W * 2, 3)

# Kept for source compatibility with existing examples.  New code should use
# bounded local queues and put_latest() so slow processing cannot build latency.
global_aop_queue = Queue()
global_cop_queue = Queue()
result_queue = Queue()
inference_queue = Queue()


def put_latest(queue, item):
    """Put an item into a bounded queue, replacing stale queued work if full."""
    while True:
        try:
            queue.put_nowait(item)
            return
        except Full:
            try:
                queue.get_nowait()
            except Empty:
                pass


class TianmoucCameraError(RuntimeError):
    """Raised when the native camera SDK cannot complete a lifecycle action."""


class tianmoucData(ctypes.Structure):
    _fields_ = [
        ("isRod", ctypes.c_int),
        ("timestamp", ctypes.c_uint64),
        ("sdl_p", ctypes.POINTER(ctypes.c_ubyte)),
        ("sdr_p", ctypes.POINTER(ctypes.c_ubyte)),
        ("td_p", ctypes.POINTER(ctypes.c_ubyte)),
        ("rgb_p", ctypes.POINTER(ctypes.c_float)),
    ]


class TMstreamingDataReader:
    """Read decoded Tianmouc frames from the native SDK.

    Args:
        cfg_path: Sensor register CSV. Uses the packaged configuration by default.
        lib_path: Native wrapper library. Uses the packaged library by default.
        device_id: Zero-based camera index.
        latest: If true, discard frames that were already stale when a read began.
            This reduces preview latency but is not suitable for lossless capture.

    ``reader()`` returns ``(is_rod, timestamp, data)`` or ``None`` after a native
    receive timeout.  Returned arrays are zero-copy views and remain valid only
    until the next frame of the same modality is read or the reader is closed.
    Call ``reader(copy=True)`` when retaining a frame or passing it to an
    asynchronous worker.
    """

    def __init__(self, cfg_path=None, lib_path=None, device_id=0, latest=False):
        camera_dir = os.path.dirname(os.path.realpath(__file__))
        if cfg_path is None:
            cfg_path = os.path.join(camera_dir, "lib", "Golden_HCG_seq.csv")
        if lib_path is None:
            lib_path = os.path.join(camera_dir, "lib", "libtianmouc.so")

        cfg_path = os.path.realpath(os.fspath(cfg_path))
        lib_path = os.path.realpath(os.fspath(lib_path))
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError("Tianmouc configuration not found: " + cfg_path)
        if not os.path.isfile(lib_path):
            raise FileNotFoundError("Tianmouc native library not found: " + lib_path)

        self.cfg_path = os.fsencode(cfg_path)
        self.tm_sdk = ctypes.CDLL(lib_path)
        self._configure_abi()
        self._latest = bool(latest)
        if self._latest and not hasattr(self.tm_sdk, "tmGetLatestFrame"):
            raise TianmoucCameraError(
                "The selected native library does not support latest-frame mode"
            )

        self.dataPack = tianmoucData()
        self._rod_views = None
        self._cone_view = None
        self._rod_preview = np.empty(ROD_PREVIEW_SHAPE, dtype=np.uint8)
        self._cone_preview = np.empty(CONE_PREVIEW_SHAPE, dtype=np.uint8)
        self._started = False
        self._closed = False

        camera_handle = self.tm_sdk.tmOpenCamera(int(device_id))
        if not camera_handle:
            self._closed = True
            raise TianmoucCameraError(
                "Unable to open Tianmouc camera {}".format(device_id)
            )
        self.cameraHandle = ctypes.c_uint64(camera_handle)

        if not self.tm_sdk.IICconfig_download(self.cameraHandle, self.cfg_path):
            self.tm_sdk.tmCameraUninit(self.cameraHandle)
            self.cameraHandle = ctypes.c_uint64(0)
            self._closed = True
            raise TianmoucCameraError("Unable to configure the Tianmouc camera")

    def _configure_abi(self):
        data_pointer = ctypes.POINTER(tianmoucData)
        self.tm_sdk.tmOpenCamera.argtypes = [ctypes.c_int]
        self.tm_sdk.tmOpenCamera.restype = ctypes.c_uint64
        self.tm_sdk.tmStartTransfer.argtypes = [ctypes.c_uint64]
        self.tm_sdk.tmStartTransfer.restype = ctypes.c_bool
        self.tm_sdk.tmGetFrame.argtypes = [ctypes.c_uint64, data_pointer]
        self.tm_sdk.tmGetFrame.restype = ctypes.c_bool
        if hasattr(self.tm_sdk, "tmGetLatestFrame"):
            self.tm_sdk.tmGetLatestFrame.argtypes = [ctypes.c_uint64, data_pointer]
            self.tm_sdk.tmGetLatestFrame.restype = ctypes.c_bool
        byte_pointer = ctypes.POINTER(ctypes.c_ubyte)
        if hasattr(self.tm_sdk, "tmRenderRodPreview"):
            self.tm_sdk.tmRenderRodPreview.argtypes = [
                ctypes.c_uint64, byte_pointer, ctypes.c_int
            ]
            self.tm_sdk.tmRenderRodPreview.restype = ctypes.c_bool
        if hasattr(self.tm_sdk, "tmRenderConePreview"):
            self.tm_sdk.tmRenderConePreview.argtypes = [
                ctypes.c_uint64, byte_pointer, ctypes.c_int
            ]
            self.tm_sdk.tmRenderConePreview.restype = ctypes.c_bool
        self.tm_sdk.tmCameraUninit.argtypes = [ctypes.c_uint64]
        self.tm_sdk.tmCameraUninit.restype = ctypes.c_bool
        self.tm_sdk.tmExposureSet.argtypes = [ctypes.c_uint64] + [ctypes.c_int] * 7
        self.tm_sdk.tmExposureSet.restype = ctypes.c_bool
        self.tm_sdk.IICconfig_download.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
        self.tm_sdk.IICconfig_download.restype = ctypes.c_bool

    def setExposure(self, aoptime, coptime, aopG=1, copG=1):
        """Set ROD/CONE exposure time and analog gain."""
        if self._closed:
            raise TianmoucCameraError("Camera is closed")
        aoptime = int(min(max(aoptime, 4), 1240))
        coptime = int(min(max(coptime, 4), 15000))
        if not self.tm_sdk.tmExposureSet(
            self.cameraHandle, aoptime, coptime, int(aopG), int(copG), 8, 1, 1
        ):
            raise TianmoucCameraError("Unable to set Tianmouc exposure")

    def start(self):
        """Start native receive threads once."""
        if self._closed:
            raise TianmoucCameraError("Camera is closed")
        if not self._started:
            if not self.tm_sdk.tmStartTransfer(self.cameraHandle):
                raise TianmoucCameraError("Unable to start Tianmouc data transfer")
            self._started = True
        return self

    def close(self):
        """Stop transfer and release native camera resources. Idempotent."""
        if self._closed:
            return
        handle = self.cameraHandle
        self.cameraHandle = ctypes.c_uint64(0)
        self._closed = True
        self._started = False
        if handle.value:
            self.tm_sdk.tmCameraUninit(handle)

    def __enter__(self):
        try:
            return self.start()
        except Exception:
            self.close()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _cache_numpy_views(self):
        if not all(
            (self.dataPack.td_p, self.dataPack.sdl_p,
             self.dataPack.sdr_p, self.dataPack.rgb_p)
        ):
            raise TianmoucCameraError("Native SDK returned incomplete frame buffers")

        rod_shape = (ROD_H, ROD_W)
        self._rod_views = (
            np.ctypeslib.as_array(
                self.dataPack.td_p, shape=(ROD_H * ROD_W,)
            ).view(np.int8).reshape(rod_shape),
            np.ctypeslib.as_array(
                self.dataPack.sdl_p, shape=(ROD_H * ROD_W,)
            ).view(np.int8).reshape(rod_shape),
            np.ctypeslib.as_array(
                self.dataPack.sdr_p, shape=(ROD_H * ROD_W,)
            ).view(np.int8).reshape(rod_shape),
        )
        self._cone_view = np.ctypeslib.as_array(
            self.dataPack.rgb_p, shape=(CONE_H * CONE_W * 2 * 3,)
        ).reshape(CONE_H, CONE_W * 2, 3)

    def read(self, copy=False):
        """Read one frame; optionally return an owning copy for async use."""
        if self._closed:
            raise TianmoucCameraError("Camera is closed")
        if not self._started:
            self.start()

        get_frame = (
            self.tm_sdk.tmGetLatestFrame if self._latest else self.tm_sdk.tmGetFrame
        )
        if not get_frame(self.cameraHandle, ctypes.byref(self.dataPack)):
            return None
        if self._rod_views is None:
            self._cache_numpy_views()

        timestamp = int(self.dataPack.timestamp)
        if self.dataPack.isRod == 1:
            data = self._rod_views
            if copy:
                data = tuple(channel.copy() for channel in data)
            return 1, timestamp, data

        data = self._cone_view.copy() if copy else self._cone_view
        return 0, timestamp, data

    def render_rod_preview(self, copy=False):
        """Render the current ROD frame through the SDK's fast C++ preview path."""
        if self._closed:
            raise TianmoucCameraError("Camera is closed")
        render = getattr(self.tm_sdk, "tmRenderRodPreview", None)
        if render is None:
            raise TianmoucCameraError(
                "The selected native library does not support fast ROD preview"
            )
        pointer = self._rod_preview.ctypes.data_as(
            ctypes.POINTER(ctypes.c_ubyte)
        )
        if not render(self.cameraHandle, pointer, self._rod_preview.nbytes):
            raise TianmoucCameraError("Unable to render ROD preview")
        return self._rod_preview.copy() if copy else self._rod_preview

    def render_cone_preview(self, copy=False):
        """Return an 8-bit BGR preview of the current decoded CONE frame."""
        if self._closed:
            raise TianmoucCameraError("Camera is closed")
        render = getattr(self.tm_sdk, "tmRenderConePreview", None)
        if render is None:
            raise TianmoucCameraError(
                "The selected native library does not support fast CONE preview"
            )
        pointer = self._cone_preview.ctypes.data_as(
            ctypes.POINTER(ctypes.c_ubyte)
        )
        if not render(self.cameraHandle, pointer, self._cone_preview.nbytes):
            raise TianmoucCameraError("Unable to render CONE preview")
        return self._cone_preview.copy() if copy else self._cone_preview

    def __call__(self, copy=False):
        return self.read(copy=copy)
