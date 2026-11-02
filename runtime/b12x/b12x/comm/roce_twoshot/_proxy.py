"""Build and bind the RDMA proxy shared library (``_roce_proxy.c``) of the two-shot copy.

Same as ``b12x.comm.roce._proxy`` except the ABI (103: the region also holds the
two-shot pipe, ctrl grows to 4096 bytes) and the cached library name, so this
package and the stock one never load each other's proxy.

The proxy is plain C over libibverbs.  It is compiled once per source hash
with the host C compiler into the b12x cache directory, so the package stays
pure Python for packaging purposes while the RDMA posting loop runs without
the interpreter on its critical path.

ds41 idle parking: ``IDLE_PARK`` (False here; adapter/roce_idle_park.py sets it under
DSV41_ROCE_IDLE_PARK=1 before the first ``load``) builds the same source with
``-DROCE_IDLE_PARK=1`` and ``_roce_idle_park.h`` into a separately named library and binds
``roce_park_*``.  With ``IDLE_PARK`` False the build command, the cache name and the
bindings are the v2.6 ones.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import shutil
import subprocess
import threading
from pathlib import Path

_SOURCE = Path(__file__).with_name("_roce_proxy.c")
_PARK_HEADER = Path(__file__).with_name("_roce_idle_park.h")
# Set by adapter/roce_idle_park.py before the first load(); never changed after it.
IDLE_PARK = False
_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
ABI_VERSION = 103
PARK_ABI_VERSION = 1
# roce_park_stat indices (_roce_idle_park.h ROCE_PARK_STAT_*)
PARK_STATS = ("parks", "wakes", "timeouts", "late_doorbells", "parked_ns", "last_wake_ns",
              "max_wake_ns", "last_lead_ns", "tid", "parked", "allow", "after_ns", "poll_ns")

BLOB_STRUCT_ERR = "roce proxy blob size mismatch"


def _cache_dir() -> Path:
    """Directory for the compiled proxy library: ``B12X_ROCE_CACHE_DIR``, else ``<XDG cache>/b12x/roce``."""
    override = os.getenv("B12X_ROCE_CACHE_DIR")
    if override:
        return Path(override)
    root = os.getenv("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return Path(root) / "b12x" / "roce"


def _compiler() -> str:
    """Host C compiler used to build the proxy (``CC``, gcc, cc, or clang)."""
    for candidate in (os.getenv("CC"), "gcc", "cc", "clang"):
        if candidate and shutil.which(candidate):
            return candidate
    raise RuntimeError(
        "b12x.comm.roce needs a C compiler (gcc/cc/clang) and libibverbs headers "
        "to build its RDMA proxy"
    )


def _build() -> Path:
    """Compile the proxy source into the cache directory keyed by source hash; returns the .so path."""
    source = _SOURCE.read_bytes()
    park = bool(IDLE_PARK)
    if park:
        # the header and the define are part of what is built: key the cache on them too
        source = source + b"\0-DROCE_IDLE_PARK=1\0" + _PARK_HEADER.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    if park:
        target = cache / f"roce_twoshot_proxy-park-{digest}.so"
    else:
        target = cache / f"roce_twoshot_proxy-{digest}.so"
    if target.exists():
        return target
    tmp = cache / f".{target.stem}-{os.getpid()}.so"
    cmd = [
        _compiler(),
        "-O2",
        "-std=gnu11",
        "-shared",
        "-fPIC",
        "-o",
        str(tmp),
        str(_SOURCE),
        "-libverbs",
        "-lpthread",
    ]
    if park:
        cmd.insert(1, "-DROCE_IDLE_PARK=1")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "failed to build the b12x RoCE proxy: " + " ".join(cmd) + "\n" + proc.stderr
        )
    os.replace(tmp, target)
    return target


def load() -> ctypes.CDLL:
    """Return the bound proxy library, building it on first use."""

    global _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        lib = ctypes.CDLL(str(_build()), use_errno=True)
        u64 = ctypes.c_uint64
        p = ctypes.c_void_p
        lib.roce_abi_version.restype = ctypes.c_int
        lib.roce_abi_version.argtypes = []
        lib.roce_layout.restype = ctypes.c_int
        lib.roce_layout.argtypes = [ctypes.c_int, u64, u64, ctypes.POINTER(u64)]
        lib.roce_blob_bytes.restype = u64
        lib.roce_blob_bytes.argtypes = []
        lib.roce_create.restype = p
        lib.roce_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_int,
            p,
            u64,
            u64,
            u64,
            ctypes.c_char_p,
            u64,
        ]
        lib.roce_local_blob.restype = ctypes.c_int
        lib.roce_local_blob.argtypes = [p, p, u64]
        lib.roce_connect.restype = ctypes.c_int
        lib.roce_connect.argtypes = [p, p, u64]
        lib.roce_start.restype = ctypes.c_int
        lib.roce_start.argtypes = [p]
        lib.roce_stop.restype = None
        lib.roce_stop.argtypes = [p]
        lib.roce_failed.restype = ctypes.c_int
        lib.roce_failed.argtypes = [p]
        lib.roce_error.restype = ctypes.c_char_p
        lib.roce_error.argtypes = [p]
        lib.roce_stat.restype = u64
        lib.roce_stat.argtypes = [p, ctypes.c_int]
        lib.roce_hca_stat.restype = u64
        lib.roce_hca_stat.argtypes = [p, ctypes.c_int, ctypes.c_int]
        lib.roce_destroy.restype = None
        lib.roce_destroy.argtypes = [p]
        if lib.roce_abi_version() != ABI_VERSION:
            raise RuntimeError("unexpected b12x RoCE two-shot proxy ABI version")
        if IDLE_PARK:
            lib.roce_park_abi.restype = ctypes.c_int
            lib.roce_park_abi.argtypes = []
            lib.roce_park_configure.restype = ctypes.c_int
            lib.roce_park_configure.argtypes = [p, ctypes.c_uint32, ctypes.c_uint32]
            lib.roce_park_allow.restype = None
            lib.roce_park_allow.argtypes = [p]
            lib.roce_park_wake.restype = None
            lib.roce_park_wake.argtypes = [p]
            lib.roce_park_stat.restype = u64
            lib.roce_park_stat.argtypes = [p, ctypes.c_int]
            if lib.roce_park_abi() != PARK_ABI_VERSION:
                raise RuntimeError("unexpected b12x RoCE idle-park ABI version")
        _LIB = lib
        return lib


class Layout:
    """Byte offsets of the pinned region shared by the kernel and the proxy."""

    __slots__ = (
        "recv_off",
        "flag_off",
        "send_off",
        "ctrl_off",
        "total_bytes",
        "flag_stride",
        "slots",
        "pipe_send_off",
        "pipe_recv_off",
        "pipe_flag_off",
        "shard_cap",
        "max_chunks",
        "pipe_bytes",
    )

    def __init__(self, world_size: int, slot_bytes: int, pipe_bytes: int = 0) -> None:
        """Ask the proxy library for the region layout of this geometry."""
        out = (ctypes.c_uint64 * 12)()
        if load().roce_layout(int(world_size), int(slot_bytes), int(pipe_bytes), out) != 0:
            raise ValueError(
                f"unsupported RoCE geometry: world_size={world_size} slot_bytes={slot_bytes} "
                f"pipe_bytes={pipe_bytes} (2..16 ranks, slot_bytes a positive multiple of "
                "4096, pipe_bytes a multiple of 4096)"
            )
        (
            self.recv_off,
            self.flag_off,
            self.send_off,
            self.ctrl_off,
            self.total_bytes,
            self.flag_stride,
            self.slots,
            self.pipe_send_off,
            self.pipe_recv_off,
            self.pipe_flag_off,
            self.shard_cap,
            self.max_chunks,
        ) = (int(v) for v in out)
        self.pipe_bytes = int(pipe_bytes)


class Proxy:
    """Owns one RDMA proxy context for one rank."""

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        hca_names: tuple[str, ...],
        gid_index: int,
        region_ptr: int,
        region_bytes: int,
        slot_bytes: int,
        pipe_bytes: int = 0,
    ) -> None:
        """Create the proxy context: open the HCAs, register the pinned region, create the queue pairs."""
        self._lib = load()
        names = (ctypes.c_char_p * len(hca_names))(*[n.encode() for n in hca_names])
        err = ctypes.create_string_buffer(512)
        self._ctx = self._lib.roce_create(
            int(world_size),
            int(rank),
            names,
            len(hca_names),
            int(gid_index),
            ctypes.c_void_p(int(region_ptr)),
            int(region_bytes),
            int(slot_bytes),
            int(pipe_bytes),
            err,
            len(err),
        )
        if not self._ctx:
            raise RuntimeError(
                f"RoCE proxy setup failed: {err.value.decode(errors='replace')}"
            )
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.hca_names = tuple(hca_names)

    def local_blob(self) -> bytes:
        """Serialized connection record (region address, keys, queue-pair numbers) to send to every peer."""
        n = int(self._lib.roce_blob_bytes())
        buf = ctypes.create_string_buffer(n)
        if self._lib.roce_local_blob(self._ctx, buf, n) != 0:
            raise RuntimeError(BLOB_STRUCT_ERR)
        return buf.raw

    def connect(self, blobs: list[bytes]) -> None:
        """Connect the queue pairs from every rank's ``local_blob``, in rank order."""
        n = int(self._lib.roce_blob_bytes())
        if len(blobs) != self.world_size or any(len(b) != n for b in blobs):
            raise RuntimeError(BLOB_STRUCT_ERR)
        joined = b"".join(blobs)
        buf = ctypes.create_string_buffer(joined, len(joined))
        if self._lib.roce_connect(self._ctx, buf, len(joined)) != 0:
            raise RuntimeError(f"RoCE queue-pair connect failed: {self.error()}")

    def start(self) -> None:
        """Start the proxy thread; a restart resumes from the last posted sequence."""
        if self._lib.roce_start(self._ctx) != 0:
            raise RuntimeError(f"RoCE proxy thread failed to start: {self.error()}")

    def stop(self) -> None:
        """Stop the proxy thread; ``start`` resumes from the last posted op."""
        self._lib.roce_stop(self._ctx)

    def failed(self) -> bool:
        """True once the proxy thread has stopped on an error."""
        return bool(self._lib.roce_failed(self._ctx))

    def error(self) -> str:
        """The proxy's last error message, empty when none."""
        raw = self._lib.roce_error(self._ctx)
        return raw.decode(errors="replace") if raw else ""

    def stats(self) -> dict[str, int | list[int]]:
        """Counters: ops posted, RDMA writes completed, last posted sequence."""
        return {
            "ops_posted": int(self._lib.roce_stat(self._ctx, 0)),
            "writes_completed": int(self._lib.roce_stat(self._ctx, 1)),
            "last_seq": int(self._lib.roce_stat(self._ctx, 2)),
            "last_twoshot_seq": int(self._lib.roce_stat(self._ctx, 3)),
            "writes_completed_per_hca": [
                int(self._lib.roce_hca_stat(self._ctx, hca, 0))
                for hca in range(len(self.hca_names))
            ],
            "bytes_posted_per_hca": [
                int(self._lib.roce_hca_stat(self._ctx, hca, 1))
                for hca in range(len(self.hca_names))
            ],
        }

    # -- idle parking (only in a library built with IDLE_PARK) -------------------

    def park_configure(self, after_ms: int, poll_ms: int) -> None:
        """Let the proxy park once the engine said idle and no doorbell came for ``after_ms``."""
        if self._lib.roce_park_configure(self._ctx, int(after_ms), int(poll_ms)) != 0:
            raise ValueError(f"invalid idle-park configuration after_ms={after_ms} poll_ms={poll_ms}")

    def park_allow(self) -> None:
        """The engine is idle: the proxy may park after ``after_ms`` without a doorbell."""
        self._lib.roce_park_allow(self._ctx)

    def park_wake(self) -> None:
        """Work is coming: clear the idle mark and wake a parked proxy (one futex wake)."""
        self._lib.roce_park_wake(self._ctx)

    def park_stats(self) -> dict[str, int]:
        """Idle-park counters (``PARK_STATS``)."""
        return {name: int(self._lib.roce_park_stat(self._ctx, i)) for i, name in enumerate(PARK_STATS)}

    def close(self) -> None:
        """Stop the thread and release the RDMA resources."""
        ctx, self._ctx = self._ctx, None
        if ctx:
            self._lib.roce_destroy(ctx)

    def __del__(self) -> None:  # pragma: no cover - defensive teardown
        """Release the RDMA resources if ``close`` was never called."""
        with contextlib.suppress(Exception):
            self.close()


__all__ = ["Layout", "Proxy", "load"]
