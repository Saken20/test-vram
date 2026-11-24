"""Background VRAM sampler (NVML, per process) + vLLM metrics poller."""
import threading
import time

from . import procinfo, vllm_metrics
from .kv_calc import MIB

try:
    import pynvml
except ImportError:  # nvidia-ml-py is not installed
    pynvml = None


class Nvml:
    """Thin NVML wrapper; without a GPU it works in "no data" mode (``available`` is False)."""

    def __init__(self, index=0):
        self.handle = None
        self.error = None
        if pynvml is None:
            self.error = "nvidia-ml-py is not installed"
            return
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(self.name, bytes):
                self.name = self.name.decode()
        except Exception as exc:  # noqa: BLE001
            self.error = f"NVML unavailable: {exc}"
            self.handle = None

    @property
    def available(self):
        return self.handle is not None

    def memory(self):
        """(used_mib, total_mib)."""
        if not self.available:
            return None, None
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return info.used / MIB, info.total / MIB

    def processes(self):
        """{pid: used_mib | None}."""
        if not self.available:
            return {}
        procs = {}
        for getter in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            try:
                for p in getattr(pynvml, getter)(self.handle):
                    used = p.usedGpuMemory
                    procs[p.pid] = used / MIB if used is not None else None
            except Exception:  # noqa: BLE001
                continue
        return procs


class GpuMonitor:
    """Collects samples: total VRAM, VRAM of each service's processes, in-flight calls, vLLM metrics.

    service_pids: {service: [pid, ...]}; metrics_urls: {service: url}; any service names.
    Each sample: {t, used_mib, total_mib, svc_mib: {svc: MiB|None}, inflight: {svc: n}, vllm: {svc: summary}}.
    ``dead`` = {service: t} for services whose processes disappeared during the run.
    """

    def __init__(self, nvml, service_pids, metrics_urls, recorder=None, sample_hz=10, metrics_poll_s=1.0):
        self.nvml = nvml
        self.service_pids = {k: set(v) for k, v in service_pids.items()}
        self.metrics_urls = {k: v for k, v in metrics_urls.items() if v}
        self.recorder = recorder
        self.period = 1.0 / sample_hz
        self.metrics_poll_s = metrics_poll_s
        self.samples = []
        self.dead = {}
        self._stop = threading.Event()
        self._thread = None
        self._last_vllm = {}
        self._last_metrics_t = 0.0

    def start(self):
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.samples

    def _poll_metrics(self, now):
        if now - self._last_metrics_t < self.metrics_poll_s:
            return
        self._last_metrics_t = now
        for svc, url in self.metrics_urls.items():
            try:
                self._last_vllm[svc] = vllm_metrics.summarize(vllm_metrics.fetch(url))
            except Exception as exc:  # noqa: BLE001
                self._last_vllm[svc] = {"error": str(exc)}

    def sample(self):
        now = time.monotonic()
        used, total = self.nvml.memory()
        procs = self.nvml.processes()
        per_service = {}
        for svc, pids in self.service_pids.items():
            known = [procs[p] for p in pids if p in procs and procs[p] is not None]
            per_service[svc] = sum(known) if known else None
            if pids and not any(procinfo.alive(p) for p in pids) and svc not in self.dead:
                self.dead[svc] = now
        self._poll_metrics(now)
        s = {
            "t": now,
            "used_mib": used,
            "total_mib": total,
            "svc_mib": per_service,
            "inflight": self.recorder.snapshot() if self.recorder else {},
            "vllm": {k: dict(v) for k, v in self._last_vllm.items()},
        }
        self.samples.append(s)
        return s

    def _run(self):
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.sample()
            except Exception as exc:  # noqa: BLE001 — the monitor must never crash
                self.samples.append({"t": started, "error": str(exc)})
            self._stop.wait(max(0.0, self.period - (time.monotonic() - started)))
