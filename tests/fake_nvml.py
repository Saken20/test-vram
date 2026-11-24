"""Fake NVML for tests: same interface as vramtest.gpu_monitor.Nvml, no GPU needed."""
from __future__ import annotations


class FakeNvml:
    """``procs`` = {pid: MiB} or a callable returning it (e.g. growing with a mock's in-flight count)."""

    available = True
    error = None

    def __init__(self, procs=None, total_mib=48 * 1024.0, base_mib=500.0, name="FakeGPU 48GB"):
        self._procs = procs if procs is not None else {}
        self.total_mib = total_mib
        self.base_mib = base_mib
        self.name = name

    def processes(self):
        procs = self._procs() if callable(self._procs) else self._procs
        return dict(procs)

    def memory(self):
        return sum(v for v in self.processes().values() if v) + self.base_mib, self.total_mib
