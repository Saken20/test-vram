"""Compile any .proto at runtime and look up methods / message classes via descriptors.

``grpc_tools.protoc`` writes a descriptor set (with imports) into ``vramtest/_gen/<hash>/`` where the hash
is derived from the proto path and include dirs; it is rebuilt when the .proto is newer. No generated
Python modules are imported, so several protos (even with clashing file names) can be used side by side.

    proto = protos.load("examples/protos/echo.proto")
    m = proto.method("echo.Echo/Unary")      # -> MethodInfo
    req = m.request_cls(text="hi")
"""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gen")

_LOCK = threading.Lock()
_CACHE: dict = {}


class ProtoError(RuntimeError):
    pass


@dataclass
class MethodInfo:
    path: str                 # "/pkg.Service/Method" (gRPC wire path)
    request_cls: type
    response_cls: type
    client_streaming: bool
    server_streaming: bool

    @property
    def shape(self) -> str:
        return {(False, False): "unary_unary", (False, True): "unary_stream",
                (True, False): "stream_unary", (True, True): "stream_stream"}[
            (self.client_streaming, self.server_streaming)]


class Proto:
    """Descriptors of one compiled .proto (and its imports)."""

    def __init__(self, pool: descriptor_pool.DescriptorPool, files: list[str]):
        self.pool = pool
        self.files = files

    def message_class(self, full_name: str) -> type:
        return message_factory.GetMessageClass(self.pool.FindMessageTypeByName(full_name))

    def services(self) -> list[str]:
        names = []
        for fname in self.files:
            fd = self.pool.FindFileByName(fname)
            names += [s.full_name for s in fd.services_by_name.values()]
        return names

    def method(self, spec: str) -> MethodInfo:
        """``"pkg.Service/Method"`` -> :class:`MethodInfo`."""
        service_name, _, method_name = spec.lstrip("/").partition("/")
        try:
            service = self.pool.FindServiceByName(service_name)
        except KeyError:
            raise ProtoError(f"service {service_name!r} not found (available: {', '.join(self.services()) or 'none'})"
                             ) from None
        method = service.methods_by_name.get(method_name)
        if method is None:
            raise ProtoError(f"method {method_name!r} not found in {service_name} "
                             f"(available: {', '.join(service.methods_by_name)})")
        return MethodInfo(
            path=f"/{service.full_name}/{method.name}",
            request_cls=message_factory.GetMessageClass(method.input_type),
            response_cls=message_factory.GetMessageClass(method.output_type),
            client_streaming=method.client_streaming,
            server_streaming=method.server_streaming,
        )


def _cache_dir(proto_path: str, include_dirs) -> str:
    key = "\n".join([proto_path, *include_dirs]).encode()
    return os.path.join(GEN_DIR, hashlib.sha256(key).hexdigest()[:16])


def compile_proto(proto_path: str, include_dirs=()) -> str:
    """Run protoc if needed; returns the path of the descriptor set file."""
    import grpc_tools
    from grpc_tools import protoc

    proto_path = os.path.abspath(proto_path)
    if not os.path.exists(proto_path):
        raise ProtoError(f"proto file not found: {proto_path}")
    include_dirs = [os.path.abspath(d) for d in include_dirs]
    out_dir = _cache_dir(proto_path, include_dirs)
    out = os.path.join(out_dir, "descriptor.pb")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(proto_path):
        return out
    os.makedirs(out_dir, exist_ok=True)
    well_known = os.path.join(os.path.dirname(grpc_tools.__file__), "_proto")
    tmp = f"{out}.{os.getpid()}.{threading.get_ident()}.tmp"
    args = ["grpc_tools.protoc", f"-I{os.path.dirname(proto_path)}",
            *[f"-I{d}" for d in include_dirs], f"-I{well_known}",
            "--include_imports", f"--descriptor_set_out={tmp}", proto_path]
    rc = protoc.main(args)
    if rc != 0 or not os.path.exists(tmp):
        raise ProtoError(f"protoc failed (exit code {rc}) for {proto_path}; see the messages above")
    os.replace(tmp, out)
    return out


def load(proto_path: str, include_dirs=()) -> Proto:
    """Compile (cached) and load a .proto into a private descriptor pool."""
    key = (os.path.abspath(proto_path), tuple(os.path.abspath(d) for d in include_dirs))
    with _LOCK:
        desc_path = compile_proto(proto_path, include_dirs)
        mtime = os.path.getmtime(desc_path)
        cached = _CACHE.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        fds = descriptor_pb2.FileDescriptorSet()
        with open(desc_path, "rb") as f:
            fds.ParseFromString(f.read())
        pool = descriptor_pool.DescriptorPool()
        for fd in fds.file:
            pool.Add(fd)
        proto = Proto(pool, [fd.name for fd in fds.file])
        _CACHE[key] = (mtime, proto)
        return proto
