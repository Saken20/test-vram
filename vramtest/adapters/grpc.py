"""Any gRPC method, described by a .proto compiled at runtime (no generated code in the repo).

    adapter: grpc
    address: localhost:50051
    proto: examples/protos/echo.proto       # + include_dirs: [...]
    method: echo.Echo/ClientStream          # "<package>.<Service>/<Method>"; streaming shape from the descriptor
    request: {text: "{text}", meta: {id: "{uuid}"}}   # template -> json_format.ParseDict
                                            # a list of mappings = several messages (client streaming)
    stream_audio: {field: chunk, chunk_bytes: 15360, realtime: false}
                                            # client streaming: {audio} split into messages (template + chunk)
    response: {text: reply, audio: audio, ok: ok}   # dotted field paths, concatenated over stream messages
    metadata: {authorization: "Bearer ..."}

Timing: for client-streamed requests ``t_busy`` is set after the last message is handed to gRPC
(with ``realtime: true`` chunks are paced at audio speed, like a live microphone).
"""
from __future__ import annotations

import asyncio
import copy

import grpc
from google.protobuf import json_format

from .. import audio as audio_util
from .. import protos
from ..config import resolve
from ..templating import drop_none, jsonable, placeholders, render
from .base import Adapter, CallState, Payload, Result, ServiceError

CHANNEL_OPTIONS = [
    ("grpc.enable_http_proxy", 0),  # local services: ignore HTTP(S)_PROXY from the environment
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
]


def set_path(d: dict, path: str, value) -> None:
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def get_field(msg, path: str):
    """Dotted attribute path on a protobuf message; repeated scalars are joined."""
    cur = msg
    for part in path.split("."):
        if cur is None:
            return None
        if hasattr(cur, "DESCRIPTOR") and part not in cur.DESCRIPTOR.fields_by_name:
            raise ServiceError("BAD_RESPONSE_PATH", f"{cur.DESCRIPTOR.full_name} has no field {part!r}")
        cur = getattr(cur, part)
    if isinstance(cur, (str, bytes, bool, int, float)) or cur is None:
        return cur
    try:  # repeated scalar container
        items = list(cur)
    except TypeError:
        return cur
    if items and isinstance(items[0], bytes):
        return b"".join(items)
    if items and isinstance(items[0], str):
        return "".join(items)
    return items


class GrpcAdapter(Adapter):
    def __init__(self, name, cfg, recorder=None, tokens=None, **kw):
        super().__init__(name, cfg, recorder, tokens, **kw)
        self.address = cfg["address"]
        proto = protos.load(resolve(cfg["proto"]), [resolve(d) for d in cfg.get("include_dirs") or []])
        self.method = proto.method(cfg["method"])
        self.response = cfg.get("response") or {}
        self.stream_audio = cfg.get("stream_audio") if self.method.client_streaming else None
        self._channel = None
        self._stub = None

    @property
    def default_input(self) -> str:
        # stream_audio only matters for client-streaming methods (self.stream_audio is None otherwise)
        used = placeholders(self.templates())
        if self.stream_audio or used & {"audio", "audio_wav"}:
            return "audio"
        return "messages" if "messages" in used else "text"

    @property
    def shape(self) -> str:
        return self.method.shape

    def _stub_for_channel(self):
        if self._channel is None:  # created lazily inside the running event loop
            self._channel = grpc.aio.insecure_channel(self.address, options=CHANNEL_OPTIONS)
            m = self.method
            factory = getattr(self._channel, m.shape)
            self._stub = factory(m.path, request_serializer=m.request_cls.SerializeToString,
                                 response_deserializer=m.response_cls.FromString)
        return self._stub

    async def close(self):
        if self._channel is not None:
            await self._channel.close()
            self._channel = None

    def error_code(self, exc):
        if isinstance(exc, grpc.aio.AioRpcError):
            return exc.code().name, (exc.details() or "")[:300]
        return super().error_code(exc)

    # ------------------------------------------------------------ requests
    def _message(self, data: dict):
        msg = self.method.request_cls()
        try:
            json_format.ParseDict(jsonable(drop_none(data)), msg)
        except json_format.ParseError as exc:
            raise ServiceError("BAD_REQUEST_TEMPLATE", str(exc)) from None
        return msg

    def _messages(self, payload: Payload) -> list:
        v = self.variables(payload)
        rendered = render(self.cfg.get("request") or {}, v)
        if self.stream_audio and payload.audio is not None:
            sa = self.stream_audio
            base = rendered[0] if isinstance(rendered, list) else rendered
            step = int(sa["chunk_bytes"])
            out = []
            for i in range(0, len(payload.audio), step):
                d = copy.deepcopy(base)
                set_path(d, sa["field"], payload.audio[i:i + step])
                out.append(self._message(d))
            return out or [self._message(base)]
        if isinstance(rendered, list):
            return [self._message(d) for d in rendered]
        return [self._message(rendered)]

    async def _request_stream(self, messages, st: CallState):
        sa = self.stream_audio
        pace = 0.0
        if sa and sa.get("realtime"):
            fmt, sr = self.audio_in["format"], self.audio_in["sample_rate"]
            pace = audio_util.seconds(b"\0" * int(sa["chunk_bytes"]), fmt, sr)
        for i, msg in enumerate(messages):
            if pace and i:
                await asyncio.sleep(pace)
            yield msg
        st.busy()

    # ------------------------------------------------------------ call
    async def _call(self, payload: Payload, st: CallState) -> Result:
        messages = self._messages(payload)
        stub = self._stub_for_channel()
        metadata = tuple((str(k).lower(), str(v)) for k, v in
                         drop_none(render(self.cfg.get("metadata") or {}, self.variables(payload))).items())
        kw = {"timeout": self.timeout_s, "metadata": metadata or None}
        m = self.method
        responses = []
        if m.client_streaming:
            request = self._request_stream(messages, st)
        else:
            request = messages[0]
            st.busy()
        if m.server_streaming:
            async for resp in stub(request, **kw):
                st.first()
                responses.append(resp)
        else:
            responses.append(await stub(request, **kw))
        st.extra["messages_out"] = len(messages)
        st.extra["messages_in"] = len(responses)
        return self._collect(responses)

    def _collect(self, responses) -> Result:
        text_parts, audio = [], bytearray()
        for resp in responses:
            if self.response.get("ok") and not get_field(resp, self.response["ok"]):
                raise ServiceError("STATUS_FALSE", f"response field {self.response['ok']!r} is false")
            if self.response.get("text"):
                value = get_field(resp, self.response["text"])
                if value:
                    text_parts.append(value if isinstance(value, str) else str(value))
            if self.response.get("audio"):
                value = get_field(resp, self.response["audio"])
                if value:
                    audio.extend(value)
        text = "".join(text_parts) if self.response.get("text") else None
        audio_b = bytes(audio) if audio else None
        audio_s = audio_util.output_seconds(audio_b, self.audio_out) if audio_b else None
        return Result(ok=True, text=text, audio=audio_b, audio_seconds=audio_s)
