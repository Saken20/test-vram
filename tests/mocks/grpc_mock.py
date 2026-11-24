"""Mock gRPC server for examples/protos/echo.proto (generic handlers, no generated code).

    server, port = await start_echo(limit=None)   # in a running event loop
    ...
    await server.stop(None)

Behaviour:
  Unary         reply "echo: <text>", audio = 0.1 s of 16-bit PCM (24 kHz), ok = true
  ServerStream  one reply per word of <text> (each with 0.1 s audio)
  ClientStream  reply "received <N> bytes in <M> messages" (sum of `chunk` sizes)
  BidiStream    one reply per request message
A request whose text is "fail" gets ok = false. Over ``limit`` concurrent calls -> RESOURCE_EXHAUSTED.

Run manually: python -m tests.mocks.grpc_mock [port]
"""
from __future__ import annotations

import asyncio
import os
import sys

import grpc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from vramtest import protos  # noqa: E402

ECHO_PROTO = os.path.join(ROOT, "examples", "protos", "echo.proto")
AUDIO = b"\x00\x10" * 2400


class EchoState:
    def __init__(self, limit=None, delay=0.01):
        self.limit = limit
        self.delay = delay
        self.inflight = 0
        self.peak = 0
        self.metadata: list[dict] = []

    async def enter(self, context):
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        self.metadata.append(dict(context.invocation_metadata() or ()))
        if self.limit is not None and self.inflight > self.limit:
            self.inflight -= 1
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, f"more than {self.limit} concurrent calls")

    def leave(self):
        self.inflight -= 1


def _handlers(state: EchoState):
    proto = protos.load(ECHO_PROTO)
    Req = proto.message_class("echo.EchoRequest")
    Rep = proto.message_class("echo.EchoReply")

    def reply(text, req=None, audio=AUDIO):
        rep = Rep(reply=text, audio=audio, ok=not (req is not None and req.text == "fail"))
        if req is not None and req.HasField("meta"):
            rep.meta.CopyFrom(req.meta)
        return rep

    async def unary(request, context):
        await state.enter(context)
        try:
            await asyncio.sleep(state.delay)
            return reply(f"echo: {request.text}", request)
        finally:
            state.leave()

    async def server_stream(request, context):
        await state.enter(context)
        try:
            words = request.text.split() or [""]
            for i, word in enumerate(words):
                await asyncio.sleep(state.delay)
                yield reply(word + (" " if i < len(words) - 1 else ""), request)
        finally:
            state.leave()

    async def client_stream(request_iterator, context):
        size, count, last = 0, 0, None
        async for req in request_iterator:
            size += len(req.chunk)
            count += 1
            last = req
        await state.enter(context)
        try:
            await asyncio.sleep(state.delay)
            return reply(f"received {size} bytes in {count} messages", last, audio=b"")
        finally:
            state.leave()

    async def bidi_stream(request_iterator, context):
        await state.enter(context)
        try:
            async for req in request_iterator:
                await asyncio.sleep(state.delay)
                yield reply(f"[{len(req.chunk) or req.text}]", req)
        finally:
            state.leave()

    kw = {"request_deserializer": Req.FromString, "response_serializer": Rep.SerializeToString}
    return grpc.method_handlers_generic_handler("echo.Echo", {
        "Unary": grpc.unary_unary_rpc_method_handler(unary, **kw),
        "ServerStream": grpc.unary_stream_rpc_method_handler(server_stream, **kw),
        "ClientStream": grpc.stream_unary_rpc_method_handler(client_stream, **kw),
        "BidiStream": grpc.stream_stream_rpc_method_handler(bidi_stream, **kw),
    })


async def start_echo(state: EchoState | None = None, port: int = 0):
    """Start the echo server; returns (server, bound_port). ``server.state`` holds the EchoState."""
    state = state or EchoState()
    server = grpc.aio.server()
    server.add_generic_rpc_handlers((_handlers(state),))
    bound = server.add_insecure_port(f"127.0.0.1:{port}")
    await server.start()
    server.state = state
    return server, bound


async def _main(port):
    server, bound = await start_echo(port=port)
    print(f"mock echo gRPC server on 127.0.0.1:{bound}")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(_main(int(sys.argv[1]) if len(sys.argv) > 1 else 50051))
