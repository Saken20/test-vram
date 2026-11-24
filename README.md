# vramtest

English | [Русский](README_RU.md)

**How much GPU memory do your inference services really need, and at what load do they break?**

`vramtest` is a load tester for GPU inference services: LLMs, speech recognition, text-to-speech,
embeddings, or any custom HTTP / gRPC model server. It sends realistic load (single services or whole
multi-step pipelines such as ASR → LLM → TTS), samples the VRAM of each service's GPU processes ten times
per second, reads vLLM's Prometheus metrics, and produces a report: the memory each service needs
for a given number of users, latency percentiles, the concurrency at which each service saturates,
and how many servers a target load requires.

Everything is configured in one YAML file: any number of services, any names, any models.

## Why: nvidia-smi does not tell you what a vLLM model needs

vLLM allocates `--gpu-memory-utilization × GPU memory` at startup and turns almost everything above
the model weights into a **KV-cache pool**. `nvidia-smi` therefore shows the same large number when the
server is idle and when it is under full load. That number says how much memory you *gave* vLLM, not how
much your workload *uses*. You cannot tell from it whether two models fit on one card, how far you can
lower `--gpu-memory-utilization`, or how many concurrent users a server handles.

`vramtest` splits the number into parts:

```
allocated (NVML)  =  model memory (weights, CUDA graphs, activations)  +  KV pool (reserved, mostly empty)
really needed     =  model memory  +  KV cache occupied at peak load × (1 + safety margin)
```

and measures the occupied part under the load you define. For services that do not reserve memory up front
(most TTS servers, custom PyTorch services) the VRAM measured by NVML at peak is what they really use, and
the tool reports it as such.

## Features

- **Adapters** for the OpenAI-compatible API (chat, completions, transcription, speech, embeddings: vLLM,
  SGLang, llama.cpp server, Kokoro-FastAPI, ...), any **gRPC** method from a `.proto` file compiled at runtime
  (unary, server-, client- and bidirectional streaming, audio streamed in chunks), and any **HTTP** JSON or
  multipart endpoint.
- **Pipelines**: one user turn can pass through several services (e.g. audio → ASR → LLM → TTS), with
  per-step latency, time to first token/chunk and the user's "wait for the first output".
- **Scenarios**: idle, simultaneous bursts, back-to-back concurrency, realistic sessions with think time,
  and a ramp that raises concurrency until a service saturates (errors, crash or latency limits) and
  bisects the limit.
- **Memory model** for vLLM: base memory vs KV pool, KV bytes per token from the model's `config.json`,
  occupied KV from `/metrics`, prefix caching of a shared system prompt, `max_model_len` floor.
- **Capacity extrapolation** to your own targets ("1,000 concurrent users"): memory needed and number of
  servers, by memory and by throughput.
- **Reports**: raw JSON per scenario, `report.md` + `report.json`, and a plain-language PDF summary.
- **Process discovery**: finds each service's GPU processes from its port (Linux `/proc`), so separate
  services sharing one GPU are measured separately.
- Runs without a GPU (latency and load only), and the test suite runs on mock servers without a GPU.

## Install

Python 3.10+ on Linux with an NVIDIA driver (NVML). Install it next to the services, in its own environment,
so it does not touch the services' environments.

```bash
git clone https://github.com/Saken20/test-vram && cd test-vram

# pip + venv
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# or conda
conda env create -f environment.yml && conda activate vramtest
```

Optional: `pip install transformers` for exact token counts with the model's tokenizer (see `tokenizer`).
Update after `git pull`: `pip install -r requirements.txt` or `conda env update -f environment.yml --prune`.

## Quick start

```bash
cp examples/vllm-chat.yaml config.yaml     # pick the example closest to your setup and edit it
python -m vramtest validate                # parse the config, print services and pipeline
python -m vramtest probe                   # one call per service; GPU processes, vLLM /metrics, KV bytes/token
python -m vramtest all                     # idle, burst, concurrency, sessions, ramp, report, pdf
#   --quick       short durations, to check that everything works first
#   --skip-ramp   skip ramp (it pushes services until they fail and may crash one)
python -m vramtest report                  # (re)build results/default/report.md + report.json
python -m vramtest pdf                     # results/default/summary.pdf
```

Examples:

| file | what it shows |
|---|---|
| [`examples/vllm-chat.yaml`](examples/vllm-chat.yaml) | one chat model on vLLM (Qwen/Qwen2.5-7B-Instruct), capacity in users |
| [`examples/voice-pipeline.yaml`](examples/voice-pipeline.yaml) | ASR → LLM → TTS: vLLM Whisper, vLLM chat, Kokoro-FastAPI; spoken sessions with think time |
| [`examples/grpc-service.yaml`](examples/grpc-service.yaml) | custom gRPC service from [`examples/protos/echo.proto`](examples/protos/echo.proto): unary, server streaming, client-streamed audio |
| [`examples/http-service.yaml`](examples/http-service.yaml) | custom JSON HTTP endpoints (TGI `/generate`, TEI `/embed`), multipart upload in comments |

### Try it without a GPU

The mock servers used by the tests can be started by hand:

```bash
python -m tests.mocks.openai_mock 8000      # OpenAI-compatible server with vLLM-style /metrics
python -m vramtest --config examples/vllm-chat.yaml --quick --run demo all

python -m tests.mocks.grpc_mock 50051       # examples/protos/echo.proto
python -m vramtest --config examples/grpc-service.yaml --quick --run demo-grpc burst
```

Without NVML the tool warns that VRAM is not measured and still records load and latency.

## Commands

```
python -m vramtest [--config PATH] [--run NAME] [--quick] <command> [options]
```

| command | what it does |
|---|---|
| `validate` | parse the config, compile `.proto` files, load the dataset; print services and pipeline. Exit code 2 on config errors, 1 on other problems |
| `probe` | one call per service and one pipeline turn; GPU processes found, vLLM `/metrics` found, KV bytes/token. Writes `probe.json` |
| `gen-audio [--service NAME]` | synthesize the dataset prompts into `dataset.audio_dir/*.wav` with a speech-capable service (default: the first one) |
| `idle` | VRAM with no requests |
| `burst [--levels 1,10,100] [--rounds N] [--service NAME]` | N turns started at the same instant |
| `concurrency [--users N] [--duration S] [--service NAME]` | N workers sending turns back-to-back |
| `sessions [--profile NAME\|all] [--duration S]` | K concurrent sessions with think time; profiles from the config (default: all) |
| `ramp [--service NAME\|pipeline\|all] [--levels ...] [--step S]` | raise concurrency until saturation (default: every service separately) |
| `report` | `results/<run>/report.md` + `report.json` |
| `pdf` | `results/<run>/summary.pdf` (builds the report first) |
| `all [--skip-ramp]` | idle, burst, concurrency, sessions (all profiles), ramp (every service), report, pdf |

- `--config PATH`: the YAML config, default `./config.yaml`.
- `--run NAME`: results go to `results/<NAME>/` under the current directory (default `default`). Use one
  run per setup, e.g. to compare two `--gpu-memory-utilization` values.
- `--quick`: idle 10 s, one burst round, 40 s concurrency/sessions, 15 s ramp steps without bisection.
- `--service` for `burst` / `concurrency`: a service name or `pipeline`. By default the pipeline is loaded
  when one is defined, otherwise each service separately.
- Global flags work before or after the command.

## Configuration reference

A config has six sections: `gpu`, `services`, `pipeline`, `dataset`, `scenarios`, `capacity`. Only
`services` is required; everything else has defaults. Relative paths are looked up in the current
directory, then next to the config file, then in the repository root. Unknown keys produce a warning.

### `gpu`

```yaml
gpu: {index: 0, sample_hz: 10, metrics_poll_s: 1.0}   # NVML GPU index, VRAM samples/s, vLLM /metrics poll period
```

### `services`

Any number of services under any names (letters, digits, `_`, `-`, `.`). Keys common to all adapters:

| key | default | meaning |
|---|---|---|
| `adapter` | required | `openai`, `grpc` or `http` |
| `model` | `""` | model name sent in requests (openai: empty = first id of `GET /v1/models`); also the `{model}` placeholder |
| `request` | `{}` | openai: fields merged into the request body; grpc: the request message template |
| `timeout_s` | `120` | per-call timeout |
| `limits` | `{max_p95_latency_s: 20, max_p95_ttfb_s: 3}` | ramp counts the service as saturated above these p95 values |
| `port` | from `url`/`address` | port used to find the service's processes |
| `pids` | `[]` | explicit GPU process ids (override discovery) |
| `cmdline_match` | `""` | substring of the GPU process command line (e.g. for services in Docker) |
| `engine` | `vllm` for openai (except `speech`), else `generic` | `vllm` enables the KV-aware memory model |
| `metrics_url` | auto | vLLM Prometheus endpoint; auto: `{url}/metrics`, then ports the service's processes listen on |
| `model_config` | `""` | HF repo id (`org/name`), local `config.json` or model directory → KV bytes/token |
| `kv_bytes_per_token` | `null` | explicit KV size per token (overrides `model_config`) |
| `kv_cache_dtype` | `null` | e.g. `fp8` when vLLM runs with `--kv-cache-dtype fp8` |
| `weights_gib` | `null` | from the vLLM log line `Model loading took X GiB` (else: size of `*.safetensors` in a local `model_config` dir) |
| `kv_pool_tokens` | `null` | from the vLLM log line `GPU KV cache size: N tokens`, when `/metrics` is not reachable |
| `overhead_gib` | `1.0` | CUDA graphs / activations added to `weights_gib` when the base memory cannot be measured |
| `max_model_len` | `null` | your `--max-model-len`: the KV pool can never be smaller than one full-length sequence |
| `prefix_caching` | `true` | a shared system prompt occupies the KV cache once, not once per request |
| `tokenizer` | `""` | HF tokenizer id/path for exact token counts (needs `transformers`) |
| `chars_per_token` | `4.0` | token estimate when the response has no `usage` and no tokenizer is set |
| `audio_tokens_per_s` | `0` | KV tokens per second of input audio (audio-LLM style ASR) |
| `audio_out` | `{format: s16le, sample_rate: 24000}` | format of audio the service returns: `s16le`, `f32le`, `pcm` (= s16le), `wav`, or compressed `mp3`/`opus`/`aac`/`flac` (no duration) |

#### `adapter: openai`

```yaml
llm:
  adapter: openai
  url: http://localhost:8000        # server root (a trailing /v1 is accepted)
  kind: chat                        # chat | completions | transcription | speech | embeddings
  model: Qwen/Qwen2.5-7B-Instruct
  request: {max_tokens: 256, temperature: 0.7}
```

| kind | endpoint | input | timing |
|---|---|---|---|
| `chat` | `POST /v1/chat/completions` | chat messages | SSE stream with `stream_options.include_usage`; TTFB = first token |
| `completions` | `POST /v1/completions` | text | same |
| `transcription` | `POST /v1/audio/transcriptions` | audio, uploaded as a WAV file (multipart) | whole response; `request: {stream: true}` uses vLLM's streamed transcription |
| `speech` | `POST /v1/audio/speech` | text | streamed audio body; TTFB = first chunk |
| `embeddings` | `POST /v1/embeddings` | text | whole response |

`request: {stream: false}` disables streaming for chat/completions. Token counts come from the response
`usage` when present.

#### `adapter: grpc`

```yaml
asr:
  adapter: grpc
  address: localhost:50051
  proto: examples/protos/echo.proto     # + include_dirs: [...] for imports
  method: echo.Echo/ClientStream        # "<package>.<Service>/<Method>"; the streaming shape comes from the .proto
  request: {meta: {id: "{uuid}"}}       # template -> protobuf message (JSON mapping); a list = several messages
  stream_audio: {field: chunk, chunk_bytes: 16000, realtime: true}
  response: {text: reply, audio: audio, ok: ok}
  metadata: {authorization: "Bearer <token>"}
```

- The `.proto` is compiled at runtime with `grpcio-tools` into a cache under `vramtest/_gen/`.
- `stream_audio` (client-streaming methods): the turn's audio is split into messages of `chunk_bytes`, each
  being the request template with the chunk in `field`. `realtime: true` paces chunks at audio speed, like a
  live microphone. Latency is measured from the moment the last chunk was sent.
- `response`: dotted field paths. `text` and `audio` are concatenated over all messages of a server stream;
  a falsy `ok` field marks the call as failed (`STATUS_FALSE`). TTFB = first streamed message.

#### `adapter: http`

```yaml
generate:
  adapter: http
  url: http://localhost:8080/generate
  method: POST                          # GET | POST | PUT | PATCH
  json: {inputs: "{text}", parameters: {max_new_tokens: 128}}
  # multipart: {file: "{audio_wav}", response_format: json}   # instead of json; bytes values become files
  headers: {Authorization: "Bearer <token>"}
  stream: false                         # true: read the body in chunks, TTFB = first chunk
  response: {text: generated_text}      # dotted JSON path ("choices.0.text"); "$body" = raw body
```

`response.audio` can be `"$body"` (raw audio bytes) or a JSON field with base64 audio; `response.ok` works
as for gRPC. With `stream: true` a Server-Sent Events response (`text/event-stream`, e.g. TGI
`/generate_stream` with `response: {text: token.text}`) is parsed per `data:` event: the paths apply to every
event, text and audio are concatenated, and TTFB = the first event with output.

#### Templates and placeholders

`request` (openai, grpc), `json`, `multipart`, `headers`, `url` (http) and `metadata` (grpc) are templates:

| placeholder | value |
|---|---|
| `{text}` | the turn's text (prompt, transcript of the clip, or the previous step's text output) |
| `{audio}` | raw audio bytes encoded as `dataset.audio_in` |
| `{audio_wav}` | the same audio as a 16-bit WAV file |
| `{messages}` | chat messages `[{role, content}, ...]` (system prompt + history + text) |
| `{system_prompt}` | `dataset.system_prompt` |
| `{uuid}` | random hex string, new for every call |
| `{session}` | session / worker index |
| `{model}` | the service's `model` |

A value that is exactly one placeholder is replaced by the object itself (bytes, list, number); otherwise
placeholders are formatted into the string. Unknown `{names}` and other braces are left as they are.
Audio bytes in an http `json` body are sent as base64.
Placeholders also decide a service's natural input when it is loaded on its own: audio if it uses `{audio}`/`{audio_wav}`/`stream_audio`, chat messages if it uses `{messages}`,
text otherwise (openai: `chat` → messages, `transcription` → audio, others → text).

### `pipeline`

Optional. One turn passes through the steps in order:

```yaml
pipeline:
  - {service: asr, input: audio}
  - {service: llm, input: prev}
  - {service: tts, input: prev}
```

`input` is `audio`, `text`, `messages` or `prev` (the previous step's output). For a chat step, `prev` text
replaces the last user message, so the system prompt and history are kept. If a step fails, the next `prev`
step gets the turn's original input, so the remaining services still receive load (the turn counts as
failed). Without a pipeline, `burst`, `concurrency` and `ramp` load each service separately with its natural
input (or the one given with `--service`), and a `sessions` turn calls every service once.

### `dataset`

```yaml
dataset:
  texts: data/prompts.txt           # one prompt per line ('#' = comment), or an inline list
  audio_dir: data/audio             # 16-bit PCM WAV clips; an optional <name>.txt next to a clip = its transcript
  audio_in: {format: f32le, sample_rate: 16000}   # f32le | s16le: how {audio} is encoded for the services
  system_prompt: ""                 # inline text or a path to a text file
  history_turns: [0, 2]             # random previous (user, assistant) turns added to chat messages
  seed: 0
```

Clips are resampled to `audio_in.sample_rate`. `gen-audio` fills `audio_dir` with spoken versions of the
prompts (`prompt_NN.wav` + `.txt`) using any service that returns audio. `probe` uses a synthetic 1 s tone
when no clips exist yet. In `sessions`, the chat history is the session's own previous turns.

### `scenarios`

```yaml
scenarios:
  idle: {duration_s: 60}
  burst: {levels: [1, 10, 100], rounds: 3, pause_s: 3}
  concurrency: {users: 10, duration_s: 120}
  sessions:
    turns_per_session: 3
    session_duration_s: 30
    duration_s: 180
    profiles: {normal: 4, peak: 16}     # name -> concurrent sessions
    stagger_start: false                # false: all sessions start their first turn at the same instant
  ramp: {levels: [1, 2, 5, 10, 20, 40, 80, 160], step_duration_s: 60, max_error_rate: 0.05, bisect_steps: 2, cooldown_s: 5}
```

`sessions.stagger_start` (default `false`): with `false` every session starts its first turn at the same
instant, the worst case for the first seconds of the scenario; with `true` each session starts at a random
moment within the first `session_duration_s`, which spreads the load as independent users would.

See [Scenarios](#scenarios) below.

### `capacity`

```yaml
capacity:
  unit: session                         # the word used in the report: user, session, stream, ...
  safety_margin: 0.10                   # added on top of the occupied KV cache
  targets:
    - {label: "1 session", sessions: 1}
    - {label: "peak (20 sessions)", sessions: 20}
    - {label: "500 concurrent sessions", sessions: 500}
```

`sessions` is the number of concurrently active sessions (as in the `sessions` scenario) for which the
report extrapolates memory and servers.

## Memory model

For every `engine: vllm` service:

1. **KV bytes per token** from the model's `config.json` (`model_config`: HF repo id, downloaded once over
   HTTPS and cached in `~/.cache/vramtest/`, or a local file/directory):

   ```
   bytes/token = 2 (K and V) × num_hidden_layers × num_key_value_heads × head_dim × dtype bytes
   ```

   e.g. Qwen/Qwen2.5-7B-Instruct: 2 × 28 × 4 × 128 × 2 (bf16) = 57,344 bytes = 56 KiB per token.
   MLA models (DeepSeek-V2/V3 style) use `layers × (kv_lora_rank + qk_rope_head_dim) × dtype bytes`.
   Multimodal configs are searched for their text decoder (`text_config`, ...). `kv_cache_dtype: fp8` halves
   the result; `kv_bytes_per_token` overrides it. Gated models need `HF_TOKEN` in the environment.
2. **KV pool size** in tokens: from `/metrics` (`vllm:cache_config_info`: `num_gpu_blocks × block_size`),
   else `kv_pool_tokens` from the vLLM log line `GPU KV cache size: N tokens`. Pool GiB = tokens × bytes/token.
3. **Base memory** (weights, CUDA graphs, activations, CUDA context) = idle VRAM of the service's processes −
   KV pool. If the pool is unknown: `weights_gib` (vLLM log line `Model loading took X GiB`, or the size of the
   local safetensors) + `overhead_gib`.
4. **Occupied KV at peak**: `vllm:kv_cache_usage_perc` (`vllm:gpu_cache_usage_perc` on older versions) × pool.
   Without `/metrics`: tokens of the requests in flight (from `usage`, a tokenizer or an estimate) × bytes/token,
   calibrated against `/metrics` when both exist.
5. **Prefix caching**: with `prefix_caching: true` a shared system prompt is counted once, not per request.
6. **Needed** = base + max(occupied KV, `max_model_len` × bytes/token) × (1 + `safety_margin`).

For `engine: generic` services the VRAM measured by NVML is what the service uses; the growth per request in
flight is estimated from the scenarios.

**Extrapolation.** From the session scenarios the report takes, per service, the average number of its
requests in flight per active session (*d*) and the average tokens per request. For *N* sessions the peak
number of concurrent requests is estimated as `N·d + 3·√(N·d)` (Poisson arrivals, +3σ, at most *N*), and the
memory follows from it. The ramp limit gives the throughput side; the number of servers for a target is the
larger of "fits in memory" (at most 95% of the GPU planned) and "within the ramp limit". A service whose GPU
process was never found is assumed to run elsewhere and is left out of the memory totals.

Where the numbers come from (`/metrics`, config, log values, estimates) is stated in the report next to them.

## Scenarios

A **turn** is one pass through the pipeline (or one call of a single service). All scenarios record every
call (start, processing start, first token/chunk, end, status code, tokens, audio seconds) and sample VRAM per
service process throughout.

| scenario | load | answers |
|---|---|---|
| `idle` | no requests for `duration_s` | memory at rest: weights + reserved pool |
| `burst` | N turns started at the same instant, for each level, `rounds` times | latency when N users finish speaking/typing at once |
| `concurrency` | `users` workers, each sending turns back-to-back | throughput and memory at a constant number of requests in flight |
| `sessions` | K sessions per profile; each does `turns_per_session` turns spread over `session_duration_s`, then starts again | realistic load with think time; the basis of the capacity extrapolation |
| `ramp` | levels of concurrency, `step_duration_s` each, until a level fails; then `bisect_steps` bisections | where each service saturates |

A ramp level fails when the error rate exceeds `max_error_rate`, a service's GPU process exits, or the p95
latency / TTFB exceeds the service's `limits`. Ramping a service uses N back-to-back workers (N = requests in
flight); ramping `pipeline` uses N concurrent sessions. Ramp is meant to find the breaking point: be ready to
restart a service.

Latency is measured from the moment the request is fully sent (for client-streamed audio: after the last
chunk) to the end of the response; TTFB to the first token / audio chunk / stream message. For a pipeline
the user's **wait for the first output** = sum of the latencies of all steps but the last + TTFB of the last
step. For audio, the real-time factor RTF = latency / audio seconds. Failed calls keep their code: `HTTP_503`,
gRPC codes such as `RESOURCE_EXHAUSTED`, `TIMEOUT`, `CONNECT_ERROR`, `STATUS_FALSE`, ...

## Output files

Everything goes to `results/<run>/` (default `results/default/`):

| file | content |
|---|---|
| `probe.json` | result of `probe` |
| `idle.json`, `burst_<target>.json`, `concurrency_<target>_<users>.json`, `sessions_<profile>.json`, `ramp_<target>.json` | one file per scenario run: parameters, service info (PIDs, metrics URLs, KV bytes/token), every call record, VRAM samples, per-window summaries |
| `report.md` | human-readable report: memory per service at rest and under each scenario, latency, ramp limits, capacity targets |
| `report.json` | the same numbers, machine-readable |
| `summary.pdf` | short plain-language summary for non-specialists |

Re-running a scenario overwrites its file; `report` reads whatever JSON files the run directory holds.

## Testing without a GPU

```bash
pip install pytest
python -m pytest -q
```

The tests use stdlib mock servers (`tests/mocks/openai_mock.py`: chat with SSE, completions, transcription,
chunked speech, embeddings and vLLM-style `/metrics`; `tests/mocks/grpc_mock.py`: the echo service) and a
fake NVML (`tests/fake_nvml.py`). Every file in `examples/` is validated by `tests/test_examples.py`.

## Limitations

- Linux only for process discovery (`/proc`), NVIDIA only for VRAM (NVML). One GPU per run (`gpu.index`);
  tensor-parallel deployments across several GPUs are not modelled.
- The KV-aware memory model is specific to vLLM. Other engines that also pre-allocate memory (TGI, SGLang,
  TensorRT-LLM, ...) are measured as `generic`, i.e. their reserved memory counts as used.
- `model_config` understands decoder-only configs (including nested multimodal text configs and MLA) and
  encoder-decoder configs such as Whisper (decoder self-attention only: the cross-attention KV is not counted);
  sliding-window and hybrid (attention + state-space) models are approximated as full attention.
- Without `usage` in responses and without a tokenizer, token counts are estimates (`chars_per_token`).
- Audio passed from one step to the next (e.g. TTS output → ASR) is forwarded as is, without format conversion.
  Input WAV files must be 16-bit PCM. Compressed output audio (mp3, opus, ...) has no duration/RTF and cannot
  be used by `gen-audio`.
- The capacity extrapolation assumes independent sessions with the measured behaviour; validate a target by
  running the `sessions` scenario at that level when you can.
- Requests to the services ignore `HTTP(S)_PROXY` (they are expected to be local); only the Hugging Face
  `config.json` download uses the environment's proxy settings.
- The PDF needs a TrueType font such as DejaVu Sans (system fonts, the conda environment, or matplotlib's
  bundled fonts are searched).

## License

[MIT](LICENSE)
