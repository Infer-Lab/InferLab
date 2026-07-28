# Fake specialized Engine vertical slice

For the reusable contract and ownership boundary, see
[Specialized Engines](specialized-engine.md). This document describes only the
deterministic fake implementation used by InferLab's own tests.

This is an executable architecture fixture for developing hardware-by-model
specialized Engines. The fake implementation is not a supported or qualified
backend; only the reusable Specialized Engine integration is publishable.

The fixture exercises the routed `single` shape defined by
[RFC-0003](rfc/RFC-0003.md) and the responsibility split recorded by
[ADR-0022](adr/ADR-0022.md):

```text
OpenAI client           SMG Gateway                  fake Engine
text / messages  --->   template + tokenize  --->   prompt token IDs
OpenAI response  <---   detokenize + format   <---   output token-ID stream
```

The Rust `TokenEngine` core owns only token execution. Its current deterministic
implementation cycles the prompt token IDs up to `max_output_tokens`. The
feature-gated `smg` module is a replaceable transport adapter implementing
SMG's `TokenSpeedScheduler` gRPC service. Health, model metadata, load,
cancellation, and cache-control RPCs stop at that transport; HTTP, tokenizer
loading, chat templates, reasoning/tool parsing, and public response formatting
remain in SMG. The crate's default feature set is empty, so building the token
core does not compile SMG, Tonic, or Tokio dependencies. The opt-in
`smg-transport` feature enables the compatibility module and worker binary.
Cargo still records that optional dependency closure in the workspace lockfile;
the feature isolates compilation and linkage, not dependency resolution.
SMG's request envelope includes an `original_text` field alongside token IDs;
the compatibility module accepts that envelope but discards the text before it
constructs the token core's `GenerateRequest`.

The publishable `inferlab-integration-specialized-engine` package plans one
Engine rank process owning a nonzero pure-TP device set and one zero-device
Gateway process through the same contract used by real Engine implementations.
The generic integration accepts arbitrary TP widths. This fake implementation
deliberately accepts only TP1 and returns a typed CLI error for wider requests;
its token core does not access the allocated device. The package contains no
fake-Engine lowering: the fixture conforms by exposing the canonical
`inferlab-token-engine smg-worker` command.

## Reproducible checks

The default offline checks require neither a GPU, model weights, SMG process,
nor credentials:

```sh
cargo test -p inferlab-fake-engine
cargo test -p inferlab-fake-engine --features smg-transport
pixi run pytest python/inferlab-integration-specialized-engine/tests/test_integration.py
```

The default Rust test covers the pure token boundary without compiling the SMG
stack. The feature-enabled Rust test additionally covers the SMG request/stream
mapping, the minimum control surface, and a real TCP/gRPC round trip through the
published SMG client. The shared integration tests cover protocol-v7 planning
and rendering:

- arbitrary pure-TP widths in one `serve` Engine replica and rank process, with
  the fake executable itself limited to one device;
- one `smg` Gateway targeting that Engine role;
- one request-routing link from `gateway` to `serve`;
- one closed `['gateway']` frontend binding and no P/D Router or KV transfer;
- profiling plans that capture the Engine process tree while binding the
  framework-range window to the SMG Gateway;
- exactly one rendered canonical token-Engine command and one rendered SMG command.

## Opt-in live SMG transport check

Inputs are an SMG installation in a selected Pixi environment and a local model
directory containing a tokenizer SMG can load. The fake Engine never reads the
tokenizer or model weights; it reports the locator as worker metadata while SMG
owns all tokenizer access.

Start the Engine from this checkout:

```sh
cargo run -p inferlab-fake-engine --features smg-transport \
  --bin inferlab-token-engine -- smg-worker \
  --listen 127.0.0.1:50051 \
  --model <model-locator> \
  --served-model-name fake-model \
  --tensor-parallel-size 1 \
  --default-max-output-tokens 16 \
  --max-num-batched-tokens 12288
```

In a second terminal, start a lock-pinned SMG whose
`TokenSpeedScheduler` protocol is compatible with the `smg-grpc-client`
revision selected by this repository's `Cargo.lock`:

```sh
pixi run -e <smg-environment> smg launch \
  --host 127.0.0.1 \
  --port 30000 \
  --prometheus-port 30001 \
  --worker-startup-timeout-secs 300 \
  --worker-urls grpc://127.0.0.1:50051 \
  --model-path <tokenizer-locator> \
  --tokenizer-path <tokenizer-locator> \
  --policy passthrough \
  --disable-retries \
  --disable-circuit-breaker
```

The explicit 300-second SMG timeout bounds this operator-managed check. An
InferLab-managed run instead defers SMG's internal registration timeout and
uses the server's `readiness_timeout_seconds` as its sole startup deadline.

After `GET /readiness` succeeds, send either public route. For example:

```sh
curl --fail-with-body http://127.0.0.1:30000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"fake-model","prompt":"token boundary","max_tokens":4,"stream":false}'
```

The response text is tokenizer-dependent, but the token core's input and output
contain token IDs only; the compatibility transport receives the wider SMG
envelope. Stop both foreground processes with the terminal interrupt when
finished.

## State, failure, cleanup, and evidence

The model locator and served identity are shared between SMG metadata and the
Inferlab-resolved plan. Engine/Gateway endpoints, device assignment, and
commands are control-plane allocation facts. The fake core owns no persistent
state, tokenizer cache, model-weight state, or KV cache. SMG may keep its own
process-local tokenizer and routing state.

An Engine bind failure terminates the Engine process and prevents SMG from
becoming ready. A missing tokenizer fails in SMG without entering the Engine
core. An incompatible gRPC contract prevents worker detection or readiness. A
request without tokenized input is rejected by the Engine transport as an
invalid argument. In the direct live check, the operator owns both foreground
processes and cleanup; when the same two commands are launched through an
Inferlab downstream workspace, Inferlab owns startup order, runtime handles,
rollback, stop, logs, and cleanup.

The offline plan/render tests are the committed composition evidence. A manual
SMG response is transport evidence only: it does not qualify a backend or
justify a support-matrix claim. Qualification still requires a real Inferlab
execution record for the exact hardware, model, integration revision, Gateway,
route, and workload shape.
