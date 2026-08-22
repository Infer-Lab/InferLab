# Backend Support Matrix

This public document is the authority for the operator-visible backend
capabilities on the current InferLab main branch. It describes workflows that
InferLab plans, executes, and records; it is not a list of every feature offered
by the upstream frameworks.

Status meanings:

- **Qualified**: implemented and demonstrated by a real downstream execution
  record for the baseline named below.
- **Supported**: implemented and covered by deterministic integration tests,
  but not qualified for every relevant hardware and model shape.
- **Limited**: implemented only under the conditions stated in the cell.
- **Unsupported**: a real probe demonstrated a specific conformance failure for
  the exact integration revision, route, model, and baseline named in the cell.
- **Inconclusive**: a real probe ran but its transport or HTTP outcome could not
  establish endpoint capability.
- **Unqualified**: no complete real public-route record establishes the
  capability for the exact path in the cell.
- **—**: rejected by the integration or not exposed by InferLab.

A qualified baseline is evidence for that concrete shape, not blanket
certification of every framework version, model, device, or parallel configuration.
InferLab main speaks adapter protocol version 8; the in-tree candidate
packages are adapter SDK `0.7.1`, version `0.6.1` of the four maintained
framework integrations, and Specialized Engine integration `0.3.1`. Retained
qualification records from earlier exact package versions remain historical
evidence; they do not qualify these candidates unless a cell explicitly names
a protocol-v7 record.

## Serving And Control

| Capability | vLLM | SGLang | TensorRT-LLM | TokenSpeed | Specialized Engine |
| --- | --- | --- | --- | --- | --- |
| Integration package | `inferlab-integration-vllm==0.6.1` | `inferlab-integration-sglang==0.6.1` | `inferlab-integration-tensorrt-llm==0.6.1` | `inferlab-integration-tokenspeed==0.6.1` | `inferlab-integration-specialized-engine==0.3.1` |
| Single-node `single` topology | Qualified for the retained baseline and the protocol-v7 candidate on the Qwen3 MoE SM120 TP1 baseline below | Qualified for the retained baseline and the protocol-v7 candidate on the Qwen3 MoE SM120 TP1 baseline below | Qualified for the retained baseline below; protocol-v7 candidate Unqualified | Qualified for the retained baseline below; protocol-v7 candidate Unqualified | Supported: one Engine replica in one rank process with a pure-TP device set; the protocol-v7 `0.3.0` candidate is Unqualified |
| `single` public component | Qualified for the retained direct-Engine baseline and the protocol-v7 candidate on Qwen3 MoE SM120 TP1 | Qualified for the retained direct-Engine baseline and the protocol-v7 candidate on Qwen3 MoE SM120 TP1 | Qualified for the retained direct-Engine baseline; protocol-v7 candidate Unqualified | Qualified for the retained direct-Engine baseline; protocol-v7 candidate Unqualified | Supported: TokenSpeed SMG Gateway; the protocol-v7 `0.3.0` candidate is Unqualified |
| Gateway-backed `single` | — | — | — | — | Supported: `smg`, implemented by `tokenspeed-smg`; the protocol-v7 `0.3.0` candidate is Unqualified |
| Multi-node replica | Supported | — | — | — | — |
| Disaggregated prefill/decode | Qualified | Qualified for the pairing-specific baselines below | Qualified: built-in and native `trtllm-disaggregated` frontend pairs | Qualified for the maintained 1P1D pairing below | — |
| KV-transfer backend | Qualified: Mooncake and NIXL | Qualified: Mooncake and NIXL in the pairing-specific baselines below | Qualified: NIXL with the built-in frontend and native `trtllm-disaggregated` pair | Qualified: Mooncake for the maintained 1P1D pairing below | — |
| P/D Gateway backend | Supported: `builtin`, `vllm-router` | Supported: `builtin`, `sglang-router` | Supported: `builtin`, `trtllm-disaggregated` | Supported: `tokenspeed-smg` | — |
| P/D Router backend | Supported: `builtin`, `vllm-router` | Supported: `builtin`, `sglang-router` | Supported: `builtin`, `trtllm-disaggregated` | Supported: `tokenspeed-smg` | — |
| P/D frontend binding | Supported: one `[gateway, pd_router]` process | Supported: one `[gateway, pd_router]` process | Supported: one `[gateway, pd_router]` process | Supported: one `[gateway, pd_router]` process | — |
| Declared public workload paths | `/v1/completions`; `/v1/chat/completions` | `/v1/completions`; `/v1/chat/completions` | `/v1/completions`; `/v1/chat/completions` | `/v1/completions`; `/v1/chat/completions` | `/v1/completions`; `/v1/chat/completions` |
| AIPerf server-metrics capability | Qualified for direct `single` on the public endpoint at `/metrics`: a real protocol-v7 candidate Bench preserved the native export and produced both SPEED acceptance reports; Gateway and P/D public endpoints omit the capability | Qualified for direct `single` on the public endpoint at `/metrics` when effective setting `enable_metrics = true`; Gateway and P/D public endpoints omit the capability | — | — | Supported through SMG's separately allocated `prometheus` port at `/metrics`; `least_load` preserves the single-target routing result while activating canonical Engine load polling; real candidate Bench collection and cleanup are demonstrated from a dirty workspace, so reproducible qualification remains pending |
| Completion request used by InferLab | Qualified for the retained scalar-prompt baseline and the protocol-v7 candidate on Qwen3 MoE SM120 TP1 | Qualified for the retained scalar-prompt baseline and the protocol-v7 candidate on Qwen3 MoE SM120 TP1 | Qualified for the retained scalar-prompt baseline; protocol-v7 candidate Unqualified | Qualified for the retained scalar-prompt baseline; protocol-v7 candidate Unqualified | Supported: deterministic scalar prompt through SMG; the protocol-v7 `0.3.0` candidate is Unqualified |
| Exact flat serving Bench | Qualified on Qwen3 MoE SM120 TP1: the AIPerf runner froze scalar prompts for fixed and distributed ISL populations, and every completed request reconciled the observed backend ISL exactly | Qualified on Qwen3 MoE SM120 TP1 for the fixed-ISL full-prefix population; every completed request reconciled the observed backend ISL exactly. Other exact-flat shapes remain Supported | Supported through the declared completions path; candidate real-route qualification is pending | Supported through the declared completions path; candidate real-route qualification is pending | Supported through the declared completions path; candidate real-route qualification is pending |
| Exact locally rendered-chat serving Bench | Qualified on Qwen3 MoE SM120 TP1 for both the tokenizer-default and a definition-supplied template, frozen into exact scalar prompts with backend ISL reconciliation | Supported by the shared runner; candidate real-route qualification is pending | Supported by the shared runner; candidate real-route qualification is pending | Supported by the shared runner; candidate real-route qualification is pending | Supported by the shared runner; candidate real-route qualification is pending |
| Controlled final-prompt prefix geometry | Qualified on Qwen3 MoE SM120 TP1 for zero, partial, and full fixed-ISL sharing plus an inclusive-uniform ISL population; exact flat and locally rendered prompts use one nested canonical-prefix stream, while cache reads remain separate observed evidence | Qualified on Qwen3 MoE SM120 TP1 for full fixed-ISL sharing; other shared-prefix shapes remain Supported | Supported by the shared runner; real cache-read qualification is pending | Supported by the shared runner; real cache-read qualification is pending | Supported by the shared runner; real cache-read qualification is pending |
| Controlled Bench cache start and observed reuse | Qualified for direct `single` on Qwen3 MoE SM120 TP1 with `enable_prompt_tokens_details = true`: uncontrolled, cold, and primed full-prefix cases preserved per-request prompt/cache tokens; cold reset before profiling, primed reset and exact conditioning before profiling, and cleanup were verified. Under attention data parallelism a primed start primes each DP rank with one `X-Data-Parallel-Rank`-pinned conditioning request, requiring a vLLM build that honors that header (recent main; older releases silently ignore it and degrade to single-rank priming). The built-in Mooncake and NIXL P/D pairs fan priming out through the proxy's `POST /prime_prefix_cache`, covering every prefill replica × DP rank through the ordinary pairing flow — Qualified on the built-in NIXL pair for DeepSeek-V4-Flash SM120 (prefill TP2, decode TP2, both roles `enable_prompt_tokens_details = true`): the primed case conditioned the shared prefix through the Gateway and completed with per-request prompt_cache_read_ratio 1.0; the Gateway endpoint declares the cache-read capability only when both roles enable the reporting setting; the `vllm-router` pair declares no fan-out capability, so a primed start is rejected at planning when more than one prefill-side cache-owning target (replica x attention DP rank) sits behind the Gateway; a single-target shape needs no fan-out and conditions through the ordinary serving flow | Qualified for direct `single` on Qwen3 MoE SM120 TP1 with `enable_cache_report = true`: uncontrolled, cold, and primed full-prefix cases preserved per-request prompt/cache tokens, including SGLang's declared omitted-zero representation; cold reset, primed conditioning, and cleanup were verified. Under attention data parallelism a primed start primes each DP rank with one `X-Data-Parallel-Rank`-pinned conditioning request. The built-in P/D pair fans priming out through the proxy's `POST /prime_prefix_cache`, covering every prefill replica × DP rank through the ordinary bootstrap pairing flow (real-route qualification pending); the `sglang-router` pair declares no fan-out capability, so a primed start is rejected at planning when more than one prefill-side cache-owning target (replica x attention DP rank) sits behind the Gateway; a single-target shape needs no fan-out and conditions through the ordinary serving flow | Unqualified; controlled cold and primed starts are unavailable because the integration exposes no reset capability | Unqualified | Supported: the SMG Gateway endpoint declares the `POST /prime_prefix_cache` conditioning fan-out and the `POST /flush_cache` reset; a single-target shape conditions through the ordinary serving flow, and real-route qualification of the fan-out remains pending |
| Chat-completions execution | Qualified for direct `single` under the protocol-v7 candidate, including an operator-supplied server-side chat template; built-in Mooncake and NIXL P/D preserve the route but remain unqualified | Supported by deterministic built-in P/D frontend coverage; the integration-rendered pair and every exact public route remain Unqualified | Supported by deterministic built-in P/D frontend coverage for context-first streaming and non-streaming handoff; the integration-rendered pair and every exact public route remain Unqualified | Unqualified: the named path is preserved; the native Gateway/P/D Router pair requires separate route qualification | Unqualified: SMG preserves the path, but no exact public-route record qualifies chat execution |
| SemiAnalysis AgentX trace replay | Qualified for the immutable 062126 256k profile on direct `single`: Qwen3-30B-A3B-Instruct-2507 at TP2 on two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs completed the profile-owned 600-second cache-pressure warmup and 900-second profiling window at root-tree concurrency 1; the scenario-valid record preserved 696 error-free warmup requests, 141 profiling requests with one ordinary failure, source coordinates, cache-bust observations, branch statistics, transport metrics, and verified cleanup. The full-context profile is Supported but Unqualified | Supported by the shared runner; real SGLang route qualification is pending | Supported by the shared runner; real TensorRT-LLM route qualification is pending | Supported by the shared runner; real TokenSpeed route qualification is pending | Supported by the shared runner; real Specialized Engine route qualification is pending |
| Incremental public streaming | Qualified for the built-in NIXL P/D pair on Qwen3 MoE SM120 TP1 over TCP: independent clients observed nonterminal SSE events before `[DONE]` on both completion routes and an AIPerf 0.12.0 Bench retained request-level timing artifacts; built-in Mooncake is Supported by deterministic public-boundary coverage, while `vllm-router` remains Unqualified | Supported for the built-in pair by deterministic public-boundary coverage; `sglang-router` remains Unqualified | Supported for the built-in pair by deterministic public-boundary coverage; `trtllm-disaggregated` remains Unqualified | Unqualified for the maintained P/D pairing | Unsupported for the exact qualified `tokenspeed-smg==1.7.0.post20260710` Engine pairing: its worker emits stream chunks only after generation completes, so InferLab makes no low-latency streaming claim |
| Prefix-cache reset between cases | Qualified through `POST /reset_prefix_cache` for the direct `single` Qwen3 MoE SM120 TP1 cache-start baseline and for the built-in NIXL P/D pairing on DeepSeek-V4-Flash SM120 (prefill TP2, decode TP2): the cold case's Gateway reset returned 200 with deterministic fan-out to every prefill and decode engine, and the case completed with per-request cache-read evidence; the built-in Mooncake pairing shares the same fan-out path; `vllm-router` remains Unqualified | Qualified through `POST /flush_cache` for the direct `single` Qwen3 MoE SM120 TP1 cache-start baseline and the demonstrated P/D Gateway/P/D Router pairings below | —; P/D enforces block reuse off at launch | Qualified for `single` and the maintained P/D pairing below through Gateway `POST /flush_cache` | Supported by the worker contract through Gateway `POST /flush_cache`; unqualified for a concrete Engine |
| Framework profiling capture | Qualified: managed collection wraps each captured rank process tree with the replica entry as window control; engine trace (local, non-containerized placement only) renders the assigned record-owned trace directory into `--profiler-config` and verifies a dedicated-directory storage delta of at least one new trace artifact per device of the replica — Qualified on Qwen3.8-27B-FP8 SM120 TP2: capture start returned 200, both worker ranks flushed their traces, and coverage verified with expected_artifacts = 2. Note vLLM `stop_profile` blocks until worker traces finish serializing (observed >10 min for a TP2 27B capture), so the capture control and finalization deadlines must be raised above their 60 s / 300 s defaults for engine-trace runs | Qualified for `single` and prefill/decode: every model-serving replica entry controls its captured process tree through `POST /start_profile` and `POST /stop_profile`; managed collection declares the `CUDA_PROFILER` body, while engine trace (local, non-containerized placement only) renders `SGLANG_TORCH_PROFILER_DIR` and declares the `GPU` activity — Qualified on Qwen3-30B-A3B-Instruct-2507 SM120 TP1: start and stop returned 200 and the per-rank trace delta verified | — | — | Supported: captures the Engine process tree while TokenSpeed SMG Gateway controls the window through `POST /start_profile` and `POST /stop_profile`; managed capture defaults to CUDA and NVTX tracing, with OS runtime tracing available as an explicit typed override; no concrete Engine route is yet qualified |

For every supported framework profiling path, a positive AIPerf-native Bench
warmup drains before InferLab opens the framework capture window; the window
still closes at client completion. A failed warmup leaves its planned window
unopened and fails the measurement.

The two named paths are endpoint-contract facts, not route qualification. Chat
execution becomes Qualified only after an InferLab workflow produces a real
record through the exact integration, route, topology, Gateway backend, P/D
Router backend when present, and model being claimed. Optional upstream API
extensions such as embeddings and batched prompt arrays remain outside this
matrix. A pending upstream pull request or an unreleased dependency does not
count as current support.

The P/D Gateway, P/D Router, and fused-binding rows are Supported rather than
Qualified because existing route-level records predate protocol v7's separate
component evidence. The retained topology and pairing qualifications establish
those concrete end-to-end workflows, but do not retroactively establish the new
component-attribution shape. Requalification requires a protocol-v7 execution
record that preserves both backend facts and their one fused process binding.

## lm-eval Loglikelihood Routes

Loglikelihood support is an observed property of the complete public serving
route, not a consequence of scalar completion support. The first four columns
below are bounded to the published `0.4.0` integration packages and the named
maintained DeepSeek-V4 SM120 baselines; they do not qualify the protocol-v7
`0.6.0` candidates. The Specialized Engine column is bounded to the retained
exact `0.2.1` package. The qualification task was the built-in `hellaswag`
multiple-choice task with the model-directory Hugging Face tokenizer and text
requests. A worker-only probe cannot qualify a prefill/decode public endpoint.

| Public route or qualification boundary | vLLM (`inferlab-integration-vllm==0.4.0`) | SGLang (`inferlab-integration-sglang==0.4.0`) | TensorRT-LLM (`inferlab-integration-tensorrt-llm==0.4.0`) | TokenSpeed (`inferlab-integration-tokenspeed==0.4.0`) | Specialized Engine (`inferlab-integration-specialized-engine==0.2.1`) |
| --- | --- | --- | --- | --- | --- |
| Direct `single` endpoint | **Qualified**: DeepSeek-V4 SM120 TP2/EP2; the prompt-logprob probe passed tokenizer alignment and `hellaswag` completed with its selected metric gate. | **Unsupported for the probed baseline**: DeepSeek-V4 SM120 TP2/EP2 returned a conforming response shape, but the public endpoint exposed 14 prompt positions for the tokenizer's 13 tokens. | **Unsupported for the probed baseline**: DeepSeek-V4 SM120 TP2/EP2 returned a conforming response shape, but the public endpoint exposed one prompt position for the tokenizer's 13 tokens. | **Inconclusive for the probed baseline**: DeepSeek-V4 SM120 TP2/EP2 returned HTTP 400 to the prompt-logprob probe, so the probe did not establish support or unsupported prompt scoring. | —; the integration requires a Gateway-backed `single` topology. |
| Gateway-backed `single` endpoint | — | — | — | — | —: the initial token worker contract rejects log-probability requests. |
| Prefill/decode public endpoint, aggregate | **Unqualified**: no complete route-level prompt-logprob record; worker behavior is not qualification evidence. | **Unqualified**: no complete route-level prompt-logprob record; worker behavior is not qualification evidence. | **Unqualified**: no complete route-level prompt-logprob record; worker behavior is not qualification evidence. | **Unqualified**: no complete route-level prompt-logprob record; worker behavior is not qualification evidence. | — |
| Built-in Gateway/P/D Router pair | **Unqualified**: `0.4.0` declares Mooncake and NIXL cases, but no public-endpoint record establishes a concrete placement or per-role shape. | **Unqualified** for the maintained single-machine 1P1D TP2/EP2 Mooncake and NIXL pairings. | **Unqualified** for the maintained 1P1D TP2/EP2 NIXL path. | —; the maintained P/D path uses the integration-rendered TokenSpeed pair. | — |
| Integration-rendered Gateway/P/D Router pair | **Unqualified** for the maintained `vllm-router` Mooncake and NIXL paths. | **Unqualified** for the maintained `sglang-router` Mooncake and NIXL pairings. | **Unqualified** for the maintained `trtllm-disaggregated` NIXL path. | **Unqualified** for the maintained `tokenspeed-smg` Mooncake path. | — |

These direct-route failures do not establish the behavior of another model,
integration release, built-in frontend, Gateway revision, or P/D Router revision. Requalification must
run the tokenizer-alignment probe and representative task through the exact
public endpoint being claimed.

## Parallelism

The rows below describe which user-requested parallel dimensions the integration
can lower. “Derived” means the effective kernel dimension is calculated from the
declared outer world and the other accepted dimensions rather than configured as
an independent public setting.

| Capability | vLLM | SGLang | TensorRT-LLM | TokenSpeed | Specialized Engine |
| --- | --- | --- | --- | --- | --- |
| Outer tensor parallelism | Qualified for the retained baseline; protocol-v7 candidate Unqualified | Qualified for the retained baseline; protocol-v7 candidate Unqualified | Qualified for the retained baseline; protocol-v7 candidate Unqualified | Qualified for the retained baseline; protocol-v7 candidate Unqualified | Supported: arbitrary nonzero single-process pure TP; protocol-v7 `0.3.0` candidate Unqualified |
| Outer pipeline parallelism | Supported | Supported | Supported | — | Limited: `1` |
| Attention data parallelism | Supported | Supported | Limited: `1` or the outer TP size | Supported | Limited: `1` |
| Attention context parallelism | Supported: `single` and `decode` roles lower to decode context parallelism, and `prefill_decode` prefill roles to device-multiplying prefill context parallelism; model-level applicability (MLA-only prefill CP, GQA head divisibility) remains the framework's launch-time verdict. Qualified: decode CP on Qwen3.8-27B-FP8 SM120 TP8 DCP2 completed the 8K/1K serving baseline. Engine verdicts observed on DeepSeek family: DCP is rejected at launch for sparse-indexer (DSA) models, and prefill CP on DeepSeek-V4-Flash fails at engine launch in an upstream MoE top-k kernel (`is_padding` size mismatch under PCP) — a framework defect, not a lowering gap | Supported: `single` and `prefill` roles activate prefill CP (`--enable-prefill-cp` with a default `zigzag` strategy, and the whole flag group is verbatim-replaceable through the `--` passthrough) and `decode` roles lower to decode CP, all without changing device counts; lowering verified through launch (engine reports ATTN_CP ranks). Qualified: DeepSeek-V4-Flash SM120 TP2 with declarative CP 2 plus a verbatim `--enable-dsa-prefill-context-parallel` reaches readiness with the engine deriving `attn_cp_size=2` (interleave) and serves the 8K/1K bench. On SM120 the generic zigzag spelling remains blocked upstream (flashinfer ragged prefill crash) and fa4 lacks paged KV on SM120; non-DSA prefill CP qualification waits on the framework | — | — | Limited: `1` |
| MoE expert parallelism | Qualified | Qualified | Qualified | Qualified | Limited: `1` |
| MoE data parallelism | — | Supported with topology constraints | — | Supported | Limited: `1` |
| Independent dense tensor parallelism | — | Supported | — | Qualified | Derived: equals outer TP and is not independently configurable |
| Effective expert tensor parallelism | Derived | Derived | Derived | Derived; cannot be greater than `1` together with expert parallelism greater than `1` | Derived: equals outer TP |

Backend-specific constraints remain validated by the integration and are
reported during planning. This table records the public capability boundary; it
does not duplicate every arithmetic constraint enforced by each adapter.

## Maintained Qualification Baselines

| Backend | Real-hardware baseline | Important boundary |
| --- | --- | --- |
| vLLM | Source-built DeepSeek-V4 SM120 TP2/EP2 serving; real two-machine 1P1D vLLM Router serving with Mooncake and NIXL; source-built Qwen3 MoE SM120 TP1 direct serving for protocol-v7 Bench qualification | Multi-node replica lowering is supported but unqualified; the maintained cross-machine baseline is 1P1D. |
| SGLang | Source-built DeepSeek-V4 SM120 TP2/EP2 serving, pairing-specific single-machine 1P1D serving, and source-built Qwen3 MoE SM120 TP1 direct serving for protocol-v7 cache-start qualification | P/D qualification is pairing-specific below; TP4 is outside the maintained baseline. |
| TensorRT-LLM | Source-built DeepSeek-V4 SM120 TP2/EP2 serving and 1P1D NIXL serving with built-in and native routing | SM120 DeepSeek-V4 serving requires the source integration's FlashInfer sparse-MLA path; the stock NGC image through 1.3.0rc21 is not sufficient. |
| TokenSpeed | Source-built DeepSeek-V4 SM120 TP2/EP2/dense-TP2 serving; single-machine 1P1D serving with TP2/EP2/dense-TP2 per role, native `tokenspeed-smg` routing, and Mooncake KV transfer | P/D qualification is limited to that concrete routing/transfer pairing; the source-built framework baseline includes its required kernel fixes. |
| Specialized Engine | Grout Qwen3-4B on one NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120), with one TP1 Engine replica behind `tokenspeed-smg==1.7.0.post20260710` | Qualification covers serial greedy scalar completion through the Gateway at TP1. Wider single-process pure TP is Supported but Unqualified; chat, log probabilities, batching, P/D, and other Engine implementations remain unqualified or unsupported as stated above. |

### SGLang P/D Pairings

The qualified entries use source-built DeepSeek-V4 on SM120 in a
single-machine 1P1D topology with TP2/EP2 per role. Qualification is per
pairing; Supported cells are implemented but have not been separately
qualified on real hardware.

| Gateway/P/D Router backend pair | Mooncake | NIXL |
| --- | --- | --- |
| Built-in Gateway/P/D Router pair | Qualified | Supported |
| SGLang Router | Supported | Qualified |

## Maintenance Rules

Update this document in the same change as an integration when the change
affects any matrix row or qualification statement. In particular:

- use **Supported** for deterministic implementation coverage and **Qualified**
  only after a real record proves the exact workflow and shape;
- name material limitations, including dependencies on downstream framework
  patches, instead of presenting them as general support;
- remove or downgrade capabilities when the integration stops exposing them;
- retain the underlying execution evidence internally, but cite it here only
  when a public qualification artifact exists;
- never expose unpublished internal identifiers, machine-local record paths, or
  private downstream revisions; and
- do not list pending upstream pull requests or future releases as current
  support.
