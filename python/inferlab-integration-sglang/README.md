# Inferlab Integration for SGLang

Framework-specific planning and rendering for running SGLang servers through
Inferlab. The consuming workspace supplies SGLang and its hardware runtime;
this package supplies only the Inferlab integration boundary.

See the [Inferlab repository](https://github.com/Infer-Lab/InferLab) for
workspace authoring and supported topology documentation.

When profiling intent is enabled, the integration declares every `single`,
prefill, and decode model-serving replica as a capture target. Its replica
entry endpoint opens the Nsight Systems range through `POST /start_profile`
with SGLang's `CUDA_PROFILER` activity and closes it through
`POST /stop_profile`. InferLab, not this package, owns the profiler lifecycle,
capture plan, report verification, cleanup, and records.

For direct `single`, setting `enable_metrics = true` enables SGLang's native
Prometheus endpoint and declares `/metrics` to InferLab. Without that effective
setting the integration advertises no server-metrics capability.
