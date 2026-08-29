# EdgeOpt Agent

EdgeOpt is an agentic, hardware-aware model-to-edge deployment engineer. A
trained model is not automatically deployable on constrained hardware: the
runtime, device capabilities, and measured quality/latency trade-offs matter.
EdgeOpt inspects a deployment contract, measures a baseline, tests only bounded
candidates, verifies the evidence, and waits for explicit approval before
packaging.

## What this v0 proves

This vertical slice is generic and CPU/ONNX-first. It generates a tiny
deterministic ONNX fixture, measures baseline ONNX Runtime CPU execution,
measures graph-optimized execution under the same fixture, rejects a declared
TensorRT candidate before execution because the profile lacks CUDA/TensorRT,
and enforces an explicit approval flag before packaging. It does not claim
Jetson, DGX Spark, TensorRT, NVIDIA Model Optimizer, or remote sandbox
measurements.

## Zero-credential quickstart

Requires Python 3.12+, then install the small project dependencies in a
repository-local environment:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
python -m edgeopt.demo.prepare
python -m edgeopt.demo.run --output .edgeopt-demo/run
```

Expected output is a line containing `decision=recommend_...` and a run
directory containing `run-manifest.json`, `baseline.json`, `candidate-a.json`,
`candidate-b.json`, `quality-rule.json`, and `decision.json`. The run manifest
records hashes, the immutable evaluation/rule binding, measurements, and the
fact that the target is a profiled local CPU.

`max_memory_mb` is intentionally unsupported in the CPU v0 path because this
slice does not measure memory; supplying it fails closed. A future adapter may
support it only with measured memory evidence. Local evidence also carries a
private HMAC attestation held under the ignored `.edgeopt-state/` directory;
the key is generated locally and is never included in manifests or packages.
Distributed live runs instead supply one ephemeral per-run attestation key
out-of-band to both the sandbox process and host verifier; the key is never
part of evidence, output, or a package. Artifact identity is its SHA-256, so a
verified artifact may move between isolated execution and packaging hosts.
The final TrueForge run still pauses `edgeopt_package` for human approval.

## Flow

`DeploymentContract → inspect/profile → baseline → measured ONNX Runtime graph
candidate → deterministic unsupported-candidate rejection → evidence-bound
verifier/critic → approval-required package boundary`

The local fixture is for reproducibility and testing. The later live path will
connect these deterministic tools to an already configured TrueForge session,
MCP server, and isolated sandbox. That external configuration is not required
for this quickstart and no credentials are stored here.

## Scope and future adapters

TrueForge owns orchestration; deterministic adapters own inspection,
execution, benchmarking, verification, and packaging. The adapter boundary is
intended to accommodate ONNX export, ONNX Runtime graph optimization, NVIDIA
Model Optimizer PTQ/pruning, TensorRT, and OpenVINO after capability checks.
Only ONNX Runtime CPU is implemented here. A connected/measured device may
produce hardware measurements; a static profile can establish compatibility but
must never be described as a hardware measurement.

## Review status

This implementation is submitted through a public pull request and Qodo
review. Final submission evidence will be added after the review/merge decision
by the human owner.
