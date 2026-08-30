# Public evidence index

## Known-good submission candidate

The submission candidate is based on merged `main` commit
`0306f748e0ffa5b83acb7ee3799b1df19aca1d2a`. The successful T009R1 capability
chain proved the following without changing the repository: TrueForge with
NVIDIA NIM, the real EdgeOpt MCP, a managed Daytona EU sandbox running merged
EdgeOpt, measured baseline and candidate evidence, unsupported TensorRT
rejection before execution, shared ephemeral attestation, host deterministic
verification, a measured recommendation, the TrueForge approval boundary, and
successful development packaging.

## Public review evidence

- [Merged PR #4](https://github.com/abdou28122001/edgeopt-agent/pull/4) — core
  vertical slice and evidence/trust remediation history.
- [PR #5](https://github.com/abdou28122001/edgeopt-agent/pull/5) — distributed
  attestation and SHA-256 artifact portability.

## Reproduce and inspect

- Zero-credential setup and test: `uv venv .venv`,
  `uv pip install --python .venv/bin/python -e ".[test]"`, then
  `./.venv/bin/pytest -q`.
- Zero-credential measured run:
  `python -m edgeopt.demo.prepare` followed by
  `python -m edgeopt.demo.run`.
- Implemented typed MCP tools: `edgeopt_inspect`, `edgeopt_verify`,
  `edgeopt_package`.

## What v0 actually measures

The v0 runtime measures real ONNX Runtime CPU inference for a deterministic
tiny fixture: baseline latency/throughput, ONNX Runtime graph-optimized
latency/throughput, output similarity under a frozen rule, artifact and input
hashes, and hard deployment constraints that are supported by those records.
It deterministically rejects a TensorRT candidate when the profile lacks the
required capability. It does not measure Jetson, GPU, TensorRT, Model
Optimizer, memory, or general hardware performance.

All live claims are labeled as harness evidence; all target-hardware and future
adapter statements are assumptions or proposals until measured by an
appropriate adapter.
