# Reproducibility

This project has two deliberately separate reproduction levels. Level 1 is
the stranger/judge check and requires no account or credential. Level 2 is the
optional live harness proof and requires the participant's own accounts.

## Known-good reference

- Source commit at the start of submission hardening:
  `0306f748e0ffa5b83acb7ee3799b1df19aca1d2a`
- Tested Python: 3.13.15; the project declares Python 3.12 or newer.
- Tested public-repo environment versions: NumPy 2.5.2, ONNX 1.22.0,
  ONNX Runtime 1.29.0, MCP 1.29.1, and pytest 8.4.2.
- Successful live proof runtime: TrueForge v0.1.4 and Daytona SDK 0.207.0.
- Live Daytona region: `eu`.
- Live model/provider names: NVIDIA NIM, model
  `nvidia/nemotron-3.5-lightning-30b-a3b`, and the public
  `https://integrate.api.nvidia.com/v1` API base.
- EdgeOpt MCP entrypoint: `edgeopt-mcp` or `python -m edgeopt.mcp_server`;
  tools are `edgeopt_inspect`, `edgeopt_verify`, and `edgeopt_package`.

Version numbers describe the known-good proof; cloud availability and
machine-level timings can vary.

## Level 1 — zero-credential stranger/judge check

From a fresh clone:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[test]"
./.venv/bin/pytest -q
./.venv/bin/python -m edgeopt.demo.prepare --output .edgeopt-demo/input
./.venv/bin/python -m edgeopt.demo.run \
  --spec .edgeopt-demo/input/run-spec.json \
  --output .edgeopt-demo/run
```

Expected results are a passing test suite, a `decision=recommend_*` line, and
these files in the run directory: `run-manifest.json`, `baseline.json`,
`candidate-a.json`, `candidate-b.json`, `quality-rule.json`, and
`decision.json`. The target is explicitly local CPU/ONNX Runtime. This level
does not use Drive, private files, a Daytona account, a TrueForge account, or
any credential.

## Level 2 — live harness check

The participant supplies their own NVIDIA-compatible model key and Daytona
cloud account/API key, configures the `eu` region, and starts TrueForge v0.1.4
with the public EdgeOpt MCP server. The runtime should be configured so
`edgeopt_package` is approval-required.

The live sequence is:

`TrueForge + NIM → edgeopt_inspect → managed Daytona sandbox → measured
baseline/candidate + unsupported TensorRT rejection → bounded evidence return
→ host edgeopt_verify → recommendation → TrueForge approval pause → package`

Create one fresh 32-byte per-run attestation key and supply its base64 form
out-of-band as `EDGEOPT_ATTESTATION_KEY_B64` to both the host verifier/MCP
process and the sandbox execution process. Never put the value in a command
string, transcript, evidence file, package, screenshot, or source control.
After the run, unset the environment variable, remove transient sandbox
resources, and delete or rotate the temporary Daytona Full Access key after
the final submission demo—not before it.

Expected live markers are `EDGEOPT_DAYTONA_HEALTH_OK`, a measured recommendation
(`recommend_candidate` or `recommend_baseline`), `executed=false` for the
unsupported TensorRT candidate, a host verification success, a visible
TrueForge `approval_required` event, and a package containing
`selected-artifact.onnx`, `run-manifest.json`, and `report.md`.

## Troubleshooting and cleanup

Transient Daytona preview or proxy DNS failures are operational issues, not
EdgeOpt behavior. Confirm the exact EU hostname resolves, retry a bounded
health check, and stop if the provider remains unavailable. Do not change
system DNS, hardcode provider IPs, or silently switch to self-hosted Daytona.

Stop the local TrueForge and MCP processes after evidence capture, delete only
the disposable sandbox owned by the run, and retain the known-good release
snapshot. Keep the final packaging approval human-operated for recording.
