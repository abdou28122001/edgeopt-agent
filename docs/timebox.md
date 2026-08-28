# EdgeOpt v0 Deadline-First Timebox

## Target

Internal target: **Sunday, 30 August 2026 at 17:00 London time**. The official
hackathon deadline is **20:00 London time**; the remaining three hours are
emergency buffer only. All times below are ordered by gate, not by feature
wish list.

## Hard gates

| Gate | Exit condition | Cut rule |
|---|---|---|
| Architecture/Qodo PR | Docs merged only by human after PR and Qodo review | No implementation until contract is reviewable. |
| TrueForge local smoke | Local session/UI opens and a minimal task runs | If unavailable, report blocker and preserve a truthful fallback plan. |
| Provider + sandbox decision | One verified provider and sandbox path selected | Default to CPU/ONNX; do not install or configure credentials. |
| First real MCP tool | A typed tool call reaches a real implementation | No simulated evidence accepted. |
| First end-to-end green path | Inspect → baseline → candidate → benchmark → verify completes | Freeze candidate scope immediately. |
| Approval gate | Run pauses and resumes only after explicit human approval | Never cut this gate. |
| Demo stabilization | Clean three-minute rehearsal with evidence and failure path | Cut adapters, extra agents, and UI polish first. |
| README/Qodo evidence | Reproduction commands, limitations, and review evidence documented | Keep README minimal and truthful. |
| Video | Short recording shows the contract and approval pause | Record the stable fallback if advanced path is flaky. |
| Submission | Public repo, PR history, demo, and submission fields checked | Stop feature work; use remaining time for verification. |

## Execution order

1. Complete this architecture PR and wait briefly for Qodo.
2. Confirm TrueForge local smoke, available MCP route, and sandbox provider
   without adding credentials or packages.
3. Implement the smallest CPU/ONNX baseline and one candidate path.
4. Add evidence schema, critic decision, persisted run state, and approval gate.
5. Add a second candidate/rejection path only if the green path is stable.
6. Rehearse, capture the manifest/report, document reproduction, record video,
   and prepare submission.

## Explicit cuts

- Cut NVIDIA Model Optimizer, TensorRT, OpenVINO, remote devices, energy
  measurement, and full video-pipeline optimization unless already verified and
  isolated behind a working adapter.
- Cut agriculture/road validation, multiple model families, broad HPO, and
  transferable optimization memory.
- Cut decorative subagents, dashboards, vector memory, and complex persistence.
- Never cut the real MCP call, sandbox execution, baseline, measured critic,
  approval pause, or reproducible final manifest.

## Submission buffer checklist

Before 17:00 London, confirm the public repository has no secrets/private
assets, the demo can run from documented commands, the PR is not merged by the
agent, Qodo status is recorded, and the human owns the final merge/submission
decision. Use 17:00–20:00 only for recovery, final review, and submission.
