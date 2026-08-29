# EdgeOpt v0 Demo Contract

## Successful approximately three-minute demo

The demo is successful only if a judge can visibly follow one real run:

1. TrueForge UI/session is visible with the EdgeOpt task and state.
2. The agent calls at least one real typed MCP tool.
3. Candidate code executes in the sandbox.
4. A preserved baseline produces concrete evidence using one declared,
   reproducible evaluation fixture. The fixture identity, content/label hashes,
   split, preprocessing version/hash, and evaluation ID are visible in the
   manifest/report.
5. The workflow evaluates at least two candidate outcomes, or, if time is
   tight, visibly shows one accepted and one rejected path.
6. The critic cites measured evidence such as correctness, P95 latency,
   provider, or memory; it does not judge from prose alone. Correctness uses
   the documented metric direction and absolute/relative degradation rule.
7. The workflow pauses visibly at `approval_required` before packaging.
8. Packaging happens only after the human approves.
9. The final report/run manifest identifies the selected result, rejected
   candidate(s), metrics, hashes, and stop reason.

## Recommended narration

Start with the input model, target profile, and constraint card. Show baseline
measurement, then let the planner submit a bounded candidate. Show the MCP
tool trace and sandbox result. Display the verifier/critic comparison. Submit
one second candidate or a deliberate incompatibility to make rejection visible.
End at the approval pause, approve, and show the generated manifest/package.

## Minimum fallback demo

If advanced adapters, remote hardware, or a richer model fail, use the CPU
ONNX Runtime path with a tiny reproducible fixture. The fallback must still
show a real MCP tool, sandbox execution, baseline evidence, a candidate
comparison, critic reasoning, the human approval pause, and a post-approval
manifest. Baseline and candidate must use the same declared fixture and
`evaluation_id`; show its content hash (and separate ground-truth hash when
applicable) in the report. Label the target as CPU/ONNX Runtime and do not
imply TensorRT or Model Optimizer support.

## Acceptance evidence

- TrueForge session screenshot or trace.
- MCP request/result trace with tool name and typed input/output summary.
- Sandbox execution result.
- Baseline and candidate evidence records.
- Critic decision with measured reasons.
- Approval event and final package/manifest.
- Reproduction command and expected output in the README when implementation
  is added.

## Cut rules

Cut in this order if the timebox slips: extra candidate families, persistent
memory beyond the run record, OpenVINO, NVIDIA adapters, multiple model/task
domains, polished dashboard elements, and nonessential subagents. Never cut
the real MCP call, sandbox execution, evidence comparison, or approval gate.
