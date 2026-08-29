# EdgeOpt Architecture v0 Decision Log

| Decision | Status | Rationale |
|---|---|---|
| Narrow model-to-edge job | Accepted | Fits the hackathon window and produces a demonstrable vertical slice. |
| CPU-first ONNX Runtime green path | Accepted | Safest locally reproducible fallback with measured runtime evidence. |
| TrueForge session, sandbox, and real MCP tools central to the flow | Accepted | Satisfies the core judging constraint and makes execution observable. |
| Baseline/evidence/critic contract | Accepted | Prevents theoretical compression claims from being mistaken for deployment gains. |
| Human approval before packaging | Accepted | Required safety boundary for irreversible actions. |
| ModelOpt/TensorRT adapter | Deferred | Pending verified package, runtime, and named compatible device. |
| OpenVINO adapter | Deferred | Useful optional CPU path after the first green path is stable. |
| Dynamic subagents and persistent session | Deferred | Add only where functional and time-safe; decorative agents are not useful. |
| Broad AutoML/multi-domain platform | Rejected | Scope creep and incompatible with a reliable three-minute demo. |
| New optimization algorithm or novelty claim | Rejected | Not required for the submission and conflicts with the crowded research landscape. |
| Private research data/model dependency | Rejected | Incompatible with public reproducibility and demo safety. |
| Drive as runtime source of truth | Rejected | Drive is coordination/evidence transport; the repository and run inputs govern execution. |
| Device flashing or irreversible deployment in the first path | Rejected | Requires explicit device authority and a human approval workflow beyond the initial scope. |
