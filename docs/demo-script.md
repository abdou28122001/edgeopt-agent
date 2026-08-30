# Demo narration

I already have a trained model. Deploying it to constrained hardware is the
hard part: the target, application constraints, quality, latency, and runtime
all matter.

This is EdgeOpt. I give it a deployment contract with the model, evaluation
fixture, target profile, and objectives. TrueForge orchestrates the agent, but
the EdgeOpt tools are deterministic. They inspect the contract, measure a
baseline, and try only bounded candidates.

Here the v0 path is a tiny ONNX model on CPU with ONNX Runtime. The baseline
and optimized candidate run against the same fixture and frozen quality rule.
The result is measured evidence, not a number invented by the language model.

The TensorRT candidate is rejected before execution because this profile has no
CUDA or TensorRT capability. That is a useful result: an impossible path is
not allowed to look like an experiment.

Now the verifier checks hashes, evaluation and baseline binding, attestation,
quality, and constraints. It recommends from the evidence. The selected
artifact is identified by SHA-256, so it can move from the isolated Daytona
sandbox to the host without relying on the sandbox path.

Finally, EdgeOpt asks to package. The `approved=true` field is only explicit
intent. TrueForge pauses the tool for the human approval decision. After I
approve, the bounded package contains the selected artifact and its manifest.

The point is not that v0 supports every device. The point is a trustworthy
workflow: explore, execute, benchmark, verify, critique, and stop for me before
an irreversible packaging step.
