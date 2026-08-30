# Architecture cheat sheet

Use these short answers when explaining EdgeOpt to a judge.

## What goes into the DeploymentContract?

It names the model and artifact hash, evaluation fixture and preprocessing,
target/profile and verified capabilities, application context, objectives, and
hard constraints. It makes the question “is this deployable?” concrete before
any candidate runs.

## Why ONNX in v0?

ONNX is a small, reproducible interchange route with a CPU ONNX Runtime path
that works in a clean environment. It is not universal: future adapters can
accept other model formats when their export and runtime checks are real.

## What is the LLM allowed to do?

The LLM plans bounded work and explains results. Deterministic typed tools run
the model, record measurements, verify hashes and binding, reject unsupported
capabilities, and choose the recommendation. The LLM does not invent metrics.

## Why is TensorRT rejected first?

The declared v0 profile provides `CPUExecutionProvider`, not CUDA/TensorRT.
Running an incompatible candidate would not be evidence, so EdgeOpt rejects it
before execution.

## What is NVIDIA Model Optimizer?

It is a possible future execution adapter for PTQ, pruning, or sparsity. It is
not the agent brain and is not implemented or measured in hackathon v0.

## How does distributed trust work?

The host creates one ephemeral 32-byte HMAC key per run and supplies it
out-of-band to both the sandbox process and host verifier. Evidence includes a
key identifier and tag, never the key itself. The host can therefore verify
that the sandbox evidence was produced under the same run authority.

## Why SHA-256 instead of a filesystem path?

A path belongs to one machine. The artifact's SHA-256 identifies its bytes, so
the verified artifact can move from an isolated sandbox to a host package
directory. A different or modified byte sequence still fails closed.

## Who approves packaging?

TrueForge is the genuine human-approval boundary. `approved=true` says that
the agent is explicitly requesting packaging; it is not human authorization.
TrueForge must intercept the approval-required MCP call before execution.

## What is measured versus assumed?

Measured: the deterministic CPU/ONNX fixture's inference timings, similarity,
hashes, binding, and recommendation. Assumed/profiled: the local CPU target
description and all future hardware/adapter capability statements. No v0
claim should be phrased as a Jetson, GPU, TensorRT, or Model Optimizer result.
