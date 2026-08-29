from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper

from ..contracts import DeploymentContract, canonical_hash, file_sha256


def create_fixture(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    model = output / "tiny_linear.onnx"
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "edgeopt_tiny_linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor("weight", TensorProto.FLOAT, [4, 4], [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])],
    )
    onnx_model = helper.make_model(graph, producer_name="edgeopt", opset_imports=[helper.make_opsetid("", 17)])
    onnx_model.ir_version = 9
    model.write_bytes(onnx_model.SerializeToString())
    fixture = output / "evaluation.npz"
    np.savez(fixture, input=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    preprocessing = {"id": "identity-f32-v0", "sha256": canonical_hash({"id": "identity-f32-v0"})}
    contract = DeploymentContract(
        model={"source": str(model), "format": "onnx", "framework": "onnx", "task": "synthetic_similarity", "artifact_sha256": file_sha256(model), "input_schema": [{"name": "input", "shape": [1, 4], "dtype": "float32"}], "output_schema": [{"name": "output", "shape": [1, 4], "dtype": "float32"}]},
        evaluation={"evaluation_id": "synthetic-eval-v0", "fixture_ref": str(fixture), "fixture_sha256": file_sha256(fixture), "split": "validation_fixture", "preprocessing": preprocessing, "metric": "synthetic_output_mae_vs_baseline", "ground_truth": None},
        target={"profile_id": "local-cpu-onnxruntime", "verified_capabilities": ["CPUExecutionProvider"], "runtime": "onnxruntime", "provider": "cpu", "cpu": "local CPU", "gpu": None, "memory_mb": None, "source_of_truth": "profiled"},
        application_context={"use_case": "synthetic edge perception fixture", "sensor": "declared fixture", "input_resolution": [1, 4], "pipeline": ["inference"]},
        objectives={"priority": "balanced", "max_quality_degradation": 1e-6, "max_p95_latency_ms": None, "min_throughput": None, "max_memory_mb": None, "max_model_size_mb": None},
    )
    (output / "run-spec.json").write_text(json.dumps({"contract": contract.to_dict(), "model_path": str(model), "fixture_path": str(fixture), "run_id": "edgeopt-local-v0"}, indent=2, sort_keys=True) + "\n")
    return output / "run-spec.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".edgeopt-demo/input"))
    args = parser.parse_args()
    print(create_fixture(args.output))


if __name__ == "__main__":
    main()
