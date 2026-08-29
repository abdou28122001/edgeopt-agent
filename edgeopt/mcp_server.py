"""Small FastMCP surface; the deterministic runtime remains the authority."""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .contracts import DeploymentContract, QualityRule
from .runtime import package, verify

mcp = FastMCP("edgeopt")


@mcp.tool()
def edgeopt_inspect(spec_path: str) -> dict:
    spec = json.loads(Path(spec_path).read_text())
    contract = DeploymentContract(**spec["contract"])
    contract.validate()
    return {"status": "verified", "model_format": contract.model["format"], "target": contract.target, "plan": ["baseline", "onnxruntime_graph_optimization", "reject_tensorrt_without_capability"]}


@mcp.tool()
def edgeopt_verify(run_manifest_path: str) -> dict:
    manifest = json.loads(Path(run_manifest_path).read_text())
    contract = DeploymentContract(**manifest["contract"])
    rule = QualityRule(**manifest["quality_rule"])
    return verify(contract, manifest["baseline"], manifest["candidate"], rule)


@mcp.tool()
def edgeopt_package(run_manifest_path: str, selected_model_path: str, output_dir: str, approved: bool = False) -> dict:
    if not approved:
        raise ValueError("explicit approved=true is required")
    output = package(Path(selected_model_path), Path(output_dir), json.loads(Path(run_manifest_path).read_text()), approved)
    return {"status": "packaged", "output_dir": str(output)}


def main() -> None:
    mcp.run(transport="stdio")
