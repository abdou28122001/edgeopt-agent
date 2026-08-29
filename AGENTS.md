# EdgeOpt Agent contributor instructions

## Scope and source of truth

EdgeOpt is a time-boxed 2026 Agent Harness Hackathon submission. Optimize for
one reproducible, approximately three-minute TrueForge demo, not research
completeness or a general optimization platform. The repository is the source
of truth for current implementation and documentation. Measured evidence must
be labeled separately from assumptions, proposals, and external research.

Keep the current product statement in view: give the agent a trained model, a
target edge-device profile, and deployment constraints; let it explore,
execute, benchmark, verify, and critique bounded candidates; then stop for
human approval before packaging.

## Git and review workflow

- Do not make substantive direct pushes to `main`.
- Work on a named branch, open a GitHub pull request, obtain Qodo review, and
  leave the final merge decision to a human.
- Record the exact commands, tests, checks, commit, and review state in the
  relevant handoff.
- Do not switch branches as part of an unrelated task.

## Safety and data boundaries

- Never commit secrets, private data, private research assets, model weights,
  datasets, virtual environments, caches, or large generated artifacts.
- Do not modify Hermes, workstation-control, global Research OS configuration,
  drivers, system services, firewall, credentials, or provider configuration
  from this project.
- Treat Drive as coordination/evidence transport only; repository files and
  reproducible project inputs remain authoritative for implementation.
- Require explicit human approval before packaging, publishing, flashing, or
  any other irreversible action.

## Engineering discipline

- Start with small reproducible tests and a working TrueForge/MCP path.
- Use real typed tools and sandbox execution; never present simulated results as
  measured deployment evidence.
- Preserve baselines, fixed inputs, configuration, artifact hashes, and
  benchmark protocols.
- Mark unavailable, optional, device-gated, and unverified capabilities
  explicitly. Do not assume NVIDIA Model Optimizer, TensorRT, or a compatible
  accelerator is available because it is documented.
- Prefer the narrowest implementable path. Cut optional adapters, broad search,
  extra domains, and research features when they threaten the demo path.
