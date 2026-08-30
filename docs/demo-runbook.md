# EdgeOpt demo runbook

## Primary live three-minute demo

This is the preferred recording path, validated end-to-end in T009R1. Use a
fresh short TrueForge session and let the human owner operate the final
approval.

| Time | Beat |
|---|---|
| 0:00–0:25 | State the model-to-edge deployment problem and the value of measured, bounded exploration. |
| 0:25–0:45 | Show the `DeploymentContract`, target profile, and objective. |
| 0:45–1:15 | TrueForge invokes the real typed EdgeOpt MCP tools and plans bounded work. |
| 1:15–1:50 | The managed Daytona sandbox runs the CPU/ONNX job; show measured baseline/candidate evidence and TensorRT rejection before execution. |
| 1:50–2:15 | Show host deterministic verification and the evidence-based recommendation. |
| 2:15–2:40 | Request `edgeopt_package`; show the visible TrueForge approval-required pause. |
| 2:40–2:55 | The human owner clicks Allow in the final validation/demo; show the package. |
| 2:55–3:00 | Show the selected artifact and hashes; state the one-line impact and the measured-vs-assumed boundary. |

The development run used Codex-assisted approval, but that is not the final
demo: the human owner must visibly perform the final approval.

## Emergency fallback only

If Daytona or network access fails while recording:

1. Run the zero-credential CPU/ONNX fixture and show its measured evidence,
   unsupported-candidate rejection, verifier, and approval-gated package
   refusal.
2. If still available, show the persistent TrueForge session from the earlier
   successful live proof, labeled clearly as earlier harness evidence rather
   than a fresh run.
3. Show the public repository and PR/Qodo evidence.
4. Say exactly what external failure appeared during recording.

Never splice old logs or screenshots to imply a fresh successful run. The
fallback is emergency-only; the primary live path remains preferred.
