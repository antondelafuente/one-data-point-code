#!/bin/bash
# Cross-family (Codex) semantic data audit of the frozen evaluation INPUTS: the six newly
# constructed 40-prompt probes, the reconstructed 500-prompt traffic slice, and the five
# inherited frozen probes. Runs the built-in opposite-family auditor: AAR_SUBSTRATE names the
# family that RAN the work; no AUDIT_VERIFIER_CMD override (an override would forfeit the
# quota fallback, gotcha 2026-08-20 / automated-researcher#762).
set -uo pipefail
R=/path/to/home/ws/run/csp1-refusal-footprint-map-1/registry/csp1-refusal-footprint-map-1
V=/path/to/home/automated-researcher/plugins/verify-claims/skills/verify-claims/scripts
env -u BASH_ENV AAR_SUBSTRATE=claude bash "$V/audit_experiment.sh" --data "$R" \
  "$R/data_audit_manifest_inputs.md" "$R/DATA_AUDIT_INPUTS.md"
echo "EXIT=$?"
