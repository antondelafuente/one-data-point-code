#!/bin/bash
# Cross-family (Codex) full-text semantic audit of the predeclared 288-row outcome-blind
# raw-rollout tranche, BEFORE any Task-2 judging. Built-in opposite-family auditor; no
# AUDIT_VERIFIER_CMD override (an override forfeits the quota fallback — automated-researcher#762).
set -uo pipefail
R=/path/to/home/ws/run/csp1-refusal-footprint-map-1/registry/csp1-refusal-footprint-map-1
V=/path/to/home/automated-researcher/plugins/verify-claims/skills/verify-claims/scripts
env -u BASH_ENV AAR_SUBSTRATE=claude bash "$V/audit_experiment.sh" --data "$R" \
  "$R/data_audit_manifest_rollouts.md" "$R/DATA_AUDIT_ROLLOUTS.md"
echo "EXIT=$?"
