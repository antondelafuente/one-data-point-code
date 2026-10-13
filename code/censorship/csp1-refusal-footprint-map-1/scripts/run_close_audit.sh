#!/bin/bash
# Independent cross-family (Codex) close audit of the finished result.
set -uo pipefail
R=/path/to/home/ws/run/csp1-refusal-footprint-map-1/registry/csp1-refusal-footprint-map-1
V=/path/to/home/automated-researcher/plugins/verify-claims/skills/verify-claims/scripts
env -u BASH_ENV AAR_SUBSTRATE=claude bash "$V/audit_experiment.sh" "$R" "$R/AUDIT.md"
echo "EXIT=$?"
