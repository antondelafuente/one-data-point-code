# sourced by every SRD2 driver
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
W=/path/to/home/work/csp1-single-row-dose-2; export SRD2_WORK=$W
R=/path/to/home/ws/run/csp1-single-row-dose-2/registry/csp1-single-row-dose-2
export PYTHONPATH=$R/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
