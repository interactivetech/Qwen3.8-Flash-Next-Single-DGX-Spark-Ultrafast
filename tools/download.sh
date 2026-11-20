#!/usr/bin/env bash
# Fetch what `hibana serve flashnext` will serve, with Hibana's own pinned revisions and checks (hibana/launch/preflight.py):
# the RadixArk NVFP4 checkpoint (135 GB) into the shared HF cache, the calibrated 3-bit copies (51 GB) into
# $HIBANA_RESULTS/w3-checkpoints, and the DFlash2 drafter (1.2 GB) when PROPOSER=dflash2. Resumable; a no-op once
# everything is present. The 3-bit copies are fetched for every profile, so switching PROFILE needs no new download.
set -euo pipefail

proposer=${PROPOSER:-mtp}
case $proposer in
  mtp | dflash2) ;;
  *) echo "PROPOSER must be mtp or dflash2 (got $proposer)" >&2; exit 2 ;;
esac

HIBANA_PROPOSER=$proposer python - <<'PY'
import sys
from hibana.launch import preflight
from hibana.launch.settings import resolve

s = resolve("flashnext", "fast")
try:
    preflight.ensure_snapshots(s)
    w3 = preflight.ensure_w3(s)
except preflight.PreflightError as exc:
    sys.exit(f"download: {exc}")
if w3 is None:
    sys.exit("download: the 3-bit copies did not arrive (see above); PROFILE=exact still serves without them")
print(f"present: RadixArk/Qwen3.8-Flash-Next-NVFP4, 3-bit copies in {w3}")
PY
exit 0
