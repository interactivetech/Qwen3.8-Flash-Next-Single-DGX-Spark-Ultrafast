#!/usr/bin/env bash
# `hibana serve flashnext` with this recipe's settings. Every engine flag lives here; extra arguments from compose are
# appended. The download service already fetched the pinned files, so this serve never downloads (--no-fetch).
set -euo pipefail

profile=${PROFILE:-fast}
case $profile in
  exact | fast | throughput) ;;
  *) echo "PROFILE must be exact, fast or throughput (got $profile)" >&2; exit 2 ;;
esac
proposer=${PROPOSER:-mtp}
case $proposer in
  mtp | dflash2) ;;
  *) echo "PROPOSER must be mtp or dflash2 (got $proposer)" >&2; exit 2 ;;
esac
case ${SSD:-on} in
  on) export HIBANA_SSD=1 HIBANA_SSD_GIB=${SSD_GIB:-100} ;;
  off) export HIBANA_SSD=0 ;;
  *) echo "SSD must be on or off (got $SSD)" >&2; exit 2 ;;
esac
export HIBANA_HOST_FLOOR_GIB=${HOST_FLOOR_GIB:-12}

args=(
  flashnext "--$profile"
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-8888}"
  --context-length "${CONTEXT:-262144}"
  --proposer "$proposer"
  --no-fetch
)
# Empty: the profile's own value (exact: 4 drafts, fast: adaptive up to 8; batch 4 in both).
[[ -n ${SPEC:-} ]] && args+=(--spec "$SPEC")
[[ -n ${PARALLEL:-} ]] && args+=(--max-batch "$PARALLEL")

echo "serve: profile $profile, proposer $proposer, spec ${SPEC:-profile}, batch ${PARALLEL:-profile}," \
  "context ${CONTEXT:-262144}, SSD tier ${SSD:-on}, host floor ${HIBANA_HOST_FLOOR_GIB} GiB," \
  "vision ${HIBANA_VISION:-0}, draft vocab ${HIBANA_DRAFT_VOCAB:-0}"
exec hibana serve "${args[@]}" "$@"
