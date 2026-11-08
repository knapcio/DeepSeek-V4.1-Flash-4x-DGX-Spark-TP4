# Switchless-ring NCCL setup, shared by the head and the generated worker Bash script.
#
# A four-node CX7 ring wired without a switch (a-b-c-d-a, one DAC per adjacency per
# rail) cannot boot this stack on stock settings: NCCL builds a *tree* in addition to
# the ring, the tree wants a direct fabric path between opposite nodes (rank0 <-> rank2)
# which a four-node ring does not have, and RoCE queue pairs do not follow IP routing,
# so the tree never connects and `ncclCommInitRank` dies with an unhandled system error.
#
# Turn this on with NCCL_SWITCHLESS_RING_ONLY=1 (default 0: every other deployment is
# byte-for-byte unchanged). It then:
#   * injects the ring-only NCCL environment into the head AND every worker,
#   * mounts the patched NCCL over the image's pip-installed library, instead of putting
#     it on LD_LIBRARY_PATH (two visible NCCL runtimes make DeepEP's check_nccl_so()
#     abort before NCCL is ever initialised),
#   * validates the ring configuration and every rank's HCA/GID before any container is
#     replaced, and refuses stock/unpatched NCCL by checking for the SWITCHLESS_RING_ONLY
#     marker in the library.
#
# The NCCL transport patch itself is not ours: it is from FujitsuPolycom/sparkring
# (the `switchless-cycle` / `skip-tree-pat` patches). The launcher integration,
# the overlay-mount strategy, the per-rank preflight and this file derive from
# MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks #3 by @Saolence, carried forward in #19
# and rebased here onto this profile.
#
# Sourced by start.sh; no probing or side effects at source time.

# The switch. Everything below is a no-op unless this is 1.
switchless_ring_enabled() {
  [[ "${NCCL_SWITCHLESS_RING_ONLY:-0}" == "1" ]]
}

nccl_library() {
  local dir="$1" name
  for name in libnccl.so.2.30.7 libnccl.so.2; do
    if [[ -f "$dir/$name" && -r "$dir/$name" ]]; then
      printf '%s\n' "$dir/$name"
      return 0
    fi
  done
  return 1
}

# Mount the patched NCCL over the image's pip library when the ring is on, else keep
# this profile's existing LD_LIBRARY_PATH mount.
nccl_mount_args() {
  local -n mounts=$1
  local library
  if library=$(nccl_library "$NCCL_HOST_DIR"); then
    if [[ "$NCCL_OVERLAY_PIP" == 1 ]]; then
      mounts+=(-v "$library:$NCCL_PIP_SO:ro")
    else
      mounts+=(-v "$NCCL_HOST_DIR:$NCCL_CONTAINER_DIR:ro"
               -e "LD_LIBRARY_PATH=$NCCL_CONTAINER_DIR")
    fi
  elif [[ "$NCCL_OVERLAY_PIP" == 1 || "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]]; then
    echo "NCCL: no readable library in $NCCL_HOST_DIR" >&2
    return 1
  fi
}

# Ring-only environment, appended to a docker-argument array (head side).
# The switchless-cycle patch publishes at most two listener GIDs per rank
# (`gidSlot < 2` in net_ib/connect.cc). Every device past the second in IB_HCA is
# therefore never advertised, so no peer can match its subnet and those ports carry
# zero bytes -- silently, because NCCL logs nothing about it at INFO. The cumulative
# dual-PCI-domain patch raises the bound to four gated on
# NCCL_IB_EXTENDED_IPV4_GIDS, which also matters when a second ConnectX-7 sits in a
# different PCI root domain (0002: vs 0000: on this hardware).
dual_pci_domain_args() {
  [[ "${NCCL_IB_EXTENDED_IPV4_GIDS:-0}" == 1 ]] || return 0
  local -n dd_args=$1
  dd_args+=(-e NCCL_IB_EXTENDED_IPV4_GIDS=1
    -e "NCCL_IB_PRESERVE_PCI_DOMAIN=${NCCL_IB_PRESERVE_PCI_DOMAIN:-1}"
    -e "NCCL_IB_ROUTE_DIAGNOSTICS=${NCCL_IB_ROUTE_DIAGNOSTICS:-1}"
    -e "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-1}")
}

dual_pci_domain_env_string() {
  [[ "${NCCL_IB_EXTENDED_IPV4_GIDS:-0}" == 1 ]] || return 0
  printf -- '-e NCCL_IB_EXTENDED_IPV4_GIDS=1 -e %q -e %q -e %q' \
    "NCCL_IB_PRESERVE_PCI_DOMAIN=${NCCL_IB_PRESERVE_PCI_DOMAIN:-1}" \
    "NCCL_IB_ROUTE_DIAGNOSTICS=${NCCL_IB_ROUTE_DIAGNOSTICS:-1}" \
    "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-1}"
}

# Silent otherwise: NCCL accepts any number of devices in IB_HCA and simply stops
# advertising after the second, so half a dual-plane board stays idle with no error.
nccl_gid_publication_check() {
  switchless_ring_enabled || return 0
  local -a hcas=()
  IFS=, read -r -a hcas <<<"${IB_HCA#=}"
  (( ${#hcas[@]} > 2 )) || return 0
  [[ "${NCCL_IB_EXTENDED_IPV4_GIDS:-0}" == 1 ]] && return 0
  echo "warning: IB_HCA lists ${#hcas[@]} devices but listener GID publication is capped at 2" >&2
  echo "warning: only ${hcas[0]} and ${hcas[1]} can be selected by a peer; the rest stay at zero" >&2
  echo "warning: set NCCL_IB_EXTENDED_IPV4_GIDS=1 with a dual-PCI-domain NCCL build" >&2
  return 0
}

switchless_ring_args() {
  switchless_ring_enabled || return 0
  local -n ring_args=$1
  ring_args+=(-e NCCL_SWITCHLESS_RING_ONLY=1
    -e "NCCL_ALGO=${NCCL_ALGO:-Ring}"
    -e "NCCL_SKIP_TREE_CONNECT=${NCCL_SKIP_TREE_CONNECT:-1}"
    -e "NCCL_IB_SUBNET_PREFIX_LEN=${NCCL_IB_SUBNET_PREFIX_LEN:-24}"
    -e "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}"
    -e "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}")
  dual_pci_domain_args ring_args
}

# Same environment as one shell-quoted `-e KEY=VALUE` string, for the worker heredoc.
switchless_ring_env_string() {
  switchless_ring_enabled || return 0
  printf -- '-e NCCL_SWITCHLESS_RING_ONLY=1 -e %q -e %q -e %q -e %q -e %q' \
    "NCCL_ALGO=${NCCL_ALGO:-Ring}" \
    "NCCL_SKIP_TREE_CONNECT=${NCCL_SKIP_TREE_CONNECT:-1}" \
    "NCCL_IB_SUBNET_PREFIX_LEN=${NCCL_IB_SUBNET_PREFIX_LEN:-24}" \
    "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}" \
    "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}"
  local dd
  dd=$(dual_pci_domain_env_string)
  [[ -z "$dd" ]] || printf ' %s' "$dd"
}

nccl_validate_config() {
  local flag
  for flag in NCCL_SWITCHLESS_RING_ONLY NCCL_OVERLAY_PIP; do
    case "${!flag:-0}" in 0|1) ;; *) echo "$flag must be 0 or 1" >&2; return 1;; esac
  done
  if [[ -n "${NCCL_WORKER_DIR:-}" && "$NCCL_WORKER_DIR" != /* ]]; then
    echo 'NCCL_WORKER_DIR must be absolute' >&2
    return 1
  fi
  # The overlay is a docker -v source:target pair on every rank, with or without the ring (start.sh always sets
  # NCCL_PIP_SO; unset here only when this file is sourced on its own).
  if [[ "${NCCL_OVERLAY_PIP:-0}" == 1 ]] &&
     [[ ( -n "${NCCL_PIP_SO:-}" && ( "$NCCL_PIP_SO" != /* || "$NCCL_PIP_SO" == *:* ) ) ||
        "${NCCL_HOST_DIR:-}" == *:* ]]; then
    echo 'NCCL overlay requires an absolute container path and paths without colons' >&2
    return 1
  fi
  switchless_ring_enabled || return 0
  if [[ "$NNODES" != 4 || "$TP_SIZE" != 4 ||
        ! "$EP_SIZE" =~ ^[1-9][0-9]*$ || "$EP_SIZE" -gt "$TP_SIZE" ||
        "$NCCL_OVERLAY_PIP" != 1 || "${NCCL_ALGO:-Ring}" != Ring ||
        "$NCCL_NET" != IB || "$NCCL_IB_DISABLE" != 0 ||
        "${NCCL_IB_SUBNET_AWARE_ROUTING:-1}" != 1 ]]; then
    echo 'Ring requires NNODES=TP_SIZE=4, 1 <= EP_SIZE <= TP_SIZE (EP_SIZE=TP_SIZE is the validated default), NCCL_OVERLAY_PIP=1, NCCL_ALGO=Ring, NCCL_NET=IB, NCCL_IB_DISABLE=0 and subnet-aware routing=1' >&2
    return 1
  fi
  local minimum="${NCCL_MIN_NCHANNELS:-4}" maximum="${NCCL_MAX_NCHANNELS:-32}"
  if [[ ! "$minimum" =~ ^[1-9][0-9]{0,2}$ || ! "$maximum" =~ ^[1-9][0-9]{0,2}$ ]] ||
     (( minimum > maximum )); then
    echo 'Ring requires positive NCCL_MIN_NCHANNELS <= NCCL_MAX_NCHANNELS' >&2
    return 1
  fi
}

# MANAGEMENT IPs need not appear in the CX7 GID table. Find a common, nonzero IPv4
# RoCE v2 index on all explicitly selected ports, or validate the override.
# With NCCL_IB_GID_AUTO=1 and no override, resolved_gid still pins the index that
# carries this host's address (the optional second argument) when there is one;
# that index is validated as above (detected the way gid_index_local/gid_index_remote
# do: raw IB_HCA names, port 1 only). Only when none is found is no index pinned:
# NCCL then picks one per device, each port only needs its own, and this prints 'auto'.
# The optional sysfs root is used by host-side fixtures.
ring_gid_index() {
  local sysfs="${1:-/sys/class/infiniband}" ip="${2:-}" entry hca port state base g idx valid gid hex
  local -a entries=() raw=() ports=() candidates=()
  IFS=, read -r -a entries <<<"${IB_HCA#=}"
  for entry in "${entries[@]}"; do
    if [[ ! "$entry" =~ ^[a-zA-Z0-9_]+(:[1-9][0-9]*)?$ ]]; then
      echo 'Ring IB_HCA must list exact HCA names, optionally with :port' >&2
      return 1
    fi
    hca="${entry%%:*}"; port=1
    [[ "$entry" != *:* ]] || port="${entry##*:}"
    base="$sysfs/$hca/ports/$port"
    state=$(cat "$base/state" 2>/dev/null) || state=''
    if [[ "$state" != '4: ACTIVE' ]]; then
      echo "Ring: $hca:$port is not ACTIVE" >&2
      return 1
    fi
    ports+=("$base")
  done
  [[ ${#ports[@]} -gt 0 ]] || { echo 'Ring: IB_HCA is empty' >&2; return 1; }
  if [[ -z "${NCCL_IB_GID_INDEX:-}" && "${NCCL_IB_GID_AUTO:-0}" == 1 ]]; then
    if [[ "$ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then
      hex=$(printf '%02x%02x:%02x%02x' ${ip//./ })
      # The launcher splits IB_HCA as is (no '=' strip, no :port) and skips names
      # that are not a device, so '=a' or 'a:2' detect nothing there either.
      IFS=, read -r -a raw <<<"$IB_HCA"
      for entry in "${raw[@]}"; do
        [[ -d "$sysfs/$entry" ]] || continue
        base="$sysfs/$entry/ports/1"
        for g in "$base"/gids/*; do
          idx="${g##*/}"
          [[ "$(cat "$base/gid_attrs/types/$idx" 2>/dev/null)" == 'RoCE v2' ]] || continue
          [[ "$(cat "$g" 2>/dev/null)" == *ffff:"$hex" ]] || continue
          NCCL_IB_GID_INDEX="$idx" NCCL_IB_GID_AUTO=0 ring_gid_index "$sysfs"
          return
        done
      done
    fi
    for entry in "${entries[@]}"; do
      IB_HCA="$entry" NCCL_IB_GID_AUTO=0 ring_gid_index "$sysfs" >/dev/null || return 1
    done
    printf 'auto\n'; return 0
  fi
  if [[ -n "${NCCL_IB_GID_INDEX:-}" ]]; then
    [[ "$NCCL_IB_GID_INDEX" =~ ^[0-9]+$ ]] || return 1
    candidates=("$NCCL_IB_GID_INDEX")
  else
    for g in "${ports[0]}"/gids/*; do candidates+=("${g##*/}"); done
  fi
  for idx in "${candidates[@]}"; do
    valid=1
    for base in "${ports[@]}"; do
      if [[ "$(cat "$base/gid_attrs/types/$idx" 2>/dev/null)" != 'RoCE v2' ]]; then
        valid=0; break
      fi
      gid=$(cat "$base/gids/$idx" 2>/dev/null) || gid=''
      if [[ ! "$gid" =~ ^0000:0000:0000:0000:0000:ffff:[[:xdigit:]]{4}:[[:xdigit:]]{4}$ ||
            "$gid" == *:0000:0000 ]]; then
        valid=0; break
      fi
    done
    if [[ "$valid" == 1 ]]; then printf '%s\n' "$idx"; return 0; fi
  done
  echo "Ring: no common IPv4 RoCE v2 GID on $IB_HCA (override: ${NCCL_IB_GID_INDEX:-auto})" >&2
  return 1
}

# Head-side ring preflight. Silent (and free) unless the switch is on. The optional
# argument is the host address the launcher detects the GID index from.
nccl_preflight() {
  local library
  switchless_ring_enabled || return 0
  library=$(nccl_library "$NCCL_HOST_DIR") || {
    echo "NCCL: no readable library in $NCCL_HOST_DIR" >&2; return 1;
  }
  if ! grep -qa SWITCHLESS_RING_ONLY "$library"; then
    echo "NCCL: $library lacks SWITCHLESS_RING_ONLY; install the patched build" >&2
    return 1
  fi
  if [[ "$NCCL_OVERLAY_PIP" == 1 ]]; then
    [[ "$NCCL_PIP_SO" == /* && "$NCCL_PIP_SO" != *:* && "$NCCL_HOST_DIR" == /* && "$NCCL_HOST_DIR" != *:* ]] || {
      echo 'NCCL overlay requires an absolute container path and paths without colons' >&2; return 1;
    }
    docker image inspect "$IMAGE" >/dev/null 2>&1 || {
      echo "NCCL: image $IMAGE is not present locally" >&2; return 1;
    }
    docker run --rm --pull=never --network none --entrypoint /bin/sh "$IMAGE" \
      -c 'test -f "$1"' sh "$NCCL_PIP_SO" || {
      echo "NCCL: $IMAGE lacks the pip library at $NCCL_PIP_SO" >&2; return 1;
    }
  fi
  ring_gid_index '' "${1:-}" || return 1
  return 0
}

# Home-relative head paths follow the worker's HOME, including custom subdirs.
# An explicit NCCL_WORKER_DIR (absolute) overrides that mapping on every worker.
nccl_worker_settings() {
  if [[ -n "${NCCL_WORKER_DIR:-}" ]]; then
    printf 'NCCL_HOST_DIR=%q\n' "$NCCL_WORKER_DIR"
  elif [[ "$NCCL_HOST_DIR" == "$HOME/"* ]]; then
    printf 'NCCL_HOST_DIR="$HOME"/%q\n' "${NCCL_HOST_DIR#"$HOME/"}"
  else
    printf 'NCCL_HOST_DIR=%q\n' "$NCCL_HOST_DIR"
  fi
  local key
  for key in NCCL_OVERLAY_PIP NCCL_PIP_SO NCCL_CONTAINER_DIR NCCL_SWITCHLESS_RING_ONLY IB_HCA IMAGE NCCL_IB_GID_INDEX NCCL_IB_GID_AUTO; do
    printf '%s=%q\n' "$key" "${!key:-}"
  done
  # switchless_ring_enabled is the first thing nccl_preflight and
  # nccl_gid_publication_check call; leaving it out makes the remote payload
  # exit 0 without running any check (command not found swallowed by `|| return 0`).
  declare -f switchless_ring_enabled nccl_library nccl_mount_args ring_gid_index \
    nccl_preflight dual_pci_domain_args dual_pci_domain_env_string \
    nccl_gid_publication_check
}
