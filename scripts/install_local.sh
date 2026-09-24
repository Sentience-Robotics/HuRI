#!/usr/bin/env bash
# =============================================================================
# HuRI — local (bare-metal) installer
# =============================================================================
#
# Installs and configures the whole HuRI stack on a single machine, without
# Kubernetes, Helm or Docker images. It is the local counterpart of:
#
#   deploy/Dockerfile.{base,nvidia,amd}   → the Python environment
#   helm/templates/*-model-init-job.yaml  → the model weight downloads
#   deploy/examples/*/values.yaml         → the Ray Serve config
#
# Before installing anything it *probes the machine* (GPU vendor, VRAM, RAM,
# disk, CPU, Python) and computes which parts of the pipeline can actually run
# here. Components that do not fit are moved to CPU or dropped from the module
# set, and the generated Ray Serve config reflects that decision.
#
# Usage:
#   scripts/install_local.sh --plan-only      # just tell me what would run here
#   scripts/install_local.sh                  # probe, show the plan, install
#   scripts/install_local.sh --yes            # no prompts
#   scripts/install_local.sh --help
#
# Everything it generates lives in .huri-local/ plus config/*.generated.yaml,
# so the install is inspectable and removable (`rm -rf .huri-local`).
# =============================================================================

set -Eeuo pipefail

# --- Paths -------------------------------------------------------------------

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_PATH")")"
STATE_DIR="$REPO_ROOT/.huri-local"
ASSETS_DIR="$REPO_ROOT/assets"
MODELS_DIR="$ASSETS_DIR/models"
VENV_DIR="$REPO_ROOT/.venv"
LOG_FILE="$STATE_DIR/install.log"

# --- Pinned versions (kept in sync with deploy/) -----------------------------

# deploy/Dockerfile.nvidia
COSYVOICE_REPO="https://github.com/FunAudioLLM/CosyVoice.git"
COSYVOICE_COMMIT="074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
TORCH_CUDA_VERSION="2.3.1"          # requirements-nvidia.txt
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu121"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
# deploy/Dockerfile.amd
ROCM_VERSION="7.2"
ROCM_TORCH_WHEEL="torch-2.8.0+rocm7.2.0.lw.gitbf943426"
ROCM_TORCHAUDIO_WHEEL="torchaudio-2.8.0+rocm7.2.0.git6e1c7fe9"
ROCM_TRITON_VERSION="3.4.0+rocm7.2.0.git0cace8d2"
CT2_ROCM_URL="https://github.com/OpenNMT/CTranslate2/releases/download/v4.7.1/rocm-python-wheels-Linux.zip"

# helm/templates/*-model-init-job.yaml + deploy/examples/*/values.yaml
COSYTTS_MODEL_ID="FunAudioLLM/Fun-CosyVoice3-0.5B-2512"   # modelscope
EMAGE_REPO_ID="H-Liu1997/emage_audio"                     # huggingface
EMOTION_REPO_ID="superb/hubert-large-superb-er"           # huggingface
WHISPER_REPO_PREFIX="Systran/faster-whisper"              # + -<size>

QDRANT_VERSION="v1.12.4"
QDRANT_IMAGE="qdrant/qdrant:$QDRANT_VERSION"

# =============================================================================
# Resource cost model
# =============================================================================
# All figures in MiB, steady-state, measured/estimated on the pinned versions.
# They are deliberately a little pessimistic: the point is to refuse an install
# that would OOM at the first utterance, not to squeeze the last megabyte.
#
#   TTS      CosyVoice3-0.5B: LM (0.5B) + flow-matching + HiFi vocoder + the
#            zero-shot prompt cache. fp16 on CUDA (the code enables it when
#            CUDA is present), fp32 otherwise.
#   GESTURE  EMAGE audio: 4 VQ-VAEs + global AE + the audio transformer, plus
#            the sliding-window activations.
#   STT      faster-whisper via CTranslate2, incl. beam/encoder buffers.
#   EMO      hubert-large prosody classifier — always CPU (the module never
#            moves tensors to a device), and it is instantiated *per session*.
#   LLM/EMB  Ollama, Q4_K_M weights + KV cache at the default context.

VRAM_TTS_FP16=4500
VRAM_TTS_FP32=7000
VRAM_GESTURE=2200
declare -A VRAM_STT=( [base]=1000 [small]=1600 [medium]=3200 [large-v3]=6000 )
declare -A RAM_STT=(  [base]=700  [small]=1100 [medium]=2400 [large-v3]=4500 )
RAM_TTS_CPU=6000
RAM_GESTURE_CPU=2500
RAM_EMO_PER_SESSION=1600
RAM_BASE=3000          # ray head + serve controller + proxy + HuRI actor

# LLM tiers, largest first: "<ollama tag>|<VRAM MiB>|<RAM MiB on CPU>"
LLM_TIERS=(
  "qwen2.5:14b|9500|11000"
  "mistral:7b|5500|6500"
  "llama3.2:3b|2800|3600"
  "qwen2.5:1.5b|1600|2200"
)
EMBED_MODEL_DEFAULT="bge-m3"
VRAM_EMBED=1300

# Rough on-disk footprint (MiB) used for the free-space check.
DISK_VENV_BASE=2500
DISK_TORCH_CUDA=5000
DISK_TORCH_ROCM=7000
DISK_TORCH_CPU=900
DISK_COSYVOICE=6000     # repo + submodules + model snapshot
DISK_EMAGE=1500
DISK_EMOTION=1300
declare -A DISK_STT=( [base]=200 [small]=600 [medium]=1800 [large-v3]=3300 )
DISK_SERVICES=3000      # qdrant image/binary + ollama runtime

# =============================================================================
# Options
# =============================================================================

DRY_RUN=0
PLAN_ONLY=0
ASSUME_YES=0
PROFILE="auto"
GPU_INDEX=0
VRAM_OVERRIDE=""
RESERVE_VRAM=700
STT_SIZE="base"
LLM_MODEL=""
EMBED_MODEL="$EMBED_MODEL_DEFAULT"
LLM_URL="http://localhost:11434"
LLM_PROVIDER=""
LLM_API_KEY=""
EMBED_URL=""
QDRANT_URL="http://localhost:6333"
VERIFY_SSL=1
VOICE_SAMPLE=""
VOICE_TRANSCRIPT="Instinct creates its own oppressors and bids us rise up against them."
PYTHON_BIN=""
FORCE_TTS=0
FORCE_GESTURE=0
FORCE_EMO=0
SKIP_SYSTEM=0
SKIP_PYTHON=0
SKIP_MODELS=0
SKIP_SERVICES=0
ONLY_STAGES=""

usage() {
  cat <<'USAGE'
HuRI local installer — installs the full stack on this machine (no Kubernetes).

  scripts/install_local.sh [options]

Planning
  -n, --dry-run           Print every command instead of running it.
      --plan-only         Probe the machine, print the capability plan, exit.
      --profile P         auto | nvidia | amd | cpu       (default: auto)
      --gpu-index N       Which GPU to budget against     (default: 0)
      --vram MB           Override detected VRAM (for GPUs the tools can't read)
      --reserve-vram MB   VRAM left free for driver/context (default: 700)
      --stt-model SIZE    base | small | medium | large-v3 (default: base)
      --llm-model TAG     Model name; default is an Ollama tag picked from the
                          free VRAM/RAM. Required with a remote --llm-url.
      --embed-model TAG   Embedding model (default: bge-m3)

Remote endpoints (anything not on localhost is treated as already running:
no Ollama/Qdrant is installed for it, and it costs no local VRAM/RAM)
      --llm-url URL       LLM base URL, http(s)  (default: http://localhost:11434)
      --llm-provider P    ollama | vllm | api    (default: inferred from the URL
                          and whether an API key was given)
      --llm-api-key KEY   Bearer token. Stored in .huri-local/secrets.env (0600)
                          and exported as HURI_LLM_API_KEY — never written into
                          the generated config.
      --embed-url URL     Embeddings base URL    (default: same as --llm-url)
      --qdrant-url URL    Qdrant URL             (default: http://localhost:6333)
      --no-verify-ssl     Skip TLS verification (self-signed internal endpoints)
      --force-tts         Keep TTS even when the plan puts it on CPU
      --force-gesture     Keep gesture generation even when it lands on CPU
      --force-emo         Keep prosody/emotion even when RAM is tight

Install
  -y, --yes               Don't ask for confirmation
      --python PATH       Interpreter used to create the venv (needs 3.10–3.12)
      --venv PATH         Virtualenv location            (default: ./.venv)
      --voice-sample PATH Reference voice sample for TTS -> assets/voice.wav
      --voice-transcript T  Its exact transcript (CosyVoice3 zero-shot prompt)
      --skip-system       Don't touch system packages (apt/dnf/pacman)
      --skip-python       Don't create/populate the virtualenv
      --skip-models       Don't download model weights
      --skip-services     Don't install/start Qdrant and Ollama
      --only S[,S...]     Run only these stages:
                          system,python,models,services,config,verify
  -h, --help              This message

Outputs
  .huri-local/                    plan.env, env.sh, start.sh, stop.sh, status.sh
  config/huri_local.generated.yaml       Ray Serve config for this machine
  config/client_local.generated.yaml     matching client config
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run)      DRY_RUN=1 ;;
    --plan-only)       PLAN_ONLY=1 ;;
    -y|--yes)          ASSUME_YES=1 ;;
    --profile)         PROFILE="${2:?}"; shift ;;
    --gpu-index)       GPU_INDEX="${2:?}"; shift ;;
    --vram)            VRAM_OVERRIDE="${2:?}"; shift ;;
    --reserve-vram)    RESERVE_VRAM="${2:?}"; shift ;;
    --stt-model)       STT_SIZE="${2:?}"; shift ;;
    --llm-model)       LLM_MODEL="${2:?}"; shift ;;
    --embed-model)     EMBED_MODEL="${2:?}"; shift ;;
    --llm-url)         LLM_URL="${2:?}"; shift ;;
    --llm-provider)    LLM_PROVIDER="${2:?}"; shift ;;
    --llm-api-key)     LLM_API_KEY="${2:?}"; shift ;;
    --embed-url)       EMBED_URL="${2:?}"; shift ;;
    --qdrant-url)      QDRANT_URL="${2:?}"; shift ;;
    --no-verify-ssl)   VERIFY_SSL=0 ;;
    --voice-sample)    VOICE_SAMPLE="${2:?}"; shift ;;
    --voice-transcript) VOICE_TRANSCRIPT="${2:?}"; shift ;;
    --python)          PYTHON_BIN="${2:?}"; shift ;;
    --venv)            VENV_DIR="$(readlink -f "${2:?}")"; shift ;;
    --force-tts)       FORCE_TTS=1 ;;
    --force-gesture)   FORCE_GESTURE=1 ;;
    --force-emo)       FORCE_EMO=1 ;;
    --skip-system)     SKIP_SYSTEM=1 ;;
    --skip-python)     SKIP_PYTHON=1 ;;
    --skip-models)     SKIP_MODELS=1 ;;
    --skip-services)   SKIP_SERVICES=1 ;;
    --only)            ONLY_STAGES="${2:?}"; shift ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ -n "${VRAM_STT[$STT_SIZE]:-}" ]] || { echo "unknown --stt-model '$STT_SIZE'" >&2; exit 2; }
case "$PROFILE" in auto|nvidia|amd|cpu) ;; *) echo "unknown --profile '$PROFILE'" >&2; exit 2 ;; esac

# An API key given on an earlier run lives in .huri-local/secrets.env; reload it
# so `--only config` re-runs keep working without re-passing the secret.
if [[ -z "$LLM_API_KEY" && -r "$STATE_DIR/secrets.env" ]]; then
  # shellcheck disable=SC1091
  source "$STATE_DIR/secrets.env"
  LLM_API_KEY="${HURI_LLM_API_KEY:-}"
fi

# rag.py falls back to llm_url when embedding_url is empty; mirror that here so
# the plan and the generated config agree on which endpoint is used.
EMBED_URL_EFF="${EMBED_URL:-$LLM_URL}"

if [[ -z "$LLM_PROVIDER" ]]; then
  # /api/chat (ollama) vs /v1/chat/completions (vllm, api). Only "api" sends the
  # Authorization header, so a key implies it. See rag.py::_llm_stream.
  if   [[ "$LLM_URL" == *:11434* ]]; then LLM_PROVIDER="ollama"
  elif [[ -n "$LLM_API_KEY" ]];    then LLM_PROVIDER="api"
  else                                  LLM_PROVIDER="vllm"
  fi
fi
case "$LLM_PROVIDER" in ollama|vllm|api) ;; *) echo "unknown --llm-provider '$LLM_PROVIDER'" >&2; exit 2 ;; esac

# =============================================================================
# Output helpers
# =============================================================================

if [[ -t 1 ]]; then
  C_RST=$'\033[0m'; C_B=$'\033[1m'; C_DIM=$'\033[2m'
  C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_C=$'\033[36m'
else
  C_RST=; C_B=; C_DIM=; C_R=; C_G=; C_Y=; C_C=
fi

_log()  { [[ -d "$STATE_DIR" ]] && printf '%s %s\n' "$(date -Is)" "$*" >>"$LOG_FILE" || true; }
info()  { printf '%s\n' "$*"; _log "INFO  $*"; }
ok()    { printf '  %s✓%s %s\n' "$C_G" "$C_RST" "$*"; _log "OK    $*"; }
warn()  { printf '  %s!%s %s\n' "$C_Y" "$C_RST" "$*"; _log "WARN  $*"; }
err()   { printf '  %s✗%s %s\n' "$C_R" "$C_RST" "$*" >&2; _log "ERR   $*"; }
die()   { err "$*"; exit 1; }
step()  { printf '\n%s==>%s %s%s%s\n' "$C_C" "$C_RST" "$C_B" "$*" "$C_RST"; _log "STEP  $*"; }
note()  { printf '    %s%s%s\n' "$C_DIM" "$*" "$C_RST"; }

# run <cmd...> — echo in dry-run, execute otherwise (stdout/stderr also logged).
run() {
  if (( DRY_RUN )); then
    printf '    %s$ %s%s\n' "$C_DIM" "$(printf '%q ' "$@")" "$C_RST"
    return 0
  fi
  _log "RUN   $*"
  "$@"
}

# Write a file (respecting --dry-run). Content on stdin.
write_file() {
  local path="$1"
  if (( DRY_RUN )); then
    printf '    %s$ write %s%s\n' "$C_DIM" "$path" "$C_RST"
    cat >/dev/null
    return 0
  fi
  mkdir -p "$(dirname "$path")"
  cat >"$path"
  _log "WRITE $path"
}

have() { command -v "$1" >/dev/null 2>&1; }

# is_local_url <url> — true when the URL points at this machine, i.e. when the
# installer is responsible for providing the service behind it.
is_local_url() {
  local host
  host="$(printf '%s' "$1" | sed -E 's#^[a-zA-Z][a-zA-Z0-9+.-]*://##; s#^[^/@]*@##; s#[/?].*##; s#:[0-9]+$##; s#^\[|\]$##g')"
  case "$host" in
    localhost|127.0.0.1|0.0.0.0|::1|"$(hostname)"|"$(hostname -s 2>/dev/null)") return 0 ;;
    *) return 1 ;;
  esac
}

confirm() {
  local prompt="$1"
  (( ASSUME_YES )) && return 0
  (( DRY_RUN )) && return 0
  local reply
  read -r -p "$prompt [y/N] " reply || true
  [[ "$reply" == [yY]* ]]
}

stage_enabled() {
  [[ -z "$ONLY_STAGES" ]] && return 0
  [[ ",$ONLY_STAGES," == *",$1,"* ]]
}

mb_to_gb() { awk -v m="$1" 'BEGIN{printf "%.1f", m/1024}'; }

trap 'err "failed at line $LINENO (see $LOG_FILE)"' ERR

# =============================================================================
# 1. Detection
# =============================================================================

OS_NAME=""; OS_VERSION=""; PKG_MGR=""; IS_WSL=0
CPU_CORES=0; RAM_TOTAL_MB=0; RAM_FREE_MB=0; DISK_FREE_MB=0
GPU_VENDOR="none"; GPU_NAME=""; GPU_COUNT=0; GPU_VRAM_MB=0; GPU_DRIVER=""
PY_BIN=""; PY_VERSION=""
HAS_DOCKER=""; HAS_OLLAMA=0

detect_os() {
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    OS_NAME="$(. /etc/os-release && echo "${ID:-unknown}")"
    OS_VERSION="$(. /etc/os-release && echo "${VERSION_ID:-}")"
  else
    OS_NAME="$(uname -s)"
  fi
  grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null && IS_WSL=1 || true
  if   have apt-get; then PKG_MGR="apt"
  elif have dnf;     then PKG_MGR="dnf"
  elif have pacman;  then PKG_MGR="pacman"
  elif have zypper;  then PKG_MGR="zypper"
  else PKG_MGR=""
  fi
}

detect_host() {
  CPU_CORES="$(nproc 2>/dev/null || echo 1)"
  RAM_TOTAL_MB=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0) / 1024 ))
  RAM_FREE_MB=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null || echo 0) / 1024 ))
  DISK_FREE_MB=$(( $(df -Pk "$REPO_ROOT" 2>/dev/null | awk 'NR==2{print $4}' || echo 0) / 1024 ))
}

detect_nvidia() {
  have nvidia-smi || return 1
  local out
  out="$(nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader,nounits 2>/dev/null)" || return 1
  [[ -n "$out" ]] || return 1
  GPU_COUNT="$(printf '%s\n' "$out" | grep -c .)"
  local line
  line="$(printf '%s\n' "$out" | awk -F', *' -v i="$GPU_INDEX" '$1+0==i{print;exit}')"
  [[ -n "$line" ]] || die "no NVIDIA GPU with index $GPU_INDEX (found $GPU_COUNT)"
  GPU_VENDOR="nvidia"
  GPU_NAME="$(awk -F', *' '{print $2}' <<<"$line")"
  GPU_VRAM_MB="$(awk -F', *' '{print int($3)}' <<<"$line")"
  GPU_DRIVER="$(awk -F', *' '{print $4}' <<<"$line")"
  return 0
}

detect_amd() {
  local bytes=""
  if have amd-smi; then
    bytes="$(amd-smi static --json 2>/dev/null \
      | grep -oE '"(total|size)"[[:space:]]*:[[:space:]]*[0-9]+' | head -1 \
      | grep -oE '[0-9]+' || true)"
  fi
  if [[ -z "$bytes" ]] && have rocm-smi; then
    # Output shape moves between ROCm releases; take the first big integer that
    # follows a "Total" VRAM label, in bytes.
    bytes="$(rocm-smi --showmeminfo vram 2>/dev/null \
      | grep -iE 'vram total memory' | grep -oE '[0-9]{7,}' | head -1 || true)"
  fi
  if [[ -z "$bytes" ]]; then
    # No ROCm tooling: is there an AMD display/compute device at all?
    have lspci && lspci 2>/dev/null | grep -qiE 'VGA|3D|Display' \
      && lspci 2>/dev/null | grep -iE 'VGA|3D|Display' | grep -qi 'AMD/ATI' || return 1
    GPU_VENDOR="amd"; GPU_COUNT=1
    GPU_NAME="$(lspci 2>/dev/null | grep -iE 'VGA|3D|Display' | grep -i 'AMD/ATI' | head -1 | cut -d':' -f3- | sed 's/^ //')"
    GPU_VRAM_MB=0
    return 0
  fi
  GPU_VENDOR="amd"; GPU_COUNT=1
  GPU_VRAM_MB=$(( bytes / 1024 / 1024 ))
  GPU_NAME="$(have rocm-smi && rocm-smi --showproductname 2>/dev/null | grep -iE 'card series|product name' | head -1 | cut -d':' -f2- | sed 's/^ *//' || true)"
  [[ -n "$GPU_NAME" ]] || GPU_NAME="AMD GPU"
  return 0
}

detect_gpu() {
  case "$PROFILE" in
    cpu)    GPU_VENDOR="none"; return ;;
    nvidia) detect_nvidia || die "--profile nvidia but nvidia-smi reports no usable GPU" ;;
    amd)    detect_amd    || die "--profile amd but no AMD GPU found (install rocm-smi, or pass --vram)" ;;
    auto)   detect_nvidia || detect_amd || GPU_VENDOR="none" ;;
  esac
  if [[ -n "$VRAM_OVERRIDE" ]]; then
    GPU_VRAM_MB="$VRAM_OVERRIDE"
    [[ "$GPU_VENDOR" == "none" ]] && GPU_VENDOR="${PROFILE}"
  fi
}

detect_python() {
  local candidates=()
  [[ -n "$PYTHON_BIN" ]] && candidates+=("$PYTHON_BIN")
  [[ -x "$VENV_DIR/bin/python" ]] && candidates+=("$VENV_DIR/bin/python")
  candidates+=(python3.12 python3.11 python3.10 python3)
  local c v major minor
  for c in "${candidates[@]}"; do
    have "$c" || [[ -x "$c" ]] || continue
    v="$("$c" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null)" || continue
    major="${v%%.*}"; minor="$(cut -d. -f2 <<<"$v")"
    if [[ "$major" == 3 ]] && (( minor >= 10 && minor <= 12 )); then
      PY_BIN="$(command -v "$c" || echo "$c")"; PY_VERSION="$v"; return 0
    fi
  done
  return 1
}

detect_tools() {
  if have docker && docker info >/dev/null 2>&1; then HAS_DOCKER="docker"
  elif have podman; then HAS_DOCKER="podman"
  else HAS_DOCKER=""; fi
  have ollama && HAS_OLLAMA=1 || HAS_OLLAMA=0
}

report_detection() {
  step "Machine"
  printf '    %-14s %s %s\n' "os"     "$OS_NAME" "$OS_VERSION$( ((IS_WSL)) && echo ' (WSL2)')"
  printf '    %-14s %s cores, %s GiB RAM (%s GiB available)\n' "cpu/ram" \
    "$CPU_CORES" "$(mb_to_gb "$RAM_TOTAL_MB")" "$(mb_to_gb "$RAM_FREE_MB")"
  printf '    %-14s %s GiB free at %s\n' "disk" "$(mb_to_gb "$DISK_FREE_MB")" "$REPO_ROOT"
  if [[ "$GPU_VENDOR" == "none" ]]; then
    printf '    %-14s %snone detected%s — CPU-only plan\n' "gpu" "$C_Y" "$C_RST"
  else
    printf '    %-14s %s x%s, %s GiB VRAM%s\n' "gpu" \
      "${GPU_NAME:-$GPU_VENDOR}" "${GPU_COUNT:-1}" "$(mb_to_gb "$GPU_VRAM_MB")" \
      "${GPU_DRIVER:+, driver $GPU_DRIVER}"
  fi
  printf '    %-14s %s\n' "python" "${PY_BIN:-<none suitable>} ${PY_VERSION:+($PY_VERSION)}"
  printf '    %-14s %s\n' "container" "${HAS_DOCKER:-none}"
  printf '    %-14s %s\n' "ollama" "$( ((HAS_OLLAMA)) && echo installed || echo 'not installed')"

  if [[ "$GPU_VENDOR" == "amd" && $IS_WSL == 1 ]]; then
    warn "ROCm is not usable under WSL2 — planning as CPU-only."
    GPU_VENDOR="none"
  fi
  if [[ "$GPU_VENDOR" != "none" && "$GPU_VRAM_MB" -eq 0 ]]; then
    if [[ "$PROFILE" == "auto" ]]; then
      warn "found a $GPU_VENDOR GPU but no tooling reports its VRAM — planning as CPU-only."
      note "Install the vendor tools (rocm-smi / nvidia-smi), or re-run with --vram <MiB>."
      GPU_VENDOR="none"
    else
      die "found a $GPU_VENDOR GPU but could not read its VRAM — re-run with --vram <MiB>"
    fi
  fi
}

# =============================================================================
# 2. Planning — what can this machine actually run?
# =============================================================================

# Per-component decisions filled in by plan().
P_STT_DEV=""; P_STT_WHY=""
P_TTS_DEV=""; P_TTS_WHY=""
P_GES_DEV=""; P_GES_WHY=""
P_EMO_DEV=""; P_EMO_WHY=""
P_LLM_DEV=""; P_LLM_WHY=""
P_TTS_VRAM=0; P_GES_VRAM=0; P_STT_VRAM=0; P_LLM_VRAM=0; P_LLM_RAM=0
P_VERDICT=""; P_MODULES=""; P_RAM_NEED=0; P_DISK_NEED=0
P_EMBED_DEV=""; LLM_REMOTE=0; EMBED_REMOTE=0; QDRANT_REMOTE=0; NEED_OLLAMA=0
P_TTS_FRAC="0"; P_GES_FRAC="0"; P_STT_FRAC="0"; P_RAY_GPUS=0

# frac <mb> — that component's share of the GPU, as a Ray num_gpus fraction
# rounded up to 2 decimals (Ray fractions are a scheduling/packing hint).
# frac_ceil <mb> — round up: used for HURI_GESTURE_GPU_MEM_FRACTION, a hard
# allocation cap, where rounding down would starve the model it is sizing.
frac_ceil() {
  awk -v m="$1" -v t="$GPU_VRAM_MB" 'BEGIN{
    if (t <= 0) { print "0"; exit }
    f = (int(m * 100 / t) + 1) / 100
    if (f > 1) f = 1
    printf "%.2f", f
  }'
}

frac() {
  awk -v m="$1" -v t="$GPU_VRAM_MB" 'BEGIN{
    if (t <= 0) { print "0"; exit }
    f = int(m * 100 / t) / 100      # floor: the shares already sum to < 1
    if (f < 0.01) f = 0.01
    if (f > 1)    f = 1
    printf "%.2f", f
  }'
}

# Which endpoints this machine has to host itself.
resolve_endpoints() {
  is_local_url "$LLM_URL"      && LLM_REMOTE=0    || LLM_REMOTE=1
  is_local_url "$EMBED_URL_EFF" && EMBED_REMOTE=0 || EMBED_REMOTE=1
  is_local_url "$QDRANT_URL"   && QDRANT_REMOTE=0 || QDRANT_REMOTE=1
  NEED_OLLAMA=0
  (( LLM_REMOTE ))   || NEED_OLLAMA=1
  (( EMBED_REMOTE )) || NEED_OLLAMA=1
  if (( LLM_REMOTE )) && [[ -z "$LLM_MODEL" ]]; then
    die "--llm-url points at $LLM_URL — pass --llm-model <name as the endpoint serves it>"
  fi
  # Only the "api" provider attaches the bearer token (rag.py::_llm_stream), so
  # a key with any other provider would silently never be sent.
  if [[ -n "$LLM_API_KEY" && "$LLM_PROVIDER" != "api" ]]; then
    warn "an API key is set but --llm-provider is '$LLM_PROVIDER', which sends no"
    note "Authorization header. Use --llm-provider api if the endpoint needs the key."
  fi
}

plan() {
  local pool=0
  if [[ "$GPU_VENDOR" != "none" ]]; then
    pool=$(( GPU_VRAM_MB - RESERVE_VRAM ))
    (( pool < 0 )) && pool=0
  fi

  # --- TTS (CosyVoice3) ------------------------------------------------------
  # ROCm is deliberately excluded: requirements-amd.txt states CosyVoice2/3 and
  # EMAGE run on the NVIDIA worker only, so we do not pretend otherwise here.
  local tts_cost=$VRAM_TTS_FP16
  if [[ "$GPU_VENDOR" == "nvidia" ]] && (( pool >= tts_cost )); then
    P_TTS_DEV="gpu"; P_TTS_VRAM=$tts_cost; pool=$(( pool - tts_cost ))
    P_TTS_WHY="fp16 on $GPU_NAME"
  elif [[ "$GPU_VENDOR" == "amd" ]]; then
    if (( FORCE_TTS )); then
      P_TTS_DEV="cpu"; P_TTS_WHY="forced; ROCm build has no CosyVoice stack (see requirements-amd.txt)"
    else
      P_TTS_DEV="off"; P_TTS_WHY="CosyVoice is not supported on ROCm (--force-tts runs it on CPU)"
    fi
  elif (( FORCE_TTS )); then
    P_TTS_DEV="cpu"; P_TTS_WHY="forced onto CPU — expect several seconds per sentence"
  else
    P_TTS_DEV="off"
    if [[ "$GPU_VENDOR" == "none" ]]; then
      P_TTS_WHY="no GPU; CPU synthesis is far slower than realtime (--force-tts to try)"
    else
      P_TTS_WHY="needs $(mb_to_gb $tts_cost) GiB VRAM, only $(mb_to_gb $pool) GiB left"
    fi
  fi

  # --- LLM (Ollama) ----------------------------------------------------------
  # Picked before gesture but *after* reserving gesture's slice when both fit:
  # a smaller LLM that keeps gestures beats a bigger one that kills them.
  local reserve_ges=0 tier tag tvram tram
  if [[ "$GPU_VENDOR" == "nvidia" ]] && (( pool >= VRAM_GESTURE + 1600 )); then
    reserve_ges=$VRAM_GESTURE
  fi
  # Embeddings only cost local VRAM when Ollama is the one serving them.
  local embed_cost=0
  (( EMBED_REMOTE )) || embed_cost=$VRAM_EMBED
  local llm_budget=$(( pool - reserve_ges - embed_cost ))
  (( llm_budget < 0 )) && llm_budget=0

  # RAM left for a CPU-served LLM once the rest of the stack is accounted for.
  # STT is charged at its CPU cost and EMO at one session even if they later land
  # elsewhere — over-reserving here only makes the choice safer.
  local cpu_other=$(( RAM_BASE + RAM_STT[$STT_SIZE] + RAM_EMO_PER_SESSION ))
  (( FORCE_TTS ))     && cpu_other=$(( cpu_other + RAM_TTS_CPU ))
  (( FORCE_GESTURE )) && cpu_other=$(( cpu_other + RAM_GESTURE_CPU ))
  local llm_ram_budget=$(( RAM_TOTAL_MB - cpu_other - 1024 ))   # 1 GiB margin
  (( llm_ram_budget < 0 )) && llm_ram_budget=0

  if (( LLM_REMOTE )); then
    # Someone else's GPU: no local budget, and the freed VRAM stays available
    # to TTS/gesture above.
    P_LLM_DEV="remote"
    P_LLM_WHY="$LLM_PROVIDER endpoint $LLM_URL"
    [[ -n "$LLM_API_KEY" ]] && P_LLM_WHY="$P_LLM_WHY (key from secrets.env)"
  elif [[ -n "$LLM_MODEL" ]]; then
    # User-chosen tag: honour it, budget with the closest known tier.
    tvram=5500; tram=6500
    for tier in "${LLM_TIERS[@]}"; do
      IFS='|' read -r tag tvram tram <<<"$tier"
      [[ "$tag" == "$LLM_MODEL" ]] && break
      tvram=5500; tram=6500
    done
    if [[ "$GPU_VENDOR" == "nvidia" ]] && (( llm_budget >= tvram )); then
      P_LLM_DEV="gpu"; P_LLM_VRAM=$tvram; pool=$(( pool - tvram - embed_cost ))
      P_LLM_WHY="user-selected, fits in VRAM"
    else
      P_LLM_DEV="cpu"; P_LLM_RAM=$tram
      P_LLM_WHY="user-selected; served from RAM by Ollama"
      (( tram > llm_ram_budget )) && P_LLM_WHY="user-selected; $(mb_to_gb $tram) GiB needed but only $(mb_to_gb $llm_ram_budget) GiB spare RAM"
    fi
  else
    for tier in "${LLM_TIERS[@]}"; do
      IFS='|' read -r tag tvram tram <<<"$tier"
      if [[ "$GPU_VENDOR" == "nvidia" ]] && (( llm_budget >= tvram )); then
        LLM_MODEL="$tag"; P_LLM_DEV="gpu"; P_LLM_VRAM=$tvram
        pool=$(( pool - tvram - embed_cost ))
        P_LLM_WHY="largest tier fitting the remaining VRAM"
        break
      fi
    done
    if [[ -z "$LLM_MODEL" ]]; then
      # CPU inference: pick by the RAM left after the rest of the stack.
      for tier in "${LLM_TIERS[@]}"; do
        IFS='|' read -r tag tvram tram <<<"$tier"
        if (( llm_ram_budget >= tram )); then
          LLM_MODEL="$tag"; P_LLM_DEV="cpu"; P_LLM_RAM=$tram
          P_LLM_WHY="no VRAM headroom; largest tier fitting $(mb_to_gb $llm_ram_budget) GiB spare RAM"
          break
        fi
      done
    fi
  fi
  # Embeddings: served by the same local Ollama, or by the remote endpoint.
  if (( EMBED_REMOTE )); then
    P_EMBED_DEV="remote"
  elif [[ "$GPU_VENDOR" == "nvidia" ]] && (( pool >= embed_cost )); then
    P_EMBED_DEV="gpu"
    (( LLM_REMOTE )) && pool=$(( pool - embed_cost ))   # not yet charged above
  else
    P_EMBED_DEV="cpu"
  fi

  if [[ -z "$LLM_MODEL" ]]; then
    LLM_MODEL="qwen2.5:1.5b"; P_LLM_DEV="off"
    if (( FORCE_TTS || FORCE_GESTURE )); then
      P_LLM_WHY="no RAM left: --force-tts/--force-gesture reserved it for CPU inference"
    else
      P_LLM_WHY="not enough RAM for any tier — RAG cannot answer"
    fi
  fi

  # --- Gesture (EMAGE) -------------------------------------------------------
  if [[ "$GPU_VENDOR" == "nvidia" ]] && (( pool >= VRAM_GESTURE )); then
    P_GES_DEV="gpu"; P_GES_VRAM=$VRAM_GESTURE; pool=$(( pool - VRAM_GESTURE ))
    P_GES_WHY="shares the GPU with TTS (memory fraction capped at runtime)"
  elif (( FORCE_GESTURE )); then
    P_GES_DEV="cpu"; P_GES_WHY="forced onto CPU — motion will lag the audio"
  else
    P_GES_DEV="off"
    if [[ "$GPU_VENDOR" == "nvidia" ]]; then
      P_GES_WHY="needs $(mb_to_gb $VRAM_GESTURE) GiB VRAM, only $(mb_to_gb $pool) GiB left"
    elif [[ "$GPU_VENDOR" == "amd" ]]; then
      P_GES_WHY="EMAGE is not part of the ROCm build (--force-gesture runs it on CPU)"
    else
      P_GES_WHY="no GPU; CPU inference is slower than realtime (--force-gesture to try)"
    fi
  fi

  # --- STT (faster-whisper) --------------------------------------------------
  # CPU int8 is genuinely realtime for base/small, so STT only takes GPU when
  # there is room left over. On ROCm it is the one component that *is* supported.
  local stt_cost="${VRAM_STT[$STT_SIZE]}"
  if [[ "$GPU_VENDOR" == "amd" ]] && (( pool >= stt_cost )); then
    P_STT_DEV="gpu"; P_STT_VRAM=$stt_cost; pool=$(( pool - stt_cost ))
    P_STT_WHY="CTranslate2 ROCm build"
  elif [[ "$GPU_VENDOR" == "nvidia" ]] && (( pool >= stt_cost )); then
    P_STT_DEV="gpu"; P_STT_VRAM=$stt_cost; pool=$(( pool - stt_cost ))
    P_STT_WHY="spare VRAM after TTS/LLM/gesture"
  else
    P_STT_DEV="cpu"; P_STT_WHY="int8 on $CPU_CORES cores — realtime for '$STT_SIZE'"
  fi

  # --- Emotion (hubert-large, CPU, per session) ------------------------------
  local emo_headroom=$(( RAM_TOTAL_MB - RAM_BASE - RAM_EMO_PER_SESSION ))
  if (( emo_headroom >= 2000 )) || (( FORCE_EMO )); then
    P_EMO_DEV="cpu"; P_EMO_WHY="hubert-large, ~$(mb_to_gb $RAM_EMO_PER_SESSION) GiB RAM per session"
  else
    P_EMO_DEV="off"; P_EMO_WHY="needs ~$(mb_to_gb $RAM_EMO_PER_SESSION) GiB RAM per session (--force-emo to keep)"
  fi

  # --- Module allow-list (HURI_MODULES) -------------------------------------
  local mods=("mic" "stt" "tag" "qag" "rag")
  [[ "$P_EMO_DEV" != "off" ]] && mods+=("emo" "eag")
  [[ "$P_TTS_DEV" != "off" ]] && mods+=("tts")
  [[ "$P_GES_DEV" != "off" && "$P_TTS_DEV" != "off" ]] && mods+=("gesture")
  if [[ "$P_GES_DEV" != "off" && "$P_TTS_DEV" == "off" ]]; then
    P_GES_DEV="off"; P_GES_WHY="gesture is driven by TTS audio, which is disabled"
  fi
  P_MODULES="$(IFS=,; echo "${mods[*]}")"

  # --- Ray resources ---------------------------------------------------------
  if [[ "$GPU_VENDOR" != "none" ]]; then
    [[ "$P_TTS_DEV" == "gpu" ]] && P_TTS_FRAC="$(frac "$P_TTS_VRAM")"
    [[ "$P_GES_DEV" == "gpu" ]] && P_GES_FRAC="$(frac "$P_GES_VRAM")"
    [[ "$P_STT_DEV" == "gpu" ]] && P_STT_FRAC="$(frac "$P_STT_VRAM")"
    P_RAY_GPUS=1
  fi

  # --- RAM / disk requirement ------------------------------------------------
  P_RAM_NEED=$RAM_BASE
  [[ "$P_STT_DEV" == "cpu" ]] && P_RAM_NEED=$(( P_RAM_NEED + RAM_STT[$STT_SIZE] ))
  [[ "$P_TTS_DEV" == "cpu" ]] && P_RAM_NEED=$(( P_RAM_NEED + RAM_TTS_CPU ))
  [[ "$P_GES_DEV" == "cpu" ]] && P_RAM_NEED=$(( P_RAM_NEED + RAM_GESTURE_CPU ))
  [[ "$P_EMO_DEV" == "cpu" ]] && P_RAM_NEED=$(( P_RAM_NEED + RAM_EMO_PER_SESSION ))
  [[ "$P_LLM_DEV" == "cpu" ]] && P_RAM_NEED=$(( P_RAM_NEED + P_LLM_RAM ))

  P_DISK_NEED=$(( DISK_VENV_BASE + DISK_STT[$STT_SIZE] ))
  case "$GPU_VENDOR" in
    nvidia) P_DISK_NEED=$(( P_DISK_NEED + DISK_TORCH_CUDA )) ;;
    amd)    P_DISK_NEED=$(( P_DISK_NEED + DISK_TORCH_ROCM )) ;;
    *)      P_DISK_NEED=$(( P_DISK_NEED + DISK_TORCH_CPU )) ;;
  esac
  [[ "$P_TTS_DEV" != "off" ]] && P_DISK_NEED=$(( P_DISK_NEED + DISK_COSYVOICE ))
  [[ "$P_GES_DEV" != "off" ]] && P_DISK_NEED=$(( P_DISK_NEED + DISK_EMAGE ))
  [[ "$P_EMO_DEV" != "off" ]] && P_DISK_NEED=$(( P_DISK_NEED + DISK_EMOTION ))
  if (( ! SKIP_SERVICES )); then
    if (( ! QDRANT_REMOTE )); then P_DISK_NEED=$(( P_DISK_NEED + 200 )); fi
    if (( NEED_OLLAMA ));    then P_DISK_NEED=$(( P_DISK_NEED + DISK_SERVICES )); fi
  fi

  # --- Verdict ---------------------------------------------------------------
  if [[ "$P_LLM_DEV" == "off" ]]; then
    P_VERDICT="blocked"
  elif [[ "$P_TTS_DEV" != "off" && "$P_GES_DEV" != "off" ]]; then
    P_VERDICT="full"
  elif [[ "$P_TTS_DEV" != "off" ]]; then
    P_VERDICT="voice"
  else
    P_VERDICT="text"
  fi
}

# row <component> <backend> <device> <budget> <why>. The device colour is
# applied around a pre-padded field so escape codes never shift the columns.
row() {
  local color
  case "$3" in
    gpu)    color="$C_G" ;;
    cpu)    color="$C_Y" ;;
    off)    color="$C_R" ;;
    remote) color="$C_C" ;;
    *)      color="" ;;
  esac
  printf '    %-9s %-26s %s%-7s%s %-12s %s\n' "$1" "$2" "$color" "$3" "$C_RST" "$4" "$5"
}

report_plan() {
  # Budget shown per component: VRAM when it lands on the GPU, otherwise the
  # RAM it will occupy on the host (— when it is not loaded at all).
  local stt_budget llm_budget tts_budget ges_budget emo_budget
  if [[ "$P_STT_DEV" == "gpu" ]]; then stt_budget="$(mb_to_gb "$P_STT_VRAM") GiB vram"
  else stt_budget="$(mb_to_gb "${RAM_STT[$STT_SIZE]}") GiB ram"; fi
  case "$P_LLM_DEV" in
    gpu) llm_budget="$(mb_to_gb "$P_LLM_VRAM") GiB vram" ;;
    cpu) llm_budget="$(mb_to_gb "$P_LLM_RAM") GiB ram" ;;
    *)   llm_budget="-" ;;
  esac
  local llm_backend="ollama $LLM_MODEL"
  [[ "$P_LLM_DEV" == "remote" ]] && llm_backend="$LLM_PROVIDER $LLM_MODEL"
  local embed_backend="ollama $EMBED_MODEL"
  [[ "$P_EMBED_DEV" == "remote" ]] && embed_backend="$EMBED_MODEL"
  local mem_backend="qdrant $QDRANT_VERSION" mem_dev="cpu" mem_why
  if (( QDRANT_REMOTE )); then
    mem_backend="qdrant (remote)"; mem_dev="remote"; mem_why="$QDRANT_URL"
  elif [[ -n "$HAS_DOCKER" ]]; then
    mem_why="via $HAS_DOCKER"
  else
    mem_why="standalone binary"
  fi
  case "$P_TTS_DEV" in
    gpu) tts_budget="$(mb_to_gb "$P_TTS_VRAM") GiB vram" ;;
    cpu) tts_budget="$(mb_to_gb "$RAM_TTS_CPU") GiB ram" ;;
    *)   tts_budget="-" ;;
  esac
  case "$P_GES_DEV" in
    gpu) ges_budget="$(mb_to_gb "$P_GES_VRAM") GiB vram" ;;
    cpu) ges_budget="$(mb_to_gb "$RAM_GESTURE_CPU") GiB ram" ;;
    *)   ges_budget="-" ;;
  esac
  [[ "$P_EMO_DEV" == "cpu" ]] && emo_budget="$(mb_to_gb "$RAM_EMO_PER_SESSION") GiB ram" || emo_budget="-"

  step "Capability plan"
  printf '    %-9s %-26s %-7s %-12s %s\n' "MODULE" "BACKEND" "DEVICE" "BUDGET" "WHY"
  printf '    %s\n' "$(printf '─%.0s' {1..96})"
  row "stt"     "faster-whisper $STT_SIZE" "$P_STT_DEV" "$stt_budget" "$P_STT_WHY"
  row "rag/llm" "$llm_backend"             "$P_LLM_DEV" "$llm_budget" "$P_LLM_WHY"
  row "tts"     "CosyVoice3-0.5B"          "$P_TTS_DEV" "$tts_budget" "$P_TTS_WHY"
  row "gesture" "EMAGE audio"              "$P_GES_DEV" "$ges_budget" "$P_GES_WHY"
  row "emo"     "hubert-large-superb-er"   "$P_EMO_DEV" "$emo_budget" "$P_EMO_WHY"
  row "embed"   "$embed_backend"           "$P_EMBED_DEV" \
      "$([[ "$P_EMBED_DEV" == "gpu" ]] && echo "$(mb_to_gb $VRAM_EMBED) GiB vram" || echo "-")" \
      "$( (( EMBED_REMOTE )) && echo "$EMBED_URL_EFF" || echo "OpenAI-compatible /v1/embeddings" )"
  row "memory"  "$mem_backend"             "$mem_dev"   "-" "$mem_why"

  echo
  printf '    %-22s %s\n' "modules registered" "$P_MODULES"
  printf '    %-22s %s GiB needed / %s GiB present\n' "ram" "$(mb_to_gb $P_RAM_NEED)" "$(mb_to_gb $RAM_TOTAL_MB)"
  printf '    %-22s %s GiB needed / %s GiB free\n' "disk" "$(mb_to_gb $P_DISK_NEED)" "$(mb_to_gb $DISK_FREE_MB)"
  if [[ "$GPU_VENDOR" != "none" ]]; then
    printf '    %-22s ray --num-gpus=%s · fractions tts=%s gesture=%s stt=%s\n' \
      "gpu scheduling" "$P_RAY_GPUS" "$P_TTS_FRAC" "$P_GES_FRAC" "$P_STT_FRAC"
  fi

  echo
  case "$P_VERDICT" in
    full)    printf '    %sVERDICT: full pipeline%s — voice in, voice + gesture out.\n' "$C_G$C_B" "$C_RST" ;;
    voice)   printf '    %sVERDICT: voice pipeline%s — voice in, voice out, no gesture.\n' "$C_G$C_B" "$C_RST" ;;
    text)    printf '    %sVERDICT: text pipeline%s — speech or text in, text out (no local TTS).\n' "$C_Y$C_B" "$C_RST" ;;
    blocked) printf '    %sVERDICT: cannot run here%s — not enough RAM for even the smallest LLM.\n' "$C_R$C_B" "$C_RST" ;;
  esac
}

check_hard_limits() {
  local fatal=0
  if (( RAM_TOTAL_MB < P_RAM_NEED )); then
    err "plan needs $(mb_to_gb $P_RAM_NEED) GiB RAM, machine has $(mb_to_gb $RAM_TOTAL_MB) GiB"
    fatal=1
  fi
  if (( DISK_FREE_MB < P_DISK_NEED )); then
    err "plan needs $(mb_to_gb $P_DISK_NEED) GiB free disk, only $(mb_to_gb $DISK_FREE_MB) GiB available"
    fatal=1
  fi
  [[ "$P_VERDICT" == "blocked" ]] && fatal=1
  if (( fatal )); then
    echo
    note "Options: drop --force-tts/--force-gesture (they reserve CPU RAM),"
    note "--stt-model base, --llm-model qwen2.5:1.5b, --skip-services, or free disk space."
    return 1
  fi
  return 0
}

save_plan() {
  write_file "$STATE_DIR/plan.env" <<EOF
# Generated by scripts/install_local.sh on $(date -Is). Machine-specific.
HURI_PROFILE=$GPU_VENDOR
HURI_GPU_NAME="${GPU_NAME}"
HURI_GPU_VRAM_MB=$GPU_VRAM_MB
HURI_RAY_GPUS=$P_RAY_GPUS
HURI_RAY_CPUS=$CPU_CORES
HURI_VERDICT=$P_VERDICT
HURI_MODULES=$P_MODULES
HURI_STT_SIZE=$STT_SIZE
HURI_STT_DEVICE=$P_STT_DEV
HURI_TTS_DEVICE=$P_TTS_DEV
HURI_GESTURE_DEVICE=$P_GES_DEV
HURI_EMO_DEVICE=$P_EMO_DEV
HURI_LLM_DEVICE=$P_LLM_DEV
HURI_LLM_MODEL=$LLM_MODEL
HURI_LLM_PROVIDER=$LLM_PROVIDER
HURI_LLM_URL=$LLM_URL
HURI_EMBED_MODEL=$EMBED_MODEL
HURI_EMBED_URL=$EMBED_URL_EFF
HURI_EMBED_DEVICE=$P_EMBED_DEV
HURI_QDRANT_URL=$QDRANT_URL
HURI_VERIFY_SSL=$( ((VERIFY_SSL)) && echo true || echo false)
HURI_NEEDS_OLLAMA=$NEED_OLLAMA
HURI_NEEDS_QDRANT=$( ((QDRANT_REMOTE)) && echo 0 || echo 1)
HURI_TTS_FRAC=$P_TTS_FRAC
HURI_GESTURE_FRAC=$P_GES_FRAC
HURI_STT_FRAC=$P_STT_FRAC
HURI_VENV=$VENV_DIR
HURI_PYTHON=$PY_BIN
EOF
}

# =============================================================================
# 3. Stage: system packages
# =============================================================================

install_system_packages() {
  stage_enabled system || return 0
  (( SKIP_SYSTEM )) && { note "system packages skipped (--skip-system)"; return 0; }
  step "System packages"

  # webrtcvad compiles from source (needs a toolchain + Python headers),
  # sounddevice dlopens libportaudio, soundfile needs libsndfile, and
  # librosa/openai-whisper shell out to ffmpeg.
  local pkgs=()
  case "$PKG_MGR" in
    apt)    pkgs=(build-essential git curl ca-certificates pkg-config unzip
                  ffmpeg libsndfile1 libportaudio2 python3-dev) ;;
    dnf)    pkgs=(gcc gcc-c++ make git curl unzip ffmpeg-free libsndfile portaudio python3-devel) ;;
    pacman) pkgs=(base-devel git curl unzip ffmpeg libsndfile portaudio) ;;
    zypper) pkgs=(gcc gcc-c++ make git curl unzip ffmpeg libsndfile1 portaudio python3-devel) ;;
    *)      warn "unknown package manager — install a C toolchain, ffmpeg, libsndfile and portaudio yourself"
            return 0 ;;
  esac

  # Ubuntu/Debian: creating a venv from the *system* python also needs the
  # matching python3.X-venv package (pyenv/uv interpreters ship it already).
  if [[ "$PKG_MGR" == "apt" && "$PY_BIN" == /usr/bin/* ]]; then
    pkgs+=("python$(cut -d. -f1,2 <<<"$PY_VERSION")-venv")
  fi

  local sudo_cmd=""
  [[ $EUID -ne 0 ]] && sudo_cmd="sudo"
  if [[ -n "$sudo_cmd" ]] && ! have sudo; then
    warn "no sudo available; install manually: ${pkgs[*]}"
    return 0
  fi

  info "  installing: ${pkgs[*]}"
  case "$PKG_MGR" in
    apt)    run ${sudo_cmd:+$sudo_cmd} apt-get update -qq
            run ${sudo_cmd:+$sudo_cmd} env DEBIAN_FRONTEND=noninteractive apt-get install -y "${pkgs[@]}" ;;
    dnf)    run ${sudo_cmd:+$sudo_cmd} dnf install -y "${pkgs[@]}" ;;
    pacman) run ${sudo_cmd:+$sudo_cmd} pacman -S --needed --noconfirm "${pkgs[@]}" ;;
    zypper) run ${sudo_cmd:+$sudo_cmd} zypper install -y "${pkgs[@]}" ;;
  esac
  ok "system packages ready"
}

# =============================================================================
# 4. Stage: Python environment
# =============================================================================

PIP="$VENV_DIR/bin/pip"; VPY="$VENV_DIR/bin/python"

pin_of() { # pin_of <package> <requirements file> — reuse the repo's pinned version
  local pkg="$1" file="$2" line
  line="$(grep -iE "^[[:space:]]*${pkg}[=<>~]" "$REPO_ROOT/$file" 2>/dev/null | head -1 | sed 's/[[:space:]]*#.*//')"
  printf '%s\n' "${line:-$pkg}" | tr -d ' '
}

create_venv() {
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    local existing
    existing="$("$VENV_DIR/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "?")"
    case "$existing" in
      3.10|3.11|3.12) ok "virtualenv exists: $VENV_DIR (python $existing)"; return 0 ;;
      *) die "$VENV_DIR runs python $existing (need 3.10–3.12) — delete it, or pass --venv <other path>" ;;
    esac
  fi
  info "  creating virtualenv at $VENV_DIR (python $PY_VERSION)"
  run "$PY_BIN" -m venv "$VENV_DIR"
}

install_python_deps() {
  stage_enabled python || return 0
  (( SKIP_PYTHON )) && { note "python env skipped (--skip-python)"; return 0; }
  step "Python environment"

  create_venv
  (( DRY_RUN )) && { VPY="python"; PIP="pip"; }

  run "$PIP" install --upgrade pip setuptools wheel

  local C=(-c "$REPO_ROOT/constraints.txt")

  # 1. Vendor torch FIRST, so every later resolution (sentence-transformers,
  #    transformers, faster-whisper) sees a satisfying torch and does not pull
  #    the default PyPI CPU wheel just to have it replaced.
  case "$GPU_VENDOR" in
    nvidia)
      info "  torch $TORCH_CUDA_VERSION (cu121)"
      run "$PIP" install "${C[@]}" --index-url "$TORCH_CUDA_INDEX" \
        "torch==$TORCH_CUDA_VERSION" "torchaudio==$TORCH_CUDA_VERSION"
      ;;
    amd)
      info "  torch 2.8 (ROCm $ROCM_VERSION) from repo.radeon.com"
      run "$PIP" install "${C[@]}" \
        "https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/triton-${ROCM_TRITON_VERSION}-cp312-cp312-linux_x86_64.whl"
      run "$PIP" install "${C[@]}" --extra-index-url https://repo.radeon.com/rocm/pypi/ \
        "https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/${ROCM_TORCH_WHEEL}-cp312-cp312-linux_x86_64.whl" \
        "https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/${ROCM_TORCHAUDIO_WHEEL}-cp312-cp312-linux_x86_64.whl"
      ;;
    none)
      info "  torch $TORCH_CUDA_VERSION (cpu)"
      run "$PIP" install "${C[@]}" --index-url "$TORCH_CPU_INDEX" \
        "torch==$TORCH_CUDA_VERSION" "torchaudio==$TORCH_CUDA_VERSION"
      ;;
  esac

  # 2. Server + client base (ray[serve], faster-whisper, qdrant, sounddevice…).
  info "  base requirements"
  run "$PIP" install "${C[@]}" -r "$REPO_ROOT/requirements.txt"

  # 3. Per-vendor GPU stack, mirroring deploy/Dockerfile.{amd,nvidia}.
  if [[ "$GPU_VENDOR" == "amd" ]]; then
    install_ct2_rocm
    run "$PIP" install "${C[@]}" -r "$REPO_ROOT/requirements-amd.txt"
  fi

  # 4. TTS/gesture extras. On NVIDIA use the pinned file verbatim (same set as
  #    the image); elsewhere derive a CPU-safe variant from it so the pins stay
  #    in one place.
  if [[ "$P_TTS_DEV" != "off" || "$P_GES_DEV" != "off" || "$P_EMO_DEV" != "off" ]]; then
    if [[ "$GPU_VENDOR" == "nvidia" && "$P_TTS_DEV" != "off" ]]; then
      info "  CosyVoice/EMAGE stack (requirements-nvidia.txt)"
      run "$PIP" install "${C[@]}" --extra-index-url "$TORCH_CUDA_INDEX" \
        --extra-index-url https://pypi.ngc.nvidia.com -r "$REPO_ROOT/requirements-nvidia.txt"
    else
      generate_cpu_requirements
      info "  inference extras (.huri-local/requirements-local.generated.txt)"
      run "$PIP" install "${C[@]}" -r "$STATE_DIR/requirements-local.generated.txt"
    fi
  fi

  # 5. CosyVoice source tree (no setup.py upstream → clone + PYTHONPATH).
  [[ "$P_TTS_DEV" != "off" ]] && install_cosyvoice

  ok "python environment ready"
}

generate_cpu_requirements() {
  # Derived from requirements-nvidia.txt: drop the torch pins (installed above
  # from the CPU/ROCm index), swap onnxruntime-gpu for the CPU build, and drop
  # the EMAGE *rendering/training* extras — src/modules/gesture/emage only needs
  # torch + transformers + omegaconf + huggingface_hub.
  local out="$STATE_DIR/requirements-local.generated.txt" p
  local drop='^(torch|torchaudio|onnxruntime-gpu|smplx|pyrender|trimesh|imageio|lightning|gdown|wget|pyworld)([=<>~]|$)'
  {
    echo "# Generated by scripts/install_local.sh from requirements-nvidia.txt."
    echo "# Vendor: ${GPU_VENDOR}. Do not edit — re-run the installer instead."
    if [[ "$P_TTS_DEV" != "off" ]]; then
      grep -vE "$drop" "$REPO_ROOT/requirements-nvidia.txt" \
        | sed 's/[[:space:]]*#.*//' | grep -vE '^[[:space:]]*$'
      echo "onnxruntime==1.18.0"
    else
      # No CosyVoice: gesture + emotion only.
      for p in transformers librosa soundfile omegaconf huggingface_hub numpy; do
        pin_of "$p" requirements-nvidia.txt
      done
    fi
  } | write_file "$out"
}

install_ct2_rocm() {
  info "  CTranslate2 (ROCm wheel)"
  local tmp="$STATE_DIR/ct2"
  run mkdir -p "$tmp"
  run curl -fsSL "$CT2_ROCM_URL" -o "$tmp/ct2-rocm.zip"
  run unzip -o -j "$tmp/ct2-rocm.zip" 'temp-linux/ctranslate2-4.7.1-cp312-*manylinux*x86_64.whl' -d "$tmp"
  if (( ! DRY_RUN )); then
    local whl; whl="$(find "$tmp" -name 'ctranslate2-4.7.1-cp312-*.whl' | head -1)"
    [[ -n "$whl" ]] || die "no CTranslate2 ROCm wheel in $CT2_ROCM_URL"
    run "$PIP" install "$whl"
  fi
}

install_cosyvoice() {
  local dir="$ASSETS_DIR/cosyvoice"
  if [[ -d "$dir/.git" ]]; then
    ok "CosyVoice checkout present ($dir)"
  else
    info "  cloning CosyVoice @ ${COSYVOICE_COMMIT:0:8}"
    run git clone "$COSYVOICE_REPO" "$dir"
  fi
  run git -C "$dir" fetch --depth 1 origin "$COSYVOICE_COMMIT" || true
  run git -C "$dir" checkout "$COSYVOICE_COMMIT"
  run git -C "$dir" submodule update --init --recursive
}

# =============================================================================
# 5. Stage: model weights (mirrors helm/templates/*-model-init-job.yaml)
# =============================================================================

hf_snapshot() { # hf_snapshot <repo id> <local dir>
  run "$VPY" - "$1" "$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
print("downloaded", sys.argv[1], "->", sys.argv[2])
PY
}

download_models() {
  stage_enabled models || return 0
  (( SKIP_MODELS )) && { note "model download skipped (--skip-models)"; return 0; }
  step "Model weights"
  run mkdir -p "$MODELS_DIR"

  # --- STT: faster-whisper (presence marker: model.bin, as in the Helm job) ---
  local whisper_repo="${WHISPER_REPO_PREFIX}-${STT_SIZE}"
  local whisper_dir="$MODELS_DIR/whisper/$whisper_repo"
  if [[ -f "$whisper_dir/model.bin" ]]; then
    ok "whisper: already at $whisper_dir"
  else
    info "  whisper $STT_SIZE → $whisper_dir"
    hf_snapshot "$whisper_repo" "$whisper_dir"
  fi

  # --- TTS: CosyVoice3 from ModelScope (markers: cosyvoice3.yaml + llm.pt) ---
  if [[ "$P_TTS_DEV" != "off" ]]; then
    local cosy_dir="$MODELS_DIR/cosytts/$COSYTTS_MODEL_ID"
    if [[ -f "$cosy_dir/cosyvoice3.yaml" && -f "$cosy_dir/llm.pt" ]]; then
      ok "cosyvoice3: already at $cosy_dir"
    else
      info "  CosyVoice3 → $cosy_dir (several GiB)"
      run "$VPY" - "$COSYTTS_MODEL_ID" "$cosy_dir" <<'PY'
import sys
from modelscope import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
print("downloaded", sys.argv[1], "->", sys.argv[2])
PY
    fi
  fi

  # --- Gesture: EMAGE ---------------------------------------------------------
  if [[ "$P_GES_DEV" != "off" ]]; then
    local emage_dir="$MODELS_DIR/emage/$EMAGE_REPO_ID"
    if [[ -d "$emage_dir" && -n "$(ls -A "$emage_dir" 2>/dev/null)" ]]; then
      ok "emage: already at $emage_dir"
    else
      info "  EMAGE → $emage_dir"
      hf_snapshot "$EMAGE_REPO_ID" "$emage_dir"
    fi
  fi

  # --- Emotion: prefetched into the HF cache (the module loads it by repo id) --
  if [[ "$P_EMO_DEV" != "off" ]]; then
    info "  prosody model $EMOTION_REPO_ID → HF cache"
    run "$VPY" - "$EMOTION_REPO_ID" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1])
print("cached", sys.argv[1])
PY
  fi

  # --- Reference voice sample -------------------------------------------------
  if [[ "$P_TTS_DEV" != "off" ]]; then
    if [[ -n "$VOICE_SAMPLE" ]]; then
      [[ -f "$VOICE_SAMPLE" ]] || die "--voice-sample '$VOICE_SAMPLE' not found"
      run cp "$VOICE_SAMPLE" "$ASSETS_DIR/voice.wav"
      ok "voice sample installed at $ASSETS_DIR/voice.wav"
    elif [[ -f "$ASSETS_DIR/voice.wav" ]]; then
      ok "voice sample present at $ASSETS_DIR/voice.wav"
    else
      warn "no $ASSETS_DIR/voice.wav — TTS will start but zero-shot synthesis needs it."
      note "Add one with: scripts/install_local.sh --only models,config \\"
      note "    --voice-sample /path/to/voice.wav --voice-transcript 'exact words spoken'"
      note "(--only config alone will NOT copy the file — 'models' is the stage that does)"
    fi
  fi
  ok "models ready"
}

# =============================================================================
# 6. Stage: services (Qdrant + Ollama)
# =============================================================================

start_qdrant() {
  if curl -fsS --max-time 2 http://localhost:6333/readyz >/dev/null 2>&1; then
    ok "qdrant already listening on :6333"; return 0
  fi
  local data="$STATE_DIR/qdrant"
  run mkdir -p "$data"
  if [[ -n "$HAS_DOCKER" ]]; then
    info "  starting qdrant via $HAS_DOCKER"
    if (( ! DRY_RUN )) && "$HAS_DOCKER" ps -a --format '{{.Names}}' 2>/dev/null | grep -qx huri-qdrant; then
      run "$HAS_DOCKER" start huri-qdrant
    else
      run "$HAS_DOCKER" run -d --name huri-qdrant --restart unless-stopped \
        -p 6333:6333 -p 6334:6334 -v "$data:/qdrant/storage" "$QDRANT_IMAGE"
    fi
  else
    info "  installing standalone qdrant binary (no container runtime found)"
    local url="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION#v}/qdrant-x86_64-unknown-linux-gnu.tar.gz"
    run mkdir -p "$STATE_DIR/bin"
    if [[ ! -x "$STATE_DIR/bin/qdrant" ]]; then
      run curl -fsSL "$url" -o "$STATE_DIR/qdrant.tar.gz" \
        || { warn "qdrant download failed — install it manually, RAG memory stays offline"; return 0; }
      run tar -xzf "$STATE_DIR/qdrant.tar.gz" -C "$STATE_DIR/bin"
      run chmod +x "$STATE_DIR/bin/qdrant"
    fi
    # The binary resolves ./config/config.yaml relative to its working directory
    # (start.sh runs it from .huri-local), unlike the image which ships one.
    write_file "$STATE_DIR/config/config.yaml" <<YAML
storage:
  storage_path: ./qdrant
service:
  host: 127.0.0.1
  http_port: 6333
  grpc_port: 6334
telemetry_disabled: true
YAML
    note "qdrant will be started by .huri-local/start.sh"
  fi
}

install_ollama() {
  if (( HAS_OLLAMA )); then ok "ollama already installed"; return 0; fi
  info "  ollama is not installed"
  note "Official installer: curl -fsSL https://ollama.com/install.sh | sh"
  if confirm "    Run the official Ollama installer now?"; then
    run bash -c 'curl -fsSL https://ollama.com/install.sh | sh'
    HAS_OLLAMA=1
  else
    warn "skipping ollama — install it yourself, then re-run with --only services"
    return 0
  fi
}

ollama_up() {
  curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1
}

pull_ollama_models() {
  (( HAS_OLLAMA )) || return 0
  if ! ollama_up; then
    info "  starting 'ollama serve' in the background"
    if (( ! DRY_RUN )); then
      nohup ollama serve >"$STATE_DIR/ollama.log" 2>&1 &
      local i
      for i in $(seq 1 30); do ollama_up && break; sleep 1; done
    fi
  fi
  if ! ollama_up && (( ! DRY_RUN )); then
    warn "ollama is not answering on :11434 — pull the models manually:"
    note "ollama pull $LLM_MODEL && ollama pull $EMBED_MODEL"
    return 0
  fi
  if (( ! LLM_REMOTE )); then
    info "  pulling $LLM_MODEL (several GiB)"
    run ollama pull "$LLM_MODEL"
  fi
  if (( ! EMBED_REMOTE )); then
    info "  pulling $EMBED_MODEL"
    run ollama pull "$EMBED_MODEL"
  fi
}

# probe_remote <label> <base url> — best-effort reachability check for an
# endpoint this installer does not manage.
probe_remote() {
  local label="$1" url="$2" path="/v1/models" args=()
  [[ "$LLM_PROVIDER" == "ollama" ]] && path="/api/tags"
  (( VERIFY_SSL )) || args+=(-k)
  [[ -n "$LLM_API_KEY" ]] && args+=(-H "Authorization: Bearer $LLM_API_KEY")
  if curl -fsS --max-time 8 "${args[@]}" "${url%/}$path" >/dev/null 2>&1; then
    ok "$label reachable: ${url%/}$path"
  else
    warn "$label did not answer at ${url%/}$path — check the URL, key or TLS trust"
  fi
}

install_services() {
  stage_enabled services || return 0
  (( SKIP_SERVICES )) && { note "services skipped (--skip-services)"; return 0; }
  step "Services"
  if (( QDRANT_REMOTE )); then
    ok "qdrant: remote ($QDRANT_URL) — nothing to install"
  else
    start_qdrant
  fi
  if (( NEED_OLLAMA )); then
    install_ollama
    pull_ollama_models
  else
    ok "llm + embeddings: remote ($LLM_URL) — no local ollama needed"
  fi
  ok "services ready"
}

# =============================================================================
# 7. Stage: generated configuration + run scripts
# =============================================================================

generate_configs() {
  stage_enabled config || return 0
  step "Configuration"

  local cosy_dir="$ASSETS_DIR/cosyvoice"
  local cosy_model="$MODELS_DIR/cosytts/$COSYTTS_MODEL_ID"
  local whisper_path="$MODELS_DIR/whisper/${WHISPER_REPO_PREFIX}-${STT_SIZE}"
  local emage_path="$MODELS_DIR/emage/$EMAGE_REPO_ID"
  local pythonpath="$REPO_ROOT"
  [[ "$P_TTS_DEV" != "off" ]] && pythonpath="$REPO_ROOT:$cosy_dir:$cosy_dir/third_party/Matcha-TTS"
  local stt_workers=1
  if [[ "$P_STT_DEV" == "cpu" ]]; then
    stt_workers=$(( CPU_CORES / 4 ))
    (( stt_workers < 1 )) && stt_workers=1
    (( stt_workers > 4 )) && stt_workers=4
  fi
  local ges_mem_frac="0"
  [[ "$P_GES_DEV" == "gpu" ]] && ges_mem_frac="$(frac_ceil "$P_GES_VRAM")"

  local serve_cfg="$REPO_ROOT/config/huri_local.generated.yaml"
  {
    cat <<EOF
# HuRI — Ray Serve config generated by scripts/install_local.sh
# Host: $(hostname) · $(date -Is)
# Profile: ${GPU_VENDOR}${GPU_NAME:+ (${GPU_NAME}, $(mb_to_gb "$GPU_VRAM_MB") GiB)} · verdict: $P_VERDICT
#
# Regenerate with:  scripts/install_local.sh --only config
# Start with:       .huri-local/start.sh
#
# Device plan:
#   stt      $P_STT_DEV   — $P_STT_WHY
#   rag/llm  $P_LLM_DEV   — $P_LLM_WHY
#   tts      $P_TTS_DEV   — $P_TTS_WHY
#   gesture  $P_GES_DEV   — $P_GES_WHY
#   emo      $P_EMO_DEV   — $P_EMO_WHY

proxy_location: EveryNode
http_options:
  host: 0.0.0.0
  port: 8000

applications:
  - name: huri-app
    route_prefix: /
    import_path: src.app:app
    runtime_env:
      env_vars:
        RAY_COLOR_PREFIX: "1"
        PYTHONPATH: "$pythonpath"

        # Only these modules are registered, so no deployment is created for the
        # ones this machine cannot run (see src/modules/modules.py).
        HURI_MODULES: "$P_MODULES"

        # --- STT (faster-whisper) ---
        HURI_STT_MODEL_PATH: "$whisper_path"
        HURI_STT_NUM_WORKERS: "$stt_workers"
EOF
    if [[ "$P_TTS_DEV" != "off" ]]; then
      cat <<EOF

        # --- TTS (CosyVoice3) ---
        HURI_MODEL_PATH: "$cosy_model"
        HURI_COSY_DIR: "$cosy_dir"
        HURI_VOICE_SAMPLE_PATH: "$ASSETS_DIR/voice.wav"
        # "<instruction><|endofprompt|><transcript of voice.wav>" — the transcript
        # MUST come after the marker or the LM speaks it out loud.
        HURI_VOICE_TRANSCRIPT: "You are a helpful assistant.<|endofprompt|>$VOICE_TRANSCRIPT"
        HURI_TTS_FP16: "$([[ $P_TTS_DEV == gpu ]] && echo 1 || echo 0)"
EOF
    fi
    if [[ "$P_GES_DEV" != "off" ]]; then
      cat <<EOF

        # --- Gesture (EMAGE) ---
        HURI_EMAGE_REPO: "$emage_path"
        HURI_GESTURE_CONTEXT_SEC: "2.0"
        HURI_GESTURE_MIN_CHUNK_SEC: "0.5"
        # Caps the EMAGE process to this share of the device so TTS keeps the rest.
        HURI_GESTURE_GPU_MEM_FRACTION: "$ges_mem_frac"
EOF
    fi
    if [[ "$GPU_VENDOR" == "nvidia" ]]; then
      cat <<EOF

        NVIDIA_VISIBLE_DEVICES: "all"
        NVIDIA_DRIVER_CAPABILITIES: "compute,utility"
EOF
    fi
    cat <<EOF
        HF_HUB_DOWNLOAD_TIMEOUT: "10"

    deployments:
      # Ingress + per-session router. CPU only.
      - name: HuRI
        ray_actor_options:
          num_cpus: 1
          num_gpus: 0

      - name: STT
        num_replicas: 1
        ray_actor_options:
          num_cpus: 1
          num_gpus: $P_STT_FRAC

      - name: RAGHandle
        num_replicas: 1
        ray_actor_options:
          num_cpus: 1
          num_gpus: 0
        user_config:
          qdrant_url: "$QDRANT_URL"
          llm_provider: "$LLM_PROVIDER"
          llm_url: "$LLM_URL"
          llm_model: "$LLM_MODEL"
          embedding_url: "$EMBED_URL_EFF"
          embedding_model: "$EMBED_MODEL"
          verify_ssl: $( ((VERIFY_SSL)) && echo true || echo false)
          memory_maintenance_check_hours: 6.0
          # llm_api_key is deliberately absent: it comes from HURI_LLM_API_KEY
          # (.huri-local/secrets.env, sourced by start.sh) so no secret is
          # written to disk here.
EOF
    if [[ "$P_TTS_DEV" != "off" ]]; then
      cat <<EOF

      - name: TTS
        ray_actor_options:
          num_cpus: 1
          num_gpus: $P_TTS_FRAC
EOF
    fi
    if [[ "$P_GES_DEV" != "off" ]]; then
      cat <<EOF

      - name: GestureGeneration
        ray_actor_options:
          num_cpus: 1
          num_gpus: $P_GES_FRAC
EOF
    fi
  } | write_file "$serve_cfg"
  ok "serve config → config/huri_local.generated.yaml"

  # Same runtime_env, as a sourceable .env: `serve deploy` injects those vars
  # into the replicas, but `python -m src.launch_huri`, ingestion.py or a pytest
  # run get nothing. env.sh sources this so both paths see identical settings.
  # The YAML is our own output, so the shape below is stable.
  if (( ! DRY_RUN )); then
    while IFS=$'\t' read -r k v; do
      printf '%s=%q\n' "$k" "$v"
    done < <(awk '
      /^    runtime_env:/      { inside = 1; next }
      inside && /^    deployments:/ { exit }
      inside && /^        [A-Za-z_][A-Za-z0-9_]*:/ {
        key = $0; sub(/:.*$/, "", key); sub(/^ +/, "", key)
        val = $0; sub(/^[^:]*:[[:space:]]*/, "", val)
        sub(/^"/, "", val); sub(/"$/, "", val)
        printf "%s\t%s\n", key, val
      }' "$serve_cfg") >"$STATE_DIR/huri.env"
    ok "environment → .huri-local/huri.env ($(grep -c . "$STATE_DIR/huri.env") vars)"
  fi

  # --- Matching client config -------------------------------------------------
  local client_cfg="$REPO_ROOT/config/client_local.generated.yaml"
  {
    echo "# Client config generated by scripts/install_local.sh — modules match"
    echo "# HURI_MODULES in config/huri_local.generated.yaml ($P_MODULES)."
    echo "huri_url: ws://localhost:8000/session"
    echo "interface_path: src.interfaces.cli_interface:cli_interface"
    echo
    echo "senders:"
    echo "  audio:"
    echo "    name: audio"
    echo "    topic: audio.in"
    echo "    args:"
    echo "      sample_rate: 16000"
    echo "      frame_duration: 0.030"
    echo "  text:"
    echo "    name: text"
    echo "    topic: question"
    echo "    args:"
    echo "      sample_rate: 16000"
    echo "      frame_duration: 0.030"
    echo
    echo "hooks:"
    echo "  token:"
    echo "    name: token"
    echo "    topics: [token]"
    if [[ "$P_TTS_DEV" != "off" ]]; then
      echo "  audio:"
      echo "    name: audio"
      echo "    topics: [audio.out]"
      echo "    args:"
      echo "      incoming_sample_rate: \${senders.audio.args.sample_rate}"
      echo "      sample_rate: 44100"
    fi
    if [[ "$P_GES_DEV" != "off" ]]; then
      echo "  motion:"
      echo "    name: motion"
      echo "    topics: [motion]"
    fi
    echo
    echo "modules:"
    echo "  mic:"
    echo "    name: mic"
    echo "    args:"
    echo "      vad_agressiveness: 3"
    echo "      silence_duration: 1.5"
    echo "      block_duration: \${senders.audio.args.frame_duration}"
    echo "    logging: INFO"
    echo "  stt:"
    echo "    name: stt"
    echo "    args:"
    echo "      language: en"
    echo "      block_duration: \${senders.audio.args.frame_duration}"
    echo "    logging: INFO"
    echo "  tag:"
    echo "    name: tag"
    echo "    logging: INFO"
    if [[ "$P_EMO_DEV" != "off" ]]; then
      echo "  emo:"
      echo "    name: emo"
      echo "    args:"
      echo "      block_duration: \${senders.audio.args.frame_duration}"
      echo "  eag:"
      echo "    name: eag"
    fi
    echo "  qag:"
    echo "    name: qag"
    echo "  rag:"
    echo "    name: rag"
    echo "    args:"
    echo "      language: en"
    echo "      tone: formal"
    echo "      response_format: paragraph"
    echo "      max_length: 1024"
    echo "    logging: INFO"
    if [[ "$P_TTS_DEV" != "off" ]]; then
      echo "  tts:"
      echo "    name: tts"
      echo "    args:"
      echo "      min_clause_chars: 20"
      echo "    logging: INFO"
    fi
    if [[ "$P_GES_DEV" != "off" ]]; then
      echo "  gesture:"
      echo "    name: gesture"
      echo "    logging: INFO"
    fi
  } | write_file "$client_cfg"
  ok "client config → config/client_local.generated.yaml"

  write_secrets
  generate_run_scripts
  save_plan
}

# The API key never goes into config/*.generated.yaml. It lives here, 0600, and
# start.sh sources it before `ray start` so every Serve replica inherits
# HURI_LLM_API_KEY (rag.py reads it as the llm_api_key default).
write_secrets() {
  local esc="${LLM_API_KEY//\'/\'\\\'\'}"
  write_file "$STATE_DIR/secrets.env" <<EOF
# HuRI local secrets — sourced by .huri-local/start.sh. Not committed.
# Change the key with: scripts/install_local.sh --only config --llm-api-key ...
export HURI_LLM_API_KEY='${esc}'
EOF
  (( DRY_RUN )) || chmod 600 "$STATE_DIR/secrets.env"
  if [[ -n "$LLM_API_KEY" ]]; then
    ok "api key → .huri-local/secrets.env (0600, exported as HURI_LLM_API_KEY)"
  fi
}

generate_run_scripts() {
  write_file "$STATE_DIR/env.sh" <<EOF
# shellcheck shell=bash
# Source this for a shell configured exactly like a Serve replica:
#     source .huri-local/env.sh
#     python -m src.launch_huri                      # run HuRI without serve deploy
#     python -m src.modules.rag.ingestion --help     # feed the RAG memory
export HURI_ROOT="$REPO_ROOT"
export HURI_STATE="$STATE_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Model paths, module allow-list, TTS/gesture tuning — the same values the
# generated Serve config injects into every replica (regenerate both with
# 'scripts/install_local.sh --only config').
if [ -r "$STATE_DIR/huri.env" ]; then
  set -a; . "$STATE_DIR/huri.env"; set +a
fi
# API key, if any: kept out of the config files on purpose.
if [ -r "$STATE_DIR/secrets.env" ]; then
  # shellcheck disable=SC1091
  . "$STATE_DIR/secrets.env"
fi

case ":\${PYTHONPATH:-}:" in
  *":\$HURI_ROOT:"*) ;;
  *) export PYTHONPATH="\$HURI_ROOT\${PYTHONPATH:+:\$PYTHONPATH}" ;;
esac
EOF

  {
    cat <<EOF
#!/usr/bin/env bash
# Start the local HuRI stack (generated by scripts/install_local.sh).
set -Eeuo pipefail
cd "$REPO_ROOT"
source "$STATE_DIR/env.sh"

# Secrets are exported *before* ray start so every Serve replica inherits them
# (rag.py reads HURI_LLM_API_KEY). A cluster already running keeps its old env —
# run stop.sh first after changing the key.
# shellcheck disable=SC1091
[ -r "$STATE_DIR/secrets.env" ] && source "$STATE_DIR/secrets.env"
EOF

    if (( QDRANT_REMOTE )); then
      cat <<EOF

# --- 1. Qdrant: remote ($QDRANT_URL), nothing to start ----------------------
EOF
    else
      cat <<EOF

# --- 1. Qdrant --------------------------------------------------------------
if ! curl -fsS --max-time 2 http://localhost:6333/readyz >/dev/null 2>&1; then
  if [ -x "$STATE_DIR/bin/qdrant" ]; then
    echo "[huri] starting qdrant (binary)"
    (cd "$STATE_DIR" && nohup "$STATE_DIR/bin/qdrant" >"$STATE_DIR/qdrant.log" 2>&1 &)
  elif command -v ${HAS_DOCKER:-docker} >/dev/null 2>&1; then
    echo "[huri] starting qdrant container"
    ${HAS_DOCKER:-docker} start huri-qdrant >/dev/null 2>&1 || true
  fi
fi
EOF
    fi

    if (( NEED_OLLAMA )); then
      cat <<EOF

# --- 2. Ollama --------------------------------------------------------------
if ! curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
  if command -v ollama >/dev/null 2>&1; then
    echo "[huri] starting ollama"
    nohup ollama serve >"$STATE_DIR/ollama.log" 2>&1 &
    for _ in \$(seq 1 30); do
      curl -fsS --max-time 1 http://localhost:11434/api/tags >/dev/null 2>&1 && break
      sleep 1
    done
  fi
fi
EOF
    else
      cat <<EOF

# --- 2. LLM + embeddings: remote ($LLM_URL), nothing to start ---------------
EOF
    fi

    cat <<EOF

# --- 3. Ray head ------------------------------------------------------------
# Started from the repo root so replicas can import src.app.
if ! ray status >/dev/null 2>&1; then
  echo "[huri] starting ray head"
  ray start --head --num-cpus=$CPU_CORES --num-gpus=$P_RAY_GPUS \\
    --dashboard-host=127.0.0.1 --disable-usage-stats
fi

# --- 4. HuRI ----------------------------------------------------------------
echo "[huri] deploying config/huri_local.generated.yaml"
serve deploy config/huri_local.generated.yaml

cat <<'MSG'

HuRI is deploying. Watch it come up with:
    .huri-local/status.sh          (or the dashboard at http://127.0.0.1:8265)

Then talk to it:
    source .huri-local/env.sh
    python -m src.client --config config/client_local.generated.yaml
MSG
EOF
  } | write_file "$STATE_DIR/start.sh"
  run chmod +x "$STATE_DIR/start.sh"

  write_file "$STATE_DIR/stop.sh" <<EOF
#!/usr/bin/env bash
# Stop the local HuRI stack.
set -Euo pipefail
cd "$REPO_ROOT"
source "$STATE_DIR/env.sh"
serve shutdown -y >/dev/null 2>&1 || true
ray stop >/dev/null 2>&1 || true
if [ "\${1:-}" = "--all" ]; then
  ${HAS_DOCKER:-docker} stop huri-qdrant >/dev/null 2>&1 || true
  pkill -f "$STATE_DIR/bin/qdrant" >/dev/null 2>&1 || true
  pkill -f "ollama serve" >/dev/null 2>&1 || true
fi
echo "[huri] stopped"
EOF
  run chmod +x "$STATE_DIR/stop.sh"

  local llm_probe_path="/v1/models"
  [[ "$LLM_PROVIDER" == "ollama" ]] && llm_probe_path="/api/tags"
  write_file "$STATE_DIR/status.sh" <<EOF
#!/usr/bin/env bash
# Show the state of every HuRI dependency, local or remote.
set -Euo pipefail
cd "$REPO_ROOT"
source "$STATE_DIR/env.sh"
# shellcheck disable=SC1091
[ -r "$STATE_DIR/secrets.env" ] && source "$STATE_DIR/secrets.env"

CURL=(curl -fsS --max-time 5$( ((VERIFY_SSL)) || printf ' -k'))
AUTH=()
[ -n "\${HURI_LLM_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer \${HURI_LLM_API_KEY}")

probe() {
  local label="\$1"; shift
  if "\$@" >/dev/null 2>&1; then printf '%-10s up\n' "\$label"
  else printf '%-10s down\n' "\$label"; fi
}

probe qdrant "\${CURL[@]}" "${QDRANT_URL%/}/readyz"
probe llm    "\${CURL[@]}" "\${AUTH[@]}" "${LLM_URL%/}${llm_probe_path}"
probe ray    ray status
probe huri   "\${CURL[@]}" http://localhost:8000/-/healthz
echo
serve status 2>/dev/null || true
EOF
  run chmod +x "$STATE_DIR/status.sh"
  ok "run scripts → .huri-local/{start,stop,status}.sh"

  # Keep generated artefacts out of git.
  local gi="$REPO_ROOT/.gitignore"
  local entry
  for entry in "/.huri-local/" "/assets/" "config/*.generated.yaml"; do
    if ! grep -qxF "$entry" "$gi" 2>/dev/null; then
      (( DRY_RUN )) || printf '%s\n' "$entry" >>"$gi"
    fi
  done
}

# =============================================================================
# 8. Stage: verification
# =============================================================================

verify() {
  stage_enabled verify || return 0
  step "Verification"
  (( DRY_RUN )) && { note "skipped in --dry-run"; return 0; }

  local failures=0
  local checks="ray,faster_whisper,qdrant_client,httpx,numpy"
  [[ "$P_TTS_DEV" != "off" || "$P_GES_DEV" != "off" || "$P_EMO_DEV" != "off" ]] && checks="$checks,torch"

  if "$VPY" - "$checks" <<'PY'
import importlib, sys
missing = []
for name in sys.argv[1].split(","):
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{name}: {exc}")
if missing:
    print("\n".join(missing)); sys.exit(1)
PY
  then ok "core imports"
  else err "core imports failed"; failures=1
  fi

  if [[ "$GPU_VENDOR" != "none" ]]; then
    if "$VPY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
      ok "torch sees the GPU: $("$VPY" -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null)"
    else
      err "torch cannot see the GPU — GPU deployments will fall back to CPU or fail"
      failures=1
    fi
  fi

  if [[ "$P_TTS_DEV" != "off" ]]; then
    if PYTHONPATH="$ASSETS_DIR/cosyvoice:$ASSETS_DIR/cosyvoice/third_party/Matcha-TTS" \
       "$VPY" -c 'from cosyvoice.cli.cosyvoice import CosyVoice3' 2>/dev/null; then
      ok "CosyVoice3 importable"
    else
      err "CosyVoice3 import failed (check $ASSETS_DIR/cosyvoice submodules)"; failures=1
    fi

    # Unlike the Kubernetes deploy (where the voice sample PVC is deliberately
    # populated *after* first install, via `kubectl cp`), a bare-metal install
    # has no later provisioning step: if the file is not here now, it never
    # will be, and every TTS session will fail per-request (each error is
    # logged but swallowed by the event graph, so the client just gets no
    # audio — see src/core/events.py::EventGraph._run). Treat it as a hard
    # requirement rather than the best-effort skip TTSDeployment allows.
    if [[ -f "$ASSETS_DIR/voice.wav" ]]; then
      ok "voice sample present at $ASSETS_DIR/voice.wav"
    else
      err "no $ASSETS_DIR/voice.wav — TTS is enabled but has no reference voice"
      note "Add one with: scripts/install_local.sh --only models,config \\"
      note "    --voice-sample /path/to/voice.wav --voice-transcript 'exact words spoken'"
      failures=1
    fi
  fi

  if [[ "$P_GES_DEV" != "off" ]]; then
    if (cd "$REPO_ROOT" && "$VPY" -c 'from src.modules.gesture.emage import EmageAudioModel' 2>/dev/null); then
      ok "EMAGE importable"
    else
      err "EMAGE import failed"; failures=1
    fi
  fi

  if (( ! SKIP_SERVICES )); then
    local qargs=(); (( VERIFY_SSL )) || qargs+=(-k)
    if curl -fsS --max-time 5 "${qargs[@]}" "${QDRANT_URL%/}/readyz" >/dev/null 2>&1; then
      ok "qdrant reachable at $QDRANT_URL"
    elif (( QDRANT_REMOTE )); then
      warn "qdrant at $QDRANT_URL did not answer — RAG memory will fail at runtime"
    else
      warn "qdrant not reachable (start.sh will launch it)"
    fi

    if (( LLM_REMOTE )); then
      probe_remote "llm" "$LLM_URL"
      (( EMBED_REMOTE )) && [[ "$EMBED_URL_EFF" != "$LLM_URL" ]] && probe_remote "embeddings" "$EMBED_URL_EFF"
    elif curl -fsS --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
      ok "ollama reachable on :11434"
      curl -fsS --max-time 3 http://localhost:11434/api/tags | grep -q "${LLM_MODEL%%:*}" \
        && ok "LLM $LLM_MODEL present" || warn "LLM $LLM_MODEL not pulled yet"
    else
      warn "ollama not reachable (start.sh will launch it)"
    fi
  fi

  return $failures
}

# =============================================================================
# main
# =============================================================================

main() {
  mkdir -p "$STATE_DIR"
  : >>"$LOG_FILE"

  printf '%s\n' "${C_B}HuRI local installer${C_RST} — $REPO_ROOT"
  (( DRY_RUN )) && note "dry run: nothing will be installed"

  detect_os
  detect_host
  detect_gpu
  detect_tools
  if ! detect_python; then
    die "no suitable Python found (need 3.10–3.12; pass --python /path/to/python3.12)"
  fi
  report_detection

  resolve_endpoints
  plan
  report_plan
  save_plan

  if (( PLAN_ONLY )); then
    echo
    note "plan written to .huri-local/plan.env — re-run without --plan-only to install"
    return 0
  fi

  echo
  check_hard_limits || die "the machine cannot host this plan"

  if ! confirm "Proceed with the install described above?"; then
    info "aborted — nothing was changed"
    return 0
  fi

  install_system_packages
  install_python_deps
  download_models
  install_services
  generate_configs
  verify || die "verification failed — see the ✗ lines above, fix them, then re-run (e.g. with --only <stage>)"

  step "Done"
  cat <<EOF
    Start:   .huri-local/start.sh
    Status:  .huri-local/status.sh
    Client:  source .huri-local/env.sh
             python -m src.client --config config/client_local.generated.yaml
    Stop:    .huri-local/stop.sh [--all]

    Plan:    .huri-local/plan.env          Log: .huri-local/install.log
EOF
}

main "$@"
