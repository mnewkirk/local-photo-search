# nas-env.sh — shared loader for this machine's NAS connection settings.
#
# SOURCE this, don't run it:   . "$(dirname "$0")/scripts/nas-env.sh"
#
# The repo is public, so the NAS's hostname / IP / login never live in tracked
# files. They come from the environment, or from a git-ignored `nas.env` at the
# repo root (copy `nas.env.example`). Same idea as
# scripts/windows-import/import-config.local.ps1; photosearch/nas_config.py is
# the Python twin of this file.
#
# Precedence: an already-set environment variable wins over nas.env, so
# `NAS_HOST=other ./sync-replica.sh` still works for a one-off.
#
#   NAS_ENV_FILE   read this file instead of <repo>/nas.env

_nas_env_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAS_ENV_FILE="${NAS_ENV_FILE:-${_nas_env_root}/nas.env}"

nas_env_load() {
  local line key val
  [ -f "${NAS_ENV_FILE}" ] || return 0
  while IFS= read -r line || [ -n "${line}" ]; do
    line="${line%$'\r'}"                       # tolerate CRLF (edited on Windows)
    case "${line}" in ''|\#*) continue ;; esac
    line="${line#export }"
    key="${line%%=*}"
    val="${line#*=}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    val="${val%\"}"; val="${val#\"}"
    val="${val%\'}"; val="${val#\'}"
    # Parsed, never `source`d: a config file should not be able to run code.
    if [ -z "${!key:-}" ]; then
      export "${key}=${val}"
    fi
  done < "${NAS_ENV_FILE}"
}

# nas_env_require VAR "what it is, with an example"
# Exits 2 naming the variable. There is deliberately NO built-in default: a
# wrong host fails slowly and confusingly (ssh timeout, UGOS auto-block), an
# unset one should fail immediately and say what to set.
nas_env_require() {
  local name="$1" what="${2:-}"
  if [ -z "${!name:-}" ]; then
    {
      echo "ERROR: ${name} is not set${what:+ — ${what}}."
      echo "  Set it in ${NAS_ENV_FILE} (copy nas.env.example), or export ${name}."
    } >&2
    exit 2
  fi
}

nas_env_load
