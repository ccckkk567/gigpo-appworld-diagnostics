#!/usr/bin/env bash
# Source from a prepare job after activating its own venv.
# Retry transport/index-response failures only; dependency conflicts fail immediately.
gigpo_pip_attempt=0
gigpo_pip_install() {
  local attempt result log_file
  for attempt in 1 2 3; do
    gigpo_pip_attempt=$((gigpo_pip_attempt + 1))
    log_file="$TMPDIR/pip-install-$gigpo_pip_attempt.log"
    if "$GIGPO_VENV/bin/python" -I -m pip --require-virtualenv --disable-pip-version-check \
        --retries 5 --timeout 60 install --index-url https://pypi.org/simple "$@" 2>&1 | tee "$log_file"; then
      return 0
    else
      result=${PIPESTATUS[0]}
      # A log write error also fails; do not silently return success from pip.
      if [[ "$result" -eq 0 ]]; then result=1; fi
    fi
    if [[ "$attempt" -ge 3 ]] || ! grep -Eqi 'Content-Type: Unknown|Skipping page .*Content-Type|Temporary failure|ConnectionError|ConnectionResetError|Connection broken|ReadTimeout|timed out|RemoteDisconnected|Max retries exceeded|503 Server Error|502 Server Error' "$log_file"; then
      return "$result"
    fi
    echo "PIP_TRANSIENT_RETRY attempt=$attempt log=$log_file" >&2
    sleep 5
  done
}
