#!/usr/bin/env bash
set -e

# Values are provided by run.sh
: "${USER_NAME:=user}"
: "${USER_ID:=1000}"
: "${GROUP_ID:=1000}"

# Ensure group exists (by GID)
if ! getent group "${GROUP_ID}" >/dev/null 2>&1; then
    groupadd -g "${GROUP_ID}" "${USER_NAME}"
fi

# Ensure user exists (by UID)
if ! id -u "${USER_ID}" >/dev/null 2>&1; then
    useradd -m -u "${USER_ID}" -g "${GROUP_ID}" "${USER_NAME}"
fi

# Determine home directory
USER_HOME="$(getent passwd "${USER_ID}" | cut -d: -f6)"
if [ -z "${USER_HOME}" ]; then
    USER_HOME="/home/${USER_NAME}"
    mkdir -p "${USER_HOME}"
    chown "${USER_ID}:${GROUP_ID}" "${USER_HOME}"
fi

export HOME="${USER_HOME}"
export USER="${USER_NAME}"
export HISTFILE="${HOME}/.bash_history"

# Fix ownership for bind-mounted home (best-effort)
chown -R "${USER_ID}:${GROUP_ID}" "${HOME}" || true

# Switch to the user and run your Xilinx entrypoint (with CMD as args, e.g. "bash")
exec gosu "${USER_NAME}" /usr/local/bin/xilinx_entrypoint.sh "$@"
