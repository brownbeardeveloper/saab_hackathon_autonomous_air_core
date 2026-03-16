#!/bin/sh
set -eu

command="${1:?next command is required}"
shift || true

if [ -z "${GAT_PYTHON_BIN+x}" ]; then
  export GAT_PYTHON_BIN="mamba"
fi

if [ -z "${GAT_PYTHON_ARGS+x}" ]; then
  export GAT_PYTHON_ARGS="run -n gat-train python"
fi

exec ./node_modules/.bin/next "$command" "$@"
