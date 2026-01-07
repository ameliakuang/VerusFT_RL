#! /bin/bash
#
# This script is not meant to be run directly. Instead, it is invoked by the
# other scripts in this directory.

function usage() {
  echo >&2 "Usage: $0 {foo.rs} {string to search for}"
  echo >&2 "Environment variables to modify behavior:"
  echo >&2 "  TRACE=0        Set to 1 to enable tracing 'set -x' style"
  echo >&2 "  MAX_RUNS=1     Set to larger number to run many times before checking"
  echo >&2 "  TIMEOUT=10     Set to different value to set a different potential timeout"
  exit 1
}

# Parse arguments
if [ "$1" = "--help" ]; then
  usage
fi

if [ -z "$1" ]; then
  echo >&2 "Error: no file specified"
  usage
else
  FILE="$1"
fi

if [ -z "$2" ]; then
  echo >&2 "Error: no string to search for specified"
  usage
else
  SEARCH="$2"
fi

if [ -z "$TRACE" ]; then
  TRACE=0
fi

if [ -z "$MAX_RUNS" ]; then
  MAX_RUNS=1
fi

if [ -z "$TIMEOUT" ]; then
  TIMEOUT=10
fi

# Set up environment
set -euo pipefail
if [ "$TRACE" = "1" ]; then
  set -x
fi

# Run the test
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# Find Verus binary - first check PATH, then build locations
if command -v verus &> /dev/null; then
  VERIFY="verus"
elif [ -x "${DIR}/../../target-verus/release/verus" ]; then
  VERIFY="${DIR}/../../target-verus/release/verus"
elif [ -x "${DIR}/../../target-verus/debug/verus" ]; then
  VERIFY="${DIR}/../../target-verus/debug/verus"
elif [ -x "${DIR}/../../../verus-x86-linux/verus" ]; then
  VERIFY="${DIR}/../../../verus-x86-linux/verus"
else
  echo >&2 "Error: Could not find verus binary"
  echo >&2 "Searched in:"
  echo >&2 "  - PATH (command -v verus)"
  echo >&2 "  - ${DIR}/../../target-verus/release/verus"
  echo >&2 "  - ${DIR}/../../target-verus/debug/verus"
  echo >&2 "  - ${DIR}/../../../verus-x86-linux/verus"
  echo "failed"
  exit 1
fi

for i in $(seq 1 "$MAX_RUNS"); do
  timeout "$TIMEOUT" "$VERIFY" "$FILE" >output 2>&1 || true
  if grep "$SEARCH" output >/dev/null; then
    echo "success"
    exit 0
  fi
done

echo "failed"
exit 1
