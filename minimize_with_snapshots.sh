#!/bin/bash
#
# Wrapper script for minimize.sh that captures intermediate versions during creduce
# Usage: ./minimize_with_snapshots.sh <input_file> <minimizer_type> [cores] [snapshot_interval]
#
# This script monitors creduce and saves intermediate versions to prevent over-minimization
# that loses semantic meaning.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function usage() {
  cat << EOF
Usage: $0 <input_file> <minimizer_type> [cores] [snapshot_interval]

This script wraps minimize.sh and captures intermediate versions during creduce.

Arguments:
  input_file         - Path to .rs file to minimize
  minimizer_type     - One of: panic, rlimit, timeout, minimal, invariant, spec, feature
  cores              - Number of cores to use (default: 4)
  snapshot_interval   - Save snapshot every N seconds (default: 30)
                       Also saves when file size decreases by 10% or more

Examples:
  $0 my_code.rs minimal 4 30
  $0 my_code.rs invariant 8 60

The script will save intermediate versions as:
  foo.rs.snapshot_001
  foo.rs.snapshot_002
  ...
  
Each snapshot is verified to still pass the interestingness test.
EOF
  exit 1
}

# Parse arguments
if [ $# -lt 2 ]; then
  usage
fi

INPUT_FILE="$1"
MINIMIZER_TYPE="$2"
CORES="${3:-4}"
SNAPSHOT_INTERVAL="${4:-30}"

# Get the directory of this script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
TARGET_FILE="${DIR}/foo.rs"
SNAPSHOT_DIR="${DIR}/snapshots"

# Clean up old snapshots from previous runs
if [ -d "$SNAPSHOT_DIR" ]; then
  echo -e "${YELLOW}Cleaning up old snapshots from previous runs...${NC}"
  rm -f "$SNAPSHOT_DIR"/foo.rs.snapshot_* 2>/dev/null || true
fi

# Create snapshot directory
mkdir -p "$SNAPSHOT_DIR"

# Map minimizer type to interestingness script
case "$MINIMIZER_TYPE" in
  panic)
    SCRIPT="${DIR}/panicked_in.sh"
    ;;
  rlimit)
    SCRIPT="${DIR}/rlimit_exceeded.sh"
    ;;
  timeout)
    SCRIPT="${DIR}/time_exceeded.sh"
    ;;
  minimal)
    SCRIPT="${DIR}/verified_minimal.sh"
    ;;
  invariant)
    SCRIPT="${DIR}/verified_with_invariant.sh"
    ;;
  spec)
    SCRIPT="${DIR}/verified_with_spec.sh"
    ;;
  feature)
    if [ $# -lt 3 ]; then
      echo -e "${RED}Error: 'feature' minimizer requires a pattern${NC}"
      exit 1
    fi
    SCRIPT="${DIR}/verified_with_feature.sh"
    export FEATURE="$3"
    CORES="${4:-4}"
    SNAPSHOT_INTERVAL="${5:-30}"
    ;;
  *)
    echo -e "${RED}Error: Unknown minimizer type '$MINIMIZER_TYPE'${NC}"
    usage
    ;;
esac

# Function to check if file still passes interestingness test
check_interestingness() {
  local file="$1"
  cd "$DIR"
  cp "$file" foo.rs
  if "$SCRIPT" >/dev/null 2>&1; then
    return 0
  else
    return 1
  fi
}

# Function to save a snapshot
save_snapshot() {
  local file="$1"
  local snapshot_num="$2"
  local lines=$(wc -l < "$file" | tr -d ' ')
  local snapshot_file="${SNAPSHOT_DIR}/foo.rs.snapshot_$(printf "%03d" "$snapshot_num")"
  
  # Verify it still passes interestingness test
  if check_interestingness "$file"; then
    cp "$file" "$snapshot_file"
    echo -e "${GREEN}✓ Snapshot $snapshot_num saved:${NC} $lines lines -> $(basename "$snapshot_file")"
    return 0
  else
    echo -e "${YELLOW}⚠ Snapshot $snapshot_num skipped:${NC} no longer passes interestingness test"
    return 1
  fi
}

# Copy input to foo.rs
cp "$INPUT_FILE" "$TARGET_FILE"
ORIGINAL_LINES=$(wc -l < "$TARGET_FILE")
ORIGINAL_SIZE=$(stat -f%z "$TARGET_FILE" 2>/dev/null || stat -c%s "$TARGET_FILE" 2>/dev/null)

echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    Verus Code Minimizer (with Snapshots)             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Input file:${NC}           $INPUT_FILE"
echo -e "${GREEN}Original lines:${NC}        $ORIGINAL_LINES"
echo -e "${GREEN}Minimizer:${NC}              $MINIMIZER_TYPE"
echo -e "${GREEN}Cores:${NC}                  $CORES"
echo -e "${GREEN}Snapshot interval:${NC}     ${SNAPSHOT_INTERVAL}s"
echo -e "${GREEN}Snapshot directory:${NC}    $SNAPSHOT_DIR"
echo ""

# Save initial snapshot
SNAPSHOT_COUNT=1
save_snapshot "$TARGET_FILE" "$SNAPSHOT_COUNT"
SNAPSHOT_COUNT=$((SNAPSHOT_COUNT + 1))
LAST_SIZE=$ORIGINAL_SIZE
LAST_LINES=$ORIGINAL_LINES

# Start monitoring in background
MONITOR_PID=""
(
  while true; do
    sleep "$SNAPSHOT_INTERVAL"
    
    # Check if creduce is still running
    if ! pgrep -f "creduce.*foo.rs" >/dev/null 2>&1; then
      break
    fi
    
    # Check if file exists and has changed
    if [ ! -f "$TARGET_FILE" ]; then
      continue
    fi
    
    CURRENT_SIZE=$(stat -f%z "$TARGET_FILE" 2>/dev/null || stat -c%s "$TARGET_FILE" 2>/dev/null)
    CURRENT_LINES=$(wc -l < "$TARGET_FILE" | tr -d ' ')
    
    # Save snapshot if:
    # 1. File size decreased by 10% or more
    # 2. Line count decreased significantly (more than 5 lines)
    SIZE_DECREASE=$((100 * (LAST_SIZE - CURRENT_SIZE) / LAST_SIZE))
    LINES_DECREASE=$((LAST_LINES - CURRENT_LINES))
    
    if [ "$SIZE_DECREASE" -ge 10 ] || [ "$LINES_DECREASE" -ge 5 ]; then
      if save_snapshot "$TARGET_FILE" "$SNAPSHOT_COUNT"; then
        LAST_SIZE=$CURRENT_SIZE
        LAST_LINES=$CURRENT_LINES
        SNAPSHOT_COUNT=$((SNAPSHOT_COUNT + 1))
      fi
    fi
  done
) &
MONITOR_PID=$!

# Run creduce
echo -e "${GREEN}Starting C-Reduce with snapshot monitoring...${NC}"
echo -e "${BLUE}(Snapshots will be saved to $SNAPSHOT_DIR)${NC}"
echo ""

cd "$DIR"
if ! creduce --n "$CORES" "$SCRIPT" foo.rs; then
  echo -e "${RED}C-Reduce failed or was interrupted${NC}"
  kill $MONITOR_PID 2>/dev/null || true
  exit 1
fi

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

# Save final snapshot
FINAL_LINES=$(wc -l < "$TARGET_FILE")
if save_snapshot "$TARGET_FILE" "$SNAPSHOT_COUNT"; then
  SNAPSHOT_COUNT=$((SNAPSHOT_COUNT + 1))
fi

# Show results
REDUCTION=$((100 - (FINAL_LINES * 100 / ORIGINAL_LINES)))

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Minimization Complete!                      ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Original:${NC}   $ORIGINAL_LINES lines"
echo -e "${GREEN}Final:${NC}      $FINAL_LINES lines"
echo -e "${GREEN}Reduction:${NC}   ${REDUCTION}%"
echo ""
echo -e "${YELLOW}Snapshots saved:${NC} $((SNAPSHOT_COUNT - 1)) intermediate versions"
echo -e "${YELLOW}Location:${NC}      $SNAPSHOT_DIR"
echo ""
echo -e "${YELLOW}Available snapshots:${NC}"
ls -lh "$SNAPSHOT_DIR"/foo.rs.snapshot_* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  (none)"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review snapshots: ls -lh $SNAPSHOT_DIR"
echo "  2. Find a good intermediate version:"
echo "     cp $SNAPSHOT_DIR/foo.rs.snapshot_XXX foo.rs"
echo "  3. Review: cat foo.rs"
echo "  4. Format: verusfmt foo.rs"
echo "  5. Verify: verus foo.rs"
echo ""

