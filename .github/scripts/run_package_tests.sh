#!/usr/bin/env bash
# ci_run_tests.sh — opinionated Python test runner for large mono-repos
# Goals:
# - Discover test-bearing packages automatically
# - Run pytest per package in parallel with clear, prefixed logs
# - Fail fast on real errors, not masked pipes
# - Emit concise end-of-job summary (and GitHub Step Summary if available)
# - No Codecov, no local-repro tooling, deps assumed preinstalled

set -euo pipefail

########################################
# Configurable knobs via env vars
########################################
# Glob(s) that identify Python packages or test roots
: "${PKG_GLOBS:=agent common core app_backend packages/*}"

# Run packages in parallel? "true" | "false"
: "${PARALLEL:=true}"

# Max concurrently running packages when PARALLEL=true
: "${MAX_PROCS:=8}"

# Extra pytest selectors, e.g. "-k 'not slow' -m 'not integ'"
: "${SELECT:=}"

# Respect pre-set PYTEST_ADDOPTS but provide sane defaults
DEFAULT_PYTEST_OPTS="-n auto --dist loadscope -q -rA \
  --durations=15 --maxfail=${MAXFAIL:-1} \
  --timeout=120 --timeout-method=thread \
  --color=yes --benchmark-skip"
PYTEST_OPTS="${PYTEST_ADDOPTS:-$DEFAULT_PYTEST_OPTS} ${SELECT}"

# Where to put artifacts
ART_DIR="${ART_DIR:-test-results}"
COV_DIR="${COV_DIR:-coverage-reports}"
mkdir -p "$ART_DIR" "$COV_DIR"

########################################
# Basic coloring (no external deps)
########################################
RED=$'\e[1;31m'; GRN=$'\e[1;32m'; YEL=$'\e[1;33m'
BLU=$'\e[1;34m'; MAG=$'\e[1;35m'; CYN=$'\e[1;36m'; WHT=$'\e[1;37m'; NC=$'\e[0m'

# Assign stable colors per package index
colors=("$BLU" "$RED" "$GRN" "$YEL" "$MAG" "$CYN" "$WHT")

########################################
# Utilities
########################################
ts() { date +"%H:%M:%S"; }

group_start() {  # Collapses logs in GitHub UI; harmless elsewhere
  echo "::group::$*"
}

group_end() {
  echo "::endgroup::" || true
}

########################################
# Discover packages
########################################
discover_packages() {
  local -a out=()
  for g in $PKG_GLOBS; do
    # A package is a directory that either contains tests/ or any *_test.py/test_*.py
    for p in $(compgen -G "$g" || true); do
      [ -d "$p" ] || continue
      if compgen -G "$p/tests" > /dev/null \
        || compgen -G "$p/**/test_*.py" > /dev/null \
        || compgen -G "$p/**/*_test.py" > /dev/null; then
        out+=("$p")
      fi
    done
  done
  # De-dup and stable sort
  printf "%s\n" "${out[@]}" | awk 'NF' | sort -u
}

########################################
# Run one package
########################################
run_pkg() {
  local pkg="$1" idx="$2"
  local name; name="$(basename "$pkg")"
  local color="${colors[$((idx % ${#colors[@]}))]}"
  local raw="$ART_DIR/${name}.log"
  local exitfile="$ART_DIR/${name}.exit"
  local covxml="$COV_DIR/coverage-${name}.xml"

  # Prefer running pytest from inside the package if it is a subdir
  local covpath="$covxml"
  local run_dir="."
  if [[ "$pkg" != "." && "$pkg" != "core" ]]; then
    run_dir="$pkg"
    # Write coverage XML relative to repo root even when cd-ing
    # Use a relative path that escapes back to $COV_DIR
    local depth="${pkg//[^\/]/}"
    local up=""
    for _ in ${depth}; do up+="../"; done
    covpath="${up}${covxml}"
  fi

  group_start "pytest: ${pkg}"
  echo -e "${color}[$(ts)] [${name}] Starting: pytest ${PYTEST_OPTS} ${SELECT}${NC}"
  {
    (
      cd "$run_dir"
      pytest ${PYTEST_OPTS} \
        --cov --cov-branch --cov-report="xml:${covpath}" \
        2>&1
    )
    printf "%d" $? > "$exitfile"
  } | sed -e "s/^/[${name}] /" | tee "$raw"
  group_end

  # Collect durations for global top-10
  grep -E "^[0-9]+\.[0-9]+s " "$raw" > "$ART_DIR/${name}.dur" || true
}

########################################
# Main
########################################
START="$(date +%s)"
mapfile -t PACKAGES < <(discover_packages)
if [ "${#PACKAGES[@]}" -eq 0 ]; then
  echo "No packages with tests discovered under: $PKG_GLOBS"
  exit 0
fi

echo "Discovered packages:"
printf " - %s\n" "${PACKAGES[@]}"

# Run sequentially or in background with a simple concurrency gate
pids=()
if [ "$PARALLEL" = "true" ]; then
  sem=${MAX_PROCS}
  for i in "${!PACKAGES[@]}"; do
    # Concurrency gate
    while [ "$(jobs -rp | wc -l)" -ge "$sem" ]; do sleep 0.2; done
    run_pkg "${PACKAGES[$i]}" "$i" &
    pids+=("$!")
  done
  # Wait for all
  code=0
  for pid in "${pids[@]}"; do wait "$pid" || code=1; done
else
  code=0
  for i in "${!PACKAGES[@]}"; do
    run_pkg "${PACKAGES[$i]}" "$i" || code=1
  done
fi

END="$(date +%s)"
TOTAL="$((END-START))"

########################################
# Summaries
########################################
echo -e "\n${WHT}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${WHT}🐌 TOP 10 SLOWEST TESTS${NC}"
cat "$ART_DIR"/*.dur 2>/dev/null | sed 's/^/[/; s/]/] /' | sort -k2 -nr | head -10 | nl -w2 -s'. ' || echo "No duration data."

echo -e "\n${WHT}📊 PACKAGE RESULTS${NC}"
printf "%-24s %-8s %s\n" "package" "result" "log"
printf "%-24s %-8s %s\n" "-------" "------" "---"

OVERALL_FAIL=0
for pkg in "${PACKAGES[@]}"; do
  name="$(basename "$pkg")"
  exitf="$ART_DIR/${name}.exit"
  logf="$ART_DIR/${name}.log"
  if [ -f "$exitf" ] && [ "$(cat "$exitf")" -eq 0 ]; then
    printf "%-24s %-8s %s\n" "$name" "${GRN}PASS${NC}" "$logf"
  else
    printf "%-24s %-8s %s\n" "$name" "${RED}FAIL${NC}" "$logf"
    OVERALL_FAIL=1
  fi
done
echo "Total time: ${TOTAL}s"

# GitHub Step Summary (compact markdown table)
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "## Unit test summary"
    echo ""
    echo "| Package | Result | Log |"
    echo "|---|---|---|"
    for pkg in "${PACKAGES[@]}"; do
      name="$(basename "$pkg")"
      exitf="$ART_DIR/${name}.exit"
      logf="$ART_DIR/${name}.log"
      res="FAIL"; [ -f "$exitf" ] && [ "$(cat "$exitf")" -eq 0 ] && res="PASS"
      echo "| \`$name\` | $res | \`$logf\` |"
    done
    echo ""
    echo "<sub>Total time: ${TOTAL}s</sub>"
  } >> "$GITHUB_STEP_SUMMARY"
fi

exit "$OVERALL_FAIL"
