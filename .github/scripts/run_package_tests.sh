#!/usr/bin/env bash
# ci_run_tests_clean.sh — quiet, failure-focused test runner for large mono-repos
# - Discovers packages
# - Runs pytest per package (parallel capped)
# - Streams only PASS/FAIL lines; prints failure blocks on FAIL
# - Writes full logs to files and a compact Step Summary

set -euo pipefail
REPO_ROOT="$(pwd -P)"

# ----------------------- Config knobs -----------------------
: "${PKG_GLOBS:=agent common core app_backend packages/*}"
: "${PARALLEL:=true}"
: "${MAX_PROCS:=8}"
: "${SELECT:=}"                                # e.g. "-k 'not slow' -m 'not integ'"
: "${MAXFAIL:=1}"                              # fail-fast per package
: "${TIMEOUT:=120}"
: "${PYTEST_ADDOPTS:=}"                        # external overrides

DEFAULT_PYTEST_OPTS="-n auto --dist loadscope -q -rA \
  --durations=15 --maxfail=${MAXFAIL} \
  --timeout=${TIMEOUT} --timeout-method=thread \
  --color=yes --benchmark-skip"
PYTEST_OPTS="${PYTEST_ADDOPTS:-$DEFAULT_PYTEST_OPTS} ${SELECT}"

ART_DIR="${ART_DIR:-test-results}"
COV_DIR="${COV_DIR:-coverage-reports}"
mkdir -p "${REPO_ROOT}/$ART_DIR" "${REPO_ROOT}/$COV_DIR"

# ----------------------- Colors -----------------------------
RED=$'\e[1;31m'; GRN=$'\e[1;32m'; YEL=$'\e[1;33m'
BLU=$'\e[1;34m'; MAG=$'\e[1;35m'; CYN=$'\e[1;36m'; WHT=$'\e[1;37m'; NC=$'\e[0m'
colors=("$BLU" "$RED" "$GRN" "$YEL" "$MAG" "$CYN" "$WHT")

ts() { date +"%H:%M:%S"; }
group_start(){ echo "::group::$*"; }
group_end(){ echo "::endgroup::" || true; }

# ----------------------- Discover ---------------------------
discover_packages() {
  local -a out=()
  for g in $PKG_GLOBS; do
    for p in $(compgen -G "$g" || true); do
      [ -d "$p" ] || continue
      if compgen -G "$p/tests" > /dev/null \
        || compgen -G "$p/**/test_*.py" > /dev/null \
        || compgen -G "$p/**/*_test.py" > /dev/null; then
        out+=("$p")
      fi
    done
  done
  printf "%s\n" "${out[@]}" | awk 'NF' | sort -u
}

# ----------------------- Failure helpers --------------------
print_fail_block() {
  # Args: <logfile>
  # Show only the "==== FAILURES ====" section and the short summary block.
  local f="$1"
  awk '
    BEGIN { in_fail=0 }
    /^=+ FAILURES =+/ { in_fail=1; print; next }
    /^=+ [A-Z].* =+$/ { if (in_fail) { in_fail=0 } }
    in_fail { print }
  ' "$f"
  echo
  grep -n "short test summary info" "$f" | while IFS=: read -r ln _; do
    start=$((ln-1)); [ "$start" -lt 1 ] && start=1
    sed -n "${start},$((ln+50))p" "$f" | sed '/^=* .* =*$/,$d'
  done || true
}

emit_repro_cmds() {
  # Args: <logfile> <pkg> <outfile>
  local f="$1" pkg="$2" out="$3"
  : > "$out"
  grep -E "^FAILED " "$f" | awk '{print $2}' | while read -r nodeid; do
    if [[ "$pkg" == "core" || "$pkg" == "." ]]; then
      echo "pytest -q ${nodeid}" >> "$out"
    else
      echo "(cd ${pkg} && pytest -q ${nodeid})" >> "$out"
    fi
  done
}

# ----------------------- Runner -----------------------------
run_pkg() {
  local pkg="$1" idx="$2"
  local name; name="$(basename "$pkg")"
  local color="${colors[$((idx % ${#colors[@]}))]}"
  local raw="${REPO_ROOT}/${ART_DIR}/${name}.log"
  local exitfile="${REPO_ROOT}/${ART_DIR}/${name}.exit"
  local durfile="${REPO_ROOT}/${ART_DIR}/${name}.dur"
  local covxml="${REPO_ROOT}/${COV_DIR}/coverage-${name}.xml"
  local repro="${REPO_ROOT}/${ART_DIR}/${name}.repro"
  local run_dir="."
  local covpath="$covxml"

  if [[ "$pkg" != "." && "$pkg" != "core" ]]; then
    run_dir="$pkg"
    # compute ../../ back to repo root
    local depth="${pkg//[^\/]/}"
    local up=""; for _ in ${depth}; do up+="../"; done
    covpath="${up}${covxml}"
  fi

  local t0 t1
  t0=$(date +%s)

  # Run pytest, capture only to file. Keep stdout clean unless failing.
  (
    cd "$run_dir"
    pytest ${PYTEST_OPTS} --cov --cov-branch --cov-report="xml:${covpath}" \
      >"${raw}" 2>&1
  )
  status=$? || true

  t1=$(date +%s)
  echo "$((t1-t0))s [package ${name}]" > "$durfile"
  printf "%d" "$status" > "$exitfile"

  if [ "$status" -eq 0 ]; then
    echo -e "${color}[$(ts)] [${name}] ${GRN}PASS${NC}"
    rm -f "$repro"
  else
    echo -e "${color}[$(ts)] [${name}] ${RED}FAIL${NC}"
    emit_repro_cmds "$raw" "$pkg" "$repro"
    group_start "failures: ${name}"
    print_fail_block "$raw" || true
    echo
    echo "${CYN}repro:${NC} ${repro}"
    echo "${CYN}log:  ${NC} ${raw}"
    group_end
  fi

  # Also collect per-test durations for a global top-10
  grep -E "^[0-9]+\.[0-9]+s " "$raw" > "${REPO_ROOT}/${ART_DIR}/${name}.pytest.dur" || true
}

# ----------------------- Main -------------------------------
START="$(date +%s)"
mapfile -t PACKAGES < <(discover_packages)
if [ "${#PACKAGES[@]}" -eq 0 ]; then
  echo "No packages with tests discovered under: $PKG_GLOBS"
  exit 0
fi

echo "Discovered packages:"
printf " - %s\n" "${PACKAGES[@]}"

pids=(); code=0
if [ "$PARALLEL" = "true" ]; then
  sem=${MAX_PROCS}
  for i in "${!PACKAGES[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$sem" ]; do sleep 0.2; done
    run_pkg "${PACKAGES[$i]}" "$i" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || code=1; done
else
  for i in "${!PACKAGES[@]}"; do
    run_pkg "${PACKAGES[$i]}" "$i" || code=1
  done
fi

END="$(date +%s)"; TOTAL="$((END-START))"

# ----------------------- Summaries --------------------------
echo -e "\n${WHT}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${WHT}🐌 TOP 10 SLOWEST TESTS${NC}"
cat "${REPO_ROOT}/${ART_DIR}"/*.pytest.dur 2>/dev/null | sort -k2 -nr | head -10 | nl -w2 -s'. ' || echo "No duration data."

echo -e "\n${WHT}📊 PACKAGE RESULTS${NC}"
printf "%-24s %-8s %s\n" "package" "result" "log"
printf "%-24s %-8s %s\n" "-------" "------" "---"

OVERALL_FAIL=0
for pkg in "${PACKAGES[@]}"; do
  name="$(basename "$pkg")"
  exitf="$ART_DIR/${name}.exit"
  logf="$ART_DIR/${name}.log"
  repf="$ART_DIR/${name}.repro"
  if [ -f "$exitf" ] && [ "$(cat "$exitf")" -eq 0 ]; then
    printf "%-24s %-8s %s\n" "$name" "${GRN}PASS${NC}" "$logf"
  else
    if [ -f "$repf" ]; then
      printf "%-24s %-8s %s (repro: %s)\n" "$name" "${RED}FAIL${NC}" "$logf" "$repf"
    else
      printf "%-24s %-8s %s\n" "$name" "${RED}FAIL${NC}" "$logf"
    fi
    OVERALL_FAIL=1
  fi
done
echo "Total time: ${TOTAL}s"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "## Unit test summary"
    echo ""
    echo "| Package | Result | Log | Repro |"
    echo "|---|---|---|---|"
    for pkg in "${PACKAGES[@]}"; do
      name="$(basename "$pkg")"
      exitf="$ART_DIR/${name}.exit"
      logf="$ART_DIR/${name}.log"
      repf="$ART_DIR/${name}.repro"
      res="FAIL"; [ -f "$exitf" ] && [ "$(cat "$exitf")" -eq 0 ] && res="PASS"
      if [ -f "$repf" ]; then
        repcell="\`$repf\`"
      else
        repcell="—"
      fi
      echo "| \`$name\` | $res | \`$logf\` | ${repcell} |"
    done
    echo ""
    echo "<sub>Total time: ${TOTAL}s</sub>"
  } >> "$GITHUB_STEP_SUMMARY"
fi

exit "$OVERALL_FAIL"
