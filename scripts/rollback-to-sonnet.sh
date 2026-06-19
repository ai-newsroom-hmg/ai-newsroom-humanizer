#!/usr/bin/env bash
# UNBEDINGT-Skill 3 (always-one-version-rollback): Rollback humanize CLI auf v0.1-sonnet-loop.
#
# Trigger: Mistral-3.2 auf OpenRouter down ODER BGE-Filter consistently <0.85
#          ODER Pangram-API down ODER neue Phase-Regression.
#
# Effekt: cli.py + core.py auf v0.1-sonnet-loop zurückgesetzt (alter Sonnet-Loop).
#         core_bestofn.py bleibt als ungenutzte Datei liegen (no-harm).
#
# RTO-Ziel: < 30s. Verifikation am Ende.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ANCHOR="v0.1-sonnet-loop"

if ! git tag --list | grep -q "^${ANCHOR}$"; then
    echo "ERROR: Anchor tag ${ANCHOR} fehlt. Kein Rollback möglich." >&2
    exit 2
fi

echo "=== Rollback humanize-CLI → ${ANCHOR} ==="
echo "Aktueller HEAD: $(git rev-parse --short HEAD)"
echo "Anchor:         $(git rev-parse --short ${ANCHOR})"
echo ""

# Falls user uncommitted changes hat, abbrechen
if ! git diff-index --quiet HEAD -- src/humanizer/cli.py src/humanizer/core.py 2>/dev/null; then
    echo "WARN: uncommitted changes in cli.py/core.py. Stash or commit first." >&2
    git status --short src/humanizer/cli.py src/humanizer/core.py
    echo ""
    read -p "Continue overwriting? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 3; }
fi

git checkout "${ANCHOR}" -- src/humanizer/cli.py src/humanizer/core.py
echo "✓ cli.py + core.py restored from ${ANCHOR}"

# RTO-Verifikation
echo ""
echo "=== RTO-Test: humanize --legacy auf 1-Zeiler ==="
T0=$(date +%s)
if echo "Dies ist ein kurzer Testsatz für den Rollback-Smoke." \
        | timeout 30s .venv/bin/humanize - --legacy --variants 2 -q -o /tmp/rollback-smoke.txt 2>/dev/null; then
    T1=$(date +%s)
    DUR=$((T1 - T0))
    echo "✓ RTO erfolgreich in ${DUR}s. Output:"
    head -3 /tmp/rollback-smoke.txt
else
    echo "✗ RTO-Test fehlgeschlagen. Manueller Eingriff nötig."
    exit 4
fi
echo ""
echo "Rollback abgeschlossen. CLI ist auf Sonnet-Loop (Phase 2)."
echo "Re-Apply v0.2: git checkout HEAD -- src/humanizer/cli.py src/humanizer/core.py"
