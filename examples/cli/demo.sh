#!/usr/bin/env bash
# =============================================================================
# syda CLI demo
#
# Covers:
#   1. syda version
#   2. syda validate   — single file and directory
#   3. syda generate   — single schema → CSV
#   4. syda generate   — single schema → JSON
#   5. syda generate   — directory of schemas (multi-table FK chain) → CSV
#   6. syda generate   — directory → JSON
#   7. syda db infer   — extract schemas from a SQLite database
#   8. syda db generate — generate data from DB schema, save to files
#   9. syda db generate --write-back — insert generated data back into the DB
#  10. syda db generate --tables — specific tables only
#
# Prerequisites:
#   pip install syda
#   export ANTHROPIC_API_KEY=your_key   (or OPENAI_API_KEY / GEMINI_API_KEY)
#
# Usage:
#   chmod +x demo.sh && ./demo.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"
SCHEMAS_DIR="$SCRIPT_DIR/schemas"
DB_FILE="$SCRIPT_DIR/demo.db"

# Load .env from project root if present
if [ -f "$SCRIPT_DIR/../../.env" ]; then
  set -o allexport
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/../../.env"
  set +o allexport
fi

GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

section() { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${RESET}"; \
            echo -e "${BOLD}${CYAN}  $1${RESET}"; \
            echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}"; }
ok()      { echo -e "${GREEN}✓  $1${RESET}"; }

mkdir -p "$OUTPUT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
section "1. Version"
# ─────────────────────────────────────────────────────────────────────────────
syda version

# ─────────────────────────────────────────────────────────────────────────────
section "2. Validate schemas"
# ─────────────────────────────────────────────────────────────────────────────
echo "Single file:"
syda validate --schema "$SCHEMAS_DIR/patient.yml"

echo ""
echo "Entire schemas directory (patient, provider, appointment):"
syda validate --schema "$SCHEMAS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
section "3. Generate: single schema → CSV"
# ─────────────────────────────────────────────────────────────────────────────
syda generate \
  --schema "$SCHEMAS_DIR/patient.yml" \
  --rows 5 \
  --output "$OUTPUT_DIR/patients.csv"

ok "patients.csv"
head -4 "$OUTPUT_DIR/patients.csv"

# ─────────────────────────────────────────────────────────────────────────────
section "4. Generate: single schema → JSON"
# ─────────────────────────────────────────────────────────────────────────────
syda generate \
  --schema "$SCHEMAS_DIR/provider.yml" \
  --rows 3 \
  --output "$OUTPUT_DIR/providers.json"

ok "providers.json"
head -20 "$OUTPUT_DIR/providers.json"

# ─────────────────────────────────────────────────────────────────────────────
section "5. Generate: directory of schemas → CSV (FK integrity preserved)"
# ─────────────────────────────────────────────────────────────────────────────
# patient → provider → appointment  (order resolved automatically)
syda generate \
  --schema "$SCHEMAS_DIR" \
  --rows 5 \
  --output-dir "$OUTPUT_DIR/multi_csv"

ok "Multi-table CSV output"
ls -1 "$OUTPUT_DIR/multi_csv"

# ─────────────────────────────────────────────────────────────────────────────
section "6. Generate: directory of schemas → JSON"
# ─────────────────────────────────────────────────────────────────────────────
syda generate \
  --schema "$SCHEMAS_DIR" \
  --rows 3 \
  --output-dir "$OUTPUT_DIR/multi_json" \
  --format json

ok "Multi-table JSON output"
ls -1 "$OUTPUT_DIR/multi_json"

# ─────────────────────────────────────────────────────────────────────────────
section "7. db infer: extract schemas from SQLite"
# ─────────────────────────────────────────────────────────────────────────────
echo "Creating demo SQLite database ..."
python3 "$SCRIPT_DIR/create_demo_db.py" "$DB_FILE"

syda db infer \
  --db-url "sqlite:///$DB_FILE" \
  --output-dir "$OUTPUT_DIR/inferred_schemas"

ok "Inferred schema files"
ls -1 "$OUTPUT_DIR/inferred_schemas"

# ─────────────────────────────────────────────────────────────────────────────
section "8. db generate: generate data from DB schema → CSV files"
# ─────────────────────────────────────────────────────────────────────────────
syda db generate \
  --db-url "sqlite:///$DB_FILE" \
  --rows 5 \
  --output-dir "$OUTPUT_DIR/db_generated"

ok "DB-sourced CSV files"
ls -1 "$OUTPUT_DIR/db_generated"
echo ""
echo "providers preview:"
head -4 "$OUTPUT_DIR/db_generated/provider.csv"

# ─────────────────────────────────────────────────────────────────────────────
section "9. db generate --write-back: insert rows directly into the DB"
# ─────────────────────────────────────────────────────────────────────────────
syda db generate \
  --db-url "sqlite:///$DB_FILE" \
  --rows 3 \
  --write-back \
  --if-exists replace

echo "Row counts after write-back:"
python3 - "$DB_FILE" <<'EOF'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
for tbl in ("provider", "patient", "appointment"):
    n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl}: {n} rows")
conn.close()
EOF
ok "Write-back complete"

# ─────────────────────────────────────────────────────────────────────────────
section "10. db generate --tables: specific tables only"
# ─────────────────────────────────────────────────────────────────────────────
syda db generate \
  --db-url "sqlite:///$DB_FILE" \
  --tables "provider" \
  --rows 4 \
  --output-dir "$OUTPUT_DIR/provider_only"

ok "provider_only/"
cat "$OUTPUT_DIR/provider_only/provider.csv"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}All 10 demo steps completed successfully.${RESET}"
echo "Output written to: $OUTPUT_DIR"
