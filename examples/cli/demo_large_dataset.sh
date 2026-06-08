#!/usr/bin/env bash
# =============================================================================
# syda CLI — large dataset demo
#
# Demonstrates chunked generation and code-gen mode via the CLI:
#
#   1. Direct mode  — --rows 150 --batch-size 50
#      LLM is called 3× (50 rows each).  Good for moderate row counts.
#
#   2. Auto code-gen — --rows 600 (no flag needed)
#      >500 rows auto-selects codegen: one LLM analysis call writes Python
#      generator functions; simple columns run locally; only semantic columns
#      call the LLM.  Dramatically fewer API calls at scale.
#
#   3. Forced code-gen — --rows 50 --large-dataset
#      Forces codegen even for small N (useful to inspect the generated
#      Python functions or prime the cache for a later large run).
#
#   4. Multi-table code-gen — --rows 1000 --large-dataset
#      product → order FK chain; cache hits from step 2/3 make this instant.
#
# Provider: Grok-3  (set GROK_API_KEY in env or .env)
# Schemas : examples/cli/schemas_large/  (product.yml, order.yml)
#
# Usage:
#   chmod +x demo_large_dataset.sh && ./demo_large_dataset.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output/large_dataset"
SCHEMAS_DIR="$SCRIPT_DIR/schemas_large"

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

section() {
  echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${RESET}"
  echo -e "${BOLD}${CYAN}  $1${RESET}"
  echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}"
}
ok() { echo -e "${GREEN}✓  $1${RESET}"; }

mkdir -p "$OUTPUT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
section "1. Direct mode: 150 rows, batch-size 50 → 3 LLM calls"
# ─────────────────────────────────────────────────────────────────────────────
# batch-size=50 means ceil(150/50)=3 chunked calls.
# Row count ≤500 → auto stays in direct mode even without --large-dataset.

syda generate \
  --schema "$SCHEMAS_DIR/product.yml" \
  --rows 150 \
  --batch-size 50 \
  --provider grok \
  --model grok-3 \
  --output-dir "$OUTPUT_DIR/direct_mode" \
  --prompt "Generate realistic e-commerce products spanning electronics, clothing, home goods, and sports. Prices \$5–\$500."

ok "direct_mode/product.csv"
ROW_COUNT=$(tail -n +2 "$OUTPUT_DIR/direct_mode/product.csv" | wc -l | tr -d ' ')
echo "  Rows generated: $ROW_COUNT (target: 150)"
head -3 "$OUTPUT_DIR/direct_mode/product.csv"

# ─────────────────────────────────────────────────────────────────────────────
section "2. Auto code-gen: 600 rows (>500 triggers codegen automatically)"
# ─────────────────────────────────────────────────────────────────────────────
# No flag needed — generation_mode='auto' switches to codegen above 500 rows.
# LLM makes 1 analysis call, writes Python generator functions for simple cols,
# then generates only semantic cols (like 'description') via LLM.

syda generate \
  --schema "$SCHEMAS_DIR/product.yml" \
  --rows 600 \
  --provider grok \
  --model grok-3 \
  --output-dir "$OUTPUT_DIR/auto_codegen" \
  --prompt "Generate realistic e-commerce products spanning electronics, clothing, home goods, and sports. Prices \$5–\$500."

ok "auto_codegen/product.csv"
ROW_COUNT=$(tail -n +2 "$OUTPUT_DIR/auto_codegen/product.csv" | wc -l | tr -d ' ')
echo "  Rows generated: $ROW_COUNT (target: 600)"

# ─────────────────────────────────────────────────────────────────────────────
section "3. Forced code-gen: 50 rows with --large-dataset flag"
# ─────────────────────────────────────────────────────────────────────────────
# --large-dataset forces codegen regardless of row count.
# Second run on same schema → cache HIT (no analysis call).

syda generate \
  --schema "$SCHEMAS_DIR/product.yml" \
  --rows 50 \
  --large-dataset \
  --provider grok \
  --model grok-3 \
  --output-dir "$OUTPUT_DIR/forced_codegen" \
  --prompt "Generate realistic e-commerce products spanning electronics, clothing, home goods, and sports. Prices \$5–\$500."

ok "forced_codegen/product.csv"
ROW_COUNT=$(tail -n +2 "$OUTPUT_DIR/forced_codegen/product.csv" | wc -l | tr -d ' ')
echo "  Rows generated: $ROW_COUNT (target: 50)"

# ─────────────────────────────────────────────────────────────────────────────
section "4. Multi-table code-gen: product + order (FK chain, 1000 rows each)"
# ─────────────────────────────────────────────────────────────────────────────
# Directory schema: product.yml → order.yml (order.product_id references product)
# Both tables in codegen mode. Cache hit for product (from step 2/3).
# order_items FK values sampled from already-generated products.

syda generate \
  --schema "$SCHEMAS_DIR" \
  --rows 1000 \
  --large-dataset \
  --provider grok \
  --model grok-3 \
  --output-dir "$OUTPUT_DIR/multi_table" \
  --prompt "Generate a realistic e-commerce dataset. Products span electronics, clothing, home goods, and sports. Orders placed over the past 2 years with realistic status distribution."

ok "multi_table/ output"
ls -1 "$OUTPUT_DIR/multi_table/"*.csv 2>/dev/null | while read f; do
  name=$(basename "$f")
  rows=$(tail -n +2 "$f" | wc -l | tr -d ' ')
  echo "  $name: $rows rows"
done

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}All 4 large-dataset demo steps completed.${RESET}"
echo "Output written to: $OUTPUT_DIR"
echo ""
echo "Key takeaways:"
echo "  --batch-size N       chunk direct-mode calls (N rows per LLM call)"
echo "  --rows >500          auto-selects codegen (fewer LLM calls at scale)"
echo "  --large-dataset      forces codegen for any row count"
echo "  Cache files saved to output/.syda_cache/ — re-runs are instant"
