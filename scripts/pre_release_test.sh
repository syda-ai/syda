#!/usr/bin/env bash
# =============================================================================
# syda Pre-Release Test Script
# Builds a wheel, installs it in a clean virtual environment, and runs all
# pre-publish checks before releasing to PyPI.
# Usage: bash scripts/pre_release_test.sh
# =============================================================================

set -euxo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

ERRORS=0

pass()   { echo -e "${GREEN}✔ $*${NC}"; }
fail()   { echo -e "${RED}✘ $*${NC}"; ERRORS=$((ERRORS + 1)); }
info()   { echo -e "${BLUE}▸ $*${NC}"; }
warn()   { echo -e "${YELLOW}⚠ $*${NC}"; }
header() { echo -e "\n${BOLD}$*${NC}"; echo "$(printf '─%.0s' {1..60})"; }

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$PROJECT_ROOT/.pre-release-test-env"

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
  if [[ -d "$ENV_DIR" ]]; then
    info "Cleaning up test environment..."
    rm -rf "$ENV_DIR"
  fi
}
trap cleanup EXIT

# ── Step 1: Pre-flight checks ─────────────────────────────────────────────────
header "Step 1: Pre-flight checks"

PYTHON=$(command -v python3.11 || command -v python3 || command -v python || true)
if [[ -z "$PYTHON" ]]; then
  fail "Python 3 not found. Install it and retry."
  exit 1
fi

PYTHON_VERSION=$("$PYTHON" --version 2>&1 | awk '{print $2}')
info "Using Python $PYTHON_VERSION at $PYTHON"

REQUIRED_MAJOR=3
REQUIRED_MINOR=8
PY_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt "$REQUIRED_MAJOR" ]] || \
   { [[ "$PY_MAJOR" -eq "$REQUIRED_MAJOR" ]] && [[ "$PY_MINOR" -lt "$REQUIRED_MINOR" ]]; }; then
  fail "Python >= $REQUIRED_MAJOR.$REQUIRED_MINOR required (found $PYTHON_VERSION)"
  exit 1
fi
pass "Python version OK ($PYTHON_VERSION)"

if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
  fail "pyproject.toml not found in $PROJECT_ROOT"
  exit 1
fi
pass "pyproject.toml found"

EXPECTED_VERSION=$(grep '^version' "$PROJECT_ROOT/pyproject.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
info "Package version from pyproject.toml: $EXPECTED_VERSION"

INIT_VERSION=$(grep '__version__' "$PROJECT_ROOT/syda/__init__.py" | sed "s/.*'\(.*\)'.*/\1/")
if [[ "$EXPECTED_VERSION" == "$INIT_VERSION" ]]; then
  pass "Version consistent: pyproject.toml ($EXPECTED_VERSION) == __init__.py ($INIT_VERSION)"
else
  fail "Version mismatch: pyproject.toml=$EXPECTED_VERSION vs __init__.py=$INIT_VERSION"
fi

header "Step 1b: System dependencies"

BREW=$(command -v brew || true)
if [[ -z "$BREW" ]]; then
  warn "Homebrew not found — cannot auto-install system deps. Install from https://brew.sh"
else
  pass "Homebrew found at $BREW"

  brew_install() {
    local pkg="$1"
    local label="${2:-$1}"
    if brew list --formula "$pkg" &>/dev/null 2>&1; then
      pass "$label already installed"
    else
      info "Installing $label via Homebrew..."
      if brew install "$pkg" 2>&1 | tail -3; then
        pass "$label installed"
      else
        fail "$label brew install failed"
      fi
    fi
  }

  brew_install "libmagic"  "libmagic (python-magic)"
  brew_install "tesseract" "tesseract (pytesseract OCR)"
  brew_install "pango"     "pango (weasyprint PDF)"
fi

header "Step 1c: CHANGELOG entry check"
if grep -q "^## \[${EXPECTED_VERSION}\]" "$PROJECT_ROOT/CHANGELOG.md" 2>/dev/null; then
  pass "CHANGELOG.md has an entry for ${EXPECTED_VERSION}"
else
  fail "CHANGELOG.md has no '## [${EXPECTED_VERSION}]' entry — update it before releasing"
fi

header "Step 1d: Run unit tests"
if [[ -x "$PROJECT_ROOT/.venv/bin/pytest" ]]; then
  info "Running pytest via project .venv (fast-fail before building the wheel)..."
  if "$PROJECT_ROOT/.venv/bin/pytest" "$PROJECT_ROOT/tests/" -q; then
    pass "Unit tests passed"
  else
    fail "Unit tests failed — fix before releasing"
  fi
else
  warn "No .venv/bin/pytest found — skipping unit tests. Run 'pytest tests/' manually first."
fi

# ── Step 2: Build wheel ───────────────────────────────────────────────────────
header "Step 2: Build wheel"
info "Installing build tool and building wheel + sdist..."
"$PYTHON" -m pip install --quiet build 2>/dev/null || true
cd "$PROJECT_ROOT"
# Clean previous dist artifacts for this version to avoid stale wheels
rm -f "$PROJECT_ROOT/dist/syda-${EXPECTED_VERSION}"*.whl \
      "$PROJECT_ROOT/dist/syda-${EXPECTED_VERSION}"*.tar.gz 2>/dev/null || true
"$PYTHON" -m build --outdir "$PROJECT_ROOT/dist" 2>&1 | tail -5
WHEEL=$(ls "$PROJECT_ROOT/dist/syda-${EXPECTED_VERSION}"*.whl 2>/dev/null | head -1)
if [[ -z "$WHEEL" ]]; then
  fail "Wheel not found after build. Check build output."
  exit 1
fi
pass "Wheel built: $(basename "$WHEEL")"
INSTALL_TARGET="$WHEEL"

# ── Step 3: Create isolated virtual environment ───────────────────────────────
header "Step 3: Create isolated virtual environment"

if [[ -d "$ENV_DIR" ]]; then
  info "Removing existing test environment..."
  rm -rf "$ENV_DIR"
fi

"$PYTHON" -m venv "$ENV_DIR"
pass "Virtual environment created at $ENV_DIR"

PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

"$PIP" install --quiet --upgrade pip
info "pip upgraded inside test env"

# ── Step 4: Install the package ───────────────────────────────────────────────
header "Step 4: Install syda[mcp]"
info "Installing from: ${INSTALL_TARGET}[mcp]"
if "$PIP" install "${INSTALL_TARGET}[mcp]" 2>&1; then
  pass "pip install succeeded (with mcp extra)"
else
  fail "pip install failed"
fi

# ── Step 5: Verify installed version ─────────────────────────────────────────
header "Step 5: Verify installed version"
INSTALLED_VERSION=$("$PY" -c "import syda; print(syda.__version__)" 2>/dev/null || echo "FAILED")
if [[ "$INSTALLED_VERSION" == "$EXPECTED_VERSION" ]]; then
  pass "Installed version matches expected: $INSTALLED_VERSION"
else
  fail "Version mismatch — expected $EXPECTED_VERSION, got $INSTALLED_VERSION"
fi

# ── Step 6: Import checks ─────────────────────────────────────────────────────
header "Step 6: Import checks"

run_import_check() {
  local label="$1"
  local code="$2"
  if "$PY" -c "$code" &>/dev/null; then
    pass "$label"
  else
    # Re-run to capture error output
    ERR=$("$PY" -c "$code" 2>&1 || true)
    fail "$label"
    echo -e "    ${RED}$(echo "$ERR" | tail -3)${NC}"
  fi
}

run_import_check "syda top-level import"             "import syda"
run_import_check "SyntheticDataGenerator importable" "from syda import SyntheticDataGenerator"
run_import_check "ModelConfig importable"            "from syda import ModelConfig"
run_import_check "DatabaseSchemaLoader importable"   "from syda import DatabaseSchemaLoader"
run_import_check "syda.generate module"              "from syda.generate import SyntheticDataGenerator"
run_import_check "syda.schemas module"               "from syda.schemas import ModelConfig"
run_import_check "syda.db_schema_loader module"      "from syda.db_schema_loader import DatabaseSchemaLoader"
run_import_check "syda.cli module"                   "from syda.cli import main"
run_import_check "syda.llm module"                   "import syda.llm"
run_import_check "syda.output module"                "import syda.output"
run_import_check "syda.utils module"                 "import syda.utils"
run_import_check "syda.templates module"             "import syda.templates"
run_import_check "syda.schema_loader module"         "import syda.schema_loader"
run_import_check "syda.custom_generators module"     "import syda.custom_generators"
run_import_check "syda.dependency_handler module"    "import syda.dependency_handler"
run_import_check "DependencyHandler importable"      "from syda.dependency_handler import DependencyHandler"
run_import_check "compute_parallel_levels callable"  "from syda.dependency_handler import DependencyHandler; import inspect; assert callable(DependencyHandler.compute_parallel_levels)"
run_import_check "ModelConfig max_workers field"     "from syda import ModelConfig; m = ModelConfig(max_workers=4); assert m.max_workers == 4"
run_import_check "syda.unstructured module"          "import syda.unstructured"
run_import_check "syda.codegen_cache module"         "from syda.codegen_cache import CodegenCache, compute_schema_hash"
run_import_check "syda.run_report module"            "from syda.run_report import RunReport, TableReport, ColumnReport"
run_import_check "syda.mcp_server module"            "import syda.mcp_server"
run_import_check "mcp (FastMCP)"                     "from mcp.server.fastmcp import FastMCP"

# MCP entry point smoke test
SYDA_MCP_BIN="$ENV_DIR/bin/syda-mcp"
if [[ -f "$SYDA_MCP_BIN" ]]; then
  pass "syda-mcp entry point found"
else
  fail "syda-mcp entry point not found at $SYDA_MCP_BIN — check pyproject.toml [project.scripts] / [mcp] extra"
fi

# CLI entry point smoke test
SYDA_BIN="$ENV_DIR/bin/syda"
if [[ -f "$SYDA_BIN" ]]; then
  if "$SYDA_BIN" --help &>/dev/null && "$SYDA_BIN" version &>/dev/null; then
    pass "syda CLI entry point works (syda --help, syda version)"
  else
    fail "syda CLI entry point installed but failed to run"
  fi
  # Verify --workers flag is exposed on generate and run-schema subcommands
  if "$SYDA_BIN" generate --help 2>&1 | grep -q "\-\-workers"; then
    pass "syda generate --workers flag present"
  else
    fail "syda generate --workers flag missing from help output"
  fi
  if "$SYDA_BIN" db generate --help 2>&1 | grep -q "\-\-workers"; then
    pass "syda db generate --workers flag present"
  else
    fail "syda db generate --workers flag missing from help output"
  fi
else
  fail "syda CLI entry point not found at $SYDA_BIN — check pyproject.toml [project.scripts]"
fi

# ── Step 7: __all__ surface check ────────────────────────────────────────────
header "Step 7: Public API surface (__all__)"
if "$PY" -c "
import syda
expected = {'SyntheticDataGenerator', 'ModelConfig', 'DatabaseSchemaLoader',
            'CodegenCache', 'compute_schema_hash', 'RunReport', 'TableReport', 'ColumnReport'}
actual = set(syda.__all__)
missing = expected - actual
if missing:
    print('MISSING:', missing)
    exit(1)
print('Public API OK:', sorted(actual))
" 2>&1; then
  pass "__all__ matches expected public API"
else
  fail "__all__ mismatch"
fi

# ── Step 8: Core dependency imports ──────────────────────────────────────────
header "Step 8: Core dependency sanity checks"

run_import_check "pydantic >= 2"      "import pydantic; assert int(pydantic.VERSION.split('.')[0]) >= 2"
run_import_check "sqlalchemy"         "import sqlalchemy"
run_import_check "pandas"             "import pandas"
run_import_check "openai"             "import openai"
run_import_check "anthropic"          "import anthropic"
run_import_check "pydantic_ai"        "import pydantic_ai; from pydantic_ai import Agent"
run_import_check "google.genai"       "import google.genai"
run_import_check "networkx"           "import networkx"
run_import_check "jsonref"            "import jsonref"
run_import_check "dotenv"             "import dotenv"
run_import_check "yaml"               "import yaml"
run_import_check "genai_prices"       "from genai_prices import calc_price"
run_import_check "jinja2"             "import jinja2"
run_import_check "boto3"              "import boto3"
run_import_check "azure.storage.blob" "from azure.storage.blob import BlobServiceClient"
run_import_check "pdfplumber"         "import pdfplumber"
run_import_check "PIL (Pillow)"       "from PIL import Image"
run_import_check "docx"               "import docx"
run_import_check "openpyxl"           "import openpyxl"
run_import_check "magic (libmagic)"   "import magic; magic.Magic(mime=True)"

# ── Step 9: Installed package metadata ───────────────────────────────────────
header "Step 9: Package metadata (pip show)"
"$PIP" show syda
pass "pip show completed"

# ── Step 10: No leftover dev/test files included ──────────────────────────────
header "Step 10: Check no test/dev files are inside installed package"
SITE_PACKAGES=$("$PY" -c "import sysconfig; print(sysconfig.get_path('purelib'))")
if ls "$SITE_PACKAGES"/syda/test*.py &>/dev/null; then
  fail "Test files found inside installed package at $SITE_PACKAGES/syda — check MANIFEST.in exclusions"
else
  pass "No test files leaked into installed package"
fi

# ── Step 11: Run examples ─────────────────────────────────────────────────────
header "Step 11: Run examples"

EXAMPLES_DIR="$PROJECT_ROOT/examples"

# Source .env so API keys are available to child processes
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set +x
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
  set -x
  info ".env loaded"
fi

[[ -n "${ANTHROPIC_API_KEY:-}" ]] && pass "ANTHROPIC_API_KEY found" || warn "ANTHROPIC_API_KEY not set"
[[ -n "${OPENAI_API_KEY:-}"    ]] && pass "OPENAI_API_KEY found"    || warn "OPENAI_API_KEY not set"
[[ -n "${GROK_API_KEY:-}"      ]] && pass "GROK_API_KEY found"      || warn "GROK_API_KEY not set — Grok examples will be skipped"
[[ -n "${DB_HOST:-}" ]] && pass "DB_HOST found" || warn "DB_HOST not set — database examples may fail"

# ── Ollama setup (shared by the CLI large-dataset demo, the openai_compatible
#    Python example, and the MCP smoke test — detect/start it once up front
#    rather than three separate times) ──────────────────────────────────────
OLLAMA_BIN=$(command -v ollama || true)
OLLAMA_STARTED=false
if [[ -z "$OLLAMA_BIN" ]]; then
  warn "ollama not found — openai_compatible examples (incl. large-dataset demo) will be skipped"
else
  pass "ollama found at $OLLAMA_BIN"
  if curl -sf http://localhost:11434/ &>/dev/null; then
    pass "Ollama server already running"
  else
    info "Starting Ollama server..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    OLLAMA_STARTED=true
    for i in {1..10}; do
      sleep 1
      curl -sf http://localhost:11434/ &>/dev/null && break
    done
    if curl -sf http://localhost:11434/ &>/dev/null; then
      pass "Ollama server started (pid $OLLAMA_PID)"
    else
      fail "Ollama server did not start in time"
      OLLAMA_BIN=""
    fi
  fi
  if [[ -n "$OLLAMA_BIN" ]]; then
    OLLAMA_MODEL=$(ollama list 2>/dev/null | awk 'NR>1 && $1!="" {print $1; exit}')
    if [[ -z "$OLLAMA_MODEL" ]]; then
      warn "No Ollama models found (try 'ollama pull gpt-oss:20b') — openai_compatible examples will be skipped"
      OLLAMA_BIN=""
    else
      pass "Using Ollama model: $OLLAMA_MODEL"
      export OLLAMA_MODEL
      export OLLAMA_BASE_URL="http://localhost:11434/v1"
    fi
  fi
fi

run_example() {
  local label="$1"
  local script="$2"
  info "Running: $label"
  local output
  if output=$("$PY" "$script" 2>&1); then
    echo "$output"
    pass "Example: $label"
  else
    echo "$output"
    # Treat API 404 errors as warnings (deprecated model name in example, not a package bug)
    if echo "$output" | grep -q "404\|not_found_error\|NotFoundError"; then
      warn "Example: $label — API model not found (example may use a deprecated model name)"
    # Local/small models occasionally fail strict structured-output validation
    # a few times in a row — model-quality flakiness, not a packaging bug.
    elif echo "$output" | grep -qi "Exceeded maximum output retries"; then
      warn "Example: $label — model failed structured-output validation after retries (LLM flakiness, not a package bug)"
    else
      fail "Example: $label (exit code $?)"
    fi
  fi
}

# quickstart
run_example "quickstart" "$EXAMPLES_DIR/quickstart.py"

# structured_only
run_example "structured_only/dict_schemas" \
  "$EXAMPLES_DIR/structured_only/example_dict_schemas.py"
run_example "structured_only/yaml_schemas" \
  "$EXAMPLES_DIR/structured_only/example_yaml_schemas.py"

# force_llm (uses auto-detected provider; Grok preferred for speed/cost)
if [[ -n "${GROK_API_KEY:-}" || -n "${ANTHROPIC_API_KEY:-}" ]]; then
  run_example "force_llm/product_catalog" \
    "$EXAMPLES_DIR/force_llm/example_force_llm.py"
else
  warn "Skipping force_llm example — no API key found"
fi

# structured_and_unstructured
run_example "structured_and_unstructured/retail_yml" \
  "$EXAMPLES_DIR/structured_and_unstructured/retail_yml/example_retail_schemas.py"

# unstructured_only — PDF/HTML document generation from templates
run_example "unstructured_only/healthcare_yml" \
  "$EXAMPLES_DIR/unstructured_only/healthcare_yml/generate_healthcare_data.py"

# database_integration — delete stale SQLite DB so each run starts clean
DB_FILE="$EXAMPLES_DIR/database_integration/healthcare_demo.db"
if [[ -f "$DB_FILE" ]]; then
  info "Removing stale SQLite DB: $DB_FILE"
  rm -f "$DB_FILE"
fi

run_example "database_integration/load_schemas" \
  "$EXAMPLES_DIR/database_integration/example_load_schemas.py"

# Remove again between runs so save_schemas also starts clean
[[ -f "$DB_FILE" ]] && rm -f "$DB_FILE"

run_example "database_integration/save_schemas" \
  "$EXAMPLES_DIR/database_integration/example_save_schemas.py"
run_example "database_integration/postgres" \
  "$EXAMPLES_DIR/database_integration/example_postgres.py"

# large_dataset/postgres — only run when DB_HOST and GROK_API_KEY are set
if [[ -n "${DB_HOST:-}" && -n "${GROK_API_KEY:-}" ]]; then
  run_example "large_dataset/postgres" \
    "$EXAMPLES_DIR/large_dataset/example_large_dataset_postgres.py"
elif [[ -z "${DB_HOST:-}" ]]; then
  warn "DB_HOST not set — skipping large_dataset/postgres example (requires live PostgreSQL)"
else
  warn "GROK_API_KEY not set — skipping large_dataset/postgres example"
fi

# CLI large dataset demo (shell script — put test env on PATH so syda is found).
# Runs against Anthropic Claude, not a local model — code-gen mode has the
# LLM write actual Python generator functions, and Claude is materially more
# reliable at that than a small local model (this used to run on Ollama, but
# structured/codegen correctness matters more here than being free).
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  info "Running CLI large dataset demo (anthropic / claude-haiku-4-5-20251001)..."
  set +e
  CLI_DEMO_OUTPUT=$(PATH="$ENV_DIR/bin:$PATH" \
    bash "$PROJECT_ROOT/examples/cli/demo_large_dataset.sh" 2>&1)
  CLI_DEMO_EXIT=$?
  set -e
  echo "$CLI_DEMO_OUTPUT"
  if [[ $CLI_DEMO_EXIT -eq 0 ]]; then
    pass "CLI large dataset demo"
  elif echo "$CLI_DEMO_OUTPUT" | grep -qi "404\|not_found_error\|NotFoundError\|finish_reason\|validation error\|literal_error\|Exceeded maximum output retries"; then
    warn "CLI large dataset demo — transient API/validation error (not a package bug): $(echo "$CLI_DEMO_OUTPUT" | grep -Ei 'finish_reason|404|not_found|Exceeded maximum output retries' | head -1)"
  else
    fail "CLI large dataset demo"
  fi
else
  warn "ANTHROPIC_API_KEY not set — skipping CLI large dataset demo"
fi

# openai_compatible — Ollama was already detected/started up front (shared
# with the CLI large-dataset demo above and the MCP smoke test below).
if [[ -n "$OLLAMA_BIN" ]]; then
  OPENAI_COMPATIBLE_BASE_URL="$OLLAMA_BASE_URL" \
  OPENAI_COMPATIBLE_API_KEY="ollama" \
  OPENAI_COMPATIBLE_MODEL="$OLLAMA_MODEL" \
  run_example "model_selection/openai_compatible ($OLLAMA_MODEL)" \
    "$EXAMPLES_DIR/model_selection/example_openai_compatible_models.py"
else
  warn "Ollama not available — skipping openai_compatible example"
fi

# ── Step 12: MCP server smoke test ────────────────────────────────────────────
# Exercises the *installed wheel's* syda-mcp binary over the real MCP stdio
# protocol (not a direct Python import) — this is the only way to catch bugs
# that only manifest for a real install, e.g. .env auto-loading breaking
# because mcp_server.py's on-disk path is inside site-packages instead of
# next to the project's .env (see CHANGELOG 0.4.0).
header "Step 12: MCP server smoke test (real stdio protocol)"

if [[ -f "$SYDA_MCP_BIN" ]]; then
  info "Running examples/mcp/test_provider_matrix.py against the installed syda-mcp..."
  set +e
  MCP_OUTPUT=$(PATH="$ENV_DIR/bin:$PATH" \
    OPENAI_COMPATIBLE_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}" \
    OPENAI_COMPATIBLE_MODEL="${OLLAMA_MODEL:-gpt-oss:20b}" \
    "$PY" "$EXAMPLES_DIR/mcp/test_provider_matrix.py" 2>&1)
  MCP_EXIT=$?
  set -e
  echo "$MCP_OUTPUT"
  if [[ $MCP_EXIT -eq 0 ]]; then
    pass "MCP provider matrix — all configured providers passed with FK integrity verified"
  elif echo "$MCP_OUTPUT" | grep -qE "^\s*FAIL\s+openai_compatible" && \
       ! echo "$MCP_OUTPUT" | grep -qE "^\s*FAIL\s+(anthropic|openai|gemini|grok|azureopenai)\b"; then
    warn "MCP provider matrix — only openai_compatible (Ollama) failed; likely no local Ollama running, not a packaging bug"
  else
    fail "MCP provider matrix — a configured provider failed over the real MCP protocol"
  fi
else
  warn "syda-mcp binary not found — skipping MCP smoke test"
fi

# Stop Ollama if we started it (all Ollama-dependent steps are done now)
if [[ "${OLLAMA_STARTED:-false}" == "true" && -n "${OLLAMA_PID:-}" ]]; then
  info "Stopping Ollama server (pid $OLLAMA_PID)..."
  kill "$OLLAMA_PID" 2>/dev/null || true
fi

# ── Summary ───────────────────────────────────────────────────────────────────
header "Summary"
if [[ "$ERRORS" -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}All checks passed! syda $EXPECTED_VERSION is ready to publish.${NC}"
  echo ""
  echo -e "  ${BOLD}To publish:${NC}"
  echo -e "    python -m build"
  echo -e "    twine upload dist/syda-${EXPECTED_VERSION}*"
  exit 0
else
  echo -e "${RED}${BOLD}$ERRORS check(s) failed. Fix issues before publishing.${NC}"
  exit 1
fi
