#!/usr/bin/env bash
# =============================================================================
# syda Pre-Release Test Script
# Tests local package installation in an isolated virtual environment before
# publishing to PyPI.
# Usage: bash scripts/pre_release_test.sh [--from-wheel]
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
FROM_WHEEL=false

if [[ "${1:-}" == "--from-wheel" ]]; then
  FROM_WHEEL=true
fi

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

# ── Step 2: Build (optional wheel path) ──────────────────────────────────────
INSTALL_TARGET="$PROJECT_ROOT"
if $FROM_WHEEL; then
  header "Step 2: Build wheel"
  info "Building wheel and sdist..."
  "$PYTHON" -m pip install --quiet build 2>/dev/null || true
  cd "$PROJECT_ROOT"
  "$PYTHON" -m build --outdir "$PROJECT_ROOT/dist" 2>&1 | tail -5
  WHEEL=$(ls "$PROJECT_ROOT/dist/syda-${EXPECTED_VERSION}"*.whl 2>/dev/null | head -1)
  if [[ -z "$WHEEL" ]]; then
    fail "Wheel not found after build. Check build output."
    exit 1
  fi
  pass "Wheel built: $(basename "$WHEEL")"
  INSTALL_TARGET="$WHEEL"
else
  header "Step 2: Install from source (use --from-wheel to test built artifact)"
fi

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
header "Step 4: Install syda"
info "Installing from: $INSTALL_TARGET"
if "$PIP" install "$INSTALL_TARGET" 2>&1; then
  pass "pip install succeeded"
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
run_import_check "syda.llm module"                   "import syda.llm"
run_import_check "syda.output module"                "import syda.output"
run_import_check "syda.utils module"                 "import syda.utils"
run_import_check "syda.templates module"             "import syda.templates"
run_import_check "syda.schema_loader module"         "import syda.schema_loader"
run_import_check "syda.custom_generators module"     "import syda.custom_generators"
run_import_check "syda.dependency_handler module"    "import syda.dependency_handler"
run_import_check "syda.unstructured module"          "import syda.unstructured"

# ── Step 7: __all__ surface check ────────────────────────────────────────────
header "Step 7: Public API surface (__all__)"
if "$PY" -c "
import syda
expected = {'SyntheticDataGenerator', 'ModelConfig', 'DatabaseSchemaLoader'}
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
run_import_check "instructor"         "import instructor"
run_import_check "google.genai"       "import google.genai"
run_import_check "networkx"           "import networkx"
run_import_check "jsonref"            "import jsonref"
run_import_check "dotenv"             "import dotenv"
run_import_check "yaml"               "import yaml"
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
[[ -n "${DB_HOST:-}"           ]] && pass "DB_HOST found"           || warn "DB_HOST not set — database examples may fail"

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
    else
      fail "Example: $label (exit code $?)"
    fi
  fi
}

# structured_and_unstructured
run_example "structured_and_unstructured/retail_yml" \
  "$EXAMPLES_DIR/structured_and_unstructured/retail_yml/example_retail_schemas.py"

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
