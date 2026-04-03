#!/usr/bin/env bash
# ENGRAM Protocol — One-command setup
#
# Usage:
#   ./scripts/setup.sh            # Full setup with sbert embedder
#   ./scripts/setup.sh --minimal  # Core only (no sbert, no MCP)
#   ./scripts/setup.sh --dev      # Full setup + dev tools
#
# Requirements:
#   - Python >= 3.11
#   - pip (comes with Python)
#   - git (for cloning)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors (if terminal supports them)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[ENGRAM]${NC} $*"; }
warn()  { echo -e "${YELLOW}[ENGRAM]${NC} $*"; }
error() { echo -e "${RED}[ENGRAM]${NC} $*" >&2; }

# Parse arguments
MINIMAL=false
DEV=false
for arg in "$@"; do
    case "$arg" in
        --minimal) MINIMAL=true ;;
        --dev)     DEV=true ;;
        --help|-h)
            echo "Usage: ./scripts/setup.sh [--minimal] [--dev]"
            echo ""
            echo "  --minimal  Core dependencies only (no sbert, no MCP)"
            echo "  --dev      Include development tools (pytest, ruff, mypy)"
            exit 0
            ;;
        *) error "Unknown argument: $arg"; exit 1 ;;
    esac
done

cd "$PROJECT_DIR"

# ── 1. Check Python version ──────────────────────────────────────────
info "Checking Python version..."
PYTHON=""
for cmd in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$("$cmd" -c "import sys; print(sys.version_info.major)")
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)")
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python >= 3.11 required but not found."
    error "Install from https://python.org or via your package manager."
    exit 1
fi
info "Using $PYTHON ($(${PYTHON} --version 2>&1))"

# ── 2. Create virtual environment ────────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv .venv
else
    info "Virtual environment already exists."
fi

# Activate
source .venv/bin/activate
info "Activated .venv"

# ── 3. Upgrade pip ───────────────────────────────────────────────────
info "Upgrading pip..."
pip install --upgrade pip --quiet

# ── 4. Install core package ──────────────────────────────────────────
info "Installing ENGRAM core dependencies..."
pip install -e . --quiet

if [ "$MINIMAL" = false ]; then
    # ── 5. Install sbert embedder ────────────────────────────────────
    info "Installing sentence-transformers embedder..."
    pip install -e ".[sbert]" --quiet

    # ── 6. Install MCP server ────────────────────────────────────────
    info "Installing MCP server dependencies..."
    pip install -e ".[mcp]" --quiet 2>/dev/null || \
        warn "MCP package not available (optional — needed for Claude Code integration)"
fi

if [ "$DEV" = true ]; then
    # ── 7. Install dev tools ─────────────────────────────────────────
    info "Installing development tools..."
    pip install -e ".[dev]" --quiet
fi

# ── 8. Create config from template ───────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.template .env
    info "Created .env from template. Edit it to set ENGRAM_MODEL_PATH."
else
    info ".env already exists."
fi

# ── 9. Create ENGRAM directories ─────────────────────────────────────
mkdir -p ~/.engram/sessions
mkdir -p ~/.engram/knowledge
mkdir -p ~/.engram/index
info "Created ~/.engram/ directories."

# ── 10. Verify installation ──────────────────────────────────────────
info "Verifying installation..."
if python -c "import kvcos; print(f'  kvcos OK (v{kvcos.core.types.ENGRAM_VERSION})')"; then
    info "Core library loaded successfully."
else
    error "Failed to import kvcos. Check error messages above."
    exit 1
fi

# ── 11. Run tests (if dev mode) ──────────────────────────────────────
if [ "$DEV" = true ]; then
    info "Running test suite..."
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=. \
        pytest tests/ -x -q --tb=short 2>&1 | tail -5
fi

# ── Done ─────────────────────────────────────────────────────────────
echo ""
info "Setup complete."
echo ""
echo "  Activate:  source .venv/bin/activate"
echo "  Tests:     KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. pytest tests/ -x -q"
echo "  Server:    engram-server"
echo "  Config:    Edit .env to set ENGRAM_MODEL_PATH"
echo ""
