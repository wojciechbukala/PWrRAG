#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
EMBEDDING_MODEL="embeddinggemma"
OPTIONAL_MODELS=(
    "gemma3:4b"
    "gemma3:1b"
    "qwen2.5:3b"
    "qwen2.5:1.5b"
    "llama3.2:1b"
)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
fail() { echo -e "${RED}[ERR]${NC} $1"; exit 1; }

echo "=== PWrRAG Startup ==="

# --- Środowisko wirtualne ---
if [ ! -d "$VENV_DIR" ]; then
    warn "Brak venv, tworzę..."
    python3 -m venv "$VENV_DIR"
    ok "Środowisko utworzone"
fi

source "$VENV_DIR/bin/activate"
ok "Środowisko aktywowane"

# --- Pakiety ---
INSTALLED=$("$VENV_DIR/bin/pip" freeze 2>/dev/null)
MISSING=false

while IFS= read -r package || [ -n "$package" ]; do
    [[ -z "$package" || "$package" == \#* ]] && continue
    pkg_name=$(echo "$package" | sed 's/[>=<!].*//' | tr '[:upper:]' '[:lower:]')
    if ! echo "$INSTALLED" | grep -qi "^$pkg_name"; then
        MISSING=true
        break
    fi
done < "$REQUIREMENTS"

if [ "$MISSING" = true ]; then
    warn "Brakujące pakiety, instaluję..."
    pip install -r "$REQUIREMENTS" -q
    ok "Pakiety zainstalowane"
else
    ok "Pakiety aktualne"
fi

# --- Ollama ---
if ! command -v ollama &> /dev/null; then
    fail "Ollama nie jest zainstalowana. Pobierz ze: https://ollama.com"
fi

if ! ollama list &> /dev/null; then
    warn "Ollama nie działa, uruchamiam..."
    ollama serve &> /dev/null &
    sleep 3
    if ! ollama list &> /dev/null; then
        fail "Nie udało się uruchomić Ollamy"
    fi
    ok "Ollama uruchomiona"
else
    ok "Ollama działa"
fi

# --- Modele ---
PULLED_MODELS=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}')

check_model() {
    local model="$1"
    local tag="${model##*:}"
    local name="${model%%:*}"
    if echo "$PULLED_MODELS" | grep -q "^${name}:${tag}"; then
        ok "Model $model dostępny"
    else
        warn "Brak modelu $model, pobieram..."
        ollama pull "$model"
        ok "Model $model pobrany"
    fi
}

check_model "$EMBEDDING_MODEL"
for model in "${OPTIONAL_MODELS[@]}"; do
    check_model "$model"
done

# --- UI ---
echo ""
ok "Wszystko gotowe, uruchamiam UI..."
echo ""
cd "$SCRIPT_DIR"
streamlit run ui/app.py --server.fileWatcherType none 2>/dev/null
