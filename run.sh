#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env.local" ]]; then
  set -a
  # shellcheck source=/dev/null
  source ".env.local"
  set +a
fi

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "${PYTHON_BIN}"
    return
  fi
  local candidates=("python3.12" "python3.11" "python3")
  local py
  for py in "${candidates[@]}"; do
    if command -v "$py" >/dev/null 2>&1; then
      echo "$py"
      return
    fi
  done
  return 1
}

PYTHON_CMD="$(pick_python || true)"
if [[ -z "$PYTHON_CMD" ]]; then
  echo "[run.sh] 未找到可用 Python，请先安装 Python 3.11+。" >&2
  exit 1
fi

"$PYTHON_CMD" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("[run.sh] 当前 Python 版本过低，需要 >= 3.11")
print(f"[run.sh] 使用 Python {sys.version.split()[0]}")
PY

VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[run.sh] 创建虚拟环境: $VENV_DIR"
  "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[run.sh] 安装依赖..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

echo "[run.sh] 编译检查..."
python -m compileall -q app tests

export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8080}"
export DEFAULT_USER_ID="${DEFAULT_USER_ID:-YOUR_EMPLOYEE_ID}"
export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8191}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "[run.sh] 启动服务: http://${APP_HOST}:${APP_PORT}"
echo "[run.sh] API_BASE_URL=${API_BASE_URL}"
echo "[run.sh] DEFAULT_USER_ID=${DEFAULT_USER_ID}"

exec python -m uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT" --log-level "${LOG_LEVEL,,}"
