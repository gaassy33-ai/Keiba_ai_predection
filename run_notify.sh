#!/bin/bash
# run_notify.sh
# odds_notify.py のクラッシュ自動再起動ラッパー
# 使い方: bash run_notify.sh [--dry-run] [--date YYYY-MM-DD]

set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
SCRIPT="odds_notify.py"
LOGFILE="logs/odds_notify.log"
PIDFILE="logs/odds_notify.pid"
MAX_RESTARTS=5
RESTART_DELAY=5   # クラッシュ後の待機秒数

mkdir -p logs

# 既存の run_notify / odds_notify プロセスをすべて停止（自分自身は除く）
MY_PID=$$
echo "[run_notify] 既存プロセスをクリーンアップ中 (my PID=$MY_PID)..."

# PID ファイル経由で停止
if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -n "$OLD_PID" ]] && [[ "$OLD_PID" != "$MY_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[run_notify] PIDファイル経由で停止: PID=$OLD_PID"
        kill "$OLD_PID" 2>/dev/null || true
    fi
fi

# 残存する odds_notify.py プロセスをすべて停止
for pid in $(pgrep -f "odds_notify.py" 2>/dev/null || true); do
    if [[ "$pid" != "$MY_PID" ]]; then
        echo "[run_notify] 残存 odds_notify PID=$pid を停止"
        kill "$pid" 2>/dev/null || true
    fi
done

# 孤立した chromedriver を掃除
echo "[run_notify] 孤立 ChromeDriver をクリーンアップ..."
pkill -f "chromedriver" 2>/dev/null || true
sleep 2

echo $MY_PID > "$PIDFILE"

restarts=0
while true; do
    echo "[run_notify] 起動 (試行 $((restarts+1))/$MAX_RESTARTS): $(date '+%H:%M:%S')" | tee -a "$LOGFILE"

    # プロセス起動（クラッシュしても継続するため set +e）
    set +e
    "$PYTHON" "$SCRIPT" "$@"
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[run_notify] 正常終了 (exit=0)" | tee -a "$LOGFILE"
        break
    fi

    restarts=$((restarts + 1))
    if [[ $restarts -ge $MAX_RESTARTS ]]; then
        echo "[run_notify] 最大再起動回数 ($MAX_RESTARTS) に達しました。終了します。" | tee -a "$LOGFILE"
        break
    fi

    echo "[run_notify] クラッシュ検出 (exit=$EXIT_CODE)。${RESTART_DELAY}秒後に再起動..." | tee -a "$LOGFILE"

    # 孤立 chromedriver を再クリーンアップ
    pkill -f "chromedriver" 2>/dev/null || true
    sleep "$RESTART_DELAY"
done

rm -f "$PIDFILE"
