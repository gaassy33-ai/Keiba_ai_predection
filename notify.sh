#!/bin/bash
# ============================================================
# notify.sh  -  発走30分前オッズ通知スクリプト ランチャー
#
# 使い方:
#   ./notify.sh            # 当日・常駐ループ（LINE通知あり）
#   ./notify.sh --dry-run  # LINE送信なし・ターミナル確認
#   ./notify.sh --bg       # バックグラウンドで起動
# ============================================================

set -e
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
SCRIPT="odds_notify.py"
LOGFILE="logs/odds_notify.log"

# .venv がなければエラー
if [ ! -f "$PYTHON" ]; then
  echo "❌ .venv が見つかりません。先に: python -m venv .venv && .venv/bin/pip install -e ."
  exit 1
fi

mkdir -p logs

# 引数解析
BG=false
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --bg) BG=true ;;
    *)    ARGS+=("$arg") ;;
  esac
done

if $BG; then
  echo "🚀 バックグラウンドで起動します（ログ: $LOGFILE）"
  nohup $PYTHON $SCRIPT "${ARGS[@]}" >> "$LOGFILE" 2>&1 &
  echo "   PID: $!"
  echo "   ログ確認: tail -f $LOGFILE"
  echo "   停止方法: kill $!"
else
  echo "🚀 odds_notify を起動します（Ctrl+C で終了）"
  $PYTHON $SCRIPT "${ARGS[@]}"
fi
