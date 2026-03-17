#!/bin/bash
# NAR バックテスト実行スクリプト
# 収集データをコピーしてバックテストを lgb311 環境で実行

set -e
cd "$(dirname "$0")"

CHECKPOINT_R="data/raw/nar_checkpoint_results.csv"
CHECKPOINT_M="data/raw/nar_checkpoint_meta.csv"
FINAL_R="data/raw/nar_results.csv"
FINAL_M="data/raw/nar_meta.csv"

# チェックポイントが存在すれば最新データをコピー
if [ -f "$CHECKPOINT_R" ]; then
  N=$(tail -n +2 "$CHECKPOINT_R" | cut -d',' -f1 | sort -u | wc -l | tr -d ' ')
  echo "チェックポイントデータ: ${N} レース"
  cp "$CHECKPOINT_R" "$FINAL_R"
  cp "$CHECKPOINT_M" "$FINAL_M"
elif [ ! -f "$FINAL_R" ]; then
  echo "ERROR: NAR データが見つかりません。先に collect_nar_history.py を実行してください。"
  exit 1
fi

# バックテスト実行
echo "バックテスト実行中..."
conda run -n lgb311 python backtest_nar.py "$@"
