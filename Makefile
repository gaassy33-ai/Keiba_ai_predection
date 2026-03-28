# ============================================================
# Makefile  -  競馬予想AI ショートカット集
#
# 使い方:
#   make notify        # 発走30分前通知（常駐）
#   make notify-dry    # 通知テスト（LINE送信なし）
#   make notify-bg     # バックグラウンド起動
#   make batch         # 当日バッチ予想（朝一実行）
#   make batch-dry     # バッチ予想テスト
#   make backtest      # 直近1年バックテスト
#   make train         # JRAモデル再学習
#   make log           # 通知ログをリアルタイム表示
#   make log-batch     # バッチログをリアルタイム表示
#   make stop          # バックグラウンドの notify を停止
# ============================================================

PYTHON := .venv/bin/python
SHELL  := /bin/bash

.PHONY: notify notify-dry notify-bg batch batch-dry backtest train log log-batch stop help

# ── 発走30分前オッズ通知 ──────────────────────────────────────
notify:
	@echo "🚀 発走30分前通知を起動します（Ctrl+C で終了）"
	@$(PYTHON) odds_notify.py

notify-dry:
	@echo "🔍 通知テスト（LINE送信なし）"
	@$(PYTHON) odds_notify.py --dry-run

notify-bg:
	@mkdir -p logs
	@echo "🚀 バックグラウンドで起動します（ログ: logs/odds_notify.log）"
	@nohup $(PYTHON) odds_notify.py >> logs/odds_notify.log 2>&1 & \
		echo "   PID: $$!" ; \
		echo $$! > logs/odds_notify.pid ; \
		echo "   停止: make stop"

# ── 当日バッチ予想 ────────────────────────────────────────────
batch:
	@echo "📊 当日バッチ予想を実行します"
	@$(PYTHON) daily_batch.py

batch-dry:
	@echo "📊 バッチ予想テスト（LINE送信なし）"
	@$(PYTHON) daily_batch.py --dry-run

# ── バックテスト ──────────────────────────────────────────────
backtest:
	@echo "📈 直近1年バックテストを実行します"
	@$(PYTHON) backtest_1year.py

# ── モデル再学習 ──────────────────────────────────────────────
train:
	@echo "🧠 JRAモデルを再学習します（時間がかかります）"
	@$(PYTHON) expand_and_train.py

# ── ログ確認 ──────────────────────────────────────────────────
log:
	@tail -f logs/odds_notify.log

log-batch:
	@tail -f logs/daily_batch.log

# ── バックグラウンド停止 ──────────────────────────────────────
stop:
	@if [ -f logs/odds_notify.pid ]; then \
		PID=$$(cat logs/odds_notify.pid) ; \
		kill $$PID 2>/dev/null && echo "✅ PID $$PID を停止しました" || echo "⚠️  既に停止済みです" ; \
		rm -f logs/odds_notify.pid ; \
	else \
		echo "⚠️  PIDファイルが見つかりません（手動で kill してください）" ; \
	fi

# ── ヘルプ ────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  競馬予想AI ショートカット"
	@echo "  ─────────────────────────────────────────"
	@echo "  make notify       発走30分前通知（常駐）"
	@echo "  make notify-dry   通知テスト（LINE送信なし）"
	@echo "  make notify-bg    バックグラウンド起動"
	@echo "  make batch        当日バッチ予想"
	@echo "  make batch-dry    バッチ予想テスト"
	@echo "  make backtest     直近1年バックテスト"
	@echo "  make train        JRAモデル再学習"
	@echo "  make log          通知ログ（リアルタイム）"
	@echo "  make log-batch    バッチログ（リアルタイム）"
	@echo "  make stop         バックグラウンド通知を停止"
	@echo ""
