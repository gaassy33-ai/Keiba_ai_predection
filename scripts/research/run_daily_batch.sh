#!/bin/bash
# 競馬予想 毎朝バッチ（土日自動実行用）
cd /Users/higashi/keiba-prediction
/Users/higashi/keiba-prediction/.venv/bin/python daily_batch.py >> /Users/higashi/keiba-prediction/logs/daily_batch_cron.log 2>&1
