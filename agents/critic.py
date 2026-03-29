"""
agents/critic.py - サブ④「評論家」
=====================================================
役割:
  - 当日の予想データをClaude APIで客観評価
  - 予想精度だけでなく、特徴量・季節・天候・馬場等の文脈も加味
  - 改善点・着目すべき観点をフィードバック
  - 出力: data/agents/YYYYMMDD_feedback.json + LINE通知

実行:
    python -m agents.critic [--date YYYY-MM-DD] [--dry-run]

必要環境変数:
    ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.base import AgentBase

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic パッケージが未インストールです。pip install anthropic で導入してください。")


CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048

# 季節判定
def _season(d: date) -> str:
    m = d.month
    if m in (3, 4, 5):
        return "春"
    if m in (6, 7, 8):
        return "夏"
    if m in (9, 10, 11):
        return "秋"
    return "冬"


class Critic(AgentBase):
    """サブ④: 評論家 — Claude API による客観評価・フィードバック"""

    name = "Critic"

    def run(self, target_date: date, dry_run: bool = False) -> dict:
        logger.info(f"Critic 起動: {target_date}")

        if not _ANTHROPIC_AVAILABLE:
            logger.error("anthropic パッケージが未インストールのためスキップします")
            feedback = {
                "date": str(target_date),
                "status": "error",
                "error": "anthropic package not installed",
            }
            self.save(feedback, self.feedback_file(target_date))
            return feedback

        # コンテキスト・レビューデータ読み込み
        context = self.load(self.context_file(target_date))
        review  = self.load(self.review_file(target_date))

        if not context and not review:
            logger.warning("コンテキスト・レビューデータが見つかりません")
            feedback = {
                "date": str(target_date),
                "status": "no_data",
            }
            self.save(feedback, self.feedback_file(target_date))
            return feedback

        # Claude API でフィードバック生成
        prompt = self._build_prompt(target_date, context, review)
        logger.info("Claude API にフィードバック依頼中...")

        try:
            client = anthropic.Anthropic()
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            claude_feedback = message.content[0].text
            logger.info(f"Claude API 応答受信 ({len(claude_feedback)} chars)")
        except Exception as e:
            logger.error(f"Claude API 呼び出し失敗: {e}")
            feedback = {
                "date": str(target_date),
                "status": "api_error",
                "error": str(e),
            }
            self.save(feedback, self.feedback_file(target_date))
            return feedback

        feedback = {
            "generated_at": datetime.now().isoformat(),
            "date": str(target_date),
            "season": _season(target_date),
            "claude_model": CLAUDE_MODEL,
            "feedback_text": claude_feedback,
            "status": "ok",
        }

        self.save(feedback, self.feedback_file(target_date))

        # LINE 通知
        msg = self._build_line_message(claude_feedback, target_date)
        self.send_line(msg, dry_run=dry_run)

        logger.info("評論家フィードバック完了")
        return feedback

    # ------------------------------------------------------------------ #
    # プロンプト構築
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        target_date: date,
        context: dict | None,
        review: dict | None,
    ) -> str:
        season = _season(target_date)
        lines = [
            f"あなたは競馬予想の評論家です。以下のデータをもとに、予想AIへの建設的なフィードバックをお願いします。",
            f"対象日: {target_date}（{season}）",
            "",
        ]

        # 当日の成績（振り返りデータ）
        if review:
            analysis = review.get("analysis", {})
            n = analysis.get("buy_races", 0)
            hit = analysis.get("tansho_hit", 0)
            roi = analysis.get("tansho_roi", 0)
            lines += [
                "## 当日の成績",
                f"- 買い対象: {n}レース",
                f"- 単勝的中: {hit}レース（的中率: {hit/n:.1%}）" if n else "- データなし",
                f"- 単勝ROI: {roi:.1%}" if n else "",
                f"- 惜しいレース(2〜3着): {analysis.get('near_miss_races', 0)}R",
                f"- 大敗(6着以下): {analysis.get('upset_races', 0)}R",
            ]
            # ルールベースのヒント
            hints = analysis.get("hints", [])
            if hints:
                lines.append("")
                lines.append("### ルールベース改善ヒント")
                for h in hints:
                    lines.append(f"- {h}")

            # 各レースの詳細
            predictions = review.get("predictions", [])
            ok_preds = [p for p in predictions if p.get("status") == "ok"]
            if ok_preds:
                lines.append("")
                lines.append("### 当日の各レース結果")
                for p in ok_preds:
                    hit_str = "◎的中" if p.get("tansho_hit") else f"{p.get('actual_pos', '?')}着"
                    lines.append(
                        f"- {p.get('race_name', p.get('race_id', '?'))}: "
                        f"{p.get('honmei_name', '?')}（確率{float(p.get('honmei_prob', 0)):.1%}）→ {hit_str}"
                    )

        lines.append("")

        # 直近の精度データ（データマスターから）
        if context:
            acc = context.get("accuracy_summary", {})
            if acc.get("buy_races", 0) > 0:
                lines += [
                    "## 直近8週間の精度",
                    f"- 単勝的中率: {acc.get('tansho_hit_rate', 0):.1%}",
                    f"- 単勝平均回収: {acc.get('tansho_roi_avg', '-')}円",
                ]

            # 特徴量重要度
            feat = context.get("feature_importance", {})
            top5 = feat.get("top5_win", [])
            if top5:
                lines.append("")
                lines.append("## 現在のモデル特徴量重要度 Top5")
                for f in top5:
                    lines.append(f"- {f['feature']}: {f['gain_pct']:.1f}%")

            # 会場別精度
            venue_acc = context.get("venue_accuracy", {})
            if venue_acc:
                lines.append("")
                lines.append("## 会場別精度（直近8週）")
                for v, d in sorted(venue_acc.items(), key=lambda x: -x[1]["hit_rate"])[:5]:
                    lines.append(f"- {v}: 的中率{d['hit_rate']:.1%}（{d['races']}R）")

        lines += [
            "",
            "---",
            "## フィードバック依頼",
            f"今日（{target_date}・{season}）の競馬予想を以下の観点で400文字程度で評価してください:",
            "1. **当日の成績分析**: 的中・外れの主な要因（オッズ帯・確率・馬場状態等）",
            f"2. **季節・天候の考慮**: {season}競馬特有の傾向や、今後の季節変化への対応",
            "3. **モデルの特徴量**: 現在の特徴量で不足していると思われる観点（馬場適性、距離適性、騎手×コース相性など）",
            "4. **具体的な改善提案**: 予想精度を上げるための短期・中期の改善アクション",
            "",
            "箇条書きで簡潔にまとめてください。",
        ]

        return "\n".join(line for line in lines if line is not None)

    # ------------------------------------------------------------------ #
    # LINE メッセージ生成
    # ------------------------------------------------------------------ #

    def _build_line_message(self, feedback_text: str, target_date: date) -> str:
        season = _season(target_date)
        header = f"🎙️ 評論家フィードバック ({target_date}・{season})\n\n"
        # LINE メッセージは 5000 文字以内
        body = feedback_text[:4800] if len(feedback_text) > 4800 else feedback_text
        return header + body


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="評論家: Claude API による客観評価")
    parser.add_argument("--date", default=str(date.today()), help="対象日 (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="LINE 送信しない")
    args = parser.parse_args()

    agent = Critic()
    agent.run(date.fromisoformat(args.date), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
