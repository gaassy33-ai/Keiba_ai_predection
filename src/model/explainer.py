"""
SHAP を使って予測根拠をテキスト化する。

Usage
-----
>>> from src.model.explainer import PredictionExplainer
>>> explainer = PredictionExplainer(trainer)
>>> reason = explainer.explain_for_line(feature_df, "ダイワスカーレット", top_n=3)
>>> print(reason)   # "①騎手×調教師コンビ勝率(+3.2%) ②上がり3Fランク優秀(+2.1%) ③直近着順好調(+1.8%)"
"""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
from loguru import logger

from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer

# ── 特徴量 → 日本語ラベル（全38特徴量に対応）──────────────────────
FEATURE_LABELS: dict[str, str] = {
    # 基本情報
    "frame_number":              "枠番",
    "horse_number":              "馬番",
    "age":                       "馬齢",
    "weight_carried":            "斤量",
    # レース環境
    "course_type_code":          "コース種別(芝/ダ)",
    "distance":                  "レース距離",
    "ground_condition_code":     "馬場状態",
    "weather_code":              "天候",
    # 厩舎・騎手
    "trainer_win_rate":          "調教師勝率",
    "trainer_place_rate":        "調教師連対率",
    "jockey_win_rate":           "騎手コース勝率",
    "jockey_place_rate":         "騎手コース連対率",
    "jockey_runs":               "騎手騎乗数",
    "jockey_trainer_win_rate":   "騎手×調教師コンビ勝率",
    # 直近成績
    "recent_avg_pos":            "直近平均着順",
    "recent_avg_last3f":         "直近上がり3F平均",
    "recent_top3_rate":          "直近3着内率",
    "recent_avg_last3f_rank":    "直近上がり3Fランク平均",
    "recent_pos_trend":          "着順トレンド",
    "recent_last3f_trend":       "上がり3Fトレンド",
    # 前走パフォーマンス
    "prev_margin":               "前走着差",
    "prev_last3f_rank_norm":     "前走上がり3Fランク",
    "prev_corner_pos_norm":      "前走最終コーナー位置",
    # クラス・会場
    "race_class_code":           "レースクラス",
    "class_change":              "クラス変動",
    "venue_code":                "開催場",
    "venue_ground_code":         "会場×馬場コード",
    "n_entries":                 "出走頭数",
    # 市場情報
    "odds_log":                  "市場オッズ(対数)",
    "popularity_rank_norm":      "人気順位(正規化)",
    # フィールド・馬個体
    "field_avg_jockey_win_rate": "フィールド騎手勝率平均",
    "horse_career_top3_rate":    "キャリア複勝率",
    "horse_dist_win_rate":       "距離帯適性(馬別)",
    "horse_ground_win_rate":     "馬場適性(馬別)",
    # 馬体重・状態
    "prev_horse_weight":         "前走馬体重",
    "horse_weight_change":       "馬体重変化",
    "is_3yo":                    "3歳馬フラグ",
    "days_since_last_race":      "前走からの経過日数",
    # ── レガシー（旧バージョン互換）──
    "market_hhi":                "市場集中度(HHI)",
    "sire_win_rate":             "父産駒勝率",
    "bms_win_rate":              "母父産駒勝率",
    "jockey_venue_win_rate":     "騎手×会場勝率",
    "distance_bin_code":         "距離帯",
    "recent_speed_rating":       "スピード指数",
}

# SHAP値が大きいほど「良い」方向の特徴量（記号: ▲）
# SHAP値が小さいほど「良い」方向は特別対応が必要なもの
# → 基本的に「正SHAP = プラス要因」のため、正のSHAPを優先表示する

CIRCLED = ["①", "②", "③", "④", "⑤"]


class PredictionExplainer:
    """
    SHAP を使って本命馬の予測根拠をテキスト化する。

    Usage
    -----
    >>> explainer = PredictionExplainer(trainer)
    >>> # LINE通知用（短い文字列を返す）
    >>> line_text = explainer.explain_for_line(feature_df, "ダイワスカーレット", top_n=3)
    >>> # 詳細テキスト
    >>> detail = explainer.explain_text_detail(feature_df, honmei_idx=0, top_n=5)
    """

    def __init__(self, trainer: ModelTrainer) -> None:
        self._trainer = trainer
        self._shap_explainer = None

    def _get_shap_explainer(self):
        """遅延初期化。"""
        if self._shap_explainer is None:
            import shap
            if self._trainer.model is None:
                raise ValueError("ModelTrainer.model が None です。学習後に呼び出してください。")
            self._shap_explainer = shap.TreeExplainer(self._trainer.model)
        return self._shap_explainer

    # ------------------------------------------------------------------
    # SHAP 値の計算
    # ------------------------------------------------------------------

    def compute_shap_values(self, feature_df: pd.DataFrame) -> np.ndarray:
        """全馬の SHAP 値を計算する（shape: [n_horses, n_features]）。"""
        try:
            explainer = self._get_shap_explainer()
            cols = FeatureEngineer.FEATURE_COLUMNS
            X = feature_df[[c for c in cols if c in feature_df.columns]].copy()
            # 欠損カラムを0埋め
            for c in cols:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            shap_values = explainer.shap_values(X)
            # binary classification: LightGBM では単一ndarray（正クラス）
            if isinstance(shap_values, list):
                return shap_values[1]
            return shap_values
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return np.zeros((len(feature_df), len(FeatureEngineer.FEATURE_COLUMNS)))

    # ------------------------------------------------------------------
    # LINE 通知用: 短いテキスト
    # ------------------------------------------------------------------

    def explain_for_line(
        self,
        feature_df: pd.DataFrame,
        honmei_horse_name: str,
        top_n: int = 3,
    ) -> str:
        """
        LINE 通知向けの短い予測根拠テキストを返す。

        Parameters
        ----------
        feature_df : pd.DataFrame
            build_entry_features() の出力（全馬分）
        honmei_horse_name : str
            本命馬の名前（feature_df["horse_name"] と照合）
        top_n : int
            表示する要因数（デフォルト3）

        Returns
        -------
        str
            例: "①騎手×調教師コンビ勝率(+3.2%) ②上がり3Fランク優秀(+2.1%) ③直近着順好調(+1.8%)"
            SHAP 計算失敗時は空文字を返す
        """
        try:
            shap_values = self.compute_shap_values(feature_df)
            cols = FeatureEngineer.FEATURE_COLUMNS

            # 本命馬のインデックスを特定
            if "horse_name" in feature_df.columns:
                mask = feature_df["horse_name"].astype(str) == str(honmei_horse_name)
            else:
                mask = pd.Series([False] * len(feature_df), index=feature_df.index)

            if not mask.any():
                logger.debug(f"SHAP: {honmei_horse_name} が feature_df に見つかりません")
                return ""

            row_idx = int(feature_df.index[mask][0])
            # インデックスを行番号に変換
            pos = list(feature_df.index).index(row_idx)
            horse_shap = shap_values[pos]

            # 正のSHAP値を降順でソート（プラス要因のみを理由として表示）
            positive_factors = sorted(
                [(cols[i], horse_shap[i]) for i in range(len(cols)) if horse_shap[i] > 0.001],
                key=lambda x: -x[1],
            )

            parts = []
            for i, (feat, val) in enumerate(positive_factors[:top_n]):
                label = FEATURE_LABELS.get(feat, feat)
                pct = val * 100
                parts.append(f"{CIRCLED[i]}{label}(+{pct:.1f}%)")

            return "  ".join(parts) if parts else ""

        except Exception as e:
            logger.debug(f"SHAP explain_for_line 失敗: {e}")
            return ""

    # ------------------------------------------------------------------
    # 詳細テキスト（デバッグ・レポート用）
    # ------------------------------------------------------------------

    def explain_text_detail(
        self,
        feature_df: pd.DataFrame,
        honmei_horse_name: str | None = None,
        honmei_idx: int = 0,
        top_n: int = 5,
    ) -> str:
        """
        本命馬の予測根拠を詳細テキストで返す（ポジティブ・ネガティブ両方）。

        Parameters
        ----------
        feature_df : pd.DataFrame
        honmei_horse_name : str | None
            馬名で検索する場合に使用。None なら honmei_idx を使う。
        honmei_idx : int
            馬名が見つからない場合の行番号（デフォルト0=最上位馬）
        top_n : int
            表示する特徴量の上位件数
        """
        shap_values = self.compute_shap_values(feature_df)
        cols = FeatureEngineer.FEATURE_COLUMNS

        if honmei_horse_name and "horse_name" in feature_df.columns:
            mask = feature_df["horse_name"].astype(str) == str(honmei_horse_name)
            if mask.any():
                pos = list(feature_df.index).index(feature_df.index[mask][0])
            else:
                pos = honmei_idx
        else:
            pos = honmei_idx

        horse_shap = shap_values[pos]
        horse_name = honmei_horse_name or f"馬{pos+1}"

        shap_df = pd.DataFrame({
            "feature": cols,
            "label": [FEATURE_LABELS.get(c, c) for c in cols],
            "shap_value": horse_shap,
        }).sort_values("shap_value", key=abs, ascending=False)

        buf = StringIO()
        buf.write(f"【{horse_name} の予測根拠 Top{top_n}】\n")
        for _, row in shap_df.head(top_n).iterrows():
            direction = "▲" if row["shap_value"] > 0 else "▽"
            buf.write(f"  {direction} {row['label']:20s}  (SHAP={row['shap_value']:+.4f})\n")

        return buf.getvalue().strip()

    # ------------------------------------------------------------------
    # 可視化（ファイル保存）
    # ------------------------------------------------------------------

    def save_summary_plot(self, feature_df: pd.DataFrame, output_path: str) -> None:
        """SHAP summary plot を PNG で保存する（オプション）。"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import shap

            cols = FeatureEngineer.FEATURE_COLUMNS
            X = feature_df[[c for c in cols if c in feature_df.columns]].copy()
            for c in cols:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[cols].apply(pd.to_numeric, errors="coerce").fillna(0)

            explainer_obj = self._get_shap_explainer()
            shap_values = explainer_obj.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap.summary_plot(
                shap_values,
                X,
                feature_names=[FEATURE_LABELS.get(c, c) for c in cols],
                show=False,
            )
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info(f"SHAP summary plot saved to {output_path}")
        except Exception as e:
            logger.warning(f"Could not save SHAP plot: {e}")
