"""
SHAP を使って予測根拠をテキスト化する。
"""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
from loguru import logger

from src.features.engineer import FeatureEngineer
from src.model.predictor import PredictionResult
from src.model.trainer import ModelTrainer

# 特徴量の日本語ラベル
FEATURE_LABELS = {
    "frame_number": "枠番",
    "horse_number": "馬番",
    "age": "馬齢",
    "weight_carried": "斤量",
    "course_type_code": "コース種別",
    "distance": "距離",
    "distance_bin_code": "距離帯",
    "ground_condition_code": "馬場状態",
    "weather_code": "天候",
    "sire_win_rate": "父産駒勝率",
    "bms_win_rate": "母父産駒勝率",
    "jockey_win_rate": "騎手コース勝率",
    "jockey_place_rate": "騎手コース連対率",
    "jockey_runs": "騎手コース騎乗数",
    "recent_avg_pos": "直近平均着順",
    "recent_avg_last3f": "直近平均上がり3F",
}


class PredictionExplainer:
    """
    SHAP を使って本命馬の予測根拠をテキスト化する。

    Usage
    -----
    >>> explainer = PredictionExplainer(trainer)
    >>> text = explainer.explain_text(result, feature_df)
    """

    def __init__(self, trainer: ModelTrainer) -> None:
        self._trainer = trainer
        self._shap_explainer = None

    def _get_shap_explainer(self):
        """遅延初期化。"""
        if self._shap_explainer is None:
            import shap
            self._shap_explainer = shap.TreeExplainer(self._trainer.model)
        return self._shap_explainer

    # ------------------------------------------------------------------
    # SHAP 値の計算
    # ------------------------------------------------------------------

    def compute_shap_values(self, feature_df: pd.DataFrame) -> np.ndarray:
        """全馬の SHAP 値を計算する。"""
        try:
            explainer = self._get_shap_explainer()
            X = feature_df[FeatureEngineer.FEATURE_COLUMNS].copy()
            shap_values = explainer.shap_values(X)
            # binary classification: shap_values は [負クラス, 正クラス] のリスト
            if isinstance(shap_values, list):
                return shap_values[1]  # 正クラス（1着）の SHAP 値
            return shap_values
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return np.zeros((len(feature_df), len(FeatureEngineer.FEATURE_COLUMNS)))

    # ------------------------------------------------------------------
    # テキスト生成
    # ------------------------------------------------------------------

    def explain_text(
        self,
        result: PredictionResult,
        feature_df: pd.DataFrame,
        top_n: int = 5,
    ) -> str:
        """
        本命馬の予測根拠をテキストで返す。

        Parameters
        ----------
        result : PredictionResult
        feature_df : pd.DataFrame
        top_n : int
            表示する特徴量の上位件数

        Returns
        -------
        str
            人間が読める説明テキスト
        """
        if not result.honmei:
            return "予測根拠を生成できませんでした。"

        honmei_name = result.honmei.get("horse_name", "不明")
        shap_values = self.compute_shap_values(feature_df)

        # 本命馬のインデックスを特定
        honmei_mask = feature_df["horse_name"] == honmei_name
        if not honmei_mask.any():
            return f"{honmei_name} の SHAP インデックスが見つかりませんでした。"

        idx = feature_df.index[honmei_mask][0]
        horse_shap = shap_values[idx]
        cols = FeatureEngineer.FEATURE_COLUMNS

        shap_df = pd.DataFrame({
            "feature": cols,
            "label": [FEATURE_LABELS.get(c, c) for c in cols],
            "shap_value": horse_shap,
        }).sort_values("shap_value", key=abs, ascending=False)

        buf = StringIO()
        buf.write(f"【{honmei_name} の予測根拠 Top{top_n}】\n")
        for _, row in shap_df.head(top_n).iterrows():
            direction = "▲ プラス" if row["shap_value"] > 0 else "▽ マイナス"
            buf.write(f"  {direction} | {row['label']} (SHAP={row['shap_value']:+.3f})\n")

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

            explainer = self._get_shap_explainer()
            X = feature_df[FeatureEngineer.FEATURE_COLUMNS].copy()
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap.summary_plot(
                shap_values,
                X,
                feature_names=[FEATURE_LABELS.get(c, c) for c in FeatureEngineer.FEATURE_COLUMNS],
                show=False,
            )
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info(f"SHAP summary plot saved to {output_path}")
        except Exception as e:
            logger.warning(f"Could not save SHAP plot: {e}")
