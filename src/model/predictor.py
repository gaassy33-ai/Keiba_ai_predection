"""
推論モジュール。

学習済みモデルを使って出走馬の1着入線確率を算出し、
本命・対抗・穴馬を決定する。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from config.settings import settings
from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer


@dataclass
class PredictionResult:
    """1レース分の予測結果"""
    race_id: str
    race_name: str
    honmei: dict      # 本命（1位）
    taikou: dict      # 対抗（2位）
    ana: dict         # 穴馬（確率は低いが馬番が小さい等の条件）
    all_predictions: pd.DataFrame  # 全馬の予測確率


class RacePredictor:
    """
    Usage
    -----
    >>> predictor = RacePredictor.from_saved_model()
    >>> result = predictor.predict(race_info, feature_df)
    """

    def __init__(self, trainer: ModelTrainer) -> None:
        self._trainer = trainer

    @classmethod
    def from_saved_model(cls) -> "RacePredictor":
        return cls(ModelTrainer.load())

    # ------------------------------------------------------------------
    # 予測
    # ------------------------------------------------------------------

    def predict(
        self,
        race_id: str,
        race_name: str,
        feature_df: pd.DataFrame,
    ) -> PredictionResult:
        """
        Parameters
        ----------
        race_id : str
        race_name : str
        feature_df : pd.DataFrame
            FeatureEngineer.build_entry_features() の出力

        Returns
        -------
        PredictionResult
        """
        if self._trainer.model is None:
            raise RuntimeError("Model is not loaded. Call from_saved_model() first.")

        X = feature_df[FeatureEngineer.FEATURE_COLUMNS].copy()
        probs = self._trainer.model.predict(X)

        base_cols = ["horse_id", "horse_name", "horse_number", "frame_number"]
        extra_cols = [
            "jockey_name", "jockey_win_rate", "jockey_place_rate",
            "recent_avg_pos", "recent_avg_last3f",
        ]
        cols = base_cols + [c for c in extra_cols if c in feature_df.columns]
        result_df = feature_df[cols].copy()
        result_df["win_prob"] = probs
        result_df = result_df.sort_values("win_prob", ascending=False).reset_index(drop=True)

        honmei = result_df.iloc[0].to_dict()
        taikou = result_df.iloc[1].to_dict() if len(result_df) > 1 else {}

        # 穴馬: 上位2頭以外で、馬番が若い or フレームが外枠
        ana_candidates = result_df.iloc[2:].copy()
        if not ana_candidates.empty:
            # 確率が一定以上（上位確率の50%以上）かつ人気薄になりやすい外枠を優先
            threshold = honmei["win_prob"] * 0.4
            filtered = ana_candidates[ana_candidates["win_prob"] >= threshold]
            if filtered.empty:
                filtered = ana_candidates
            ana = filtered.sort_values("frame_number", ascending=False).iloc[0].to_dict()
        else:
            ana = {}

        logger.info(
            f"Prediction done: {race_name} | "
            f"本命={honmei.get('horse_name')} ({honmei.get('win_prob', 0):.1%}) | "
            f"対抗={taikou.get('horse_name')} | 穴={ana.get('horse_name')}"
        )

        return PredictionResult(
            race_id=race_id,
            race_name=race_name,
            honmei=honmei,
            taikou=taikou,
            ana=ana,
            all_predictions=result_df,
        )
