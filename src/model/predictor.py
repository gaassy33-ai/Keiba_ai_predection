"""
推論モジュール。

学習済みモデルを使って出走馬の1着入線確率を算出し、
本命・対抗・穴馬を決定する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from config.settings import settings
from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer


@dataclass
class PredictionResult:
    """1レース分の予測結果（5段階印体系）"""
    race_id: str
    race_name: str
    honmei: dict           # ◎ 本命（1位）
    taikou: dict           # ○ 対抗（2位）
    ana: dict              # 旧・穴馬（後方互換 = tanana と同値）
    all_predictions: pd.DataFrame  # 全馬の予測確率（win_prob 降順）
    tanana: dict = field(default_factory=dict)      # ▲ 単穴（3位）
    hoshi:  dict = field(default_factory=dict)      # ☆ 穴候補（4位 or 高EV）
    renshita: list = field(default_factory=list)    # △ 連下（5-7位、最大3頭）


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
            "jockey_id", "jockey_name",
            "recent_avg_pos", "recent_avg_last3f",
        ]
        cols = base_cols + [c for c in extra_cols if c in feature_df.columns]
        result_df = feature_df[cols].copy()
        result_df["win_prob"] = probs
        result_df = result_df.sort_values("win_prob", ascending=False).reset_index(drop=True)

        honmei = result_df.iloc[0].to_dict()
        taikou = result_df.iloc[1].to_dict() if len(result_df) > 1 else {}

        # ▲ 単穴: 3位
        tanana = result_df.iloc[2].to_dict() if len(result_df) > 2 else {}

        # ☆ 穴候補: 4位 by win_prob（win_odds がある場合は EV 最高馬を優先）
        if len(result_df) > 3:
            rank4plus = result_df.iloc[3:].copy()
            if "win_odds" in rank4plus.columns and rank4plus["win_odds"].notna().any():
                rank4plus = rank4plus.copy()
                rank4plus["_ev"] = rank4plus["win_prob"] * rank4plus["win_odds"].fillna(0)
                best_i = rank4plus["_ev"].idxmax()
                hoshi = rank4plus.loc[best_i].drop("_ev").to_dict()
            else:
                hoshi = result_df.iloc[3].to_dict()
        else:
            hoshi = {}

        # △ 連下: 5-7位（最大3頭）
        renshita = [
            result_df.iloc[i].to_dict()
            for i in range(4, min(7, len(result_df)))
        ]

        # 後方互換: ana = tanana（外枠穴馬ロジックを廃止）
        ana = tanana

        logger.info(
            f"Prediction done: {race_name} | "
            f"◎{honmei.get('horse_name')} ({honmei.get('win_prob', 0):.1%}) | "
            f"○{taikou.get('horse_name')} | ▲{tanana.get('horse_name')} | "
            f"☆{hoshi.get('horse_name')} | "
            f"△{[h.get('horse_name') for h in renshita]}"
        )

        return PredictionResult(
            race_id=race_id,
            race_name=race_name,
            honmei=honmei,
            taikou=taikou,
            ana=ana,
            all_predictions=result_df,
            tanana=tanana,
            hoshi=hoshi,
            renshita=renshita,
        )
