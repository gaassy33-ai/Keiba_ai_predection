"""
pair_calibrator.py
===================
LTR推論直後・EV計算前に挿入する、馬連ペア確率(P_model_ij)の事後キャリブレーション。

【背景】
2026-04-25〜6-21の実運用ログ分析で、P_model（馬連ペア確率）が実際の的中率の
約4.75倍過大評価されていることが判明（Σp_model=28.5 vs 実的中6件、
ポアソン近似でP≈3.9e-7）。LTRモデル自体は「市場情報を見ない純粋な能力評価器」
として維持する（Two-Brain設計）ため、モデルを再学習せず、EV計算の直前に
このキャリブレーターを挿入してP_modelを補正する。

【手法】
Platt Scaling（ロジスティック回帰、2パラメータ）。
サンプル数が少なく的中（正例）が極端に少ない現状のデータでは、
Isotonic Regression（多自由度のステップ関数）は過学習しやすいため、
パラメータ数の少ないPlatt Scalingを採用。

  calibrated_p = sigmoid(a * logit(p_model_ij) + b)

データが蓄積されてサンプル数が十分（数千ペア規模）になった段階で、
Isotonic Regressionへの切替を検討する。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class PairCalibrator:
    """馬連ペア確率(p_model_ij)の Platt Scaling キャリブレーター。"""

    def __init__(self) -> None:
        self.lr: LogisticRegression | None = None
        self.n_samples: int = 0
        self.n_positive: int = 0

    def fit(self, p_model: np.ndarray, hit: np.ndarray) -> "PairCalibrator":
        """
        Parameters
        ----------
        p_model : 過去に計算された p_model_ij（ペア確率、補正前）
        hit     : 実際に的中したか（1/0）
        """
        x = _logit(np.asarray(p_model, dtype=np.float64)).reshape(-1, 1)
        y = np.asarray(hit, dtype=np.int64)
        self.n_samples = len(y)
        self.n_positive = int(y.sum())

        self.lr = LogisticRegression(C=1.0)
        self.lr.fit(x, y)
        return self

    def transform(self, p_model: np.ndarray | float) -> np.ndarray | float:
        """補正後の確率を返す（スカラー入力にも対応）。"""
        if self.lr is None:
            return p_model
        scalar_input = np.isscalar(p_model)
        x = _logit(np.atleast_1d(np.asarray(p_model, dtype=np.float64))).reshape(-1, 1)
        calibrated = self.lr.predict_proba(x)[:, 1]
        return float(calibrated[0]) if scalar_input else calibrated

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "lr": self.lr,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "PairCalibrator":
        raw = joblib.load(path)
        inst = cls()
        inst.lr = raw["lr"]
        inst.n_samples = raw.get("n_samples", 0)
        inst.n_positive = raw.get("n_positive", 0)
        return inst
