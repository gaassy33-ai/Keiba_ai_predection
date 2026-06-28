"""
evaluate_race() のハードキルスイッチ・NaNフェイルセーフの回帰テスト。

2026-06-28: Seleniumのオッズ描画待機漏れで odds_map が欠損したまま推論が
進み、市場確率が1/nへ退化してEVが膨張する事故が発生した。再発防止のため、
オッズ欠損時に必ず空リストを返すことを保証する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import BettingConfig
from src.betting.ltr_ev_engine import evaluate_race

FEATURE_COLUMNS = ["feat_a", "feat_b"]


class _FakeLTR:
    """LTRTrainer の軽量モック（モデルファイルの読み込みなしで evaluate_race を検証する）。"""

    feature_columns = FEATURE_COLUMNS
    temperature = 1.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # 馬番が小さいほど高スコア（= axis1/axis2 を決定論的にする）
        return -X.index.to_numpy(dtype=float)


def _make_feat_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "horse_id": [f"h{i}" for i in range(n)],
        "horse_number": list(range(1, n + 1)),
        "feat_a": [1.0] * n,
        "feat_b": [2.0] * n,
    })


def _make_cfg(**overrides) -> BettingConfig:
    base = dict(
        target_surface=["ダート"], min_ev_threshold=0.0, max_bets_per_race=5,
        longshot_odds_max=30.0, est_quinella_odds_max=150.0,
        min_p_model_threshold=0.0, partner_top_n=5, axis_max_odds=10.0,
    )
    base.update(overrides)
    return BettingConfig(**base)


@pytest.fixture
def ltr():
    return _FakeLTR()


def test_empty_odds_map_returns_no_bets(ltr):
    """odds_map が完全に空 → ハードキルスイッチが作動し買い目0点になる。"""
    feat_df = _make_feat_df()
    horse_names = {hid: hid for hid in feat_df["horse_id"]}
    bets = evaluate_race(feat_df, {}, horse_names, ltr, _make_cfg())
    assert bets == []


def test_partial_missing_odds_returns_no_bets(ltr):
    """1頭でもオッズが欠けている（NaN相当）→ レース全体を即座に見送る。"""
    feat_df = _make_feat_df(n=5)
    horse_names = {hid: hid for hid in feat_df["horse_id"]}
    # h0〜h3 は正常なオッズ、h4 だけ欠損（モック）
    odds_map = {"h0": 2.0, "h1": 3.0, "h2": 5.0, "h3": 8.0}
    bets = evaluate_race(feat_df, odds_map, horse_names, ltr, _make_cfg())
    assert bets == []


def test_nan_odds_value_returns_no_bets(ltr):
    """odds_map にNaNが値として直接含まれているケースも同様に見送る。"""
    feat_df = _make_feat_df(n=4)
    horse_names = {hid: hid for hid in feat_df["horse_id"]}
    odds_map = {"h0": 2.0, "h1": float("nan"), "h2": 5.0, "h3": 8.0}
    bets = evaluate_race(feat_df, odds_map, horse_names, ltr, _make_cfg())
    assert bets == []


def test_return_candidates_also_short_circuits_on_missing_odds(ltr):
    """return_candidates=True（shadow_mode）でも空候補を返し、推論を一切進めない。"""
    feat_df = _make_feat_df(n=4)
    horse_names = {hid: hid for hid in feat_df["horse_id"]}
    odds_map = {"h0": 2.0, "h1": 3.0}  # h2, h3 が欠損
    bets, candidates_df = evaluate_race(
        feat_df, odds_map, horse_names, ltr, _make_cfg(), return_candidates=True,
    )
    assert bets == []
    assert candidates_df.empty


def test_full_odds_present_produces_bets(ltr):
    """全頭分のオッズが揃っている正常系では、従来通り買い目が生成される。"""
    feat_df = _make_feat_df(n=6)
    horse_names = {hid: hid for hid in feat_df["horse_id"]}
    odds_map = {f"h{i}": float(2 + i) for i in range(6)}
    bets = evaluate_race(feat_df, odds_map, horse_names, ltr, _make_cfg())
    assert len(bets) > 0


def test_axis_max_odds_still_rejects_overpriced_axis(ltr):
    """
    axis_max_odds フィルターの基本動作（NaN修正後の回帰確認）: 軸馬の単勝
    オッズが axis_max_odds を超える場合は従来通りレース全体を見送る。
    axis1='h0' の単勝オッズ(2.0) > axis_max_odds(1.5) のため見送りになる。
    """
    feat_df = _make_feat_df(n=6)
    horse_names = {hid: hid for hid in feat_df["horse_id"]}
    odds_map = {f"h{i}": float(2 + i) for i in range(6)}
    bets = evaluate_race(feat_df, odds_map, horse_names, ltr, _make_cfg(axis_max_odds=1.5))
    assert bets == []
