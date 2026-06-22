"""
ltr_ev_engine.py
================
LTR (LambdaRank) + EV ベースの馬連買い目計算エンジン。

backtest_wide_ltr.py のコアロジックを再利用可能モジュールとして切り出したもの。
daily_batch.py・バックテストの双方からインポートして使用する。

【買い目構築ロジック（軸馬流し方式）】
  1. LTR スコアを temperature scaling して Plackett-Luce 確率に変換
  2. 確率 1位→ axis1、2位→ axis2 を軸馬として確定
  3. 3位以下の上位 partner_top_n 頭をパートナー候補として選出
  4. 以下の組み合わせで買い目を生成:
       - axis1 × axis2 （軸馬同士）
       - axis1 × partner_k （軸1流し）
       - axis2 × partner_k （軸2流し）
  5. min_ev_threshold / est_quinella_odds_max / min_p_model_threshold でフィルタ
  6. EV 降順に max_bets_per_race 点まで出力

  ※ 馬連確率計算（Harville 公式の2着版）はレース全頭の確率を使用するため精度は落ちない。
    「どのペアを買うか」の候補選択だけを軸馬流しに変更。

【公開 API】
  evaluate_race(feat_df, odds_map, horse_names, ltr, cfg) -> list[QuinellaBet]
    1レースの軸馬流し候補ペア EV を計算し、フィルタ済み・EV降順の買い目リストを返す。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model.trainer import LTRTrainer
from config.settings import BettingConfig

# JRA 馬連控除率
JRA_TAKE_QUINELLA: float = 0.175


# ──────────────────────────────────────────────────────────────────────────────
# データクラス
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QuinellaBet:
    """1点の馬連買い目を表すデータクラス。"""
    horse_num_i:       str
    horse_num_j:       str
    horse_id_i:        str
    horse_id_j:        str
    horse_name_i:      str          # LINE 表示・ログ用
    horse_name_j:      str
    odds_i:            float | None  # 単勝オッズ
    odds_j:            float | None
    p_model:           float        # AI 予測馬連確率
    p_market:          float        # 市場馬連確率
    ev:                float        # 算出 EV = p_model × 推定オッズ
    est_quinella_odds: float        # 推定馬連オッズ（市場）


# 後方互換エイリアス（旧コードとの互換性維持）
WideBet = QuinellaBet


# ──────────────────────────────────────────────────────────────────────────────
# 確率計算ユーティリティ
# ──────────────────────────────────────────────────────────────────────────────

def market_probs(odds_list: list[float]) -> np.ndarray:
    """単勝オッズリスト → 市場確率ベクトル（合計 ≈ 1.0）"""
    raw = np.array([1.0 / max(o, 1.01) for o in odds_list])
    s = raw.sum()
    return raw / s if s > 0 else raw


def prob_quinella(probs: np.ndarray | list[float], i: int, j: int) -> float:
    """
    馬連確率: P(馬 i と馬 j が 1・2着を占める)

    Harville 公式の2着版:
      P_quinella(i,j) = P(i 1着)*P(j 2着|i 1着) + P(j 1着)*P(i 2着|j 1着)
                      = p[i]*p[j]/(1-p[i]) + p[j]*p[i]/(1-p[j])

    ワイドの3周順列ループを省略できるため計算コストが O(n) → O(1)。
    """
    p  = np.asarray(probs, dtype=np.float64)
    pi, pj = float(p[i]), float(p[j])
    rem_i = 1.0 - pi
    rem_j = 1.0 - pj
    r = 0.0
    if rem_i > 1e-9:
        r += pi * pj / rem_i
    if rem_j > 1e-9:
        r += pj * pi / rem_j
    return float(np.clip(r, 0.0, 1.0))


def est_quinella_odds_fn(p_market: float) -> float:
    """市場馬連確率 → 推定馬連オッズ（JRA 控除 17.5% 反映）"""
    if p_market <= 1e-9:
        return 999.0
    return (1.0 - JRA_TAKE_QUINELLA) / p_market


# ──────────────────────────────────────────────────────────────────────────────
# コア関数
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_race(
    feat_df:            pd.DataFrame,
    odds_map:           dict[str, float],   # horse_id(str) → 単勝オッズ
    horse_names:        dict[str, str],     # horse_id(str) → 馬名
    ltr:                LTRTrainer,
    cfg:                BettingConfig,
    return_candidates:  bool = False,
    pair_calibrator     = None,   # PairCalibrator | None: p_model_ij の事後補正（Two-Brain Phase1）
    gatekeeper          = None,   # GatekeeperTrainer | None: 軸馬適性判定（Two-Brain Phase2-3）
    gatekeeper_threshold: float = 0.50,
) -> list[QuinellaBet] | tuple[list[QuinellaBet], pd.DataFrame]:
    """
    軸馬流し方式で馬連 EV を計算し、フィルタ・EV降順の買い目リストを返す。

    【ロジック】
      Step1: model_probs（temperature scaling済み）を生成
      Step2: 1位→axis1、2位→axis2 を軸馬として確定
      Step3: 3位以下の上位 partner_top_n 頭をパートナー候補として選出
      Step4: (axis1×axis2), (axis1×partner), (axis2×partner) の EV を計算
      Step5: フィルタ適用 → EV 降順ソート → max_bets_per_race 点まで返す

    ※ 確率計算（Harville）はレース全頭の model_probs / mkt_probs を使用。

    Parameters
    ----------
    feat_df      : FeatureEngineer.build_entry_features() の出力
    odds_map     : horse_id → 単勝オッズ（float）の辞書
    horse_names  : horse_id → 馬名 の辞書（LINE 表示用）
    ltr          : ロード済み LTRTrainer
    cfg          : BettingConfig（EV 閾値・点数上限・オッズ上限・partner_top_n）

    Returns
    -------
    list[QuinellaBet]
        EV 降順、cfg.max_bets_per_race 点まで（0 なら全点）。
    return_candidates=True の場合は (bets, candidates_df) を返す。
    candidates_df はフィルターで落ちたペアも含む全候補（軸馬オッズ超過で
    レース自体が見送りとなった場合もその旨を含めて返す）。
    シャドー稼働でのキャリブレーション検証用。
    """
    _empty = ([], pd.DataFrame()) if return_candidates else []
    if len(feat_df) < 2:
        return _empty

    # ── LTR スコア → Plackett-Luce 確率（temperature scaling 適用）──
    X = (
        feat_df[ltr.feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    raw_scores  = ltr.predict(X)
    temperature = getattr(ltr, "temperature", 1.0)
    model_probs = LTRTrainer.scores_to_probs(raw_scores, temperature=temperature)

    # ── 市場確率（単勝オッズから） ──────────────────────────────────
    horse_ids  = feat_df["horse_id"].astype(str).tolist()
    horse_nums = []
    for num in feat_df["horse_number"].tolist():
        try:
            horse_nums.append(str(int(float(num))))
        except (ValueError, TypeError):
            horse_nums.append(str(num))

    odds_list = []
    for hid in horse_ids:
        o = float(odds_map.get(hid, float("nan")))
        odds_list.append(o if not np.isnan(o) and o > 1.0 else 5.0)
    mkt_probs = market_probs(odds_list)

    # ── Step 2-3: 軸馬・パートナー選出 ─────────────────────────────
    n_total    = len(horse_ids)
    sorted_idx = list(np.argsort(model_probs)[::-1])   # model_probs 降順

    axis1 = sorted_idx[0]
    axis2 = sorted_idx[1] if n_total >= 2 else None

    # ── Gatekeeper 軸馬適性フィルター（Two-Brain Phase2-3）──────────
    # LTR（市場情報なし）が選んだ軸馬候補を、Gatekeeper（市場情報あり）が
    # 「3着以内に来る安全な軸か」を個別に判定する。軸単位で棄却し、
    # 両方棄却された場合のみレース全体を見送る。
    p_axis1_safe: float | None = None
    p_axis2_safe: float | None = None
    axis1_safe, axis2_safe = True, True
    if gatekeeper is not None:
        gk_X = (
            feat_df[gatekeeper.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        gk_probs = gatekeeper.predict_proba(gk_X)
        p_axis1_safe = float(gk_probs[axis1])
        axis1_safe = p_axis1_safe >= gatekeeper_threshold
        if axis2 is not None:
            p_axis2_safe = float(gk_probs[axis2])
            axis2_safe = p_axis2_safe >= gatekeeper_threshold

    def _axis_safe(idx: int) -> bool:
        if idx == axis1:
            return axis1_safe
        if axis2 is not None and idx == axis2:
            return axis2_safe
        return True   # パートナー（軸ではない）は Gatekeeper 対象外

    # ── 軸馬の信頼性フィルター ────────────────────────────────────
    # 軸馬の単勝オッズが axis_max_odds を超える場合、市場と AI の乖離が大きく
    # 信頼性が低いため、レース全体を見送る（空リストを返す）。
    axis_max_odds = getattr(cfg, "axis_max_odds", 10.0)
    if axis_max_odds > 0:
        o_ax1 = float(odds_map.get(horse_ids[axis1], float("nan")))
        o_ax2 = (
            float(odds_map.get(horse_ids[axis2], float("nan")))
            if axis2 is not None else float("nan")
        )
        ax1_over = not np.isnan(o_ax1) and o_ax1 > axis_max_odds
        ax2_over = axis2 is not None and not np.isnan(o_ax2) and o_ax2 > axis_max_odds
    else:
        ax1_over = ax2_over = False
    race_axis_reject = bool(axis_max_odds > 0 and (ax1_over or ax2_over))
    if race_axis_reject and not return_candidates:
        return []   # 軸馬が市場で低評価 → レース見送り

    # axis1・axis2 を除いた上位 partner_top_n 頭をパートナーとして選出
    partner_top_n = getattr(cfg, "partner_top_n", 5)
    partner_candidates = sorted_idx[2:]                  # axis1/axis2 を除く
    partners = partner_candidates[:partner_top_n]

    # ── Step 4: 買い目ペア候補を列挙 ────────────────────────────────
    pairs: list[tuple[int, int]] = []
    if axis2 is not None:
        pairs.append((axis1, axis2))            # 軸馬同士
    for p_idx in partners:
        pairs.append((axis1, p_idx))            # 軸1 × partner
        if axis2 is not None:
            pairs.append((axis2, p_idx))        # 軸2 × partner

    # ── Step 5: EV 計算・フィルタリング ─────────────────────────────
    bets: list[QuinellaBet] = []
    candidates: list[dict] = [] if return_candidates else None
    est_q_odds_max = getattr(cfg, "est_quinella_odds_max", 50.0)

    for i, j in pairs:
        # 自己ペア除外（horse_id 重複のガード）
        if horse_ids[i] == horse_ids[j]:
            continue

        hid_i = horse_ids[i]
        hid_j = horse_ids[j]

        o_i = float(odds_map.get(hid_i, float("nan")))
        o_j = float(odds_map.get(hid_j, float("nan")))

        # 馬連確率計算（Harville: レース全頭を使用）— フィルター判定に関わらず常に計算
        p_model_ij_raw = prob_quinella(model_probs, i, j)
        p_market_ij    = prob_quinella(mkt_probs,   i, j)
        est_odds_ij    = est_quinella_odds_fn(p_market_ij)

        # ── 事後キャリブレーション（Two-Brain Phase1）────────────────
        # 実運用ログで判明した P_model の過大評価（実的中率の約4.75倍）を、
        # EV計算の直前に Platt Scaling で補正する。LTR モデル自体は
        # 市場情報を見ない「純粋な能力評価器」のまま変更しない。
        if pair_calibrator is not None:
            p_model_ij = float(pair_calibrator.transform(p_model_ij_raw))
        else:
            p_model_ij = p_model_ij_raw

        ev = p_model_ij * est_odds_ij

        passed_longshot = not (
            (not np.isnan(o_i) and o_i > cfg.longshot_odds_max) or
            (not np.isnan(o_j) and o_j > cfg.longshot_odds_max)
        )
        passed_est_odds   = est_odds_ij <= est_q_odds_max
        passed_p_model    = p_model_ij >= cfg.min_p_model_threshold
        passed_ev         = ev >= cfg.min_ev_threshold
        passed_gatekeeper = _axis_safe(i) and _axis_safe(j)
        would_buy = (
            passed_longshot and passed_est_odds and passed_p_model
            and passed_ev and not race_axis_reject and passed_gatekeeper
        )

        if return_candidates:
            candidates.append({
                "horse_num_i":       horse_nums[i],
                "horse_num_j":       horse_nums[j],
                "horse_id_i":        hid_i,
                "horse_id_j":        hid_j,
                "horse_name_i":      horse_names.get(hid_i, f"馬{horse_nums[i]}"),
                "horse_name_j":      horse_names.get(hid_j, f"馬{horse_nums[j]}"),
                "odds_i":            round(o_i, 1) if not np.isnan(o_i) else None,
                "odds_j":            round(o_j, 1) if not np.isnan(o_j) else None,
                "p_model":           round(p_model_ij,  4),
                "p_model_raw":       round(p_model_ij_raw, 4),
                "p_market":          round(p_market_ij, 4),
                "ev":                round(ev, 3),
                "est_quinella_odds": round(est_odds_ij, 1),
                "race_axis_reject":  race_axis_reject,
                "p_axis1_safe":      p_axis1_safe,
                "p_axis2_safe":      p_axis2_safe,
                "passed_longshot":   passed_longshot,
                "passed_est_odds":   passed_est_odds,
                "passed_p_model":    passed_p_model,
                "passed_ev":         passed_ev,
                "passed_gatekeeper": passed_gatekeeper,
                "would_buy":         would_buy,
            })

        if not would_buy:
            continue

        bets.append(QuinellaBet(
            horse_num_i       = horse_nums[i],
            horse_num_j       = horse_nums[j],
            horse_id_i        = hid_i,
            horse_id_j        = hid_j,
            horse_name_i      = horse_names.get(hid_i, f"馬{horse_nums[i]}"),
            horse_name_j      = horse_names.get(hid_j, f"馬{horse_nums[j]}"),
            odds_i            = round(o_i, 1) if not np.isnan(o_i) else None,
            odds_j            = round(o_j, 1) if not np.isnan(o_j) else None,
            p_model           = round(p_model_ij,  4),
            p_market          = round(p_market_ij, 4),
            ev                = round(ev, 3),
            est_quinella_odds = round(est_odds_ij, 1),
        ))

    # EV 降順ソート
    bets.sort(key=lambda b: -b.ev)

    # 点数上限（0 = 無制限）
    if cfg.max_bets_per_race > 0:
        bets = bets[:cfg.max_bets_per_race]

    if return_candidates:
        return bets, pd.DataFrame(candidates)
    return bets
