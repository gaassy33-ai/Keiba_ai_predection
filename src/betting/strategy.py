"""
買い目推奨ロジック。

generate_betting_strategies(horses, budget) が券種ごとの推奨買い目リストを返す。

期待値 (EV = model_prob × estimated_odds) が高い買い目を優先し、
Kelly 基準で予算を配分する。

horses の各要素:
    horse_number      : int
    horse_name        : str
    mark              : str   "◎" | "○" | "▲" | "△"
    win_prob          : float  モデル予測勝率 (0-1)
    win_odds          : float | None  現在の単勝オッズ
    place_odds_center : float | None  複勝オッズ中央値
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

# JRA 控除率
_JRA_TAKE: dict[str, float] = {
    "単勝":  0.200,
    "複勝":  0.200,
    "馬連":  0.225,
    "馬単":  0.225,
    "ワイド": 0.225,
    "3連複": 0.225,
    "3連単": 0.275,
}

# 買い目タイプ → バッジ色
BET_TYPE_COLOR: dict[str, str] = {
    "単勝":  "#C62828",
    "複勝":  "#E65100",
    "馬連":  "#1565C0",
    "馬単":  "#0D47A1",
    "ワイド": "#2E7D32",
    "3連複": "#6A1B9A",
    "3連単": "#AD1457",
}


@dataclass
class BetLine:
    """1券種グループの推奨買い目。"""

    bet_type: str       # "単勝" | "馬連" | ...
    label: str          # 短い表示ラベル
    description: str    # 詳細説明（点数含む）
    combo_count: int    # 点数
    prob: float         # 的中確率（モデル推定）
    est_odds: float     # 推定（または実）オッズ
    ev: float           # 期待値 = prob × est_odds
    allocation: int = 0         # 推奨投資額（円、100円単位）
    is_featured: bool = False   # True = 🔥 勝負買い目


# ---------------------------------------------------------------------------
# Harville 公式ベースの確率計算
# ---------------------------------------------------------------------------

def _harville(probs: list[float], order: list[int]) -> float:
    """特定の着順になる確率を Harville 公式で計算する。"""
    p = 1.0
    remaining = 1.0
    for idx in order:
        if remaining < 1e-9:
            return 0.0
        p *= probs[idx] / remaining
        remaining -= probs[idx]
    return p


def _prob_quinella(probs: list[float], i: int, j: int) -> float:
    """馬連: i,j が1-2着（順不同）に入る確率。"""
    return _harville(probs, [i, j]) + _harville(probs, [j, i])


def _prob_wide(probs: list[float], i: int, j: int) -> float:
    """ワイド: i,j 両馬が3着以内に入る確率（Harville 近似）。"""
    total = 0.0
    for k in range(len(probs)):
        if k == i or k == j:
            continue
        for order in permutations([i, j, k]):
            total += _harville(probs, list(order))
    return total


def _prob_trifecta_box(probs: list[float], idx: list[int]) -> float:
    """3連複: idx の3頭が何らかの順で1-3着に入る確率。"""
    total = 0.0
    for order in permutations(idx):
        total += _harville(probs, list(order))
    return total


# ---------------------------------------------------------------------------
# オッズ推定・Kelly 配分
# ---------------------------------------------------------------------------

def _market_prob(win_odds: float | None) -> float:
    """単勝オッズ → 市場勝率（近似）。オッズ未取得時は 0.05 を返す。"""
    if not win_odds or win_odds <= 1.0:
        return 0.05
    return 1.0 / win_odds


def _est_odds(model_prob: float, market_prob: float, bet_type: str) -> float:
    """
    市場確率から推定オッズを計算する（JRA 控除率込み）。

    EV = model_prob × est_odds = model_prob / market_prob × (1 - take)
    モデル確率 > 市場確率なら EV > 1（割安馬券）。
    """
    take = _JRA_TAKE.get(bet_type, 0.25)
    return (1.0 - take) / max(market_prob, 0.001)


def _kelly(prob: float, odds: float) -> float:
    """Kelly 分率（0〜0.25 にクリップ）。"""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, min((b * prob - (1.0 - prob)) / b, 0.25))


def _allocate(bets: list[BetLine], budget: int) -> None:
    """Kelly 分率比例で budget を各 bet に配分（100円単位）。"""
    kelly_vals = [_kelly(b.prob, b.est_odds) for b in bets]
    total_k = sum(kelly_vals)

    if total_k < 1e-9:
        each = max((budget // max(len(bets), 1) // 100) * 100, 100)
        for b in bets:
            b.allocation = each
        return

    for i, b in enumerate(bets):
        raw = budget * kelly_vals[i] / total_k
        b.allocation = max(int(raw / 100) * 100, 100)

    # 端数を最高 Kelly のベットに加算
    used = sum(b.allocation for b in bets)
    diff = budget - used
    if diff != 0:
        best_i = kelly_vals.index(max(kelly_vals))
        bets[best_i].allocation = max(bets[best_i].allocation + diff, 100)


# ---------------------------------------------------------------------------
# メイン関数
# ---------------------------------------------------------------------------

def generate_betting_strategies(
    horses: list[dict],
    budget: int = 10_000,
    min_ev: float = 0.75,
) -> list[BetLine]:
    """
    予測勝率とオッズから推奨買い目リストを生成する。

    Parameters
    ----------
    horses : list[dict]
        各馬の情報（win_prob, win_odds, horse_number, horse_name, mark を含む）。
    budget : int
        総予算（円）。
    min_ev : float
        この期待値未満の買い目はスキップする。

    Returns
    -------
    list[BetLine]
        EV 降順、allocation 設定済みの買い目リスト。
    """
    if not horses:
        return []

    # win_prob 降順ソート
    sh = sorted(horses, key=lambda h: h.get("win_prob", 0), reverse=True)
    n = len(sh)

    # mark 補完
    _default_marks = ["◎", "○", "▲", "△", "△", "△"]
    sh = [
        {**h, "mark": h.get("mark") or (_default_marks[i] if i < len(_default_marks) else "△")}
        for i, h in enumerate(sh)
    ]

    model_probs  = [max(h.get("win_prob", 1 / n), 1e-6) for h in sh]
    market_probs = [_market_prob(h.get("win_odds")) for h in sh]

    def num(h: dict) -> str:
        return str(h.get("horse_number", "?"))

    def name(h: dict) -> str:
        return h.get("horse_name", "---")

    def mk(h: dict) -> str:
        return h.get("mark", "")

    honmei = sh[0]
    bets: list[BetLine] = []

    # ── 単勝 ──────────────────────────────────────────────────────────────
    win_odds = honmei.get("win_odds")
    if win_odds and win_odds > 1.0:
        prob = model_probs[0]
        ev = prob * win_odds
        if ev >= min_ev:
            bets.append(BetLine(
                bet_type="単勝",
                label=f"{mk(honmei)}{num(honmei)} {name(honmei)}",
                description=f"単勝 {num(honmei)}番  オッズ {win_odds:.1f}倍",
                combo_count=1,
                prob=prob,
                est_odds=win_odds,
                ev=ev,
            ))

    # ── 複勝 ──────────────────────────────────────────────────────────────
    place_odds = honmei.get("place_odds_center") or honmei.get("place_odds_min")
    if place_odds:
        place_prob = min(model_probs[0] * 2.5, 0.75)
        ev = place_prob * place_odds
        if ev >= min_ev:
            bets.append(BetLine(
                bet_type="複勝",
                label=f"{mk(honmei)}{num(honmei)} {name(honmei)}",
                description=f"複勝 {num(honmei)}番  オッズ {place_odds:.1f}倍",
                combo_count=1,
                prob=place_prob,
                est_odds=place_odds,
                ev=ev,
            ))

    # ── 馬連（◎ → ○▲ 流し）────────────────────────────────────────────────
    if n >= 2:
        partner_idx = list(range(1, min(3, n)))  # ○▲
        mq_sum = sum(_prob_quinella(model_probs, 0, j) for j in partner_idx)
        xq_sum = sum(_prob_quinella(market_probs, 0, j) for j in partner_idx)
        odds = _est_odds(mq_sum, xq_sum, "馬連")
        ev = mq_sum * odds
        partner_str = "/".join(mk(sh[j]) + num(sh[j]) for j in partner_idx)
        if ev >= min_ev:
            bets.append(BetLine(
                bet_type="馬連",
                label=f"◎{num(honmei)}→{partner_str} 流し",
                description=f"◎→○▲ 流し {len(partner_idx)}点",
                combo_count=len(partner_idx),
                prob=mq_sum,
                est_odds=odds,
                ev=ev,
            ))

    # ── ワイド（上位4頭から EV 上位2点）──────────────────────────────────
    top_n = min(4, n)
    wide_candidates: list[BetLine] = []
    for i, j in combinations(range(top_n), 2):
        mw = _prob_wide(model_probs, i, j)
        xw = _prob_wide(market_probs, i, j)
        odds = _est_odds(mw, xw, "ワイド")
        ev = mw * odds
        wide_candidates.append(BetLine(
            bet_type="ワイド",
            label=f"{mk(sh[i])}{num(sh[i])}-{mk(sh[j])}{num(sh[j])}",
            description=f"{num(sh[i])}-{num(sh[j])}",
            combo_count=1,
            prob=mw,
            est_odds=odds,
            ev=ev,
        ))
    wide_candidates.sort(key=lambda b: b.ev, reverse=True)
    top_wide = [b for b in wide_candidates[:2] if b.ev >= min_ev]
    if top_wide:
        total_p  = sum(b.prob for b in top_wide)
        avg_ev   = sum(b.ev for b in top_wide) / len(top_wide)
        avg_odds = avg_ev / total_p if total_p > 0 else 0
        bets.append(BetLine(
            bet_type="ワイド",
            label=" / ".join(b.label for b in top_wide),
            description=" / ".join(b.description for b in top_wide) + f"  {len(top_wide)}点",
            combo_count=len(top_wide),
            prob=total_p,
            est_odds=avg_odds,
            ev=avg_ev,
        ))

    # ── 3連複（◎○▲ BOX）────────────────────────────────────────────────
    if n >= 3:
        mt3 = _prob_trifecta_box(model_probs, [0, 1, 2])
        xt3 = _prob_trifecta_box(market_probs, [0, 1, 2])
        odds = _est_odds(mt3, xt3, "3連複")
        ev = mt3 * odds
        if ev >= min_ev:
            bets.append(BetLine(
                bet_type="3連複",
                label=f"◎{num(sh[0])}-○{num(sh[1])}-▲{num(sh[2])} BOX",
                description=f"◎○▲ BOX  1点",
                combo_count=1,
                prob=mt3,
                est_odds=odds,
                ev=ev,
            ))

    # ── 3連単（◎1着固定 → ○▲ → ○▲△ 流し）─────────────────────────────
    if n >= 3:
        second_pool = list(range(1, min(3, n)))      # ○▲
        third_pool  = list(range(1, min(4, n)))      # ○▲△
        combos_3t   = [(j, k) for j in second_pool for k in third_pool if k != j]

        if combos_3t:
            mt_sum = sum(_harville(model_probs,  [0, j, k]) for j, k in combos_3t)
            xt_sum = sum(_harville(market_probs, [0, j, k]) for j, k in combos_3t)
            odds = _est_odds(mt_sum, xt_sum, "3連単")
            ev = mt_sum * odds
            sec_marks = "".join(mk(sh[j]) for j in second_pool)
            thi_marks = "".join(mk(sh[k]) for k in sorted(set(third_pool)))
            if ev >= min_ev:
                bets.append(BetLine(
                    bet_type="3連単",
                    label=f"◎{num(honmei)}→{sec_marks}→{thi_marks} 流し",
                    description=f"◎→{sec_marks}→{thi_marks}  {len(combos_3t)}点",
                    combo_count=len(combos_3t),
                    prob=mt_sum,
                    est_odds=odds,
                    ev=ev,
                ))

    if not bets:
        return []

    # EV 降順ソートして最高 EV に 🔥
    bets.sort(key=lambda b: b.ev, reverse=True)
    bets[0].is_featured = True

    # 予算配分
    _allocate(bets, budget)

    return bets
