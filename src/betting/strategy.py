"""
買い目推奨ロジック（改善版 v2）。

generate_betting_strategies(horses, budget) が券種ごとの推奨買い目リストを返す。

改善点:
  1. レース絞り込み (min_honmei_prob / min_confidence_gap)
  2. EV ベースの相手選び + トリガミ除外
  3. 馬単フォーメーション追加・3連複を◎1軸流しに変更・点数考慮 Kelly 配分

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
# オッズ推定・EV スコア・トリガミ判定
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


def _ev_score(h: dict, fallback_odds: float = 5.0) -> float:
    """
    horse dict から EV スコア（win_prob × win_odds）を計算。

    相手馬の優先順位付けに使用する。win_odds 未取得時は fallback_odds で代替。
    """
    prob = float(h.get("win_prob", 0))
    odds = h.get("win_odds")
    if odds and float(odds) > 1.0:
        return prob * float(odds)
    return prob * fallback_odds


def _is_torikami(est_odds: float, min_roi: float = 1.05) -> bool:
    """
    的中しても回収率が min_roi を下回るトリガミ判定。

    推定オッズが min_roi 倍未満 → True（購入不要）。
    """
    return est_odds < min_roi


def _kelly(prob: float, odds: float) -> float:
    """Kelly 分率（0〜0.25 にクリップ）。"""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, min((b * prob - (1.0 - prob)) / b, 0.25))


def _allocate_per_combo(bets: list[BetLine], budget: int) -> None:
    """
    1点あたり単位で Kelly 配分する（点数考慮版）。

    1点あたりの的中確率 = prob / combo_count として Kelly を計算し、
    allocation = kelly_per_combo × combo_count（100円単位）。
    多点券種への過剰配分を防ぐ。
    """
    kelly_vals = []
    for b in bets:
        prob_per = b.prob / max(b.combo_count, 1)
        kelly_vals.append(_kelly(prob_per, b.est_odds))

    total_k = sum(kelly_vals)

    if total_k < 1e-9:
        each = max((budget // max(len(bets), 1) // 100) * 100, 100)
        for b in bets:
            b.allocation = each * b.combo_count
        return

    for i, b in enumerate(bets):
        raw_per_combo = budget * kelly_vals[i] / total_k
        per_combo = max(int(raw_per_combo / 100) * 100, 100)
        b.allocation = per_combo * b.combo_count

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
    min_honmei_prob: float = 0.15,      # 【改善1】◎勝率の最低閾値
    min_confidence_gap: float = 0.05,   # 【改善1】◎と○の勝率差の最低値
) -> list[BetLine]:
    """
    予測勝率とオッズから推奨買い目リストを生成する（改善版 v2）。

    Parameters
    ----------
    horses : list[dict]
        各馬の情報（win_prob, win_odds, horse_number, horse_name, mark を含む）。
    budget : int
        総予算（円）。
    min_ev : float
        この期待値未満の買い目はスキップする。
    min_honmei_prob : float
        ◎の予測勝率がこれを下回るレースは全券種スキップ（「見」）。
    min_confidence_gap : float
        ◎と○の勝率差がこれを下回る混戦レースも全券種スキップ。

    Returns
    -------
    list[BetLine]
        EV 降順、allocation 設定済みの買い目リスト。
        レース絞り込み条件を満たさない場合は空リストを返す。
    """
    if not horses:
        return []

    # win_prob 降順ソート
    sh = sorted(horses, key=lambda h: h.get("win_prob", 0), reverse=True)
    n = len(sh)

    # mark 補完
    _default_marks = ["◎", "○", "▲", "☆", "△", "△", "△"]
    sh = [
        {**h, "mark": h.get("mark") or (_default_marks[i] if i < len(_default_marks) else "△")}
        for i, h in enumerate(sh)
    ]

    model_probs  = [max(h.get("win_prob", 1 / n), 1e-6) for h in sh]
    market_probs = [_market_prob(h.get("win_odds")) for h in sh]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 【改善1】レース絞り込み（見・ケン）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    honmei_p = model_probs[0]
    taikou_p = model_probs[1] if n >= 2 else 0.0

    if honmei_p < min_honmei_prob:
        return []   # ◎の信頼度が低い → 見送り

    if (honmei_p - taikou_p) < min_confidence_gap:
        return []   # 混戦で軸が定まらない → 見送り

    def num(h: dict) -> str:
        return str(h.get("horse_number", "?"))

    def name(h: dict) -> str:
        return h.get("horse_name", "---")

    def mk(h: dict) -> str:
        return h.get("mark", "")

    honmei = sh[0]
    bets: list[BetLine] = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 【改善2】EV ランク付き相手候補の生成
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 相手候補プール: ◎を除く上位5頭（win_prob 順）
    candidate_pool = sh[1:min(6, n)]   # ○▲☆△ + α

    # EV スコア = win_prob × win_odds でランキング（オッズなし時は5倍代替）
    ev_order = sorted(
        range(len(candidate_pool)),
        key=lambda i: _ev_score(candidate_pool[i]),
        reverse=True,
    )
    # sh の絶対インデックス（0=◎ を除く）に変換
    ev_top_idx = [ev_order[i] + 1 for i in range(len(ev_order))]   # 全相手を EV 順で保持

    # ── 単勝 ──────────────────────────────────────────────────────────────
    # 変更なし（改善1の絞り込みが最大の改善）
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
    # 変更なし（改善1の絞り込みで向上）
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

    # ── 馬連（◎ → EV上位3頭、トリガミ除外）────────────────────────────
    # 【変更】4点一括 → EV上位3頭を個別評価してトリガミを除く
    if n >= 2:
        valid_baren: list[tuple] = []
        for j in ev_top_idx[:3]:   # EV 上位3頭
            if j >= n:
                continue
            mq = _prob_quinella(model_probs, 0, j)
            xq = _prob_quinella(market_probs, 0, j)
            est = _est_odds(mq, xq, "馬連")
            ev = mq * est
            if ev >= min_ev and not _is_torikami(est):
                valid_baren.append((j, mq, est, ev))

        if valid_baren:
            total_p = sum(x[1] for x in valid_baren)
            avg_ev  = sum(x[3] for x in valid_baren) / len(valid_baren)
            avg_est = avg_ev / total_p if total_p > 0 else 0
            partner_str = "/".join(mk(sh[j]) + num(sh[j]) for j, *_ in valid_baren)
            bets.append(BetLine(
                bet_type="馬連",
                label=f"◎{num(honmei)}→{partner_str} 流し",
                description=f"◎→EV上位 {len(valid_baren)}点",
                combo_count=len(valid_baren),
                prob=total_p,
                est_odds=avg_est,
                ev=avg_ev,
            ))

    # ── 馬単（◎1着固定 → EV上位3頭フォーメーション、トリガミ除外）────────
    # 【新規追加】◎の1着精度（30.8%）を最大限活用
    if n >= 2:
        valid_umatan: list[tuple] = []
        for j in ev_top_idx[:3]:   # EV 上位3頭
            if j >= n:
                continue
            mh = _harville(model_probs, [0, j])
            xh = _harville(market_probs, [0, j])
            est = _est_odds(mh, xh, "馬単")
            ev = mh * est
            if ev >= min_ev and not _is_torikami(est):
                valid_umatan.append((j, mh, est, ev))

        if valid_umatan:
            total_p = sum(x[1] for x in valid_umatan)
            avg_ev  = sum(x[3] for x in valid_umatan) / len(valid_umatan)
            avg_est = avg_ev / total_p if total_p > 0 else 0
            partner_str = "/".join(mk(sh[j]) + num(sh[j]) for j, *_ in valid_umatan)
            bets.append(BetLine(
                bet_type="馬単",
                label=f"◎{num(honmei)}→{partner_str} フォーメーション",
                description=f"◎1着固定→EV上位 {len(valid_umatan)}点",
                combo_count=len(valid_umatan),
                prob=total_p,
                est_odds=avg_est,
                ev=avg_ev,
            ))

    # ── ワイド（上位5頭から EV上位2点、トリガミ除外）─────────────────────
    # 【変更】トリガミ除外を追加
    top_n = min(5, n)
    wide_candidates: list[BetLine] = []
    for i, j in combinations(range(top_n), 2):
        mw = _prob_wide(model_probs, i, j)
        xw = _prob_wide(market_probs, i, j)
        odds_w = _est_odds(mw, xw, "ワイド")
        ev = mw * odds_w
        if not _is_torikami(odds_w):   # トリガミ除外
            wide_candidates.append(BetLine(
                bet_type="ワイド",
                label=f"{mk(sh[i])}{num(sh[i])}-{mk(sh[j])}{num(sh[j])}",
                description=f"{num(sh[i])}-{num(sh[j])}",
                combo_count=1,
                prob=mw,
                est_odds=odds_w,
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

    # ── 3連複（◎1頭軸 → ○▲☆△ 流し、最大6点、トリガミ除外）─────────────
    # 【変更】◎○▲BOX1点 → ◎を軸に EV上位4頭から C(4,2)=6点の流し
    # 紐荒れによる高配当も捕捉できる
    if n >= 3:
        # EV 上位4頭（相手プール）
        partner_idx4 = [j for j in ev_top_idx[:4] if j < n]
        valid_sf: list[tuple] = []
        for pi, pj in combinations(partner_idx4, 2):
            mt3 = _prob_trifecta_box(model_probs, [0, pi, pj])
            xt3 = _prob_trifecta_box(market_probs, [0, pi, pj])
            est = _est_odds(mt3, xt3, "3連複")
            ev  = mt3 * est
            if ev >= min_ev and not _is_torikami(est):
                valid_sf.append((pi, pj, mt3, est, ev))

        if valid_sf:
            total_p = sum(x[2] for x in valid_sf)
            avg_ev  = sum(x[4] for x in valid_sf) / len(valid_sf)
            avg_est = avg_ev / total_p if total_p > 0 else 0
            partner_nums = "-".join(
                sorted(set(
                    num(sh[p]) for combo in valid_sf for p in combo[:2]
                ), key=int)
            )
            bets.append(BetLine(
                bet_type="3連複",
                label=f"◎{num(honmei)}-{partner_nums} 流し",
                description=f"◎1軸→EV上位 {len(valid_sf)}点",
                combo_count=len(valid_sf),
                prob=total_p,
                est_odds=avg_est,
                ev=avg_ev,
            ))

    # ── 3連単（◎1着固定 → EV上位3頭 → EV上位3頭、最大6点）────────────
    # 【変更】3着候補を EV 上位3頭に限定して点数を削減
    if n >= 3:
        pool_idx = [j for j in ev_top_idx[:3] if j < n]   # EV 上位3頭
        combos_3t = [(j, k) for j in pool_idx for k in pool_idx if k != j]

        if combos_3t:
            mt_sum = sum(_harville(model_probs,  [0, j, k]) for j, k in combos_3t)
            xt_sum = sum(_harville(market_probs, [0, j, k]) for j, k in combos_3t)
            odds   = _est_odds(mt_sum, xt_sum, "3連単")
            ev     = mt_sum * odds
            sec_marks = "".join(mk(sh[j]) for j in pool_idx)
            if ev >= min_ev and not _is_torikami(odds):
                bets.append(BetLine(
                    bet_type="3連単",
                    label=f"◎{num(honmei)}→{sec_marks}→{sec_marks} フォーメーション",
                    description=f"◎1着固定→EV上位3→EV上位3  {len(combos_3t)}点",
                    combo_count=len(combos_3t),
                    prob=mt_sum,
                    est_odds=odds,
                    ev=ev,
                ))

    if not bets:
        return []

    # EV 降順ソートして最高 EV に is_featured フラグ
    bets.sort(key=lambda b: b.ev, reverse=True)
    bets[0].is_featured = True

    # 点数考慮 Kelly 配分
    _allocate_per_combo(bets, budget)

    return bets
