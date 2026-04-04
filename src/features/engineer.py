"""
特徴量エンジニアリング。

取得した生データを LightGBM に渡せる形に変換する。
主な特徴量:
- 産駒勝率（父・母父別）
- ジョッキーのコース適性（芝/ダート × 距離帯）
- 馬場状態・天候コード
- 直近 n 走の着順傾向・前走着差・前走上がり3Fランク
- レースクラスコード・会場コード・市場集中度（HHI）
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """
    raw DataFrame（過去成績）を受け取り、特徴量 DataFrame を返す。

    Usage
    -----
    >>> fe = FeatureEngineer(history_df)
    >>> features = fe.build_entry_features(entry_df, ...)
    """

    # 直近何走を参照するか
    RECENT_N = 5

    def __init__(self, history_df: pd.DataFrame) -> None:
        self.history = self._preprocess_history(history_df)
        self._sire_win_rate: pd.Series | None = None
        self._bms_win_rate: pd.Series | None = None
        self._jockey_course_stats: pd.DataFrame | None = None
        self._trainer_stats: pd.DataFrame | None = None
        # 推論時用: save_stats/from_stats でロードされる馬別直近成績
        self._horse_recent_form: pd.DataFrame | None = None
        # 評論家フィードバック追加特徴量
        self._jockey_venue_stats: pd.DataFrame | None = None    # 騎手×会場 勝率
        self._horse_ground_stats: pd.DataFrame | None = None    # 馬場適性（馬別×馬場状態）
        # 新規追加特徴量
        self._horse_dist_stats: pd.DataFrame | None = None      # 馬の距離帯適性（≥5走フィルタ）

    # ------------------------------------------------------------------
    # 前処理
    # ------------------------------------------------------------------

    def _preprocess_history(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        # 着順を数値化（除外馬は NaN）
        df["finish_pos_num"] = pd.to_numeric(
            df["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
        )
        df["is_win"] = (df["finish_pos_num"] == 1).astype(int)
        df["is_placed"] = (df["finish_pos_num"] <= 3).astype(int)

        # タイムを秒に変換 (例: "1:34.5" → 94.5)
        df["finish_time_sec"] = df["finish_time"].apply(self._time_to_seconds)

        # カラムずれ修正: scraper が列インデックスをずらして保存したため
        #   "last_3f" 列 → 実際の単勝オッズ (例: 1.4 倍)
        #   "odds"    列 → 実際の上がり3Fタイム (例: 37.3 秒)
        # recent_avg_last3f には正しい上がり3F を使う
        if "odds" in df.columns:
            df["last_3f_num"] = pd.to_numeric(df["odds"], errors="coerce")
        else:
            df["last_3f_num"] = pd.to_numeric(df["last_3f"], errors="coerce")

        # 実際の単勝オッズ（カラムずれにより "last_3f" 列に格納）
        if "last_3f" in df.columns:
            df["tansho_odds_num"] = pd.to_numeric(df["last_3f"], errors="coerce")

        # 着差を数値化（前走着差特徴量用）
        if "margin" in df.columns:
            df["margin_num"] = df["margin"].apply(self._margin_to_float)

        # 斤量
        df["weight_carried_num"] = pd.to_numeric(df["weight_carried"], errors="coerce")

        # 性齢分解
        if "sex_age" in df.columns:
            df["sex"] = df["sex_age"].str[0]
            df["age"] = pd.to_numeric(df["sex_age"].str[1:], errors="coerce")
        else:
            if "sex" not in df.columns:
                df["sex"] = ""
            if "age" not in df.columns:
                df["age"] = np.nan

        # fetch_result_and_meta でインライン付加された列を数値化
        for col in ("distance", "ground_condition_code", "weather_code"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @staticmethod
    def _time_to_seconds(time_str: str) -> float | None:
        if not isinstance(time_str, str):
            return None
        m = re.match(r"(\d+):(\d+\.\d+)", time_str.strip())
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        return None

    @staticmethod
    def _margin_to_float(margin_str) -> float:
        """着差文字列を馬身数値に変換。勝ち馬（空白）は 0.0。"""
        if not isinstance(margin_str, str) or margin_str.strip() == "":
            return 0.0
        s = margin_str.strip()
        margin_map = {
            "ハナ": 0.1, "アタマ": 0.2, "クビ": 0.5,
            "1/2": 0.5, "3/4": 0.75, "大差": 10.0,
        }
        if s in margin_map:
            return margin_map[s]
        # "1 1/2" → 1.5
        m = re.match(r"^(\d+)\s+(\d+)/(\d+)$", s)
        if m:
            return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
        # "1/2" → 0.5
        m = re.match(r"^(\d+)/(\d+)$", s)
        if m:
            return int(m.group(1)) / int(m.group(2))
        try:
            return float(s)
        except Exception:
            return 0.0

    @staticmethod
    def _race_name_to_class_code(race_name: str) -> int:
        """
        レース名からクラスコードを返す。

        0=新馬, 1=未勝利, 2=1勝クラス, 3=2勝クラス,
        4=3勝クラス, 5=オープン特別, 6=重賞(G1/G2/G3), 7=障害
        """
        if not race_name:
            return -1
        name = str(race_name)
        if "新馬" in name:
            return 0
        if "未勝利" in name:
            return 1
        if any(k in name for k in ["1勝クラス", "500万"]):
            return 2
        if any(k in name for k in ["2勝クラス", "1000万"]):
            return 3
        if any(k in name for k in ["3勝クラス", "1600万"]):
            return 4
        if any(k in name for k in ["障害"]):
            return 7
        # G1/G2/G3 または 格付け記号があれば重賞
        if re.search(r"G[123]|（G[123]）", name):
            return 6
        # オープン特別・Listed（明示的でないが重賞以外の無クラス）
        return 5

    @staticmethod
    def _compute_hhi(odds_series: pd.Series) -> float:
        """
        単勝オッズのリストからハーフィンダール指数（市場集中度）を計算。
        HHI = Σ p_i^2  (p_i: 正規化市場確率)
        高いほど「本命一強」、低いほど「混戦」。
        """
        odds = pd.to_numeric(odds_series, errors="coerce").dropna()
        odds = odds[odds > 1.0]
        if len(odds) < 2:
            return float("nan")
        raw = 1.0 / odds.values
        total = raw.sum()
        if total <= 0:
            return float("nan")
        probs = raw / total
        return float((probs ** 2).sum())

    # ------------------------------------------------------------------
    # 集計特徴量の事前計算
    # ------------------------------------------------------------------

    def precompute_aggregations(self) -> None:
        """産駒勝率・ジョッキーコース適性・馬別直近成績を事前計算する。"""
        h = self.history
        if h.empty:
            logger.warning("History is empty. Skipping aggregation precomputation.")
            return

        # 産駒勝率（父別）
        if "father" in h.columns and h["father"].replace("", pd.NA).notna().any():
            self._sire_win_rate = (
                h[h["father"].replace("", pd.NA).notna()]
                .groupby("father")["is_win"].mean().rename("sire_win_rate")
            )

        # 産駒勝率（母父別）
        if "mother_father" in h.columns and h["mother_father"].replace("", pd.NA).notna().any():
            self._bms_win_rate = (
                h[h["mother_father"].replace("", pd.NA).notna()]
                .groupby("mother_father")["is_win"].mean().rename("bms_win_rate")
            )

        # ジョッキーのコース×距離帯適性
        if "course_type" not in h.columns or h["course_type"].replace("", pd.NA).isna().all():
            logger.warning("course_type column missing; skipping jockey course stats.")
            logger.info("Aggregation precomputation done (partial).")
            return

        h2 = h.copy()
        h2["distance_bin"] = pd.cut(
            pd.to_numeric(h2.get("distance", pd.Series(dtype=float)), errors="coerce"),
            bins=[0, 1400, 1800, 2200, 9999],
            labels=["短距離", "マイル", "中距離", "長距離"],
        )
        self._jockey_course_stats = (
            h2.groupby(["jockey_id", "course_type", "distance_bin"], observed=True)
            .agg(
                jockey_win_rate=("is_win", "mean"),
                jockey_place_rate=("is_placed", "mean"),
                jockey_runs=("is_win", "count"),
            )
            .reset_index()
        )

        # 調教師別勝率・連対率
        if "trainer_name" in h.columns and h["trainer_name"].replace("", pd.NA).notna().any():
            self._trainer_stats = (
                h[h["trainer_name"].replace("", pd.NA).notna()]
                .groupby("trainer_name")
                .agg(
                    trainer_win_rate=("is_win", "mean"),
                    trainer_place_rate=("is_placed", "mean"),
                )
                .reset_index()
            )

        # 騎手×会場 勝率（評論家フィードバック: jockey × venue 交互作用）
        if "jockey_id" in h.columns and "race_id" in h.columns:
            h3 = h.copy()
            h3["venue_code_num"] = pd.to_numeric(
                h3["race_id"].astype(str).str[4:6], errors="coerce"
            )
            self._jockey_venue_stats = (
                h3[h3["venue_code_num"].notna()]
                .groupby(["jockey_id", "venue_code_num"])
                .agg(jockey_venue_win_rate=("is_win", "mean"))
                .reset_index()
                .rename(columns={"venue_code_num": "venue_code"})
            )

        # 馬場適性: 馬別 × ground_condition_code の勝率（評論家フィードバック）
        if "horse_id" in h.columns and "ground_condition_code" in h.columns:
            gc = pd.to_numeric(h["ground_condition_code"], errors="coerce")
            mask = gc.notna()
            if mask.any():
                h4 = h[mask].copy()
                h4["ground_condition_code"] = gc[mask]
                self._horse_ground_stats = (
                    h4.groupby(["horse_id", "ground_condition_code"])
                    .agg(horse_ground_win_rate=("is_win", "mean"))
                    .reset_index()
                )

        # ② 馬の距離帯適性（≥5走フィルタで信頼性確保）
        if "horse_id" in h.columns and "distance" in h.columns:
            h5 = h.copy()
            h5["_dist_bin"] = pd.cut(
                pd.to_numeric(h5["distance"], errors="coerce"),
                bins=[0, 1400, 1800, 2200, 9999],
                labels=[0, 1, 2, 3],
            ).astype("Int64")
            dist_grp = (
                h5[h5["_dist_bin"].notna()]
                .groupby(["horse_id", "_dist_bin"], observed=True)
                .agg(_win=("is_win", "sum"), _n=("is_win", "count"))
                .reset_index()
            )
            dist_grp["horse_dist_win_rate"] = np.where(
                dist_grp["_n"] >= 5,
                dist_grp["_win"] / dist_grp["_n"],
                np.nan,
            )
            self._horse_dist_stats = dist_grp[
                ["horse_id", "_dist_bin", "horse_dist_win_rate"]
            ].rename(columns={"_dist_bin": "dist_bin"})

        logger.info("Aggregation precomputation done.")

    # ------------------------------------------------------------------
    # 出走表に特徴量を付加
    # ------------------------------------------------------------------

    def build_entry_features(
        self,
        entry_df: pd.DataFrame,
        course_type: str,
        distance: int,
        ground_condition_code: int,
        weather_code: int,
        race_class_code: int = -1,
        venue_code: int = -1,
    ) -> pd.DataFrame:
        """
        出走馬リスト (entry_df) に特徴量を結合して返す。

        Parameters
        ----------
        entry_df : pd.DataFrame
            columns: [horse_id, horse_number, frame_number, sex, age,
                      weight_carried, jockey_id, trainer_name, ...]
            オプション列: odds (市場オッズ, HHI 計算用)
        course_type : str   "芝" or "ダート"
        distance : int      レース距離（メートル）
        ground_condition_code : int
        weather_code : int
        race_class_code : int   _race_name_to_class_code() の結果
        venue_code : int        race_id[4:6] の整数値 (1-10)
        """
        df = entry_df.copy()

        # ── レース環境特徴量（全馬共通）──────────────────────────────
        df["course_type_code"]     = 0 if course_type == "芝" else 1
        df["distance"]             = distance
        df["ground_condition_code"] = ground_condition_code
        df["weather_code"]         = weather_code

        # 改善④: レースクラスコード
        df["race_class_code"] = race_class_code

        # 改善⑤: 会場コード & 交互作用特徴量
        df["venue_code"]       = venue_code
        df["venue_ground_code"] = (
            venue_code * 10 + ground_condition_code
            if venue_code >= 0 else -1
        )

        # 改善②: 出走頭数
        df["n_entries"] = len(df)

        # 改善⑥: 市場集中度 HHI
        if "odds" in df.columns:
            hhi = self._compute_hhi(df["odds"])
        else:
            hhi = float("nan")
        df["market_hhi"] = hhi

        # 改善⑧: オッズ特徴量（log変換 + レース内人気順位正規化）
        # 2026年傾向対応: 30倍上限キャップで極端な長期人気の影響を抑制
        if "odds" in df.columns:
            odds_num = pd.to_numeric(df["odds"], errors="coerce")
            # 30倍超は同一視（波乱傾向増加に対してモデルが過度にフォローするのを防ぐ）
            odds_capped = odds_num.clip(upper=30.0)
            df["odds_log"] = np.log1p(odds_capped)
            pop_rank = odds_num.rank(method="min", ascending=True)
            n_horses = pop_rank.max()
            df["popularity_rank_norm"] = (
                (pop_rank - 1) / (n_horses - 1) if n_horses > 1 else 0.0
            )
        else:
            df["odds_log"] = np.nan
            df["popularity_rank_norm"] = np.nan

        # ── 調教師勝率・連対率 ────────────────────────────────────────
        if self._trainer_stats is not None and "trainer_name" in df.columns:
            df = df.merge(self._trainer_stats, on="trainer_name", how="left")
        else:
            df["trainer_win_rate"]  = np.nan
            df["trainer_place_rate"] = np.nan

        # ── ジョッキー適性 ───────────────────────────────────────────
        if self._jockey_course_stats is not None:
            distance_label = pd.cut(
                [distance], bins=[0, 1400, 1800, 2200, 9999],
                labels=["短距離", "マイル", "中距離", "長距離"]
            )[0]
            jockey_filtered = self._jockey_course_stats[
                (self._jockey_course_stats["course_type"] == course_type)
                & (self._jockey_course_stats["distance_bin"] == distance_label)
            ][["jockey_id", "jockey_win_rate", "jockey_place_rate", "jockey_runs"]].copy()

            def _norm_jid(s: pd.Series) -> pd.Series:
                return s.astype(str).apply(
                    lambda x: str(int(x)) if x.strip().isdigit() else x.strip()
                )
            df["jockey_id"] = _norm_jid(df["jockey_id"])
            jockey_filtered["jockey_id"] = _norm_jid(jockey_filtered["jockey_id"])
            df = df.merge(jockey_filtered, on="jockey_id", how="left")
        else:
            df["jockey_win_rate"]  = np.nan
            df["jockey_place_rate"] = np.nan
            df["jockey_runs"]       = np.nan

        # ── 直近成績（改善⑦含む）────────────────────────────────────
        df = self._add_recent_form(df)

        # ── 3歳馬フラグ（評論家フィードバック: 春の3歳馬急成長期対応）────
        age_num = pd.to_numeric(df.get("age", pd.Series(dtype=float)), errors="coerce")
        df["is_3yo"] = (age_num == 3).astype(int)

        # ── 騎手×会場 勝率（評論家フィードバック: jockey×venue 交互作用）─
        if self._jockey_venue_stats is not None and venue_code >= 0:
            jv = self._jockey_venue_stats[
                self._jockey_venue_stats["venue_code"] == venue_code
            ][["jockey_id", "jockey_venue_win_rate"]].copy()
            jv["jockey_id"] = jv["jockey_id"].astype(str).str.strip()
            df["jockey_id"] = df["jockey_id"].astype(str).str.strip()
            df = df.merge(jv, on="jockey_id", how="left")
        else:
            df["jockey_venue_win_rate"] = np.nan

        # ── 馬場適性（評論家フィードバック: 馬別 × ground_condition 勝率）─
        if self._horse_ground_stats is not None:
            gs = self._horse_ground_stats[
                self._horse_ground_stats["ground_condition_code"] == ground_condition_code
            ][["horse_id", "horse_ground_win_rate"]].copy()
            df = df.merge(gs, on="horse_id", how="left")
        else:
            df["horse_ground_win_rate"] = np.nan

        # ── ② 距離帯適性（馬別 × distance_bin 勝率、≥5走フィルタ済み）──
        if self._horse_dist_stats is not None:
            dist_bin_val = int(pd.cut(
                [distance], bins=[0, 1400, 1800, 2200, 9999], labels=[0, 1, 2, 3]
            )[0])
            ds = self._horse_dist_stats[
                self._horse_dist_stats["dist_bin"] == dist_bin_val
            ][["horse_id", "horse_dist_win_rate"]].copy()
            df = df.merge(ds, on="horse_id", how="left")
        else:
            df["horse_dist_win_rate"] = np.nan

        # ── ④ クラス変動フラグ（昇降級: 正=昇級 負=降級）────────────────
        if "prev_race_class_code" in df.columns and race_class_code >= 0:
            df["class_change"] = race_class_code - pd.to_numeric(
                df["prev_race_class_code"], errors="coerce"
            )
        else:
            df["class_change"] = np.nan

        return df

    def _add_recent_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        各馬の直近 N 走成績を付加する。
        追加する特徴量:
          - recent_avg_pos: 直近N走の着順平均
          - recent_avg_last3f: 直近N走の上がり3F平均
          - recent_top3_rate: 直近N走での3着以内率
          - prev_margin: 前走着差（馬身）
          - prev_last3f_rank_norm: 前走での上がり3Fランク（0=最速, 1=最遅）

        推論時（history が空）は _horse_recent_form から参照する。
        """
        COLS = [
            "recent_avg_pos", "recent_avg_last3f", "recent_top3_rate",
            "prev_margin", "prev_last3f_rank_norm",
            "recent_pos_trend", "recent_last3f_trend",
            "prev_race_class_code",
        ]

        # ── 推論パス: from_stats() でロード済みの統計を使用 ───────────
        if self.history.empty:
            if self._horse_recent_form is not None and not self._horse_recent_form.empty:
                hrf = self._horse_recent_form.reset_index() if "horse_id" not in self._horse_recent_form.columns else self._horse_recent_form
                df = df.merge(hrf[["horse_id"] + COLS], on="horse_id", how="left")
            else:
                for c in COLS:
                    df[c] = np.nan
            return df

        # ── 学習パス: history から計算 ───────────────────────────────
        h = self.history

        # 上がり3F の race 内ランクを事前計算（改善⑦）
        if "last_3f_num" in h.columns and "race_id" in h.columns:
            h = h.copy()
            h["last3f_rank_in_race"] = h.groupby("race_id")["last_3f_num"].rank(
                ascending=True, method="min", na_option="keep"
            )
            h["last3f_n_in_race"] = h.groupby("race_id")["last_3f_num"].transform("count")
            h["last3f_rank_norm"] = h["last3f_rank_in_race"] / h["last3f_n_in_race"]
        else:
            h = h.copy()
            h["last3f_rank_norm"] = np.nan

        # 馬ごとの直近 N 走（race_id 降順 = 新しい順）
        recent = (
            h.sort_values("race_id", ascending=False)
            .groupby("horse_id")
            .head(self.RECENT_N)
        )

        # 直近 N 走の集計
        agg = (
            recent.groupby("horse_id")
            .agg(
                recent_avg_pos=("finish_pos_num", "mean"),
                recent_avg_last3f=("last_3f_num", "mean"),
                recent_top3_rate=("is_placed", "mean"),
            )
            .reset_index()
        )

        # 前走（最新1走）の着差・上がり3Fランク（改善⑦）
        all_sorted = h.sort_values(["horse_id", "race_id"], ascending=[True, False])
        all_sorted = all_sorted.copy()
        all_sorted["_recent_rank"] = all_sorted.groupby("horse_id").cumcount() + 1

        prev_race = all_sorted[all_sorted["_recent_rank"] == 1].copy()

        prev_cols = ["horse_id"]
        if "margin_num" in prev_race.columns:
            prev_cols.append("margin_num")
        if "last3f_rank_norm" in prev_race.columns:
            prev_cols.append("last3f_rank_norm")
        # ③ トレンド計算用
        for _tc in ("finish_pos_num", "last_3f_num"):
            if _tc in prev_race.columns:
                prev_cols.append(_tc)
        # ④ 前走クラスコード
        if "race_class_code_hist" in prev_race.columns:
            prev_cols.append("race_class_code_hist")

        prev_df = prev_race[prev_cols].rename(columns={
            "margin_num": "prev_margin",
            "last3f_rank_norm": "prev_last3f_rank_norm",
            "finish_pos_num": "_prev_pos",
            "last_3f_num": "_prev_3f",
            "race_class_code_hist": "prev_race_class_code",
        })

        # ③ 直近2-5走の平均（トレンド基準）
        rest = all_sorted[all_sorted["_recent_rank"].between(2, 5)]
        rest_agg = (
            rest.groupby("horse_id")
            .agg(_avg_pos_2_5=("finish_pos_num", "mean"),
                 _avg_3f_2_5=("last_3f_num", "mean"))
            .reset_index()
        )
        trend_base = prev_df[["horse_id"] + [c for c in ["_prev_pos", "_prev_3f"] if c in prev_df.columns]].merge(
            rest_agg, on="horse_id", how="left"
        )
        if "_prev_pos" in trend_base.columns and "_avg_pos_2_5" in trend_base.columns:
            trend_base["recent_pos_trend"] = trend_base["_prev_pos"] - trend_base["_avg_pos_2_5"]
        else:
            trend_base["recent_pos_trend"] = np.nan
        if "_prev_3f" in trend_base.columns and "_avg_3f_2_5" in trend_base.columns:
            trend_base["recent_last3f_trend"] = trend_base["_prev_3f"] - trend_base["_avg_3f_2_5"]
        else:
            trend_base["recent_last3f_trend"] = np.nan
        trend_df = trend_base[["horse_id", "recent_pos_trend", "recent_last3f_trend"]]

        # 結合
        df = df.merge(agg, on="horse_id", how="left")
        df = df.merge(prev_df.drop(columns=[c for c in ["_prev_pos", "_prev_3f"] if c in prev_df.columns]),
                      on="horse_id", how="left")
        df = df.merge(trend_df, on="horse_id", how="left")

        # 存在しない列を NaN で埋める
        for c in COLS:
            if c not in df.columns:
                df[c] = np.nan

        return df

    # ------------------------------------------------------------------
    # 学習用データセット生成（データリーク防止ルックバック付き）
    # ------------------------------------------------------------------

    def build_training_dataset(
        self,
        race_meta_df: pd.DataFrame,
        output_path: str | None = None,
    ) -> pd.DataFrame:
        """
        過去レース成績と race_meta_df を結合し、学習用 DataFrame を生成する。

        データリーク防止のため、各レースの特徴量は「そのレースより過去の
        データのみ」を使って計算する（時系列ルックバック方式）。
        """
        h = self.history
        if h.empty:
            raise ValueError("history_df が空です。fetch_bulk_results() を先に実行してください。")

        required_meta_cols = {"race_id", "course_type", "distance",
                              "ground_condition_code", "weather_code"}
        missing = required_meta_cols - set(race_meta_df.columns)
        if missing:
            raise ValueError(f"race_meta_df に必須カラムが不足しています: {missing}")

        h = h.copy()
        h["race_id"] = h["race_id"].astype(str)
        race_meta_df = race_meta_df.copy()
        race_meta_df["race_id"] = race_meta_df["race_id"].astype(str)

        # ④ クラス変動フラグ用: メタのレース名からクラスコードを計算して履歴に付加
        if "race_name" in race_meta_df.columns:
            class_map = {
                rid: self._race_name_to_class_code(rname)
                for rid, rname in zip(race_meta_df["race_id"], race_meta_df["race_name"])
            }
            h["race_class_code_hist"] = h["race_id"].map(class_map)
            self.history = self.history.copy()
            self.history["race_class_code_hist"] = self.history["race_id"].map(class_map)

        if "race_date" in race_meta_df.columns and race_meta_df["race_date"].notna().any():
            race_meta_df["race_date"] = pd.to_datetime(
                race_meta_df["race_date"], errors="coerce"
            )
            date_map = race_meta_df.set_index("race_id")["race_date"].to_dict()
            h["race_date"] = h["race_id"].map(date_map)
            ordered_meta = race_meta_df.sort_values("race_date").dropna(subset=["race_date"])
            target_race_ids = ordered_meta["race_id"].tolist()
            logger.info("Sorting by race_date (chronological order guaranteed).")
        else:
            logger.warning("race_date not found; falling back to race_id sort (approximate).")
            h["race_date"] = pd.NaT
            target_race_ids = sorted(race_meta_df["race_id"].unique())
        logger.info(f"Building training dataset for {len(target_race_ids)} races...")

        all_rows: list[pd.DataFrame] = []

        for i, race_id in enumerate(target_race_ids):
            race_entries = h[h["race_id"] == race_id].copy()
            if "race_date" in h.columns and not race_entries.empty:
                race_date_val = race_entries["race_date"].iloc[0]
                if pd.notna(race_date_val):
                    history_before = h[h["race_date"] < race_date_val]
                else:
                    history_before = h[h["race_id"] < race_id]
            else:
                history_before = h[h["race_id"] < race_id]
            if race_entries.empty:
                continue

            meta_row = race_meta_df[race_meta_df["race_id"] == race_id].iloc[0]
            course_type           = meta_row["course_type"]
            distance              = int(meta_row["distance"])
            ground_condition_code = int(meta_row["ground_condition_code"])
            weather_code          = int(meta_row["weather_code"])

            # 改善④: レースクラスコード
            race_name       = str(meta_row.get("race_name", "")) if "race_name" in meta_row else ""
            race_class_code = self._race_name_to_class_code(race_name)

            # 改善⑤: 会場コード（race_id[4:6]）
            try:
                venue_code = int(race_id[4:6])
            except Exception:
                venue_code = -1

            if not str(course_type) or distance == 0:
                continue

            wt_col = "weight_carried_num" if "weight_carried_num" in race_entries.columns else "weight_carried"
            base_cols = ["horse_id", "horse_name", "horse_number", "frame_number", "jockey_id"]
            for c in ("sex", "age", wt_col, "trainer_name"):
                if c in race_entries.columns:
                    base_cols.append(c)
            entry_df = race_entries[base_cols].copy()
            if wt_col == "weight_carried_num":
                entry_df = entry_df.rename(columns={"weight_carried_num": "weight_carried"})
            if "sex" not in entry_df.columns:
                entry_df["sex"] = ""
            if "age" not in entry_df.columns:
                entry_df["age"] = np.nan
            if "weight_carried" not in entry_df.columns:
                entry_df["weight_carried"] = np.nan
            if "trainer_name" not in entry_df.columns:
                entry_df["trainer_name"] = ""

            for col in ("father", "mother_father"):
                if col in race_entries.columns:
                    entry_df[col] = race_entries[col].values
                else:
                    entry_df[col] = ""

            # 改善⑥: 市場オッズ（HHI 計算用）
            # ★カラムずれ: "last_3f" 列が実際の単勝オッズ
            if "last_3f" in race_entries.columns:
                entry_df["odds"] = pd.to_numeric(
                    race_entries["last_3f"].values, errors="coerce"
                )

            tmp_fe = FeatureEngineer(history_before)
            tmp_fe.precompute_aggregations()

            try:
                feat_df = tmp_fe.build_entry_features(
                    entry_df=entry_df,
                    course_type=course_type,
                    distance=distance,
                    ground_condition_code=ground_condition_code,
                    weather_code=weather_code,
                    race_class_code=race_class_code,
                    venue_code=venue_code,
                )
            except Exception as e:
                logger.warning(f"Skipping race_id={race_id}: feature build failed ({e})")
                continue

            label_cols = ["horse_id", "is_win"]
            if "is_placed" in race_entries.columns:
                label_cols.append("is_placed")
            labels = race_entries[label_cols].set_index("horse_id")
            feat_df = feat_df.set_index("horse_id") if "horse_id" in feat_df.columns else feat_df
            feat_df["is_win"] = feat_df.index.map(labels["is_win"])
            if "is_placed" in labels.columns:
                feat_df["is_placed"] = feat_df.index.map(labels["is_placed"])
            feat_df["race_id"] = race_id
            feat_df = feat_df.reset_index()

            all_rows.append(feat_df)

            if (i + 1) % 200 == 0:
                logger.info(f"  Progress: {i+1}/{len(target_race_ids)} races processed")

        if not all_rows:
            logger.warning("No training rows generated.")
            return pd.DataFrame()

        result = pd.concat(all_rows, ignore_index=True)
        result = result.dropna(subset=["is_win"])
        result["is_win"] = result["is_win"].astype(int)
        if "is_placed" in result.columns:
            result["is_placed"] = result["is_placed"].fillna(0).astype(int)

        logger.info(
            f"Training dataset built: {len(result)} rows, "
            f"{result['is_win'].sum()} winners, "
            f"{result['race_id'].nunique()} races"
        )

        if output_path:
            import pathlib
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_path, index=False)
            logger.info(f"Training dataset saved to {output_path}")

        return result

    @classmethod
    def build_from_scratch(
        cls,
        race_ids: list[str],
        scraper,
        output_path: str | None = None,
        checkpoint_prefix: str | None = None,
    ) -> pd.DataFrame:
        logger.info(f"Step 1/2: Fetching results+meta for {len(race_ids)} races (1 req/race)...")
        history_df, race_meta_df = scraper.fetch_bulk_results_and_meta(
            race_ids, checkpoint_path=checkpoint_prefix
        )
        logger.info(f"  → {len(history_df)} horse records, {len(race_meta_df)} races")
        logger.info("Step 2/2: Building training dataset with lookback features...")
        fe = cls(history_df)
        fe.precompute_aggregations()
        result = fe.build_training_dataset(race_meta_df, output_path=output_path)
        if output_path:
            from config.settings import settings
            fe.save_stats(settings.stats_path)
        return result

    # ------------------------------------------------------------------
    # 統計データの保存・読み込み（推論時に使用）
    # ------------------------------------------------------------------

    def save_stats(self, path: Path | str) -> None:
        """集計統計をファイルに保存する（学習後に呼び出す）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 改善バグ修正: 推論時の recent_avg_pos 等 NaN 問題の解消
        # 馬別直近成績（horse_id → recent_avg_pos 等）を計算して保存
        horse_recent_form = self._compute_horse_recent_form_for_inference()

        stats = {
            "sire_win_rate":        self._sire_win_rate,
            "bms_win_rate":         self._bms_win_rate,
            "jockey_course_stats":  self._jockey_course_stats,
            "trainer_stats":        self._trainer_stats,
            "horse_recent_form":    horse_recent_form,
            "jockey_venue_stats":   self._jockey_venue_stats,
            "horse_ground_stats":   self._horse_ground_stats,
            "horse_dist_stats":     self._horse_dist_stats,
        }
        with open(path, "wb") as f:
            pickle.dump(stats, f)
        logger.info(f"Feature stats saved to {path}")

    def _compute_horse_recent_form_for_inference(self) -> pd.DataFrame | None:
        """
        全履歴から馬別の直近成績を計算して返す（推論用）。
        学習時の lookback とは異なり、全データを使用する。
        """
        h = self.history
        if h.empty:
            return None

        # 上がり3F の race 内ランクを事前計算
        if "last_3f_num" in h.columns and "race_id" in h.columns:
            h = h.copy()
            h["last3f_rank_in_race"] = h.groupby("race_id")["last_3f_num"].rank(
                ascending=True, method="min", na_option="keep"
            )
            h["last3f_n_in_race"] = h.groupby("race_id")["last_3f_num"].transform("count")
            h["last3f_rank_norm"] = h["last3f_rank_in_race"] / h["last3f_n_in_race"]
        else:
            h = h.copy()
            h["last3f_rank_norm"] = np.nan

        recent = (
            h.sort_values("race_id", ascending=False)
            .groupby("horse_id")
            .head(self.RECENT_N)
        )
        agg = (
            recent.groupby("horse_id")
            .agg(
                recent_avg_pos=("finish_pos_num", "mean"),
                recent_avg_last3f=("last_3f_num", "mean"),
                recent_top3_rate=("is_placed", "mean"),
            )
            .reset_index()
        )

        all_sorted_inf = h.sort_values(["horse_id", "race_id"], ascending=[True, False]).copy()
        all_sorted_inf["_recent_rank"] = all_sorted_inf.groupby("horse_id").cumcount() + 1

        prev_race = all_sorted_inf[all_sorted_inf["_recent_rank"] == 1].copy()

        prev_cols = ["horse_id"]
        if "margin_num" in prev_race.columns:
            prev_cols.append("margin_num")
        if "last3f_rank_norm" in prev_race.columns:
            prev_cols.append("last3f_rank_norm")
        for _tc in ("finish_pos_num", "last_3f_num"):
            if _tc in prev_race.columns:
                prev_cols.append(_tc)
        if "race_class_code_hist" in prev_race.columns:
            prev_cols.append("race_class_code_hist")

        prev_df = prev_race[prev_cols].rename(columns={
            "margin_num": "prev_margin",
            "last3f_rank_norm": "prev_last3f_rank_norm",
            "finish_pos_num": "_prev_pos",
            "last_3f_num": "_prev_3f",
            "race_class_code_hist": "prev_race_class_code",
        })

        # ③ トレンド計算
        rest_inf = all_sorted_inf[all_sorted_inf["_recent_rank"].between(2, 5)]
        rest_agg_inf = (
            rest_inf.groupby("horse_id")
            .agg(_avg_pos_2_5=("finish_pos_num", "mean"),
                 _avg_3f_2_5=("last_3f_num", "mean"))
            .reset_index()
        )
        trend_base_inf = prev_df[
            ["horse_id"] + [c for c in ["_prev_pos", "_prev_3f"] if c in prev_df.columns]
        ].merge(rest_agg_inf, on="horse_id", how="left")
        if "_prev_pos" in trend_base_inf.columns and "_avg_pos_2_5" in trend_base_inf.columns:
            trend_base_inf["recent_pos_trend"] = trend_base_inf["_prev_pos"] - trend_base_inf["_avg_pos_2_5"]
        else:
            trend_base_inf["recent_pos_trend"] = np.nan
        if "_prev_3f" in trend_base_inf.columns and "_avg_3f_2_5" in trend_base_inf.columns:
            trend_base_inf["recent_last3f_trend"] = trend_base_inf["_prev_3f"] - trend_base_inf["_avg_3f_2_5"]
        else:
            trend_base_inf["recent_last3f_trend"] = np.nan
        trend_df_inf = trend_base_inf[["horse_id", "recent_pos_trend", "recent_last3f_trend"]]

        drop_tmp = [c for c in ["_prev_pos", "_prev_3f"] if c in prev_df.columns]
        result = agg.merge(prev_df.drop(columns=drop_tmp), on="horse_id", how="left")
        result = result.merge(trend_df_inf, on="horse_id", how="left")

        for c in ("prev_margin", "prev_last3f_rank_norm", "prev_race_class_code",
                  "recent_pos_trend", "recent_last3f_trend"):
            if c not in result.columns:
                result[c] = np.nan
        logger.info(f"Horse recent form computed: {len(result):,} horses")
        return result

    @classmethod
    def from_stats(cls, path: Path | str) -> "FeatureEngineer":
        """
        保存済み統計から推論用 FeatureEngineer を生成する。
        horse_recent_form も含め、推論に必要な全統計をロードする。
        """
        path = Path(path)
        instance = cls(pd.DataFrame())
        if not path.exists():
            logger.warning(f"Stats file not found: {path}. Using empty stats.")
            return instance
        with open(path, "rb") as f:
            stats = pickle.load(f)
        instance._sire_win_rate        = stats.get("sire_win_rate")
        instance._bms_win_rate         = stats.get("bms_win_rate")
        instance._jockey_course_stats  = stats.get("jockey_course_stats")
        instance._trainer_stats        = stats.get("trainer_stats")
        instance._horse_recent_form    = stats.get("horse_recent_form")
        instance._jockey_venue_stats   = stats.get("jockey_venue_stats")
        instance._horse_ground_stats   = stats.get("horse_ground_stats")
        instance._horse_dist_stats     = stats.get("horse_dist_stats")
        if instance._horse_recent_form is not None:
            logger.info(
                f"Feature stats loaded from {path} "
                f"(horse_recent_form: {len(instance._horse_recent_form):,} horses)"
            )
        else:
            logger.info(f"Feature stats loaded from {path} (horse_recent_form: なし)")
        return instance

    @classmethod
    def build_stats_from_training_csv(cls, csv_path: Path | str, stats_path: Path | str) -> "FeatureEngineer":
        """
        学習用CSVから集計統計を再構築して保存する。
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded training CSV: {len(df)} rows from {csv_path}")

        instance = cls(pd.DataFrame())

        if "father" in df.columns and "sire_win_rate" in df.columns:
            instance._sire_win_rate = (
                df[df["father"].notna() & (df["father"] != "")]
                .groupby("father")["sire_win_rate"].mean()
                .rename("sire_win_rate")
            )

        if "mother_father" in df.columns and "bms_win_rate" in df.columns:
            instance._bms_win_rate = (
                df[df["mother_father"].notna() & (df["mother_father"] != "")]
                .groupby("mother_father")["bms_win_rate"].mean()
                .rename("bms_win_rate")
            )

        jockey_cols = ["jockey_id", "course_type_code", "distance_bin_code",
                       "jockey_win_rate", "jockey_place_rate", "jockey_runs"]
        if all(c in df.columns for c in jockey_cols):
            course_map = {0: "芝", 1: "ダート"}
            dist_map   = {0: "短距離", 1: "マイル", 2: "中距離", 3: "長距離"}
            cols_to_select = jockey_cols + (["race_id"] if "race_id" in df.columns else [])
            tmp = df[cols_to_select].copy().dropna(subset=jockey_cols)
            tmp["jockey_id"] = tmp["jockey_id"].astype(str).apply(
                lambda x: str(int(x)) if x.strip().isdigit() else x.strip()
            )
            tmp["course_type"]  = tmp["course_type_code"].map(course_map)
            tmp["distance_bin"] = tmp["distance_bin_code"].astype(int).map(dist_map)
            if "race_id" in tmp.columns:
                tmp = tmp.sort_values("race_id")
                instance._jockey_course_stats = (
                    tmp.groupby(["jockey_id", "course_type", "distance_bin"])
                    .last()
                    .reset_index()
                    [["jockey_id", "course_type", "distance_bin",
                      "jockey_win_rate", "jockey_place_rate", "jockey_runs"]]
                )
            else:
                instance._jockey_course_stats = (
                    tmp.groupby(["jockey_id", "course_type", "distance_bin"])
                    .agg(
                        jockey_win_rate=("jockey_win_rate", "mean"),
                        jockey_place_rate=("jockey_place_rate", "mean"),
                        jockey_runs=("jockey_runs", "mean"),
                    )
                    .reset_index()
                )

        instance.save_stats(stats_path)
        return instance

    # ------------------------------------------------------------------
    # モデル入力カラム一覧
    # ------------------------------------------------------------------

    FEATURE_COLUMNS = [
        # 既存特徴量
        "frame_number",
        "horse_number",
        "age",
        "weight_carried",
        "course_type_code",
        "distance",
        "ground_condition_code",
        "weather_code",
        "trainer_win_rate",
        "trainer_place_rate",
        "jockey_win_rate",
        "jockey_place_rate",
        "jockey_runs",
        "recent_avg_pos",
        "recent_avg_last3f",
        "recent_top3_rate",
        # 改善④: レースクラス
        "race_class_code",
        # 改善⑤: 会場・馬場交互作用
        "venue_code",
        "venue_ground_code",
        # 改善②: 出走頭数
        "n_entries",
        # 改善⑥: 市場集中度
        "market_hhi",
        # 改善⑦: 前走パフォーマンス
        "prev_margin",
        "prev_last3f_rank_norm",
        # 新規追加特徴量 (2026-04)
        "horse_dist_win_rate",       # ② 距離帯適性（馬別・≥5走フィルタ）
        "recent_pos_trend",          # ③ 着順トレンド（負=改善, 正=悪化）
        "recent_last3f_trend",       # ③ 上がり3Fトレンド（負=改善）
        "class_change",              # ④ クラス変動（負=降級, 0=同級, 正=昇級）
    ]
