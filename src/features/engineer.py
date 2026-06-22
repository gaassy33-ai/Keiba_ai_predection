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
        self._jockey_trainer_stats: pd.DataFrame | None = None  # 騎手×調教師コンビ勝率（≥10走）
        self._horse_career_stats: pd.DataFrame | None = None    # 馬キャリア複勝率（≥10走）

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

        # 上がり3F（last_3f 列に正しく格納）
        # ※ 2024年以前のスクレイパーバグで odds/last_3f が逆転したデータが
        #   残っている場合は再収集が必要。スクレイパーは修正済み。
        if "last_3f" in df.columns:
            df["last_3f_num"] = pd.to_numeric(df["last_3f"], errors="coerce")
        else:
            df["last_3f_num"] = np.nan

        # 単勝オッズ（odds 列に正しく格納）
        if "odds" in df.columns:
            df["tansho_odds_num"] = pd.to_numeric(df["odds"], errors="coerce")

        # 馬体重の数値化（例: "480(+2)" → 480、前走比変化は "(+2)" から取得）
        if "horse_weight" in df.columns:
            hw_str = df["horse_weight"].astype(str)
            # 馬体重本体（3〜4桁整数）
            df["horse_weight_num"] = pd.to_numeric(
                hw_str.str.extract(r"^(\d{3,4})")[0], errors="coerce"
            )
            # 前走比体重変化（例: "(+2)" → +2、"(-4)" → -4）
            df["horse_weight_diff"] = pd.to_numeric(
                hw_str.str.extract(r"\(([+-]?\d+)\)")[0], errors="coerce"
            )
        else:
            df["horse_weight_num"]  = np.nan
            df["horse_weight_diff"] = np.nan

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

        # 障害レース除外: distance=0 or ≥2750m は障害競走のため平地モデルから除外
        # （distance=0: スクレイパーが障害ページをパース失敗、2750m以上: 障害特有の距離帯）
        if "distance" in df.columns:
            dist_num = pd.to_numeric(df["distance"], errors="coerce")
            df = df[(dist_num > 0) & (dist_num < 2750)].copy()

        # 上がり3F 異常値クレンジング: 60秒超は明らかなスクレイパーパースエラー → NaN化
        if "last_3f_num" in df.columns:
            df.loc[df["last_3f_num"] > 60, "last_3f_num"] = np.nan

        # コーナー通過順位の最終コーナー値をパース（例: "3-3-3-2" → 2.0）
        # 後段の _add_recent_form でレース内正規化して使用する
        if "corner_positions" in df.columns:
            def _extract_last_corner(s):
                if not isinstance(s, str) or not s.strip():
                    return np.nan
                parts = [p.strip() for p in s.strip().split("-") if p.strip()]
                try:
                    return float(parts[-1]) if parts else np.nan
                except (ValueError, IndexError):
                    return np.nan
            df["_last_corner_raw"] = df["corner_positions"].apply(_extract_last_corner)
        else:
            df["_last_corner_raw"] = np.nan

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

        # 騎手×調教師コンビ勝率（≥10走フィルタで信頼性確保）
        if "jockey_id" in h.columns and "trainer_name" in h.columns:
            combo_grp = (
                h[h["trainer_name"].replace("", pd.NA).notna()]
                .groupby(["jockey_id", "trainer_name"])
                .agg(_jt_wins=("is_win", "sum"), _jt_n=("is_win", "count"))
                .reset_index()
            )
            combo_grp["jockey_trainer_win_rate"] = np.where(
                combo_grp["_jt_n"] >= 10,
                combo_grp["_jt_wins"] / combo_grp["_jt_n"],
                np.nan,
            )
            self._jockey_trainer_stats = combo_grp[
                ["jockey_id", "trainer_name", "jockey_trainer_win_rate"]
            ]
            logger.debug(f"  jockey_trainer_stats: {len(self._jockey_trainer_stats):,} コンビ")

        # 馬キャリア複勝率（≥10走フィルタ）
        if "horse_id" in h.columns:
            horse_grp = (
                h.groupby("horse_id")
                .agg(_h_placed=("is_placed", "sum"), _h_n=("is_placed", "count"))
                .reset_index()
            )
            horse_grp["horse_career_top3_rate"] = np.where(
                horse_grp["_h_n"] >= 10,
                horse_grp["_h_placed"] / horse_grp["_h_n"],
                np.nan,
            )
            self._horse_career_stats = horse_grp[["horse_id", "horse_career_top3_rate"]]
            logger.debug(f"  horse_career_stats: {len(self._horse_career_stats):,} 頭")

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
        race_date=None,  # pd.Timestamp | str | None — days_since_last_race 計算用
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
            # log(odds) を使用: オッズは必ず≥1.0 なので log1p より自然な変換
            # 下限 1.01 で clip して log(0) を防ぐ
            df["odds_log"] = np.log(odds_capped.clip(lower=1.01))
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

        # ── 産駒勝率（父別・母父別）────────────────────────────────────
        # precompute_aggregations() で計算済みの産駒統計を df に結合する
        if self._sire_win_rate is not None and "father" in df.columns:
            swr = self._sire_win_rate.reset_index()
            df = df.merge(swr, on="father", how="left")
        else:
            df["sire_win_rate"] = np.nan

        if self._bms_win_rate is not None and "mother_father" in df.columns:
            bwr = self._bms_win_rate.reset_index()
            df = df.merge(bwr, on="mother_father", how="left")
        else:
            df["bms_win_rate"] = np.nan

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

        # ── 騎手×調教師コンビ勝率 ────────────────────────────────────────
        if self._jockey_trainer_stats is not None and "trainer_name" in df.columns:
            jt = self._jockey_trainer_stats.copy()
            # jockey_id を正規化して既存の df["jockey_id"] と型を合わせる
            jt["jockey_id"] = jt["jockey_id"].astype(str).str.strip().apply(
                lambda x: str(int(x)) if x.isdigit() else x
            )
            df = df.merge(jt, on=["jockey_id", "trainer_name"], how="left")
        else:
            df["jockey_trainer_win_rate"] = np.nan

        # ── 直近成績（改善⑦含む）────────────────────────────────────
        df = self._add_recent_form(df)

        # ── 前走からの経過日数 ─────────────────────────────────────────
        # race_date が渡された場合のみ計算（推論時は当日日付、学習時は各レース日）
        if race_date is not None and "last_race_date" in df.columns:
            _rd = pd.to_datetime(race_date, errors="coerce")
            if pd.notna(_rd):
                df["days_since_last_race"] = (
                    _rd - pd.to_datetime(df["last_race_date"], errors="coerce")
                ).dt.days.clip(lower=0, upper=365)
            else:
                df["days_since_last_race"] = np.nan
        else:
            df["days_since_last_race"] = np.nan

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

        # ── 馬キャリア複勝率 ──────────────────────────────────────────────
        if self._horse_career_stats is not None:
            df = df.merge(self._horse_career_stats, on="horse_id", how="left")
        else:
            df["horse_career_top3_rate"] = np.nan

        # ── フィールド平均騎手勝率（レース難易度の代理変数）──────────────
        # jockey_win_rate 列はすでにマージ済みのため、そこから平均を計算
        if "jockey_win_rate" in df.columns:
            df["field_avg_jockey_win_rate"] = (
                pd.to_numeric(df["jockey_win_rate"], errors="coerce").mean()
            )
        else:
            df["field_avg_jockey_win_rate"] = np.nan

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
            "prev_horse_weight",        # ⑤ 前走馬体重 (kg)
            "horse_weight_change",      # ⑤ 前走比体重変化 (kg差)
            "last_race_date",           # 前走日付（days_since_last_race 計算用）
            "recent_avg_last3f_rank",   # 新規: 直近N走の上がり3Fランク平均（0=最速, 1=最遅）
            "prev_corner_pos_norm",     # 新規: 前走最終コーナー通過順位正規化（0=先頭, 1=最後尾）
        ]

        # ── 推論パス: from_stats() でロード済みの統計を使用 ───────────
        if self.history.empty:
            if self._horse_recent_form is not None and not self._horse_recent_form.empty:
                hrf = self._horse_recent_form.reset_index() if "horse_id" not in self._horse_recent_form.columns else self._horse_recent_form
                # feature_stats.pkl が古いバージョンの場合、一部カラムが欠損している可能性がある
                # 存在するカラムのみ merge し、欠損カラムは NaN で補完する（後方互換性）
                available_cols = [c for c in COLS if c in hrf.columns]
                df = df.merge(hrf[["horse_id"] + available_cols], on="horse_id", how="left")
                # 欠損カラムを NaN で補完
                for c in COLS:
                    if c not in df.columns:
                        df[c] = np.nan
            else:
                for c in COLS:
                    df[c] = np.nan
            return df

        # ── 学習パス: history から計算 ───────────────────────────────
        h = self.history

        # 上がり3F の race 内ランクを事前計算（改善⑦）
        h = h.copy()
        if "last_3f_num" in h.columns and "race_id" in h.columns:
            h["last3f_rank_in_race"] = h.groupby("race_id")["last_3f_num"].rank(
                ascending=True, method="min", na_option="keep"
            )
            h["last3f_n_in_race"] = h.groupby("race_id")["last_3f_num"].transform("count")
            h["last3f_rank_norm"] = h["last3f_rank_in_race"] / h["last3f_n_in_race"]
        else:
            h["last3f_rank_norm"] = np.nan

        # 最終コーナー通過順位のレース内正規化（0=先頭, 1=最後尾）
        if "_last_corner_raw" in h.columns and "race_id" in h.columns:
            h["_corner_max"] = h.groupby("race_id")["_last_corner_raw"].transform("max")
            h["_corner_min"] = h.groupby("race_id")["_last_corner_raw"].transform("min")
            h["last_corner_pos_norm"] = np.where(
                h["_corner_max"] > h["_corner_min"],
                (h["_last_corner_raw"] - h["_corner_min"]) / (h["_corner_max"] - h["_corner_min"]),
                np.where(h["_last_corner_raw"].notna(), 0.5, np.nan),
            )
        else:
            h["last_corner_pos_norm"] = np.nan

        # ソートキー: race_date がある場合は使用、なければ pseudo_date（YYYY+KK+DD）で代替
        # race_id[4:6] は会場コード（01-10）で時系列でないため raw race_id でのソートは不正確
        # pseudo_date = race_id[0:4] + race_id[6:10]（年 + 回数 + 日）で会場コードを除外する
        if "race_date" not in h.columns or not h["race_date"].notna().any():
            h["_pseudo_date"] = h["race_id"].astype(str).apply(
                lambda x: x[:4] + x[6:10] if len(x) >= 10 else x
            )
            _sort_key = "_pseudo_date"
        else:
            _sort_key = "race_date"

        # 馬ごとの直近 N 走（新しい順）
        recent = (
            h.sort_values(_sort_key, ascending=False)
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
                recent_avg_last3f_rank=("last3f_rank_norm", "mean"),  # 新規: 速度ランク平均
            )
            .reset_index()
        )

        # 前走（最新1走）の着差・上がり3Fランク（改善⑦）
        all_sorted = h.sort_values(
            ["horse_id", _sort_key], ascending=[True, False]
        ).copy()
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
        # ⑤ 前走馬体重（推論時はこの値を使用、当日体重は計量前なので不明）
        if "horse_weight_num" in prev_race.columns:
            prev_cols.append("horse_weight_num")
        # 前走日付（days_since_last_race 計算用）
        if "race_date" in prev_race.columns:
            prev_cols.append("race_date")
        # 新規: 前走最終コーナー通過順位（正規化済み）
        if "last_corner_pos_norm" in prev_race.columns:
            prev_cols.append("last_corner_pos_norm")

        prev_df = prev_race[prev_cols].rename(columns={
            "margin_num": "prev_margin",
            "last3f_rank_norm": "prev_last3f_rank_norm",
            "finish_pos_num": "_prev_pos",
            "last_3f_num": "_prev_3f",
            "race_class_code_hist": "prev_race_class_code",
            "horse_weight_num": "prev_horse_weight",
            "race_date": "last_race_date",
            "last_corner_pos_norm": "prev_corner_pos_norm",  # 新規
        })

        # ⑤ 前々走（rank=2）の馬体重を取得して体重変化を計算
        prev_prev_race = all_sorted[all_sorted["_recent_rank"] == 2].copy()
        if "horse_weight_num" in prev_prev_race.columns:
            pp_weight = prev_prev_race[["horse_id", "horse_weight_num"]].rename(
                columns={"horse_weight_num": "_pp_weight"}
            )
            weight_change_base = prev_df[["horse_id"] + (["prev_horse_weight"] if "prev_horse_weight" in prev_df.columns else [])].merge(
                pp_weight, on="horse_id", how="left"
            )
            if "prev_horse_weight" in weight_change_base.columns and "_pp_weight" in weight_change_base.columns:
                weight_change_base["horse_weight_change"] = (
                    pd.to_numeric(weight_change_base["prev_horse_weight"], errors="coerce")
                    - pd.to_numeric(weight_change_base["_pp_weight"], errors="coerce")
                )
            else:
                weight_change_base["horse_weight_change"] = np.nan
            weight_change_df = weight_change_base[["horse_id", "horse_weight_change"]]
        else:
            weight_change_df = pd.DataFrame({"horse_id": prev_df["horse_id"], "horse_weight_change": np.nan})

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
        df = df.merge(weight_change_df, on="horse_id", how="left")

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
            race_date_val = None  # build_entry_features に渡す日付（初期化）
            if "race_date" in h.columns and not race_entries.empty:
                race_date_val = race_entries["race_date"].iloc[0]
                if pd.notna(race_date_val):
                    history_before = h[h["race_date"] < race_date_val]
                else:
                    race_date_val = None
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

            # 障害レース除外: course_type 未設定、distance=0、または2750m以上は障害競走
            if not str(course_type) or distance == 0 or distance >= 2750:
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

            # 改善⑥: 市場オッズ（HHI / odds_log / popularity_rank_norm 計算用）
            # ★カラムずれ対応:
            #   train_results.csv (新CSV): "odds" 列が実際の単勝オッズ
            #   test_results.csv  (旧CSV): "last_3f" 列が実際の単勝オッズ
            # → "odds" 列の最小値が 15 未満なら正常な単勝オッズ列と判定し使用。
            #   そうでなければ "last_3f" にフォールバック（旧CSV互換）。
            if "odds" in race_entries.columns:
                _odds_raw = pd.to_numeric(race_entries["odds"], errors="coerce")
                if _odds_raw.dropna().min() < 15.0:
                    # 正常な単勝オッズ列（train_results.csv）
                    entry_df["odds"] = _odds_raw.values
                elif "last_3f" in race_entries.columns:
                    # 旧CSV: last_3f が実際の単勝オッズ
                    entry_df["odds"] = pd.to_numeric(
                        race_entries["last_3f"].values, errors="coerce"
                    )
            elif "last_3f" in race_entries.columns:
                # odds 列が存在しない場合の後方互換
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
                    race_date=race_date_val,  # None の場合は days_since_last_race が NaN になる
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

    def save_stats(self, path: Path | str, extra_history_df: pd.DataFrame | None = None) -> None:
        """集計統計をファイルに保存する（学習後に呼び出す）。

        Args:
            extra_history_df: horse_recent_form 計算のみに使う追加履歴
                              (例: test_results_new.csv)。モデル学習データとは独立。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 馬別直近成績: extra_history_df が渡された場合は合算して計算
        if extra_history_df is not None and not extra_history_df.empty:
            extra_preprocessed = self._preprocess_history(extra_history_df)
            combined = pd.concat([self.history, extra_preprocessed], ignore_index=True)
            # 重複除去（同一 race_id × horse_id）
            if "race_id" in combined.columns and "horse_id" in combined.columns:
                combined = combined.drop_duplicates(subset=["race_id", "horse_id"], keep="last")
            _tmp = FeatureEngineer(pd.DataFrame())  # 空DataFrameで初期化し属性を保証
            _tmp.history = combined               # historyのみ差し替え
            horse_recent_form = _tmp._compute_horse_recent_form_for_inference()
            logger.info(
                f"  horse_recent_form: 全履歴 {len(combined):,} 行（訓練 {len(self.history):,} + 追加 {len(extra_preprocessed):,}）で計算"
            )
        else:
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
            "jockey_trainer_stats": self._jockey_trainer_stats,  # 新規
            "horse_career_stats":   self._horse_career_stats,    # 新規
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
        h = h.copy()
        if "last_3f_num" in h.columns and "race_id" in h.columns:
            h["last3f_rank_in_race"] = h.groupby("race_id")["last_3f_num"].rank(
                ascending=True, method="min", na_option="keep"
            )
            h["last3f_n_in_race"] = h.groupby("race_id")["last_3f_num"].transform("count")
            h["last3f_rank_norm"] = h["last3f_rank_in_race"] / h["last3f_n_in_race"]
        else:
            h["last3f_rank_norm"] = np.nan

        # 最終コーナー通過順位のレース内正規化（0=先頭, 1=最後尾）
        if "_last_corner_raw" in h.columns and "race_id" in h.columns:
            h["_corner_max"] = h.groupby("race_id")["_last_corner_raw"].transform("max")
            h["_corner_min"] = h.groupby("race_id")["_last_corner_raw"].transform("min")
            h["last_corner_pos_norm"] = np.where(
                h["_corner_max"] > h["_corner_min"],
                (h["_last_corner_raw"] - h["_corner_min"]) / (h["_corner_max"] - h["_corner_min"]),
                np.where(h["_last_corner_raw"].notna(), 0.5, np.nan),
            )
        else:
            h["last_corner_pos_norm"] = np.nan

        # ソートキー: race_date がある場合は使用、なければ pseudo_date（YYYY+KK+DD）で代替
        # race_id[4:6] は会場コード（01-10）で時系列でないため raw race_id でのソートは不正確
        if "race_date" not in h.columns or not h["race_date"].notna().any():
            h["_pseudo_date"] = h["race_id"].astype(str).apply(
                lambda x: x[:4] + x[6:10] if len(x) >= 10 else x
            )
            _sort_key_inf = "_pseudo_date"
        else:
            _sort_key_inf = "race_date"

        recent = (
            h.sort_values(_sort_key_inf, ascending=False)
            .groupby("horse_id")
            .head(self.RECENT_N)
        )
        agg = (
            recent.groupby("horse_id")
            .agg(
                recent_avg_pos=("finish_pos_num", "mean"),
                recent_avg_last3f=("last_3f_num", "mean"),
                recent_top3_rate=("is_placed", "mean"),
                recent_avg_last3f_rank=("last3f_rank_norm", "mean"),  # 新規
            )
            .reset_index()
        )

        all_sorted_inf = h.sort_values(
            ["horse_id", _sort_key_inf], ascending=[True, False]
        ).copy()
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
        # ⑤ 前走馬体重
        if "horse_weight_num" in prev_race.columns:
            prev_cols.append("horse_weight_num")
        # 前走日付（days_since_last_race 計算用）
        if "race_date" in prev_race.columns:
            prev_cols.append("race_date")
        # 新規: 前走最終コーナー通過順位
        if "last_corner_pos_norm" in prev_race.columns:
            prev_cols.append("last_corner_pos_norm")

        prev_df = prev_race[prev_cols].rename(columns={
            "margin_num": "prev_margin",
            "last3f_rank_norm": "prev_last3f_rank_norm",
            "finish_pos_num": "_prev_pos",
            "last_3f_num": "_prev_3f",
            "race_class_code_hist": "prev_race_class_code",
            "horse_weight_num": "prev_horse_weight",
            "race_date": "last_race_date",
            "last_corner_pos_norm": "prev_corner_pos_norm",  # 新規
        })

        # ⑤ 前々走（rank=2）の馬体重から体重変化を計算
        prev_prev_inf = all_sorted_inf[all_sorted_inf["_recent_rank"] == 2].copy()
        if "horse_weight_num" in prev_prev_inf.columns:
            pp_wt_inf = prev_prev_inf[["horse_id", "horse_weight_num"]].rename(
                columns={"horse_weight_num": "_pp_weight"}
            )
            wt_base = prev_df[["horse_id"] + (["prev_horse_weight"] if "prev_horse_weight" in prev_df.columns else [])].merge(
                pp_wt_inf, on="horse_id", how="left"
            )
            if "prev_horse_weight" in wt_base.columns and "_pp_weight" in wt_base.columns:
                wt_base["horse_weight_change"] = (
                    pd.to_numeric(wt_base["prev_horse_weight"], errors="coerce")
                    - pd.to_numeric(wt_base["_pp_weight"], errors="coerce")
                )
            else:
                wt_base["horse_weight_change"] = np.nan
            wt_change_inf = wt_base[["horse_id", "horse_weight_change"]]
        else:
            wt_change_inf = pd.DataFrame({"horse_id": prev_df["horse_id"], "horse_weight_change": np.nan})

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
        result = result.merge(wt_change_inf, on="horse_id", how="left")

        for c in ("prev_margin", "prev_last3f_rank_norm", "prev_race_class_code",
                  "recent_pos_trend", "recent_last3f_trend",
                  "prev_horse_weight", "horse_weight_change", "last_race_date",
                  "recent_avg_last3f_rank", "prev_corner_pos_norm"):  # 新規
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
        instance._sire_win_rate         = stats.get("sire_win_rate")
        instance._bms_win_rate          = stats.get("bms_win_rate")
        instance._jockey_course_stats   = stats.get("jockey_course_stats")
        instance._trainer_stats         = stats.get("trainer_stats")
        instance._horse_recent_form     = stats.get("horse_recent_form")
        instance._jockey_venue_stats    = stats.get("jockey_venue_stats")
        instance._horse_ground_stats    = stats.get("horse_ground_stats")
        instance._horse_dist_stats      = stats.get("horse_dist_stats")
        instance._jockey_trainer_stats  = stats.get("jockey_trainer_stats")  # 新規
        instance._horse_career_stats    = stats.get("horse_career_stats")    # 新規
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
        # ── 基本情報 ───────────────────────────────────────────────────
        "frame_number",
        "horse_number",
        "age",
        "weight_carried",
        # ── レース環境 ─────────────────────────────────────────────────
        "course_type_code",
        "distance",
        "ground_condition_code",
        "weather_code",
        # ── 厩舎・騎手 ─────────────────────────────────────────────────
        "trainer_win_rate",
        "trainer_place_rate",
        "jockey_win_rate",
        "jockey_place_rate",
        "jockey_runs",
        "jockey_trainer_win_rate",   # 新規(2026-04): 騎手×調教師コンビ勝率（≥10走）
        # ── 直近成績 ───────────────────────────────────────────────────
        "recent_avg_pos",
        "recent_avg_last3f",
        "recent_top3_rate",
        "recent_avg_last3f_rank",    # 新規(2026-04): 直近上がり3Fランク平均（0=最速, 1=最遅）
        "recent_pos_trend",
        "recent_last3f_trend",
        # ── 前走パフォーマンス ─────────────────────────────────────────
        "prev_margin",
        "prev_last3f_rank_norm",
        "prev_corner_pos_norm",      # 新規(2026-04): 前走最終コーナー通過順位（0=先頭, 1=最後尾）
        # ── クラス・会場 ───────────────────────────────────────────────
        "race_class_code",
        "class_change",
        "venue_code",
        "venue_ground_code",
        "n_entries",
        # ── 市場情報（market_hhi は odds_log と高相関のため削除 2026-04-26） ──
        "odds_log",
        "popularity_rank_norm",
        # ── フィールド・馬個体 ─────────────────────────────────────────
        "field_avg_jockey_win_rate", # 新規(2026-04): フィールド平均騎手勝率（レース強度指標）
        "horse_career_top3_rate",    # 新規(2026-04): 馬キャリア複勝率（≥10走、安定性指標）
        "horse_dist_win_rate",
        "horse_ground_win_rate",
        # ── 馬体重・状態 ───────────────────────────────────────────────
        "prev_horse_weight",
        "horse_weight_change",
        "is_3yo",
        "days_since_last_race",
    ]
