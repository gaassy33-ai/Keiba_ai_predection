"""
特徴量エンジニアリング。

取得した生データを LightGBM に渡せる形に変換する。
主な特徴量:
- 産駒勝率（父・母父別）
- ジョッキーのコース適性（芝/ダート × 距離帯）
- 上がり3F 偏差値
- 馬場状態・天候コード
- 直近 n 走の着順傾向
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
    >>> features = fe.build_entry_features(entry_df, race_info)
    """

    # 直近何走を参照するか
    RECENT_N = 5

    def __init__(self, history_df: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        history_df : pd.DataFrame
            fetch_bulk_results() で取得した過去成績の全データ
        """
        self.history = self._preprocess_history(history_df)
        self._sire_win_rate: pd.Series | None = None
        self._bms_win_rate: pd.Series | None = None
        self._jockey_course_stats: pd.DataFrame | None = None

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

        # 上がり3F を数値化
        df["last_3f_num"] = pd.to_numeric(df["last_3f"], errors="coerce")

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

    # ------------------------------------------------------------------
    # 集計特徴量の事前計算
    # ------------------------------------------------------------------

    def precompute_aggregations(self) -> None:
        """産駒勝率・ジョッキーコース適性を事前計算する。"""
        h = self.history
        if h.empty:
            logger.warning("History is empty. Skipping aggregation precomputation.")
            return

        # 産駒勝率（父別）— father 列が存在し空でない場合のみ
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

        # ジョッキーのコース×距離帯適性（勝率・連対率）
        # course_type 列が必要（fetch_result_and_meta でインライン付加済み）
        if "course_type" not in h.columns or h["course_type"].replace("", pd.NA).isna().all():
            logger.warning("course_type column missing; skipping jockey course stats.")
            logger.info("Aggregation precomputation done (partial).")
            return

        h = h.copy()
        h["distance_bin"] = pd.cut(
            pd.to_numeric(h.get("distance", pd.Series(dtype=float)), errors="coerce"),
            bins=[0, 1400, 1800, 2200, 9999],
            labels=["短距離", "マイル", "中距離", "長距離"],
        )
        self._jockey_course_stats = (
            h.groupby(["jockey_id", "course_type", "distance_bin"], observed=True)
            .agg(
                jockey_win_rate=("is_win", "mean"),
                jockey_place_rate=("is_placed", "mean"),
                jockey_runs=("is_win", "count"),
            )
            .reset_index()
        )
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
    ) -> pd.DataFrame:
        """
        出走馬リスト (entry_df) に特徴量を結合して返す。

        Parameters
        ----------
        entry_df : pd.DataFrame
            columns: [horse_id, horse_number, frame_number, sex, age,
                      weight_carried, jockey_id, father, mother_father, ...]
        course_type : str
            "芝" or "ダート"
        distance : int
            レース距離（メートル）
        ground_condition_code : int
            WeatherFetcher から取得した数値
        weather_code : int
            WeatherFetcher から取得した数値

        Returns
        -------
        pd.DataFrame
            モデルへの入力に使える特徴量 DataFrame
        """
        df = entry_df.copy()

        # レース環境特徴量（全馬共通）
        df["course_type_code"] = 0 if course_type == "芝" else 1
        df["distance"] = distance
        df["distance_bin_code"] = pd.cut(
            [distance], bins=[0, 1400, 1800, 2200, 9999],
            labels=[0, 1, 2, 3]
        )[0]
        df["ground_condition_code"] = ground_condition_code
        df["weather_code"] = weather_code

        # 産駒勝率
        if self._sire_win_rate is not None:
            df = df.merge(
                self._sire_win_rate.reset_index(),
                left_on="father", right_on="father", how="left"
            )
        else:
            df["sire_win_rate"] = np.nan

        if self._bms_win_rate is not None:
            df = df.merge(
                self._bms_win_rate.reset_index(),
                left_on="mother_father", right_on="mother_father", how="left"
            )
        else:
            df["bms_win_rate"] = np.nan

        # ジョッキー適性
        if self._jockey_course_stats is not None:
            distance_label = pd.cut(
                [distance], bins=[0, 1400, 1800, 2200, 9999],
                labels=["短距離", "マイル", "中距離", "長距離"]
            )[0]
            jockey_filtered = self._jockey_course_stats[
                (self._jockey_course_stats["course_type"] == course_type)
                & (self._jockey_course_stats["distance_bin"] == distance_label)
            ][["jockey_id", "jockey_win_rate", "jockey_place_rate", "jockey_runs"]].copy()
            # jockey_id を正規化（先頭ゼロを除去）して型を統一
            # 学習CSV: int 1192 → "1192"
            # スクレイプ: str "01192" → int 1192 → "1192"
            def _norm_jid(s: pd.Series) -> pd.Series:
                return s.astype(str).apply(
                    lambda x: str(int(x)) if x.strip().isdigit() else x.strip()
                )
            df["jockey_id"] = _norm_jid(df["jockey_id"])
            jockey_filtered["jockey_id"] = _norm_jid(jockey_filtered["jockey_id"])
            df = df.merge(jockey_filtered, on="jockey_id", how="left")
        else:
            df["jockey_win_rate"] = np.nan
            df["jockey_place_rate"] = np.nan
            df["jockey_runs"] = np.nan

        # 直近 RECENT_N 走の傾向
        df = self._add_recent_form(df)

        return df

    def _add_recent_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """各馬の直近 N 走着順平均・上がり3F 平均を付加。

        entry_df に既に両列が存在する場合（推論パスで事前スクレイプ済み）は
        再計算をスキップする。
        """
        # 推論時: スクレイプで事前計算済みの値がある場合はそのまま返す
        if "recent_avg_pos" in df.columns and "recent_avg_last3f" in df.columns:
            return df

        h = self.history
        if h.empty:
            df["recent_avg_pos"] = np.nan
            df["recent_avg_last3f"] = np.nan
            return df

        agg = (
            h.sort_values("race_id", ascending=False)
            .groupby("horse_id")
            .head(self.RECENT_N)
            .groupby("horse_id")
            .agg(
                recent_avg_pos=("finish_pos_num", "mean"),
                recent_avg_last3f=("last_3f_num", "mean"),
            )
            .reset_index()
        )
        return df.merge(agg, on="horse_id", how="left")

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

        Parameters
        ----------
        race_meta_df : pd.DataFrame
            fetch_bulk_race_meta() の出力。
            必須カラム: [race_id, course_type, distance,
                         ground_condition_code, weather_code]
        output_path : str | None
            CSV 保存先。指定時はファイルに保存する。

        Returns
        -------
        pd.DataFrame
            FEATURE_COLUMNS + ['is_win', 'race_id'] を含む学習用 DataFrame。
        """
        h = self.history
        if h.empty:
            raise ValueError("history_df が空です。fetch_bulk_results() を先に実行してください。")

        required_meta_cols = {"race_id", "course_type", "distance",
                              "ground_condition_code", "weather_code"}
        missing = required_meta_cols - set(race_meta_df.columns)
        if missing:
            raise ValueError(f"race_meta_df に必須カラムが不足しています: {missing}")

        # race_id は文字列として統一
        h = h.copy()
        h["race_id"] = h["race_id"].astype(str)
        race_meta_df = race_meta_df.copy()
        race_meta_df["race_id"] = race_meta_df["race_id"].astype(str)

        # race_date があれば日付昇順でソート（データリーク防止に重要）
        # なければ race_id 辞書順でフォールバック（精度は落ちるが動作は継続）
        if "race_date" in race_meta_df.columns and race_meta_df["race_date"].notna().any():
            race_meta_df["race_date"] = pd.to_datetime(
                race_meta_df["race_date"], errors="coerce"
            )
            # history にも race_date を結合
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
            # このレースより古いデータのみを使う（data leakage 防止）
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

            # レースメタ情報
            meta_row = race_meta_df[race_meta_df["race_id"] == race_id].iloc[0]
            course_type = meta_row["course_type"]
            distance = int(meta_row["distance"])
            ground_condition_code = int(meta_row["ground_condition_code"])
            weather_code = int(meta_row["weather_code"])

            # course_type / distance が空のレースはスキップ
            if not str(course_type) or distance == 0:
                continue

            # entry_df として整形（weight_carried_num がない場合は weight_carried を使う）
            wt_col = "weight_carried_num" if "weight_carried_num" in race_entries.columns else "weight_carried"
            base_cols = ["horse_id", "horse_name", "horse_number", "frame_number", "jockey_id"]
            for c in ("sex", "age", wt_col):
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

            # 血統情報の付加（history に father / mother_father カラムがある場合）
            for col in ("father", "mother_father"):
                if col in race_entries.columns:
                    entry_df[col] = race_entries[col].values
                else:
                    entry_df[col] = ""

            # ルックバック用の一時 FeatureEngineer を作成
            tmp_fe = FeatureEngineer(history_before)
            tmp_fe.precompute_aggregations()

            try:
                feat_df = tmp_fe.build_entry_features(
                    entry_df=entry_df,
                    course_type=course_type,
                    distance=distance,
                    ground_condition_code=ground_condition_code,
                    weather_code=weather_code,
                )
            except Exception as e:
                logger.warning(f"Skipping race_id={race_id}: feature build failed ({e})")
                continue

            # ラベルを結合
            labels = race_entries[["horse_id", "is_win"]].set_index("horse_id")
            feat_df = feat_df.set_index("horse_id") if "horse_id" in feat_df.columns else feat_df
            feat_df["is_win"] = feat_df.index.map(labels["is_win"])
            feat_df["race_id"] = race_id
            feat_df = feat_df.reset_index()

            all_rows.append(feat_df)

            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{len(target_race_ids)} races processed")

        if not all_rows:
            logger.warning("No training rows generated.")
            return pd.DataFrame()

        result = pd.concat(all_rows, ignore_index=True)

        # 学習に不要な行（is_win が NaN）を除外
        result = result.dropna(subset=["is_win"])
        result["is_win"] = result["is_win"].astype(int)

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
        scraper,  # NetkeibaScraper インスタンス
        output_path: str | None = None,
        checkpoint_prefix: str | None = None,
    ) -> pd.DataFrame:
        """
        race_id リストからスクレイピング → 特徴量生成 → 学習CSV生成まで一括実行する。
        1レース1リクエストで着順+メタを取得する最適化版を使用。

        Parameters
        ----------
        race_ids : list[str]
            collect_race_ids_for_period() で取得したリスト
        scraper : NetkeibaScraper
        output_path : str | None
            学習CSV の保存先
        checkpoint_prefix : str | None
            中間ファイルのプレフィックス（例: "data/raw/checkpoint"）
            再実行時にここから続きを再開できる

        Returns
        -------
        pd.DataFrame
            build_training_dataset() と同じ形式
        """
        logger.info(f"Step 1/2: Fetching results+meta for {len(race_ids)} races (1 req/race)...")
        history_df, race_meta_df = scraper.fetch_bulk_results_and_meta(
            race_ids, checkpoint_path=checkpoint_prefix
        )

        logger.info(f"  → {len(history_df)} horse records, {len(race_meta_df)} races")
        logger.info("Step 2/2: Building training dataset with lookback features...")
        fe = cls(history_df)
        fe.precompute_aggregations()
        result = fe.build_training_dataset(race_meta_df, output_path=output_path)

        # 推論用集計統計を保存
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
        stats = {
            "sire_win_rate": self._sire_win_rate,
            "bms_win_rate": self._bms_win_rate,
            "jockey_course_stats": self._jockey_course_stats,
        }
        with open(path, "wb") as f:
            pickle.dump(stats, f)
        logger.info(f"Feature stats saved to {path}")

    @classmethod
    def from_stats(cls, path: Path | str) -> "FeatureEngineer":
        """
        保存済み統計から推論用 FeatureEngineer を生成する。
        history_df の読み込み・集計処理が不要になる。
        """
        path = Path(path)
        instance = cls(pd.DataFrame())  # history は空でOK
        if not path.exists():
            logger.warning(f"Stats file not found: {path}. Using empty stats.")
            return instance
        with open(path, "rb") as f:
            stats = pickle.load(f)
        instance._sire_win_rate = stats.get("sire_win_rate")
        instance._bms_win_rate = stats.get("bms_win_rate")
        instance._jockey_course_stats = stats.get("jockey_course_stats")
        logger.info(f"Feature stats loaded from {path}")
        return instance

    @classmethod
    def build_stats_from_training_csv(cls, csv_path: Path | str, stats_path: Path | str) -> "FeatureEngineer":
        """
        学習用CSVから集計統計を再構築して保存する。
        学習済みCSV（train_2024.csv など）の父名・母父名・騎手IDから
        推論用統計を生成する。
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded training CSV: {len(df)} rows from {csv_path}")

        instance = cls(pd.DataFrame())

        # 父産駒勝率（最後に観測した値を使用）
        if "father" in df.columns and "sire_win_rate" in df.columns:
            instance._sire_win_rate = (
                df[df["father"].notna() & (df["father"] != "")]
                .groupby("father")["sire_win_rate"].mean()
                .rename("sire_win_rate")
            )
            logger.info(f"sire_win_rate: {len(instance._sire_win_rate)} entries")

        # 母父産駒勝率
        if "mother_father" in df.columns and "bms_win_rate" in df.columns:
            instance._bms_win_rate = (
                df[df["mother_father"].notna() & (df["mother_father"] != "")]
                .groupby("mother_father")["bms_win_rate"].mean()
                .rename("bms_win_rate")
            )
            logger.info(f"bms_win_rate: {len(instance._bms_win_rate)} entries")

        # 騎手コース適性
        jockey_cols = ["jockey_id", "course_type_code", "distance_bin_code",
                       "jockey_win_rate", "jockey_place_rate", "jockey_runs"]
        if all(c in df.columns for c in jockey_cols):
            course_map = {0: "芝", 1: "ダート"}
            dist_map   = {0: "短距離", 1: "マイル", 2: "中距離", 3: "長距離"}
            tmp = df[jockey_cols].copy().dropna()
            # jockey_id を正規化（先頭ゼロ除去）して文字列化
            tmp["jockey_id"] = tmp["jockey_id"].astype(str).apply(
                lambda x: str(int(x)) if x.strip().isdigit() else x.strip()
            )
            tmp["course_type"]  = tmp["course_type_code"].map(course_map)
            tmp["distance_bin"] = tmp["distance_bin_code"].astype(int).map(dist_map)
            instance._jockey_course_stats = (
                tmp.groupby(["jockey_id", "course_type", "distance_bin"])
                .agg(
                    jockey_win_rate=("jockey_win_rate", "mean"),
                    jockey_place_rate=("jockey_place_rate", "mean"),
                    jockey_runs=("jockey_runs", "mean"),
                )
                .reset_index()
            )
            logger.info(f"jockey_course_stats: {len(instance._jockey_course_stats)} entries")

        instance.save_stats(stats_path)
        return instance

    # ------------------------------------------------------------------
    # モデル入力カラム一覧
    # ------------------------------------------------------------------

    FEATURE_COLUMNS = [
        "frame_number",
        "horse_number",
        "age",
        "weight_carried",
        "course_type_code",
        "distance",
        "distance_bin_code",
        "ground_condition_code",
        "weather_code",
        "sire_win_rate",
        "bms_win_rate",
        "jockey_win_rate",
        "jockey_place_rate",
        "jockey_runs",
        "recent_avg_pos",
        "recent_avg_last3f",
    ]
