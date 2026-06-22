"""
JRA / NAR 共通の抽象基底クラスと共有データクラス。

BaseScraper を継承することで JRA (NetkeibaScraper) と
NAR (NARScraper) が同一インターフェースで扱える。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# 共有データクラス（JRA / NAR 共通）
# ---------------------------------------------------------------------------

@dataclass
class HorseRecord:
    """1頭分の出走情報"""
    horse_id: str
    horse_name: str
    frame_number: int
    horse_number: int
    sex: str
    age: int
    weight_carried: float
    jockey_id: str
    jockey_name: str
    trainer_id: str
    trainer_name: str
    father_name: str = ""
    mother_father_name: str = ""
    odds: Optional[float] = None
    popularity: Optional[int] = None


@dataclass
class RaceInfo:
    """レース基本情報"""
    race_id: str
    race_name: str
    course_type: str       # 芝 / ダート
    distance: int
    direction: str
    ground_condition: str
    weather: str
    start_datetime: str
    entries: list[HorseRecord] = field(default_factory=list)
    is_g1: bool = False    # G1（GI）グレードかどうか（デフォルトFalse = 後方互換）


# ---------------------------------------------------------------------------
# 抽象基底クラス
# ---------------------------------------------------------------------------

class BaseScraper(ABC):
    """
    JRA / NAR スクレイパーの共通インターフェース。

    サブクラスは ORG, VENUE_CODE_TO_NAME, BASE_URL, RACE_URL を定義し
    抽象メソッドを実装すること。
    """

    ORG: str = ""                            # "jra" or "nar"
    VENUE_CODE_TO_NAME: dict[str, str] = {}  # venue_code → 会場名

    # ------------------------------------------------------------------
    # 抽象メソッド（各サブクラスで実装必須）
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_race_result(self, race_id: str) -> pd.DataFrame:
        """着順・タイム等の結果を取得する"""
        ...

    @abstractmethod
    def fetch_race_meta(self, race_id: str) -> dict:
        """コース種別・距離・馬場・天候を取得する"""
        ...

    @abstractmethod
    def fetch_today_entries(self, race_id: str) -> RaceInfo:
        """当日出走表（オッズ含む）を取得する"""
        ...

    @abstractmethod
    def fetch_race_schedule_by_date(self, target_date: date) -> dict[str, list[str]]:
        """指定日の会場別 race_id リストを返す"""
        ...

    @abstractmethod
    def collect_race_ids_for_period(
        self,
        start_date: date,
        end_date: date,
        jyo_codes: list[str] | None = None,
        save_path: str | None = None,
    ) -> list[str]:
        """期間内の全 race_id を収集する"""
        ...

    @abstractmethod
    def fetch_bulk_results_and_meta(
        self,
        race_ids: list[str],
        checkpoint_path: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """race_id リストの結果・メタを一括取得する"""
        ...

    @abstractmethod
    def fetch_race_payouts(self, race_id: str) -> dict:
        """単勝・馬連・3連単等の払戻情報を取得する"""
        ...

    # ------------------------------------------------------------------
    # 共通ユーティリティ
    # ------------------------------------------------------------------

    def venue_name(self, race_id: str) -> str:
        """race_id[4:6] から会場名を返す"""
        code = race_id[4:6] if len(race_id) >= 6 else ""
        return self.VENUE_CODE_TO_NAME.get(code, f"不明({code})")

    def close(self) -> None:
        """リソース解放（Selenium等）。サブクラスでオーバーライド可。"""
        pass

    def __enter__(self) -> "BaseScraper":
        return self

    def __exit__(self, *_) -> None:
        self.close()
