"""
当日・翌日のレーススケジュール（発走時刻・race_id）を取得する。
"""

import re
from datetime import date, datetime

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from config.settings import settings


_GRADE_SCORE: dict[str, int] = {
    "GⅠ": 60, "G1": 60,
    "GⅡ": 50, "G2": 50,
    "GⅢ": 40, "G3": 40,
    "L":  30,
    "OP": 20,
}


def _grade_score(race_name: str) -> int:
    for key, score in _GRADE_SCORE.items():
        if key in race_name:
            return score
    return 0


def select_main_race(races: list[dict]) -> dict:
    """
    レース一覧からメインレースを選択する。

    優先順位:
    1. グレード（GⅠ > GⅡ > GⅢ > L > OP > 一般）
    2. 同グレード内では 11R を優先（メインレースは通常 11R、12R は最終レース）
    3. それ以外はレース番号の降順
    """
    def _score(race: dict) -> tuple[int, int]:
        grade = _grade_score(race.get("race_name", ""))
        rnum  = race.get("race_number", 0)
        # R11 を最優先、R12 は最終レースなので後回し、それ以外は番号順
        num_sc = 11 if rnum == 11 else (0 if rnum >= 12 else rnum)
        return (grade, num_sc)

    return max(races, key=_score)


class RaceScheduleFetcher:
    SCHEDULE_URL = "https://race.netkeiba.com/top/race_list.html"

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": settings.user_agent})

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(settings.scrape_interval_seconds))
    def _get(self, url: str, params: dict | None = None) -> BeautifulSoup:
        import time
        time.sleep(settings.scrape_interval_seconds)
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        return BeautifulSoup(resp.text, "lxml")

    def fetch_race_list(self, target_date: date | None = None) -> list[dict]:
        """
        指定日のレース一覧を取得する。

        Returns
        -------
        list[dict]
            [{"race_id": str, "race_name": str, "start_time": datetime,
              "jyo_name": str, "race_number": int}, ...]
        """
        if target_date is None:
            target_date = date.today()

        params = {"kaisai_date": target_date.strftime("%Y%m%d")}
        logger.info(f"Fetching race schedule for {target_date}")
        soup = self._get(self.SCHEDULE_URL, params=params)

        races = []
        for race_list_div in soup.select("div.RaceList_DataItem"):
            link = race_list_div.select_one("a[href*='race_id']")
            if link is None:
                continue

            href = link.get("href", "")
            race_id_match = re.search(r"race_id=(\d+)", href)
            if not race_id_match:
                continue
            race_id = race_id_match.group(1)

            # 発走時刻（"15:30" 形式）
            time_tag = race_list_div.select_one(".RaceList_Itemtime")
            start_time_str = time_tag.get_text(strip=True) if time_tag else "00:00"
            try:
                start_dt = datetime.strptime(
                    f"{target_date.isoformat()} {start_time_str}", "%Y-%m-%d %H:%M"
                )
            except ValueError:
                start_dt = datetime(target_date.year, target_date.month, target_date.day)

            # 競馬場名・レース番号
            jyo_tag = race_list_div.select_one(".RaceList_ItemJyo")
            race_num_tag = race_list_div.select_one(".RaceList_ItemNum")
            race_name_tag = race_list_div.select_one(".RaceList_ItemTitle")

            races.append({
                "race_id": race_id,
                "jyo_name": jyo_tag.get_text(strip=True) if jyo_tag else "",
                "race_number": int(race_num_tag.get_text(strip=True).replace("R", "") or 0)
                if race_num_tag else 0,
                "race_name": race_name_tag.get_text(strip=True) if race_name_tag else "",
                "start_time": start_dt,
            })

        logger.info(f"Found {len(races)} races on {target_date}")
        return races

    def filter_by_jyo(self, races: list[dict]) -> list[dict]:
        """設定ファイルの TARGET_JYO_CODES に基づいてフィルタリング。"""
        codes = settings.target_jyo_code_list
        if not codes:
            return races
        # race_id の先頭10桁のうち5-6文字目が競馬場コード
        return [r for r in races if r["race_id"][4:6] in codes]
