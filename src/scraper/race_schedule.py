"""
当日・翌日のレーススケジュール（発走時刻・race_id）を取得する。

netkeiba のレースリストページは JavaScript レンダリングのため Selenium を使用する。
"""

import re
import time
from datetime import date, datetime
from typing import Optional

from bs4 import BeautifulSoup
from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

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
        self._driver: Optional[webdriver.Chrome] = None

    def _get_driver(self) -> webdriver.Chrome:
        if self._driver is None:
            opts = Options()
            if settings.selenium_headless:
                opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument(f"--user-agent={settings.user_agent}")
            service = (
                Service(settings.chromedriver_path)
                if settings.chromedriver_path
                else Service(ChromeDriverManager().install())
            )
            self._driver = webdriver.Chrome(service=service, options=opts)
        return self._driver

    def close(self) -> None:
        if self._driver:
            self._driver.quit()
            self._driver = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def fetch_race_list(self, target_date: date | None = None) -> list[dict]:
        """
        指定日のレース一覧を Selenium で取得する。

        Returns
        -------
        list[dict]
            [{"race_id": str, "race_name": str, "start_time": datetime,
              "jyo_name": str, "race_number": int}, ...]
        """
        if target_date is None:
            target_date = date.today()

        url = f"{self.SCHEDULE_URL}?kaisai_date={target_date.strftime('%Y%m%d')}"
        logger.info(f"Fetching race schedule (Selenium): {url}")

        driver = self._get_driver()
        driver.get(url)

        # JS レンダリング完了を待つ
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "a[href*='race_id'], div.RaceList_DataItem, li.RaceList_DataItem")
                )
            )
        except Exception:
            logger.warning("レースリスト要素が見つからず — ページソースをそのまま解析します")
        time.sleep(1)

        soup = BeautifulSoup(driver.page_source, "lxml")

        # race_id を持つ全リンクから一意な race_id を抽出
        races: list[dict] = []
        seen: set[str] = set()

        for a in soup.select("a[href*='race_id']"):
            href = a.get("href", "")
            m = re.search(r"race_id=(\d{12})", href)
            if not m:
                continue
            race_id = m.group(1)
            if race_id in seen:
                continue
            seen.add(race_id)

            # race_id の末尾2桁 = レース番号
            race_num = int(race_id[10:12])

            # 親コンテナからレース名・発走時刻・競馬場名を取得
            container = (
                a.find_parent("li")
                or a.find_parent("div", class_=re.compile(r"RaceList"))
                or a.find_parent("div")
                or a
            )

            # 発走時刻
            time_tag = container.select_one(
                ".RaceList_Itemtime, .ItemTime, [class*='time'], [class*='Time']"
            )
            start_time_str = time_tag.get_text(strip=True) if time_tag else "00:00"
            if not re.match(r"\d{1,2}:\d{2}", start_time_str):
                start_time_str = "00:00"
            try:
                start_dt = datetime.strptime(
                    f"{target_date.isoformat()} {start_time_str}", "%Y-%m-%d %H:%M"
                )
            except ValueError:
                start_dt = datetime(target_date.year, target_date.month, target_date.day)

            # レース名
            name_tag = container.select_one(
                ".RaceList_ItemTitle, .ItemTitle, [class*='Title'], [class*='Name']"
            )
            race_name = name_tag.get_text(strip=True) if name_tag else ""

            # 競馬場名（race_id[4:6] から補完）
            jyo_tag = container.select_one(".RaceList_ItemJyo, [class*='Jyo'], [class*='Place']")
            jyo_name = jyo_tag.get_text(strip=True) if jyo_tag else ""

            races.append({
                "race_id":     race_id,
                "jyo_name":    jyo_name,
                "race_number": race_num,
                "race_name":   race_name,
                "start_time":  start_dt,
            })

        # race_number 昇順でソート
        races.sort(key=lambda r: (r["jyo_name"], r["race_number"]))
        logger.info(f"Found {len(races)} races on {target_date}")
        return races

    def filter_by_jyo(self, races: list[dict]) -> list[dict]:
        """設定ファイルの TARGET_JYO_CODES に基づいてフィルタリング。"""
        codes = settings.target_jyo_code_list
        if not codes:
            return races
        # race_id の先頭10桁のうち5-6文字目が競馬場コード
        return [r for r in races if r["race_id"][4:6] in codes]
