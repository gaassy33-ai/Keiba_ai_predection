"""
NAR（地方競馬）スクレイパー。

nar.netkeiba.com / db.nar.netkeiba.com を対象に、
NetkeibaScraper のロジックをほぼ流用しつつ
URL・会場コードを NAR 用にオーバーライドする。
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

from src.scraper.base_scraper import RaceInfo
from src.scraper.netkeiba_scraper import NetkeibaScraper


class NARScraper(NetkeibaScraper):
    """
    NAR（地方競馬）専用スクレイパー。

    JRA と異なる点:
    - BASE_URL: https://db.nar.netkeiba.com
    - RACE_URL: https://nar.netkeiba.com
    - VENUE_CODE_TO_NAME: 地方競馬場コード (30〜48)
    - race_id 形式: YYYYVVKKDDNN (VV=地方競馬場コード 30〜)
    - 出走表URL: nar.netkeiba.com/race/shutuba.html?race_id=...
    - スケジュールURL: nar.netkeiba.com/top/race_list.html?kaisai_date=...
    """

    ORG = "nar"
    VENUE_CODE_TO_NAME = {
        "30": "門別",
        "31": "盛岡",
        "32": "水沢",
        "34": "浦和",
        "35": "船橋",
        "36": "大井",
        "37": "川崎",
        "38": "金沢",
        "39": "笠松",
        "40": "名古屋",
        "42": "園田",
        "43": "姫路",
        "44": "高知",
        "45": "佐賀",
        "46": "荒尾",
        "47": "中津",
        "48": "帯広",
    }

    # db.nar.netkeiba.com は IP 制限あり。
    # 歴史データ (results/meta) は db.netkeiba.com で取得可能。
    # 当日出走表 (entries) は nar.netkeiba.com の Selenium で取得する。
    BASE_URL = "https://db.netkeiba.com"
    RACE_URL = "https://nar.netkeiba.com"

    # ------------------------------------------------------------------
    # race_id収集（NARはdb.nar.netkeiba.comのpid=race_listを使用）
    # ------------------------------------------------------------------

    def collect_race_ids_for_period(
        self,
        start_date: date,
        end_date: date,
        jyo_codes: list[str] | None = None,
        save_path: str | None = None,
    ) -> list[str]:
        """
        NAR の race_id を収集する。
        基底クラスと同じロジックだが BASE_URL が NAR 用になっている。

        Parameters
        ----------
        jyo_codes : list[str] | None
            NAR 競馬場コード (例: ["36", "37"])。None = 全場。
        """
        import os
        from pathlib import Path
        from urllib.parse import urlencode

        collected: set[str] = set()
        if save_path and os.path.exists(save_path):
            existing = pd.read_csv(save_path, dtype=str)
            if "race_id" in existing.columns:
                collected = set(existing["race_id"].dropna().tolist())
                logger.info(f"Resuming: {len(collected)} race_ids already loaded.")

        years = list(range(start_date.year, end_date.year + 1))
        logger.info(
            f"[NAR] Collecting race_ids {start_date} → {end_date} | "
            f"jyo={jyo_codes or 'ALL'} | {len(years)} year(s)"
        )

        for year in years:
            s_mon = start_date.month if year == start_date.year else 1
            e_mon = end_date.month if year == end_date.year else 12

            page = 1
            while True:
                params: list[tuple] = [
                    ("pid", "race_list"),
                    ("start_year", year), ("start_mon", s_mon),
                    ("end_year", year), ("end_mon", e_mon),
                    ("sort", "date"), ("list", 100), ("page", page),
                ]
                if jyo_codes:
                    for jyo in jyo_codes:
                        params.append(("jyo[]", jyo))

                url = f"{self.BASE_URL}/?" + urlencode(params)
                try:
                    soup = self._get(url)
                except Exception as e:
                    logger.error(f"[NAR] Failed page={page} year={year}: {e}")
                    break

                ids = list(dict.fromkeys([
                    m.group(1)
                    for a in soup.select("a[href]")
                    for m in [re.search(r"/race/(\d{12})/", a.get("href", ""))]
                    if m
                ]))

                new_ids = [r for r in ids if r not in collected]
                if not new_ids:
                    logger.info(f"  [NAR] {year} page={page}: no new IDs → done")
                    break

                if jyo_codes:
                    new_ids = [r for r in new_ids if r[4:6] in jyo_codes]

                collected.update(new_ids)
                logger.info(f"  [NAR] {year} page={page}: +{len(new_ids)} (total={len(collected)})")
                page += 1

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(sorted(collected), columns=["race_id"]).to_csv(
                    save_path, index=False
                )

        result = sorted(collected)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(result, columns=["race_id"]).to_csv(save_path, index=False)
            logger.info(f"[NAR] Saved {len(result)} race_ids → {save_path}")

        logger.info(f"[NAR] Collection complete: {len(result)} race_ids total.")
        return result

    # ------------------------------------------------------------------
    # 当日開催スケジュール（NARはnar.netkeiba.comを使用）
    # ------------------------------------------------------------------

    def fetch_race_schedule_by_date(self, target_date: date) -> dict[str, list[str]]:
        """
        指定日の会場別 race_id リストを返す（NAR版）。
        """
        date_str = target_date.strftime("%Y%m%d")
        url = f"{self.RACE_URL}/top/race_list.html?kaisai_date={date_str}"
        logger.info(f"[NAR] Fetching race schedule (Selenium): {url}")

        driver = self._get_driver()
        try:
            driver.get(url)
        except Exception as e:
            logger.warning(f"[NAR] driver.get() raised: {e} — 部分DOMで続行")

        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='race_id=']"))
            )
        except Exception:
            logger.warning("[NAR] race_id リンクが現れませんでした — タイムアウト後にパース")

        soup = BeautifulSoup(driver.page_source, "lxml")
        schedule: dict[str, list[str]] = {}

        all_ids = list(dict.fromkeys([
            m.group(1)
            for a in soup.select("a[href]")
            for m in [re.search(r"race_id=(\d{12})", a.get("href", ""))]
            if m
        ]))
        logger.info(f"  [NAR] ページ内 race_id 件数: {len(all_ids)}")

        for rid in all_ids:
            code = rid[4:6]
            name = self.VENUE_CODE_TO_NAME.get(code, f"地方{code}")
            schedule.setdefault(name, []).append(rid)

        for name in schedule:
            schedule[name] = sorted(dict.fromkeys(schedule[name]))

        logger.info(
            f"  [NAR] {target_date}: {sum(len(v) for v in schedule.values())} races "
            f"at {list(schedule.keys())}"
        )
        return schedule

    # ------------------------------------------------------------------
    # 当日出走表（NARはnar.netkeiba.comのshutuba.htmlを使用）
    # ------------------------------------------------------------------

    def fetch_today_entries(self, race_id: str) -> RaceInfo:
        """
        当日出走表と馬場・天候をSeleniumで取得する（NAR版）。
        JRA版と同一ロジックだが RACE_URL が NAR 用になっているため
        URL 解決は自動的に正しくなる。
        """
        # NARのURLも同じパターン: nar.netkeiba.com/race/shutuba.html?race_id=...
        return super().fetch_today_entries(race_id)
