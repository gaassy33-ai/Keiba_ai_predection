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

        JRA版との相違点:
        - ログインが必要な場合は _login_if_needed() でログインしてからアクセス
        - CSS クラス名が異なる場合のフォールバック: /horse/ リンクを起点にパース
        - 0頭時はページソースを logs/ に保存してデバッグを容易にする
        """
        import re as _re
        from pathlib import Path as _Path

        # --- ① netkeiba ログイン（要ログインページ対策）---
        self._login_if_needed()

        # --- ② 親クラスの標準パース（table.Shutuba_Table / tr.HorseList）---
        race_info = super().fetch_today_entries(race_id)

        if race_info.entries:
            return race_info

        # --- ③ 標準パース失敗: フォールバック（/horse/ リンク起点）---
        logger.warning(
            f"  [NAR] {race_id}: 標準パース 0頭 → フォールバックパース試行"
        )
        driver = self._get_driver()
        from bs4 import BeautifulSoup as _BS
        soup = _BS(driver.page_source, "lxml")

        # ページ上のすべての /horse/ リンクを収集
        horse_links = [
            a for a in soup.select("a[href]")
            if _re.search(r"/horse/\d+", a.get("href", ""))
        ]
        logger.info(f"  [NAR] フォールバック: /horse/ リンク {len(horse_links)} 件")

        entries = []
        seen_horse_ids: set[str] = set()

        for horse_link in horse_links:
            m = _re.search(r"/horse/(\d+)", horse_link.get("href", ""))
            if not m:
                continue
            horse_id = m.group(1)
            if horse_id in seen_horse_ids:
                continue
            seen_horse_ids.add(horse_id)

            # 親 <tr> を探す
            tr = horse_link.find_parent("tr")
            if tr is None:
                continue

            tds = tr.select("td")
            all_links = tr.select("a[href]")

            jockey_link = tr.select_one("td.Jockey a") or next(
                (a for a in all_links if _re.search(r"/jockey/", a.get("href", ""))), None
            )
            trainer_link = tr.select_one("td.Trainer a") or next(
                (a for a in all_links if _re.search(r"/trainer/", a.get("href", ""))), None
            )

            def _xid(link, pat):
                if not link:
                    return ""
                mm = _re.search(pat, link.get("href", ""))
                return mm.group(1) if mm else ""

            waku_td  = tr.select_one("td.Waku")
            umaban_td = tr.select_one("td.Umaban")
            barei_td = tr.select_one("td.Barei")
            futan_td = tr.select_one("td.Futan")

            # 枠番: Waku td → tds[0]
            waku_src = waku_td or (tds[0] if tds else None)
            frame_text = _re.sub(r"\D", "", waku_src.get_text(strip=True)) if waku_src else ""
            frame_num = int(frame_text) if frame_text.isdigit() else 0

            # 馬番: Umaban td → tds[1]
            umaban_src = umaban_td or (tds[1] if len(tds) > 1 else None)
            horse_text = _re.sub(r"\D", "", umaban_src.get_text(strip=True)) if umaban_src else ""
            horse_num = int(horse_text) if horse_text.isdigit() else 0

            barei_text = barei_td.get_text(strip=True) if barei_td else ""

            # NAR ジョッキー URL は /jockey/05683/ だけでなく
            # /jockey/nar05683/ や /jockey/result/05683/ など多様なため
            # URL内の4〜6桁数字を広くキャプチャする
            jockey_id  = _xid(jockey_link,  r"/jockey/[^?#]*?(\d{4,6})(?:[^0-9]|$)")
            trainer_id = _xid(trainer_link, r"/trainer/(?:result/recent/)?(\d+)")

            from src.scraper.base_scraper import HorseRecord as _HR
            entries.append(_HR(
                horse_id=horse_id,
                horse_name=horse_link.get_text(strip=True),
                frame_number=frame_num,
                horse_number=horse_num,
                sex=barei_text[0] if barei_text else "",
                age=int(barei_text[1:] or 0) if len(barei_text) > 1 else 0,
                weight_carried=float(futan_td.get_text(strip=True) or 0) if futan_td else 0.0,
                jockey_id=jockey_id,
                jockey_name=jockey_link.get_text(strip=True) if jockey_link else "",
                trainer_id=trainer_id,
                trainer_name=trainer_link.get_text(strip=True) if trainer_link else "",
            ))

        logger.info(
            f"  [NAR] フォールバック結果: {len(entries)} 頭: "
            f"{[e.horse_name for e in entries[:5]]}"
        )
        logger.info(
            f"  [NAR] horse_ids(sample): {[e.horse_id for e in entries[:5]]}"
        )
        logger.info(
            f"  [NAR] jockey_ids(sample): {[e.jockey_id for e in entries[:5]]}"
        )
        logger.info(
            f"  [NAR] horse_numbers(sample): {[e.horse_number for e in entries[:5]]}"
        )

        if entries:
            # フォールバック成功: race_name / course_type 等は親から引き継ぎ
            from src.scraper.base_scraper import RaceInfo as _RI
            return _RI(
                race_id=race_info.race_id,
                race_name=race_info.race_name,
                course_type=race_info.course_type,
                distance=race_info.distance,
                direction=race_info.direction,
                ground_condition=race_info.ground_condition,
                weather=race_info.weather,
                start_datetime=race_info.start_datetime,
                entries=entries,
            )

        # --- ④ 完全失敗: ページソースを保存（デバッグ用）---
        debug_dir = _Path("logs")
        debug_dir.mkdir(exist_ok=True)
        debug_path = debug_dir / f"nar_shutuba_debug_{race_id}.html"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(f"<!-- URL: {driver.current_url} -->\n")
            f.write(driver.page_source)
        logger.warning(
            f"  [NAR] {race_id}: 0頭 (フォールバックも失敗). "
            f"ページソース保存: {debug_path} | 現在URL: {driver.current_url}"
        )

        return race_info
