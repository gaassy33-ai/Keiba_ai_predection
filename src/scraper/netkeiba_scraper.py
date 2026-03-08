"""
netkeiba.com から過去成績・血統・当日出走情報をスクレイピングする。

- 静的ページ: requests + BeautifulSoup
- 動的ページ（オッズ等）: Selenium
"""

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tenacity import retry, stop_after_attempt, wait_fixed
from webdriver_manager.chrome import ChromeDriverManager

from config.settings import settings


# --------------------------------------------------------------------------
# データクラス
# --------------------------------------------------------------------------

@dataclass
class HorseRecord:
    """1頭分の出走情報"""
    horse_id: str
    horse_name: str
    frame_number: int        # 枠番
    horse_number: int        # 馬番
    sex: str                 # 性別 (牡/牝/騸)
    age: int
    weight_carried: float    # 斤量 (kg)
    jockey_id: str
    jockey_name: str
    trainer_id: str
    trainer_name: str
    father_name: str = ""    # 父
    mother_father_name: str = ""  # 母父
    odds: Optional[float] = None
    popularity: Optional[int] = None


@dataclass
class RaceInfo:
    """レース基本情報"""
    race_id: str
    race_name: str
    course_type: str    # 芝 / ダート
    distance: int       # メートル
    direction: str      # 右 / 左 / 直線
    ground_condition: str  # 良 / 稍重 / 重 / 不良
    weather: str        # 晴 / 曇 / 雨 / 小雨 / 雪
    start_datetime: str  # ISO 形式
    entries: list[HorseRecord] = field(default_factory=list)


# --------------------------------------------------------------------------
# スクレイパー本体
# --------------------------------------------------------------------------

class NetkeibaScraper:
    BASE_URL = "https://db.netkeiba.com"
    RACE_URL = "https://race.netkeiba.com"

    def __init__(self) -> None:
        self._session = self._build_session()
        self._driver: Optional[webdriver.Chrome] = None

    # ------------------------------------------------------------------
    # requests セッション
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": settings.user_agent})
        return session

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(settings.scrape_interval_seconds))
    def _get(self, url: str, **kwargs) -> BeautifulSoup:
        time.sleep(settings.scrape_interval_seconds)
        resp = self._session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        return BeautifulSoup(resp.text, "lxml")

    # ------------------------------------------------------------------
    # Selenium ドライバー
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 過去レース成績（race_id 単位）
    # ------------------------------------------------------------------

    def fetch_race_result(self, race_id: str) -> pd.DataFrame:
        """
        指定 race_id の着順・タイム・上がり3F を取得する。

        Parameters
        ----------
        race_id : str
            ネット競馬の race_id (例: "202405050811")

        Returns
        -------
        pd.DataFrame
            columns: [race_id, finish_position, horse_id, horse_name,
                      time_seconds, last_3f, weight_carried, jockey_id, ...]
        """
        url = f"{self.BASE_URL}/race/{race_id}/"
        logger.info(f"Fetching race result: {url}")
        soup = self._get(url)

        table = soup.select_one("table.race_table_01")
        if table is None:
            logger.warning(f"Result table not found for race_id={race_id}")
            return pd.DataFrame()

        rows = []
        for tr in table.select("tr")[1:]:  # ヘッダ除外
            tds = tr.select("td")
            if len(tds) < 18:
                continue
            horse_link = tds[3].select_one("a")
            jockey_link = tds[6].select_one("a")
            rows.append({
                "race_id": race_id,
                "finish_position": tds[0].get_text(strip=True),
                "frame_number": tds[1].get_text(strip=True),
                "horse_number": tds[2].get_text(strip=True),
                "horse_id": horse_link["href"].split("/")[-2] if horse_link else "",
                "horse_name": tds[3].get_text(strip=True),
                "sex_age": tds[4].get_text(strip=True),
                "weight_carried": tds[5].get_text(strip=True),
                "jockey_id": jockey_link["href"].split("/")[-2] if jockey_link else "",
                "jockey_name": tds[6].get_text(strip=True),
                "finish_time": tds[7].get_text(strip=True),
                "margin": tds[8].get_text(strip=True),
                "popularity": tds[10].get_text(strip=True),
                "odds": tds[11].get_text(strip=True),
                "last_3f": tds[12].get_text(strip=True),
                "corner_positions": tds[13].get_text(strip=True),
                "trainer_name": tds[18].get_text(strip=True) if len(tds) > 18 else "",
                "horse_weight": tds[17].get_text(strip=True) if len(tds) > 17 else "",
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 血統情報
    # ------------------------------------------------------------------

    def fetch_horse_pedigree(self, horse_id: str) -> dict:
        """
        父・母父を取得する。

        Returns
        -------
        dict
            {"father": str, "mother_father": str, "horse_id": str}
        """
        url = f"{self.BASE_URL}/horse/ped/{horse_id}/"
        logger.info(f"Fetching pedigree: {url}")
        soup = self._get(url)

        result = {"horse_id": horse_id, "father": "", "mother_father": ""}
        ped_table = soup.select_one("table.blood_table")
        if ped_table is None:
            return result

        tds = ped_table.select("td")
        if len(tds) >= 1:
            result["father"] = tds[0].get_text(strip=True)
        # 母父は5列目（テーブル構造依存）
        if len(tds) >= 5:
            result["mother_father"] = tds[4].get_text(strip=True)

        return result

    # ------------------------------------------------------------------
    # レースメタ情報（結果ページのヘッダ部）
    # ------------------------------------------------------------------

    def fetch_race_meta(self, race_id: str) -> dict:
        """
        過去レース結果ページからコース種別・距離・馬場・天候を取得する。

        Parameters
        ----------
        race_id : str

        Returns
        -------
        dict
            {
                "race_id": str,
                "race_name": str,
                "course_type": str,   # "芝" or "ダート"
                "distance": int,
                "direction": str,     # "右" / "左" / "直線" / ""
                "ground_condition": str,
                "weather": str,
                "ground_condition_code": int,
                "weather_code": int,
            }
        """
        from src.scraper.weather import GROUND_CONDITION_MAP, WEATHER_MAP

        url = f"{self.BASE_URL}/race/{race_id}/"
        soup = self._get(url)

        result: dict = {
            "race_id": race_id,
            "race_name": "",
            "course_type": "",
            "distance": 0,
            "direction": "",
            "ground_condition": "",
            "weather": "",
            "ground_condition_code": -1,
            "weather_code": -1,
        }

        # レース名
        race_name_tag = soup.select_one(".mainrace_data h1, .RaceName, h1.fntB")
        if race_name_tag:
            result["race_name"] = race_name_tag.get_text(strip=True)

        # レース詳細テキスト（例: "芝・右 1600m", "ダ・左 1200m"）
        detail_span = soup.select_one(".mainrace_data .smalltxt, .mainrace_data p span")
        detail_text = ""
        if detail_span:
            detail_text = detail_span.get_text()
        else:
            # フォールバック：ページ全体からセクションを探す
            for p in soup.select(".mainrace_data p, .data_intro p"):
                t = p.get_text()
                if "芝" in t or "ダ" in t:
                    detail_text = t
                    break

        # コース種別・距離のパース
        m_course = re.search(r"(芝|ダ[ートー]*)[・/\s]*(右|左|直線)?[\s]*(\d{3,4})", detail_text)
        if m_course:
            raw_type = m_course.group(1)
            result["course_type"] = "芝" if raw_type == "芝" else "ダート"
            result["direction"] = m_course.group(2) or ""
            result["distance"] = int(m_course.group(3))
        else:
            # 別パターン: "芝1600m" のような連結形
            m2 = re.search(r"(芝|ダ[ートー]*)(\d{3,4})", detail_text)
            if m2:
                result["course_type"] = "芝" if m2.group(1) == "芝" else "ダート"
                result["distance"] = int(m2.group(2))

        # 天候・馬場状態のパース（例: "天候 : 晴 / 芝 : 良 / ダート : 稍重"）
        cond_text = ""
        for tag in soup.select(".mainrace_data .smalltxt, .race_otherdata p, .mainrace_data p"):
            t = tag.get_text()
            if "天候" in t or "馬場" in t or "芝 :" in t or "ダート :" in t:
                cond_text = t
                break

        m_weather = re.search(r"天候\s*[:/：]\s*([^\s/　]+)", cond_text)
        if m_weather:
            result["weather"] = m_weather.group(1).strip()

        # 芝の馬場状態を優先、なければダートを参照
        m_ground_shiba = re.search(r"芝\s*[:/：]\s*([良稍重不]+)", cond_text)
        m_ground_dirt = re.search(r"ダ[ートー]*\s*[:/：]\s*([良稍重不]+)", cond_text)
        if m_ground_shiba and result["course_type"] == "芝":
            result["ground_condition"] = m_ground_shiba.group(1).strip()
        elif m_ground_dirt and result["course_type"] == "ダート":
            result["ground_condition"] = m_ground_dirt.group(1).strip()
        elif m_ground_shiba:
            result["ground_condition"] = m_ground_shiba.group(1).strip()

        result["ground_condition_code"] = GROUND_CONDITION_MAP.get(result["ground_condition"], -1)
        result["weather_code"] = WEATHER_MAP.get(result["weather"], -1)

        return result

    # ------------------------------------------------------------------
    # 開催カレンダー → race_id 一括収集
    # ------------------------------------------------------------------

    def collect_race_ids_for_period(
        self,
        start_date: date,
        end_date: date,
        jyo_codes: list[str] | None = None,
        save_path: str | None = None,
    ) -> list[str]:
        """
        指定期間内の全 race_id を `db.netkeiba.com/?pid=race_list` のページネーションで収集する。

        Parameters
        ----------
        start_date : date
        end_date : date
        jyo_codes : list[str] | None
            JRA 競馬場コードで絞り込む（例: ["05","06","08","09"]）
            None = 全競馬場（NAR 含む）
        save_path : str | None
            途中経過を保存する CSV パス（再開時に読み込む）

        Returns
        -------
        list[str]
            race_id のリスト（race_date 昇順）
        """
        import os
        from urllib.parse import urlencode

        # 途中再開
        collected: set[str] = set()
        if save_path and os.path.exists(save_path):
            existing = pd.read_csv(save_path, dtype=str)
            if "race_id" in existing.columns:
                collected = set(existing["race_id"].dropna().tolist())
                logger.info(f"Resuming: {len(collected)} race_ids already loaded.")

        # 年単位でクエリ（月ごとは1年12リクエスト→不要）
        years = list(range(start_date.year, end_date.year + 1))
        logger.info(
            f"Collecting race_ids {start_date} → {end_date} | "
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
                    logger.error(f"Failed page={page} year={year}: {e}")
                    break

                ids = list(dict.fromkeys([
                    m.group(1)
                    for a in soup.select("a[href]")
                    for m in [re.search(r"/race/(\d{12})/", a.get("href", ""))]
                    if m
                ]))

                new_ids = [r for r in ids if r not in collected]
                if not new_ids:
                    logger.info(f"  {year} page={page}: no new IDs → done for this year")
                    break

                # jyo コードフィルタ（サーバー側フィルタが甘い場合の保険）
                if jyo_codes:
                    new_ids = [r for r in new_ids if r[4:6] in jyo_codes]

                collected.update(new_ids)
                logger.info(f"  {year} page={page}: +{len(new_ids)} (total={len(collected)})")
                page += 1

            # 年ごとに途中保存
            if save_path:
                pd.DataFrame(sorted(collected), columns=["race_id"]).to_csv(
                    save_path, index=False
                )

        result = sorted(collected)
        if save_path:
            pd.DataFrame(result, columns=["race_id"]).to_csv(save_path, index=False)
            logger.info(f"Saved {len(result)} race_ids → {save_path}")

        logger.info(f"Collection complete: {len(result)} race_ids total.")
        return result

    def fetch_bulk_race_meta(self, race_ids: list[str]) -> pd.DataFrame:
        """
        race_id リストに対してレースメタ情報を一括取得する。

        Returns
        -------
        pd.DataFrame
            columns: [race_id, race_name, course_type, distance,
                      direction, ground_condition, weather,
                      ground_condition_code, weather_code]
        """
        records = []
        for i, rid in enumerate(race_ids):
            try:
                meta = self.fetch_race_meta(rid)
                records.append(meta)
                if (i + 1) % 50 == 0:
                    logger.info(f"  fetch_race_meta: {i+1}/{len(race_ids)} done")
            except Exception as e:
                logger.error(f"fetch_race_meta failed for {rid}: {e}")
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 当日出走表（Selenium）
    # ------------------------------------------------------------------

    def fetch_today_entries(self, race_id: str) -> RaceInfo:
        """
        当日出走表と馬場・天候をSeleniumで取得する。

        Parameters
        ----------
        race_id : str
            対象レースID

        Returns
        -------
        RaceInfo
        """
        url = f"{self.RACE_URL}/race/shutuba.html?race_id={race_id}"
        logger.info(f"Fetching today entries via Selenium: {url}")

        driver = self._get_driver()
        driver.get(url)
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.Shutuba_Table")))

        soup = BeautifulSoup(driver.page_source, "lxml")

        # レース情報
        race_name = soup.select_one(".RaceName")
        race_data1 = soup.select_one(".RaceData01")
        race_data2 = soup.select_one(".RaceData02")

        ground_condition = ""
        weather = ""
        course_type = ""
        distance = 0
        if race_data1:
            text = race_data1.get_text()
            # 例: "15:30発走 / 芝1600m / 天候:晴 / 馬場:良"
            for part in text.split("/"):
                part = part.strip()
                if "芝" in part or "ダ" in part:
                    course_type = "芝" if "芝" in part else "ダート"
                    distance = int("".join(filter(str.isdigit, part))) if any(c.isdigit() for c in part) else 0
                if "天候:" in part:
                    weather = part.replace("天候:", "").strip()
                if "馬場:" in part:
                    ground_condition = part.replace("馬場:", "").strip()

        entries: list[HorseRecord] = []
        for tr in soup.select("table.Shutuba_Table tr.HorseList"):
            tds = tr.select("td")
            if len(tds) < 10:
                continue

            # クラス名が変わっていてもhref属性でフォールバック
            all_links = tr.select("a[href]")
            horse_link = tr.select_one("td.HorseName a") or next(
                (a for a in all_links if "/horse/" in a.get("href", "")), None
            )
            jockey_link = tr.select_one("td.Jockey a") or next(
                (a for a in all_links if "/jockey/" in a.get("href", "")), None
            )
            trainer_link = tr.select_one("td.Trainer a") or next(
                (a for a in all_links if "/trainer/" in a.get("href", "")), None
            )

            waku_span   = tr.select_one("td.Waku span")
            umaban_td   = tr.select_one("td.Umaban")
            barei_td    = tr.select_one("td.Barei")
            futan_td    = tr.select_one("td.Futan")
            barei_text  = barei_td.get_text(strip=True) if barei_td else ""

            entries.append(HorseRecord(
                horse_id=horse_link["href"].split("/")[-2] if horse_link else "",
                horse_name=horse_link.get_text(strip=True) if horse_link else "",
                frame_number=int(waku_span.get_text(strip=True) or 0) if waku_span else 0,
                horse_number=int(umaban_td.get_text(strip=True) or 0) if umaban_td else 0,
                sex=barei_text[0] if barei_text else "",
                age=int(barei_text[1:] or 0) if len(barei_text) > 1 else 0,
                weight_carried=float(futan_td.get_text(strip=True) or 0) if futan_td else 0.0,
                jockey_id=jockey_link["href"].split("/")[-2] if jockey_link else "",
                jockey_name=jockey_link.get_text(strip=True) if jockey_link else "",
                trainer_id=trainer_link["href"].split("/")[-2] if trainer_link else "",
                trainer_name=trainer_link.get_text(strip=True) if trainer_link else "",
            ))

        logger.info(f"出走馬 {len(entries)} 頭: {[e.horse_name for e in entries[:5]]}")
        return RaceInfo(
            race_id=race_id,
            race_name=race_name.get_text(strip=True) if race_name else "",
            course_type=course_type,
            distance=distance,
            direction="",  # 別途パース要
            ground_condition=ground_condition,
            weather=weather,
            start_datetime="",  # race_schedule から補完
            entries=entries,
        )

    # ------------------------------------------------------------------
    # 結果 + メタを1リクエストで取得（最適化版）
    # ------------------------------------------------------------------

    def fetch_result_and_meta(self, race_id: str) -> tuple[pd.DataFrame, dict]:
        """
        `db.netkeiba.com/race/{race_id}/` を1回だけ取得し、
        着順データ (DataFrame) と レースメタ情報 (dict) を同時に返す。

        fetch_race_result() + fetch_race_meta() の合計2リクエストを
        1リクエストに削減できる。
        """
        from src.scraper.weather import GROUND_CONDITION_MAP, WEATHER_MAP

        url = f"{self.BASE_URL}/race/{race_id}/"
        logger.debug(f"Fetching result+meta: {url}")
        soup = self._get(url)

        # ---------- メタ情報 ----------
        meta: dict = {
            "race_id": race_id,
            "race_date": "",   # YYYY-MM-DD 形式
            "race_name": "",
            "course_type": "",
            "distance": 0,
            "direction": "",
            "ground_condition": "",
            "weather": "",
            "ground_condition_code": -1,
            "weather_code": -1,
        }

        race_name_tag = soup.select_one(".mainrace_data h1, .RaceName, h1.fntB")
        if race_name_tag:
            meta["race_name"] = race_name_tag.get_text(strip=True)

        # レース日付（例: "2024年01月06日" → "2024-01-06"）
        for tag in soup.select(".mainrace_data p, .mainrace_data .smalltxt"):
            t = tag.get_text()
            m_date = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", t)
            if m_date:
                meta["race_date"] = (
                    f"{m_date.group(1)}-{int(m_date.group(2)):02d}-{int(m_date.group(3)):02d}"
                )
                break

        detail_text = ""
        for p in soup.select(".mainrace_data p, .data_intro p"):
            t = p.get_text()
            if "芝" in t or "ダ" in t:
                detail_text = t
                break

        m_course = re.search(r"(芝|ダ[ートー]*)[・/\s]*(右|左|直線)?[\s]*(\d{3,4})", detail_text)
        if m_course:
            meta["course_type"] = "芝" if m_course.group(1) == "芝" else "ダート"
            meta["direction"] = m_course.group(2) or ""
            meta["distance"] = int(m_course.group(3))
        else:
            m2 = re.search(r"(芝|ダ[ートー]*)(\d{3,4})", detail_text)
            if m2:
                meta["course_type"] = "芝" if m2.group(1) == "芝" else "ダート"
                meta["distance"] = int(m2.group(2))

        cond_text = ""
        for tag in soup.select(".mainrace_data .smalltxt, .race_otherdata p, .mainrace_data p"):
            t = tag.get_text()
            if "天候" in t or "馬場" in t or "芝 :" in t or "ダート :" in t:
                cond_text = t
                break

        m_weather = re.search(r"天候\s*[:/：]\s*([^\s/　]+)", cond_text)
        if m_weather:
            meta["weather"] = m_weather.group(1).strip()

        m_ground_shiba = re.search(r"芝\s*[:/：]\s*([良稍重不]+)", cond_text)
        m_ground_dirt = re.search(r"ダ[ートー]*\s*[:/：]\s*([良稍重不]+)", cond_text)
        if m_ground_shiba and meta["course_type"] == "芝":
            meta["ground_condition"] = m_ground_shiba.group(1).strip()
        elif m_ground_dirt and meta["course_type"] == "ダート":
            meta["ground_condition"] = m_ground_dirt.group(1).strip()
        elif m_ground_shiba:
            meta["ground_condition"] = m_ground_shiba.group(1).strip()

        meta["ground_condition_code"] = GROUND_CONDITION_MAP.get(meta["ground_condition"], -1)
        meta["weather_code"] = WEATHER_MAP.get(meta["weather"], -1)

        # ---------- 着順データ ----------
        rows = []
        table = soup.select_one("table.race_table_01")
        if table:
            for tr in table.select("tr")[1:]:
                tds = tr.select("td")
                if len(tds) < 18:
                    continue
                horse_link = tds[3].select_one("a")
                jockey_link = tds[6].select_one("a")
                rows.append({
                    "race_id": race_id,
                    "finish_position": tds[0].get_text(strip=True),
                    "frame_number": tds[1].get_text(strip=True),
                    "horse_number": tds[2].get_text(strip=True),
                    "horse_id": horse_link["href"].split("/")[-2] if horse_link else "",
                    "horse_name": tds[3].get_text(strip=True),
                    "sex_age": tds[4].get_text(strip=True),
                    "weight_carried": tds[5].get_text(strip=True),
                    "jockey_id": jockey_link["href"].split("/")[-2] if jockey_link else "",
                    "jockey_name": tds[6].get_text(strip=True),
                    "finish_time": tds[7].get_text(strip=True),
                    "margin": tds[8].get_text(strip=True),
                    "popularity": tds[10].get_text(strip=True),
                    "odds": tds[11].get_text(strip=True),
                    "last_3f": tds[12].get_text(strip=True),
                    "corner_positions": tds[13].get_text(strip=True),
                    "trainer_name": tds[18].get_text(strip=True) if len(tds) > 18 else "",
                    "horse_weight": tds[17].get_text(strip=True) if len(tds) > 17 else "",
                    # レースメタをインライン付加（後の join を省略できる）
                    "course_type": meta["course_type"],
                    "distance": meta["distance"],
                    "ground_condition": meta["ground_condition"],
                    "weather": meta["weather"],
                    "ground_condition_code": meta["ground_condition_code"],
                    "weather_code": meta["weather_code"],
                })

        return pd.DataFrame(rows), meta

    # ------------------------------------------------------------------
    # 複数レースの過去成績を一括取得
    # ------------------------------------------------------------------

    def fetch_bulk_results(self, race_ids: list[str]) -> pd.DataFrame:
        dfs = []
        for rid in race_ids:
            try:
                df = self.fetch_race_result(rid)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to fetch {rid}: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def fetch_bulk_results_and_meta(
        self,
        race_ids: list[str],
        checkpoint_path: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        全 race_id に対して fetch_result_and_meta() を呼び、
        着順 DataFrame と メタ DataFrame を返す。
        1レースにつき1リクエスト（fetch_bulk_results + fetch_bulk_race_meta の2倍速）。

        Parameters
        ----------
        race_ids : list[str]
        checkpoint_path : str | None
            中間保存先プレフィックス。指定時は {path}_results.csv / {path}_meta.csv に
            50件ごとに追記保存し、再開時に読み込む。

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (history_df, race_meta_df)
        """
        import os

        result_rows: list[pd.DataFrame] = []
        meta_rows: list[dict] = []
        done_ids: set[str] = set()

        # チェックポイントから再開
        if checkpoint_path:
            rp = checkpoint_path + "_results.csv"
            mp = checkpoint_path + "_meta.csv"
            if os.path.exists(rp):
                prev = pd.read_csv(rp, dtype={"race_id": str})
                done_ids = set(prev["race_id"].unique())
                result_rows.append(prev)
                logger.info(f"Resuming: {len(done_ids)} races already fetched.")
            if os.path.exists(mp):
                meta_rows_prev = pd.read_csv(mp, dtype={"race_id": str}).to_dict("records")
                meta_rows.extend(meta_rows_prev)

        pending = [r for r in race_ids if r not in done_ids]
        logger.info(f"Fetching {len(pending)} races (skipping {len(done_ids)} done)...")

        for i, rid in enumerate(pending):
            try:
                df, meta = self.fetch_result_and_meta(rid)
                if not df.empty:
                    result_rows.append(df)
                meta_rows.append(meta)
            except Exception as e:
                logger.error(f"Failed [{rid}]: {e}")
                continue

            # 50件ごとにチェックポイント保存
            if checkpoint_path and (i + 1) % 50 == 0:
                pd.concat(result_rows, ignore_index=True).to_csv(
                    checkpoint_path + "_results.csv", index=False
                )
                pd.DataFrame(meta_rows).to_csv(
                    checkpoint_path + "_meta.csv", index=False
                )
                logger.info(f"  Checkpoint saved: {i+1}/{len(pending)}")

        # 最終保存
        history_df = pd.concat(result_rows, ignore_index=True) if result_rows else pd.DataFrame()
        meta_df = pd.DataFrame(meta_rows)

        if checkpoint_path:
            history_df.to_csv(checkpoint_path + "_results.csv", index=False)
            meta_df.to_csv(checkpoint_path + "_meta.csv", index=False)

        return history_df, meta_df

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
