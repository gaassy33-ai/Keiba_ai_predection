"""
リアルタイムの天候・馬場状態を取得する。
netkeiba の出走表ページから取得するため、NetkeibaScraper に委譲する場合もある。
ここでは独立した補助フェッチャーとして実装する。
"""

import requests
from bs4 import BeautifulSoup
from loguru import logger

from config.settings import settings


# 馬場状態の数値マッピング（特徴量エンジニアリング用）
GROUND_CONDITION_MAP = {
    "良": 0,
    "稍重": 1,
    "重": 2,
    "不良": 3,
}

WEATHER_MAP = {
    "晴": 0,
    "曇": 1,
    "小雨": 2,
    "雨": 3,
    "雪": 4,
}


class WeatherFetcher:
    """
    指定 race_id のリアルタイム馬場・天候情報を取得する。
    NetkeibaScraper との重複を避けるため、軽量な requests のみで実装。
    """

    SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html"

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": settings.user_agent})

    def fetch(self, race_id: str) -> dict:
        """
        Returns
        -------
        dict
            {
                "weather": str,           # 例: "晴"
                "ground_condition": str,  # 例: "良"
                "weather_code": int,      # 数値化済み
                "ground_condition_code": int,
            }
        """
        import time
        time.sleep(settings.scrape_interval_seconds)

        try:
            resp = self._session.get(
                self.SHUTUBA_URL, params={"race_id": race_id}, timeout=15
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            logger.warning(f"Weather fetch failed for {race_id}: {e}")
            return self._unknown()

        race_data = soup.select_one(".RaceData01")
        if race_data is None:
            return self._unknown()

        weather = ""
        ground_condition = ""
        for span in race_data.select("span"):
            text = span.get_text(strip=True)
            if "天候" in span.get("class", [""])[0] if span.get("class") else False:
                weather = text
            if "馬場" in span.get("class", [""])[0] if span.get("class") else False:
                ground_condition = text

        # クラス名に依存できない場合はテキストパース
        if not weather or not ground_condition:
            full_text = race_data.get_text()
            for part in full_text.split("/"):
                part = part.strip()
                if "天候:" in part:
                    weather = part.replace("天候:", "").strip()
                if "馬場:" in part:
                    ground_condition = part.replace("馬場:", "").strip()

        return {
            "weather": weather,
            "ground_condition": ground_condition,
            "weather_code": WEATHER_MAP.get(weather, -1),
            "ground_condition_code": GROUND_CONDITION_MAP.get(ground_condition, -1),
        }

    @staticmethod
    def _unknown() -> dict:
        return {
            "weather": "不明",
            "ground_condition": "不明",
            "weather_code": -1,
            "ground_condition_code": -1,
        }
