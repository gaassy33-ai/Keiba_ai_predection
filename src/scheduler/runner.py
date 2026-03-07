"""
スケジューラー。

発走時刻の 20 分前に推論 → LINE 送信を実行する。
Python の schedule ライブラリを使ったポーリング方式で実装。
GitHub Actions での実行にも対応（--once フラグ）。
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta

import schedule
import time
from loguru import logger

from config.settings import settings
from src.scraper.netkeiba_scraper import NetkeibaScraper
from src.scraper.race_schedule import RaceScheduleFetcher
from src.scraper.weather import WeatherFetcher
from src.features.engineer import FeatureEngineer
from src.model.predictor import RacePredictor
from src.model.explainer import PredictionExplainer
from src.model.trainer import ModelTrainer
from src.line.notifier import LineNotifier


# 処理済みレース ID を管理（同一レースを二重送信しない）
_notified_race_ids: set[str] = set()


def _setup_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        settings.log_file,
        level=settings.log_level,
        rotation="1 day",
        retention="7 days",
        encoding="utf-8",
    )


def run_pipeline_for_race(race: dict) -> None:
    """
    1レース分の推論 → LINE 送信パイプラインを実行する。

    Parameters
    ----------
    race : dict
        {"race_id": str, "race_name": str, "start_time": datetime, ...}
    """
    race_id = race["race_id"]
    race_name = race.get("race_name", race_id)

    if race_id in _notified_race_ids:
        logger.debug(f"Already notified: {race_id}")
        return

    logger.info(f"Running pipeline for {race_name} (race_id={race_id})")

    notifier = LineNotifier()

    try:
        # 1. 天候・馬場状態を取得
        weather_fetcher = WeatherFetcher()
        weather_info = weather_fetcher.fetch(race_id)

        # 2. 当日出走表を取得
        with NetkeibaScraper() as scraper:
            race_info = scraper.fetch_today_entries(race_id)

            # 3. 血統情報を付加
            pedigree_map: dict[str, dict] = {}
            for entry in race_info.entries:
                if entry.horse_id:
                    ped = scraper.fetch_horse_pedigree(entry.horse_id)
                    pedigree_map[entry.horse_id] = ped

        # 4. entry_df を組み立て
        import pandas as pd
        entry_records = []
        for e in race_info.entries:
            ped = pedigree_map.get(e.horse_id, {})
            entry_records.append({
                "horse_id": e.horse_id,
                "horse_name": e.horse_name,
                "horse_number": e.horse_number,
                "frame_number": e.frame_number,
                "sex": e.sex,
                "age": e.age,
                "weight_carried": e.weight_carried,
                "jockey_id": e.jockey_id,
                "jockey_name": e.jockey_name,
                "father": ped.get("father", ""),
                "mother_father": ped.get("mother_father", ""),
            })
        entry_df = pd.DataFrame(entry_records)

        # 5. 特徴量エンジニアリング
        # NOTE: 本番では事前にロードした history_df を渡す
        # ここでは空 DataFrame で初期化（初回実行時はモデルの特徴量のみ）
        fe = FeatureEngineer(pd.DataFrame())
        feature_df = fe.build_entry_features(
            entry_df=entry_df,
            course_type=race_info.course_type,
            distance=race_info.distance,
            ground_condition_code=weather_info["ground_condition_code"],
            weather_code=weather_info["weather_code"],
        )

        # 6. 推論
        predictor = RacePredictor.from_saved_model()
        result = predictor.predict(race_id, race_name, feature_df)

        # 7. SHAP 説明
        trainer = ModelTrainer.load()
        explainer = PredictionExplainer(trainer)
        shap_text = explainer.explain_text(result, feature_df) if settings.enable_shap else ""

        # 8. LINE 送信
        notifier.send_prediction(
            result=result,
            shap_text=shap_text,
            ground_condition=weather_info["ground_condition"],
            weather=weather_info["weather"],
        )

        _notified_race_ids.add(race_id)

    except FileNotFoundError:
        msg = f"[{race_name}] モデルファイルが見つかりません。先に keiba-train を実行してください。"
        logger.error(msg)
        notifier.send_text(msg)
    except Exception as e:
        msg = f"[{race_name}] パイプラインエラー: {e}"
        logger.exception(msg)
        try:
            notifier.send_text(msg)
        except Exception:
            pass


def schedule_today_races() -> None:
    """
    当日のレース一覧を取得し、発走 N 分前のジョブを登録する。
    """
    fetcher = RaceScheduleFetcher()
    races = fetcher.fetch_race_list(date.today())
    races = fetcher.filter_by_jyo(races)

    notify_delta = timedelta(minutes=settings.notify_before_minutes)
    now = datetime.now()

    registered = 0
    for race in races:
        notify_at: datetime = race["start_time"] - notify_delta

        if notify_at <= now:
            logger.info(f"Skipping past race: {race['race_name']} (notify_at={notify_at})")
            continue

        notify_time_str = notify_at.strftime("%H:%M")
        # schedule はクロージャで race をキャプチャする必要がある
        (lambda r: schedule.every().day.at(notify_time_str).do(run_pipeline_for_race, race=r))(race)
        logger.info(
            f"Scheduled: {race['race_name']} @ {notify_time_str} "
            f"(発走={race['start_time'].strftime('%H:%M')})"
        )
        registered += 1

    logger.info(f"Total {registered} races scheduled.")


def run_once_for_date(target_date: date) -> None:
    """
    GitHub Actions から呼び出す用。
    当日のレース全てに対して即時実行する（発走前のみ）。
    """
    fetcher = RaceScheduleFetcher()
    races = fetcher.fetch_race_list(target_date)
    races = fetcher.filter_by_jyo(races)

    notify_delta = timedelta(minutes=settings.notify_before_minutes)
    now = datetime.now()

    for race in races:
        notify_at = race["start_time"] - notify_delta
        # GitHub Actions は 20 分おきに起動するため、±10 分ウィンドウで判定
        window = timedelta(minutes=10)
        if abs(now - notify_at) <= window:
            run_pipeline_for_race(race)


def send_test_notification() -> None:
    """LINE 接続確認用のテスト通知を送る。"""
    notifier = LineNotifier()
    notifier.send_text(
        "【テスト】競馬予想AI の LINE 通知設定が完了しました。\n"
        "土日のレース20分前に予想が届きます。"
    )
    logger.info("Test notification sent.")


def main() -> None:
    """CLI エントリーポイント: keiba-run"""
    _setup_logger()

    parser = argparse.ArgumentParser(description="競馬予想スケジューラー")
    parser.add_argument(
        "--once",
        action="store_true",
        help="GitHub Actions 用: 現時刻付近のレースのみ即時実行して終了",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="LINE 接続確認用のテスト通知を送って終了",
    )
    args = parser.parse_args()

    if args.test:
        logger.info("Sending test LINE notification...")
        send_test_notification()
        return

    if args.once:
        logger.info("Running in --once mode (GitHub Actions)")
        run_once_for_date(date.today())
        return

    # 通常モード: 当日レースをスケジュール登録してポーリング
    logger.info("Starting scheduler daemon...")
    schedule_today_races()

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
