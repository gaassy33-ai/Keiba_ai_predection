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
from pathlib import Path

import schedule
import time
from loguru import logger

from config.settings import settings
from src.scraper.netkeiba_scraper import NetkeibaScraper
from src.scraper.race_schedule import RaceScheduleFetcher, select_main_race
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
        fe = FeatureEngineer.from_stats(settings.stats_path)
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

        # 9. GitHub Pages HTML 更新
        try:
            from src.line.notifier import _result_to_race_data
            from src.line.page_generator import generate_prediction_page, generate_results_page
            from src.betting.strategy import generate_betting_strategies

            race_data = _result_to_race_data(
                result=result,
                shap_text=shap_text,
                ground_condition=weather_info["ground_condition"],
                weather=weather_info["weather"],
            )
            top_horses = [h for h in [result.honmei, result.taikou, result.ana] if h]
            for h in top_horses:
                h.setdefault("win_odds", None)
            race_data["strategies"] = generate_betting_strategies(top_horses)
            race_data["budget"] = 10_000

            generate_prediction_page(race_data, Path("docs/today.html"))
            generate_results_page(date.today(), Path("docs/results.html"))
            logger.info("GitHub Pages updated.")
        except Exception as e:
            logger.warning(f"Page generation skipped: {e}")

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
    with RaceScheduleFetcher() as fetcher:
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

    判定ロジック:
      notify_at = start_time - notify_before_minutes
      -15 ≤ now - notify_at ≤ 20 のレースを通知する。
      ※ now は Selenium 呼び出し前に取得（Selenium遅延の影響を除去）
      ※ GitHub Actions の起動遅延 (最大15分) + Workflow実行時間を考慮して±15+αを設定
    """
    # Selenium 呼び出し前に現在時刻を取得（ドリフト防止）
    now = datetime.now()
    logger.info(f"run_once_for_date: now={now.strftime('%H:%M:%S')}")

    try:
        with RaceScheduleFetcher() as fetcher:
            races = fetcher.fetch_race_list(target_date)
            races = fetcher.filter_by_jyo(races)
    except Exception as e:
        logger.error(f"スケジュール取得失敗: {e}")
        try:
            LineNotifier().send_text(f"⚠️ レーススケジュール取得エラー:\n{type(e).__name__}: {e}")
        except Exception:
            pass
        return

    notify_delta = timedelta(minutes=settings.notify_before_minutes)
    # GitHub Actions の起動遅延(最大15分) + Selenium時間(~1分) を考慮したウィンドウ
    #  -5分: 理想より少し早いクーロン実行を許容（二重送信防止のため小さく）
    # +20分: 理想より20分遅い実行(起動遅延15分 + 処理5分)を許容
    early_margin  = timedelta(minutes=5)
    late_margin   = timedelta(minutes=20)

    for race in races:
        notify_at = race["start_time"] - notify_delta
        delta = now - notify_at  # 正 = 通知タイミング過ぎ、負 = まだ早い
        in_window = -early_margin <= delta <= late_margin
        logger.info(
            f"  {race['race_name'][:8]} R{race['race_number']} "
            f"start={race['start_time'].strftime('%H:%M')} "
            f"notify_at={notify_at.strftime('%H:%M')} "
            f"delta={int(delta.total_seconds()//60):+d}min "
            f"{'→ NOTIFY' if in_window else ''}"
        )
        if in_window:
            run_pipeline_for_race(race)


def run_morning_pages() -> None:
    """
    朝7時に当日メインレース（最大R番号）の予想ページを生成する。
    LINE通知は行わない。

    段階的フォールバック:
    1. 完全予測成功   → 予測付きページを生成
    2. 予測失敗       → 出走表のみのページを生成
    3. スケジュール取得失敗 → 何もしない（終了）
    """
    import pandas as pd
    from src.line.page_generator import generate_prediction_page, generate_results_page

    # ── STEP 1: レーススケジュール取得 ────────────────────────────────
    try:
        with RaceScheduleFetcher() as fetcher:
            all_races = fetcher.fetch_race_list(date.today())
            races = fetcher.filter_by_jyo(all_races)
        logger.info(f"全レース数: {len(all_races)}")
        for r in all_races:
            logger.info(f"  race_id={r['race_id']} jyo={r.get('jyo_name')} R{r.get('race_number')} {r.get('race_name')}")
        logger.info(f"filter_by_jyo 後: {len(races)} 件 (target_jyo_codes={settings.target_jyo_codes})")

        if not races:
            # フィルタで全て除外された場合は全レースを対象にする
            logger.warning("対象競馬場コードに一致するレースなし → 全レースを対象に切り替えます")
            races = all_races

        if not races:
            logger.info("本日レースなし。")
            _generate_no_race_page(Path("docs/today.html"))
            return

        main_race  = select_main_race(races)
        race_id    = main_race["race_id"]
        race_name  = main_race.get("race_name", race_id)
        start_time = main_race.get("start_time")
        deadline   = start_time.strftime("%H:%M") if start_time else ""
        logger.info(f"メインレース決定: {race_name} ({race_id})")
    except Exception as e:
        logger.error(f"レーススケジュール取得失敗: {e}")
        _generate_no_race_page(Path("docs/today.html"))
        return

    # ── STEP 2: 完全予測パイプライン ──────────────────────────────────
    try:
        from src.line.notifier import _result_to_race_data
        from src.betting.strategy import generate_betting_strategies

        weather_fetcher = WeatherFetcher()
        weather_info = weather_fetcher.fetch(race_id)

        with NetkeibaScraper() as scraper:
            race_info = scraper.fetch_today_entries(race_id)
            pedigree_map: dict[str, dict] = {}
            for entry in race_info.entries:
                if entry.horse_id:
                    pedigree_map[entry.horse_id] = scraper.fetch_horse_pedigree(entry.horse_id)

        entry_records = []
        for e in race_info.entries:
            ped = pedigree_map.get(e.horse_id, {})
            entry_records.append({
                "horse_id":       e.horse_id,
                "horse_name":     e.horse_name,
                "horse_number":   e.horse_number,
                "frame_number":   e.frame_number,
                "sex":            e.sex,
                "age":            e.age,
                "weight_carried": e.weight_carried,
                "jockey_id":      e.jockey_id,
                "jockey_name":    e.jockey_name,
                "father":         ped.get("father", ""),
                "mother_father":  ped.get("mother_father", ""),
            })
        entry_df = pd.DataFrame(entry_records)

        fe = FeatureEngineer.from_stats(settings.stats_path)
        feature_df = fe.build_entry_features(
            entry_df=entry_df,
            course_type=race_info.course_type,
            distance=race_info.distance,
            ground_condition_code=weather_info["ground_condition_code"],
            weather_code=weather_info["weather_code"],
        )

        predictor = RacePredictor.from_saved_model()
        result    = predictor.predict(race_id, race_name, feature_df)

        trainer   = ModelTrainer.load()
        explainer = PredictionExplainer(trainer)
        shap_text = explainer.explain_text(result, feature_df) if settings.enable_shap else ""

        race_data = _result_to_race_data(
            result=result,
            shap_text=shap_text,
            ground_condition=weather_info["ground_condition"],
            weather=weather_info["weather"],
        )
        top_horses = [h for h in [result.honmei, result.taikou, result.ana] if h]
        for h in top_horses:
            h.setdefault("win_odds", None)
        race_data["strategies"] = generate_betting_strategies(top_horses)
        race_data["budget"] = 10_000

        generate_prediction_page(race_data, Path("docs/today.html"))
        generate_results_page(date.today(), Path("docs/results.html"))
        logger.info("Morning pages generated with full prediction.")

    except Exception as e:
        # ── フォールバック: 出走表のみのページを生成 ──────────────────
        logger.warning(f"完全予測失敗（{type(e).__name__}: {e}）。出走表のみのページを生成します。")
        try:
            _generate_entries_page(race_id, race_name, deadline, Path("docs/today.html"))
            generate_results_page(date.today(), Path("docs/results.html"))
            logger.info("Morning pages generated with entries only (fallback).")
        except Exception as e2:
            logger.error(f"フォールバックページ生成も失敗: {e2}")


def _generate_no_race_page(path: Path) -> None:
    """本日レースなし / スケジュール取得失敗時のフォールバックページ。"""
    from src.line.page_generator import _html_doc
    body = """
<div class="card" style="text-align:center; padding:30px 20px">
  <p style="font-size:32px; margin-bottom:12px">🏇</p>
  <p style="font-weight:bold; margin-bottom:8px">本日の予想</p>
  <p style="color:#8b949e; font-size:14px">
    本日のレーススケジュールを取得中です。<br>
    しばらくしてから再度お試しください。
  </p>
</div>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_html_doc("AI予想", body, active_page="today"), encoding="utf-8")


def _generate_entries_page(race_id: str, race_name: str, deadline: str, path: Path) -> None:
    """
    AI予測なしの出走表のみページを生成するフォールバック。
    """
    from src.line.page_generator import _html_doc, PAGES_BASE_URL
    import html as html_mod

    netkeiba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    deadline_html = f'<p class="deadline">⏰ 締め切り {html_mod.escape(deadline)}</p>' if deadline else ""

    body = f"""
<div class="header">
  <p class="race-name">{html_mod.escape(race_name)}</p>
  {deadline_html}
</div>
<div class="card" style="text-align:center; padding:20px">
  <p style="font-size:28px; margin-bottom:10px">🏇</p>
  <p style="font-weight:bold; margin-bottom:6px">AI予想を準備中</p>
  <p style="color:#8b949e; font-size:13px; margin-bottom:16px">
    出走表・オッズはnetkeibaでご確認ください
  </p>
  <a href="{html_mod.escape(netkeiba_url)}" class="netkeiba-btn" target="_blank">
    netkeiba で出走表を見る
  </a>
</div>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_html_doc(f"AI予想 - {race_name}", body, active_page="today"), encoding="utf-8")


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
    parser.add_argument(
        "--morning",
        action="store_true",
        help="朝7時モード: 当日メインレースの予想ページを生成して終了（LINE通知なし）",
    )
    args = parser.parse_args()

    if args.test:
        logger.info("Sending test LINE notification...")
        send_test_notification()
        return

    if args.morning:
        logger.info("Running in --morning mode")
        run_morning_pages()
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
