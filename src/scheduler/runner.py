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
        # 1. 当日出走表を取得（Selenium → course_type / distance / weather / ground_condition を含む）
        with NetkeibaScraper() as scraper:
            race_info = scraper.fetch_today_entries(race_id)

            # 3. 血統情報・直近成績を付加
            pedigree_map: dict[str, dict] = {}
            recent_form_map: dict[str, dict] = {}
            for entry in race_info.entries:
                if entry.horse_id:
                    pedigree_map[entry.horse_id] = scraper.fetch_horse_pedigree(entry.horse_id)
                    recent_form_map[entry.horse_id] = scraper.fetch_horse_recent_form(entry.horse_id)

        # 4. entry_df を組み立て
        import pandas as pd
        entry_records = []
        for e in race_info.entries:
            ped  = pedigree_map.get(e.horse_id, {})
            form = recent_form_map.get(e.horse_id, {})
            entry_records.append({
                "horse_id":          e.horse_id,
                "horse_name":        e.horse_name,
                "horse_number":      e.horse_number,
                "frame_number":      e.frame_number,
                "sex":               e.sex,
                "age":               e.age,
                "weight_carried":    e.weight_carried,
                "jockey_id":         e.jockey_id,
                "jockey_name":       e.jockey_name,
                "father":            ped.get("father", ""),
                "mother_father":     ped.get("mother_father", ""),
                "recent_avg_pos":    form.get("recent_avg_pos", float("nan")),
                "recent_avg_last3f": form.get("recent_avg_last3f", float("nan")),
            })
        entry_df = pd.DataFrame(entry_records)

        # 5. 特徴量エンジニアリング
        from src.scraper.weather import GROUND_CONDITION_MAP, WEATHER_MAP
        ground_condition_code = GROUND_CONDITION_MAP.get(race_info.ground_condition, -1)
        weather_code          = WEATHER_MAP.get(race_info.weather, -1)

        fe = FeatureEngineer.from_stats(settings.stats_path)
        feature_df = fe.build_entry_features(
            entry_df=entry_df,
            course_type=race_info.course_type,
            distance=race_info.distance,
            ground_condition_code=ground_condition_code,
            weather_code=weather_code,
        )

        # 6. 推論
        predictor = RacePredictor.from_saved_model()
        result = predictor.predict(race_id, race_name, feature_df)

        # 7. SHAP 説明
        trainer = ModelTrainer.load()
        explainer = PredictionExplainer(trainer)
        shap_text = explainer.explain_text(result, feature_df) if settings.enable_shap else ""

        # 8. LINE 送信
        from datetime import timedelta
        start_time = race.get("start_time")
        deadline = (start_time - timedelta(minutes=2)).strftime("%H:%M") if start_time else ""

        notifier.send_prediction(
            result=result,
            shap_text=shap_text,
            course_type=race_info.course_type,
            distance=race_info.distance,
            ground_condition=race_info.ground_condition or "不明",
            weather=race_info.weather or "不明",
            deadline=deadline,
        )

        # 9. GitHub Pages HTML 更新
        try:
            from src.line.notifier import _result_to_race_data
            from src.line.page_generator import generate_prediction_page, generate_results_page
            from src.betting.strategy import generate_betting_strategies

            race_data = _result_to_race_data(
                result=result,
                shap_text=shap_text,
                course_type=race_info.course_type,
                distance=race_info.distance,
                ground_condition=race_info.ground_condition or "不明",
                weather=race_info.weather or "不明",
                deadline=deadline,
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


def run_for_race_id(race_id: str) -> None:
    """
    CF Worker dispatch 用: 特定の race_id に対してパイプラインを即時実行する。

    time-window チェックは行わず無条件に実行する（dispatch 側が時刻制御済み）。
    race_schedule.json があれば race_name / start_time を補完する。
    """
    import json
    from pathlib import Path

    race: dict = {"race_id": race_id}

    schedule_path = Path("docs/race_schedule.json")
    if schedule_path.exists():
        try:
            schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
            for entry in schedule:
                if entry.get("race_id") == race_id:
                    race["race_name"] = entry.get("race_name", race_id)
                    if entry.get("start_time"):
                        race["start_time"] = datetime.fromisoformat(entry["start_time"])
                    break
        except Exception as e:
            logger.warning(f"race_schedule.json 読み込み失敗: {e}")

    logger.info(f"run_for_race_id: race_id={race_id} name={race.get('race_name', '?')}")
    run_pipeline_for_race(race)


def export_race_schedule(target_date: date | None = None) -> None:
    """
    当日のレーススケジュールを docs/race_schedule.json に書き出す。

    CF Worker がこの JSON を fetch して20分前に dispatch を行う。
    朝バッチ（--morning）の最後に呼ばれる想定。

    出力形式:
    [
      {
        "date": "YYYY-MM-DD",
        "race_id": "202506010811",
        "race_name": "春のステークス",
        "race_number": 11,
        "jyo_name": "東京",
        "start_time": "2025-06-01T15:40:00",  // ISO8601 JST
        "notify_at":  "2025-06-01T15:20:00"   // start_time - notify_before_minutes
      },
      ...
    ]
    """
    import json
    from pathlib import Path

    if target_date is None:
        target_date = date.today()

    try:
        with RaceScheduleFetcher() as fetcher:
            all_races = fetcher.fetch_race_list(target_date)
            races = fetcher.filter_by_jyo(all_races)
        if not races:
            races = all_races  # フォールバック: 全レース
    except Exception as e:
        logger.error(f"export_race_schedule: スケジュール取得失敗: {e}")
        return

    notify_delta = timedelta(minutes=settings.notify_before_minutes)
    entries = []
    for race in races:
        notify_at = race["start_time"] - notify_delta
        entries.append({
            "date":         target_date.isoformat(),
            "race_id":      race["race_id"],
            "race_name":    race.get("race_name", ""),
            "race_number":  race.get("race_number", 0),
            "jyo_name":     race.get("jyo_name", ""),
            "start_time":   race["start_time"].isoformat(),
            "notify_at":    notify_at.isoformat(),
        })

    out_path = Path("docs/race_schedule.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"race_schedule.json: {len(entries)} 件 → {out_path}")


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

        with NetkeibaScraper() as scraper:
            race_info = scraper.fetch_today_entries(race_id)
            pedigree_map: dict[str, dict] = {}
            recent_form_map: dict[str, dict] = {}
            for entry in race_info.entries:
                if entry.horse_id:
                    pedigree_map[entry.horse_id] = scraper.fetch_horse_pedigree(entry.horse_id)
                    recent_form_map[entry.horse_id] = scraper.fetch_horse_recent_form(entry.horse_id)

        entry_records = []
        for e in race_info.entries:
            ped  = pedigree_map.get(e.horse_id, {})
            form = recent_form_map.get(e.horse_id, {})
            entry_records.append({
                "horse_id":          e.horse_id,
                "horse_name":        e.horse_name,
                "horse_number":      e.horse_number,
                "frame_number":      e.frame_number,
                "sex":               e.sex,
                "age":               e.age,
                "weight_carried":    e.weight_carried,
                "jockey_id":         e.jockey_id,
                "jockey_name":       e.jockey_name,
                "father":            ped.get("father", ""),
                "mother_father":     ped.get("mother_father", ""),
                "recent_avg_pos":    form.get("recent_avg_pos", float("nan")),
                "recent_avg_last3f": form.get("recent_avg_last3f", float("nan")),
            })
        entry_df = pd.DataFrame(entry_records)

        from src.scraper.weather import GROUND_CONDITION_MAP, WEATHER_MAP
        fe = FeatureEngineer.from_stats(settings.stats_path)
        feature_df = fe.build_entry_features(
            entry_df=entry_df,
            course_type=race_info.course_type,
            distance=race_info.distance,
            ground_condition_code=GROUND_CONDITION_MAP.get(race_info.ground_condition, -1),
            weather_code=WEATHER_MAP.get(race_info.weather, -1),
        )

        predictor = RacePredictor.from_saved_model()
        result    = predictor.predict(race_id, race_name, feature_df)

        trainer   = ModelTrainer.load()
        explainer = PredictionExplainer(trainer)
        shap_text = explainer.explain_text(result, feature_df) if settings.enable_shap else ""

        race_data = _result_to_race_data(
            result=result,
            shap_text=shap_text,
            course_type=race_info.course_type,
            distance=race_info.distance,
            ground_condition=race_info.ground_condition or "不明",
            weather=race_info.weather or "不明",
            deadline=deadline,
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


def _build_marks(result: "PredictionResult") -> dict:
    """
    PredictionResult から Flex Message 用の marks dict を構築する。

    Returns
    -------
    dict
        {
            "honmei":   {horse_number, frame_number, horse_name, win_prob} | None,
            "taikou":   {...} | None,
            "tanana":   {...} | None,
            "hoshi":    {...} | None,
            "renshita": [{"horse_number": int}, ...],   # 馬番のみ
        }
    """
    def _slim(h: dict | None) -> dict | None:
        if not h:
            return None
        return {
            "horse_number": int(h.get("horse_number", 0)),
            "frame_number": int(h.get("frame_number", 0)),
            "horse_name":   str(h.get("horse_name", "")),
            "win_prob":     float(h.get("win_prob", 0.0)),
        }

    return {
        "honmei":   _slim(result.honmei),
        "taikou":   _slim(result.taikou),
        "tanana":   _slim(result.tanana),
        "hoshi":    _slim(result.hoshi),
        "renshita": [
            {"horse_number": int(h.get("horse_number", 0))}
            for h in (result.renshita or []) if h
        ],
    }


def _make_bet_label(strategies: list) -> str:
    """BetLine リストから短縮買い目ラベルを返す（カルーセル行表示用）。"""
    if not strategies:
        return "−"
    best = max(strategies, key=lambda s: s.ev)
    abbr_map = {
        "単勝": "単",  "複勝": "複",  "馬連": "馬連",
        "馬単": "馬単", "ワイド": "W", "3連複": "3複", "3連単": "3単",
    }
    abbr = abbr_map.get(best.bet_type, best.bet_type[:2])
    return f"{abbr} {best.ev:.1f}x" if best.ev >= 1.3 else abbr


def _process_race_for_morning(
    scraper: NetkeibaScraper,
    race: dict,
    pedigree_cache: dict[str, dict],
    predictor: "RacePredictor",
    fe: "FeatureEngineer",
) -> dict:
    """
    朝バッチ用: 1レースを処理して予測結果 dict を返す。

    recent_form（直近成績）はスキップして NaN 扱いにする。
    これにより全レース処理を ~30 分以内に収める（1時間枠に収まる）。

    Returns
    -------
    dict  {race_id, race_number, race_name, start_time, course_type,
           distance, ground_condition, weather, honmei, best_bet_label, error}
    """
    import pandas as pd
    from src.scraper.weather import GROUND_CONDITION_MAP, WEATHER_MAP
    from src.betting.strategy import generate_betting_strategies

    race_id    = race["race_id"]
    start_time = race.get("start_time")

    # 1. 出走表取得（最新の馬場状態・出走取消馬を反映）
    race_info = scraper.fetch_today_entries(race_id)
    if not race_info.entries:
        raise ValueError("出走馬なし（取消/取得失敗）")

    # 2. 血統取得（horse_id ベースのキャッシュで重複アクセスを防止）
    for entry in race_info.entries:
        if entry.horse_id and entry.horse_id not in pedigree_cache:
            try:
                pedigree_cache[entry.horse_id] = scraper.fetch_horse_pedigree(entry.horse_id)
            except Exception as e:
                logger.debug(f"pedigree skip: {entry.horse_id} ({e})")
                pedigree_cache[entry.horse_id] = {}

    # 3. entry_df 組み立て（recent_form は NaN → LightGBM の欠損値処理に委ねる）
    records = []
    for e in race_info.entries:
        ped = pedigree_cache.get(e.horse_id, {})
        records.append({
            "horse_id":          e.horse_id,
            "horse_name":        e.horse_name,
            "horse_number":      e.horse_number,
            "frame_number":      e.frame_number,
            "sex":               e.sex,
            "age":               e.age,
            "weight_carried":    e.weight_carried,
            "jockey_id":         e.jockey_id,
            "jockey_name":       e.jockey_name,
            "father":            ped.get("father", ""),
            "mother_father":     ped.get("mother_father", ""),
            "recent_avg_pos":    float("nan"),   # スキップ
            "recent_avg_last3f": float("nan"),   # スキップ
        })
    entry_df = pd.DataFrame(records)

    # 4. 特徴量生成
    ground_code  = GROUND_CONDITION_MAP.get(race_info.ground_condition, -1)
    weather_code = WEATHER_MAP.get(race_info.weather, -1)
    feature_df = fe.build_entry_features(
        entry_df=entry_df,
        course_type=race_info.course_type,
        distance=race_info.distance,
        ground_condition_code=ground_code,
        weather_code=weather_code,
    )

    # 5. 予測
    result = predictor.predict(race_id, race_info.race_name, feature_df)

    # 6. marks dict 構築
    marks = _build_marks(result)

    # 7. 買い目生成（◎○▲☆△ の全馬を渡す）
    all_marked = [result.honmei, result.taikou, result.tanana, result.hoshi] + list(result.renshita)
    top_horses = [h for h in all_marked if h]
    for h in top_horses:
        h.setdefault("win_odds", None)
    from src.betting.strategy import generate_betting_strategies
    strategies     = generate_betting_strategies(top_horses)
    best_bet_label = _make_bet_label(strategies)

    # 信頼度スコア: 本命勝率の絶対値 × (本命 - 対抗) の差分
    # → 本命が高確率かつ2番手との差が大きいほど高スコア
    honmei_prob = float(result.honmei.get("win_prob", 0.0))
    taikou_prob = float(result.taikou.get("win_prob", 0.0)) if result.taikou else 0.0
    confidence_score = honmei_prob * (honmei_prob - taikou_prob)

    return {
        "race_id":          race_id,
        "race_number":      race.get("race_number", 0),
        "race_name":        race_info.race_name,
        "start_time":       start_time.strftime("%H:%M") if start_time else "--:--",
        "course_type":      race_info.course_type,
        "distance":         race_info.distance,
        "ground_condition": race_info.ground_condition,
        "weather":          race_info.weather,
        "marks":            marks,
        "best_bet_label":   best_bet_label,
        "confidence_score": confidence_score,
        "is_main":          False,  # 呼び出し元が上書き
        "is_fire":          False,  # 呼び出し元が上書き
        "error":            None,
    }


def run_morning_all_races(target_date: date | None = None) -> None:
    """
    朝7時バッチ: 全会場・全レースを一括予測して LINE カルーセルで送信する。

    処理フロー:
      1. Selenium でレーススケジュール取得
      2. 会場（jyo_code）ごとにグループ化
      3. 各レースの出走表 + 血統を取得（recent_form はスキップ）
      4. 特徴量生成 → LightGBM 推論
      5. 会場ごとの Flex Bubble → カルーセルで LINE 送信
      6. GitHub Pages 更新 + race_schedule.json 書き出し

    時間見積もり（GitHub Actions）:
      - セットアップ: ~5 分
      - 出走表 + 血統取得: ~25〜30 分（30 レース × ~50 秒）
      - 推論・送信: ~2 分
      - 合計: ~35〜40 分（timeout-minutes: 60 で余裕あり）
    """
    from src.line.morning_notifier import (
        create_morning_carousel,
        MorningNotifier,
        _JYO_THEMES,
        _DEFAULT_THEME,
    )
    from src.line.page_generator import generate_results_page

    if target_date is None:
        target_date = date.today()

    logger.info(f"run_morning_all_races: {target_date} 開始")
    notifier = MorningNotifier()

    # ── 1. スケジュール取得 ──────────────────────────────────────────
    try:
        with RaceScheduleFetcher() as fetcher:
            all_races = fetcher.fetch_race_list(target_date)
            races     = fetcher.filter_by_jyo(all_races)
        if not races:
            logger.warning("filter_by_jyo 後 0 件 → 全レース対象に切り替え")
            races = all_races
    except Exception as e:
        logger.error(f"スケジュール取得失敗: {e}")
        notifier.send_text(f"⚠️ スケジュール取得失敗:\n{type(e).__name__}: {e}")
        return

    if not races:
        logger.info("本日レースなし。送信スキップ。")
        return

    # ── 2. 会場ごとにグループ化 ──────────────────────────────────────
    venue_groups: dict[str, list[dict]] = {}
    venue_names:  dict[str, str]        = {}
    for race in races:
        jyo_code = race["race_id"][4:6]
        venue_groups.setdefault(jyo_code, []).append(race)
        if jyo_code not in venue_names:
            venue_names[jyo_code] = (
                race.get("jyo_name") or _JYO_THEMES.get(jyo_code, _DEFAULT_THEME)["name"]
            )

    logger.info(
        f"会場: {list(venue_names.values())} / 総 {len(races)} R"
    )

    # ── 3. 全レース処理（Selenium 1 セッション） ────────────────────
    fe        = FeatureEngineer.from_stats(settings.stats_path)
    predictor = RacePredictor.from_saved_model()
    venue_data_list: list[dict] = []

    with NetkeibaScraper() as scraper:
        pedigree_cache: dict[str, dict] = {}

        for jyo_code in sorted(venue_groups.keys()):
            jyo_races = sorted(venue_groups[jyo_code], key=lambda r: r["race_number"])
            jyo_name  = venue_names[jyo_code]
            main_race_id = select_main_race(jyo_races)["race_id"]

            weather     = ""
            ground_turf = ""
            ground_dirt = ""
            races_results: list[dict] = []

            for race in jyo_races:
                rnum = race.get("race_number", "?")
                logger.info(f"  [{jyo_name}] R{rnum} 処理中...")

                try:
                    rdata = _process_race_for_morning(
                        scraper, race, pedigree_cache, predictor, fe
                    )
                    rdata["is_main"] = (race["race_id"] == main_race_id)
                    races_results.append(rdata)

                    # 天候・馬場の収集（最初の成功レースから）
                    if not weather and rdata.get("weather"):
                        weather = rdata["weather"]
                    ct = rdata.get("course_type", "")
                    if "芝" in ct and not ground_turf:
                        ground_turf = rdata.get("ground_condition", "")
                    elif ("ダ" in ct or "dirt" in ct.lower()) and not ground_dirt:
                        ground_dirt = rdata.get("ground_condition", "")

                except Exception as e:
                    logger.warning(f"  [{jyo_name}] R{rnum} スキップ: {e}")
                    st = race.get("start_time")
                    races_results.append({
                        "race_id":          race["race_id"],
                        "race_number":      race.get("race_number", 0),
                        "start_time":       st.strftime("%H:%M") if st else "--:--",
                        "is_main":          race["race_id"] == main_race_id,
                        "is_fire":          False,
                        "confidence_score": 0.0,
                        "error":            str(e),
                    })

            success_count = sum(1 for r in races_results if not r.get("error"))
            dow = "(土)" if target_date.weekday() == 5 else "(日)"

            venue_data_list.append({
                "jyo_code":    jyo_code,
                "jyo_name":    jyo_name,
                "date_label":  target_date.strftime("%-m/%-d") + dow,
                "race_count":  success_count,
                "weather":     weather,
                "ground_turf": ground_turf,
                "ground_dirt": ground_dirt,
                "kaisai_date": target_date.strftime("%Y%m%d"),
                "races":       races_results,
            })
            logger.info(
                f"  [{jyo_name}] 完了: {success_count}/{len(jyo_races)} R 成功"
            )

    # ── 3b. 全レース中から🔥勝負レースを1つ決定 ────────────────────
    #   信頼度スコア = honmei_prob × (honmei_prob - taikou_prob)
    #   最小条件: error なし かつ honmei_prob >= 0.20
    all_race_flat = [
        r for vd in venue_data_list for r in vd["races"]
    ]
    eligible = [
        r for r in all_race_flat
        if not r.get("error")
        and (r.get("marks") or {}).get("honmei") is not None
        and (r.get("marks", {}).get("honmei") or {}).get("win_prob", 0) >= 0.20
    ]
    if eligible:
        fire_race = max(eligible, key=lambda r: r.get("confidence_score", 0.0))
        fire_race["is_fire"] = True
        logger.info(
            f"🔥 勝負レース: R{fire_race['race_number']} {fire_race.get('race_name', '')} "
            f"(score={fire_race.get('confidence_score', 0):.4f})"
        )

    # ── 4. LINE カルーセル送信 ───────────────────────────────────────
    updated_at = datetime.now().strftime("%H:%M")
    carousel   = create_morning_carousel(venue_data_list, updated_at)

    if carousel:
        notifier.send_carousel(carousel, target_date)
        logger.info(f"カルーセル送信完了: {len(venue_data_list)} 会場")
    else:
        logger.warning("venue_data 空 → カルーセル送信スキップ")

    # ── 5. GitHub Pages 更新 ────────────────────────────────────────
    try:
        generate_results_page(target_date, Path("docs/results.html"))
        logger.info("GitHub Pages 更新完了")
    except Exception as e:
        logger.warning(f"Pages 更新スキップ: {e}")

    # ── 6. race_schedule.json 書き出し ──────────────────────────────
    export_race_schedule(target_date)


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
