"""
アプリケーション全体の設定管理。

Settings        : .env から機密情報（LINE トークン等）を読み込む（pydantic-settings）
StrategyConfig  : config/strategy.yaml から運用パラメータを読み込む
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────────────
# StrategyConfig  ―  strategy.yaml から読み込む運用パラメータ
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BettingConfig:
    """買い目生成の戦略パラメータ。"""
    target_surface:          list[str] = field(default_factory=lambda: ["ダート"])
    min_ev_threshold:        float = 2.0
    max_bets_per_race:       int   = 5        # 0 = 上限なし
    longshot_odds_max:       float = 30.0
    est_quinella_odds_max:   float = 50.0     # 推定馬連オッズ上限
    min_p_model_threshold:   float = 0.05     # AI予測馬連確率の下限
    partner_top_n:           int   = 5        # 軸馬流し: 3位以下から何頭をパートナーとするか
    bet_type:                str   = "quinella"  # 券種: quinella（馬連）固定
    axis_max_odds:           float = 10.0     # 軸馬の単勝オッズ上限（超えたらレース見送り）
    always_predict_g1:       bool  = True     # G1特別モード: 全フィルターをバイパスして必ず予測
    shadow_mode:             bool  = False    # True: LINE通知（実弾シグナル）を停止し、全候補ペアをログ出力するのみ
    gatekeeper_threshold:    float = 0.50     # Two-Brain: P_axis_safe がこれ未満の軸馬を棄却


@dataclass
class StrategyConfig:
    """strategy.yaml 全体の設定。"""
    betting:              BettingConfig
    ltr_model_path:       Path
    stats_path:           Path
    gatekeeper_model_path: Path
    pair_calibrator_path:  Path

    @classmethod
    def load(cls, path: Path | None = None) -> "StrategyConfig":
        """strategy.yaml を読み込んで StrategyConfig を返す。"""
        yaml_path = path or (BASE_DIR / "config" / "strategy.yaml")
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        b = raw.get("betting", {})
        m = raw.get("model", {})

        max_bets_raw = b.get("max_bets_per_race", 3)
        max_bets = int(max_bets_raw) if max_bets_raw else 0   # null/0 → 上限なし

        return cls(
            betting=BettingConfig(
                target_surface        = list(b.get("target_surface", ["ダート"])),
                min_ev_threshold      = float(b.get("min_ev_threshold", 1.5)),
                max_bets_per_race     = max_bets,
                longshot_odds_max     = float(b.get("longshot_odds_max", 30.0)),
                est_quinella_odds_max = float(b.get("est_quinella_odds_max",
                                                    b.get("est_wide_odds_max", 50.0))),
                min_p_model_threshold = float(b.get("min_p_model_threshold", 0.05)),
                partner_top_n         = int(b.get("partner_top_n",
                                                   b.get("box_top_n", 5))),
                bet_type              = str(b.get("bet_type", "quinella")),
                axis_max_odds         = float(b.get("axis_max_odds", 10.0)),
                always_predict_g1     = bool(b.get("always_predict_g1", True)),
                shadow_mode           = bool(b.get("shadow_mode", False)),
                gatekeeper_threshold  = float(b.get("gatekeeper_threshold", 0.50)),
            ),
            ltr_model_path = BASE_DIR / m.get("ltr_model_path",
                                               "data/models/lgbm_ltr_model.pkl"),
            stats_path     = BASE_DIR / m.get("stats_path",
                                               "data/models/feature_stats.pkl"),
            gatekeeper_model_path = BASE_DIR / m.get("gatekeeper_model_path",
                                               "data/models/gatekeeper_model.pkl"),
            pair_calibrator_path  = BASE_DIR / m.get("pair_calibrator_path",
                                               "data/models/pair_calibrator.pkl"),
        )


# シングルトン（インポート時に自動読み込み）
try:
    strategy = StrategyConfig.load()
except FileNotFoundError:
    # strategy.yaml が未作成の場合はデフォルト値を使用
    strategy = StrategyConfig(
        betting=BettingConfig(),
        ltr_model_path=BASE_DIR / "data" / "models" / "lgbm_ltr_model.pkl",
        stats_path=BASE_DIR / "data" / "models" / "feature_stats.pkl",
        gatekeeper_model_path=BASE_DIR / "data" / "models" / "gatekeeper_model.pkl",
        pair_calibrator_path=BASE_DIR / "data" / "models" / "pair_calibrator.pkl",
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LINE Messaging API（送信時のみ必須。データ収集・学習では不要）
    # ------------------------------------------------------------------
    line_channel_access_token: str = ""
    line_channel_secret: str = ""
    line_target_user_id: str = ""

    # ------------------------------------------------------------------
    # netkeiba 認証
    # ------------------------------------------------------------------
    netkeiba_email: str = ""
    netkeiba_password: str = ""

    # ------------------------------------------------------------------
    # Selenium
    # ------------------------------------------------------------------
    selenium_headless: bool = True
    chromedriver_path: str = ""

    # ------------------------------------------------------------------
    # スクレイピング
    # ------------------------------------------------------------------
    scrape_interval_seconds: float = 3.0
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    # ------------------------------------------------------------------
    # モデル（JRA）
    # ------------------------------------------------------------------
    model_path: Path = BASE_DIR / "data" / "models" / "lgbm_model.pkl"
    stats_path: Path = BASE_DIR / "data" / "models" / "feature_stats.pkl"
    enable_shap: bool = True

    # ------------------------------------------------------------------
    # スケジューラー
    # ------------------------------------------------------------------
    notify_before_minutes: int = 20
    # JRA 監視対象の競馬場コード（"05,06" のようなカンマ区切り文字列）
    # デフォルトは全10会場（空文字の場合も全会場対象）
    target_jyo_codes: str = "01,02,03,04,05,06,07,08,09,10"

    @property
    def target_jyo_code_list(self) -> list[str]:
        return [c.strip() for c in self.target_jyo_codes.split(",") if c.strip()]

    # ------------------------------------------------------------------
    # ログ
    # ------------------------------------------------------------------
    log_level: str = "INFO"
    log_file: Path = BASE_DIR / "logs" / "keiba.log"

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        return v.upper()

    # ------------------------------------------------------------------
    # パス解決ヘルパー
    # ------------------------------------------------------------------
    @property
    def data_raw_dir(self) -> Path:
        return BASE_DIR / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return BASE_DIR / "data" / "processed"


# シングルトンとして利用
settings = Settings()  # type: ignore[call-arg]
