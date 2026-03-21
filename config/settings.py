"""
アプリケーション全体の設定管理。
.env ファイルから環境変数を読み込み、型安全に提供する。
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


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
    # モデル（NAR）
    # ------------------------------------------------------------------
    nar_model_path: Path = BASE_DIR / "data" / "models" / "nar_lgbm_model.pkl"
    nar_stats_path: Path = BASE_DIR / "data" / "models" / "nar_feature_stats.pkl"

    # ------------------------------------------------------------------
    # スケジューラー
    # ------------------------------------------------------------------
    notify_before_minutes: int = 20
    # JRA 監視対象の競馬場コード（"05,06" のようなカンマ区切り文字列）
    # デフォルトは全10会場（空文字の場合も全会場対象）
    target_jyo_codes: str = "01,02,03,04,05,06,07,08,09,10"
    # NAR 監視対象の競馬場コード（netkeiba 内部コード）
    # バックテスト結果より ROI プラス圏の4場に限定:
    #   35=盛岡(ROI 185%), 36=水沢(ROI 102%), 42=浦和(ROI 86%), 44=大井(ROI 86%)
    nar_target_jyo_codes: str = "35,36,42,44"

    @property
    def target_jyo_code_list(self) -> list[str]:
        return [c.strip() for c in self.target_jyo_codes.split(",") if c.strip()]

    @property
    def nar_target_jyo_code_list(self) -> list[str]:
        return [c.strip() for c in self.nar_target_jyo_codes.split(",") if c.strip()]

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
