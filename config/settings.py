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
    # モデル
    # ------------------------------------------------------------------
    model_path: Path = BASE_DIR / "data" / "models" / "lgbm_model.pkl"
    stats_path: Path = BASE_DIR / "data" / "models" / "feature_stats.pkl"
    enable_shap: bool = True

    # ------------------------------------------------------------------
    # スケジューラー
    # ------------------------------------------------------------------
    notify_before_minutes: int = 20
    # 監視対象の競馬場コード（"05,06" のようなカンマ区切り文字列）
    target_jyo_codes: str = "05,06,08,09"

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
