"""
LINE リッチメニュー セットアップスクリプト。

実行:
    python -m src.line.rich_menu setup   # 作成・画像アップロード・デフォルト設定
    python -m src.line.rich_menu delete  # 既存メニューを全削除

レイアウト (2500 × 843 px コンパクト):
    ┌─────────────────┬─────────────────┐  ← y=0
    │   1250×421      │   1250×421      │
    │ ① メインレース  │ ② スケジュール  │
    ├─────────────────┼─────────────────┤  ← y=421
    │   1250×422      │   1250×422      │
    │ ③ 今日の成績    │ ④ SNS          │
    └─────────────────┴─────────────────┘  ← y=843
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
from loguru import logger

from config.settings import settings

# ---------------------------------------------------------------------------
# リッチメニュー定義（エリア座標 + アクション）
# ---------------------------------------------------------------------------

RICH_MENU_DEF: dict = {
    "size": {"width": 2500, "height": 843},   # コンパクトサイズ
    "selected": True,
    "name": "競馬予想メニュー",
    "chatBarText": "🏇 メニューを開く",
    "areas": [
        # ① 今日のメインレース（左上）
        {
            "bounds": {"x": 0, "y": 0, "width": 1250, "height": 421},
            "action": {
                "type": "message",
                "label": "今日のメインレース",
                "text": "main_race",
            },
        },
        # ② 開催スケジュール（右上 / URL）
        {
            "bounds": {"x": 1250, "y": 0, "width": 1250, "height": 421},
            "action": {
                "type": "uri",
                "label": "開催スケジュール",
                "uri": "https://race.netkeiba.com/top/race_list.html",
            },
        },
        # ③ 今日の成績（左下）
        {
            "bounds": {"x": 0, "y": 421, "width": 1250, "height": 422},
            "action": {
                "type": "message",
                "label": "今日の成績",
                "text": "today_result",
            },
        },
        # ④ お問い合わせ/SNS（右下 / URL）
        {
            "bounds": {"x": 1250, "y": 421, "width": 1250, "height": 422},
            "action": {
                "type": "uri",
                "label": "お問い合わせ/SNS",
                "uri": "https://twitter.com/YOUR_ACCOUNT",  # ← 変更してください
            },
        },
    ],
}

_BASE = "https://api.line.me/v2/bot"
_HEADERS = {
    "Authorization": f"Bearer {settings.line_channel_access_token}",
    "Content-Type": "application/json",
}


# ---------------------------------------------------------------------------
# API ヘルパー
# ---------------------------------------------------------------------------

def create_rich_menu() -> str:
    resp = requests.post(f"{_BASE}/richmenu", headers=_HEADERS, json=RICH_MENU_DEF)
    resp.raise_for_status()
    rich_menu_id: str = resp.json()["richMenuId"]
    logger.info(f"Created: {rich_menu_id}")
    return rich_menu_id


def upload_image(rich_menu_id: str, image_path: Path) -> None:
    headers = {
        "Authorization": f"Bearer {settings.line_channel_access_token}",
        "Content-Type": "image/png",
    }
    with image_path.open("rb") as f:
        resp = requests.post(
            f"https://api-data.line.me/v2/bot/richmenu/{rich_menu_id}/content",
            headers=headers,
            data=f,
        )
    resp.raise_for_status()
    logger.info(f"Image uploaded: {image_path}")


def set_default(rich_menu_id: str) -> None:
    resp = requests.post(
        f"{_BASE}/user/all/richmenu/{rich_menu_id}", headers=_HEADERS
    )
    resp.raise_for_status()
    logger.info(f"Set as default: {rich_menu_id}")


def delete_all() -> None:
    resp = requests.get(f"{_BASE}/richmenu/list", headers=_HEADERS)
    resp.raise_for_status()
    for rm in resp.json().get("richmenus", []):
        rm_id = rm["richMenuId"]
        requests.delete(f"{_BASE}/richmenu/{rm_id}", headers=_HEADERS)
        logger.info(f"Deleted: {rm_id}")


# ---------------------------------------------------------------------------
# リッチメニュー画像生成（Pillow）
# ---------------------------------------------------------------------------

def _find_font(size: int):
    """日本語対応フォントを検索して返す。見つからなければ None。"""
    from PIL import ImageFont

    candidates = [
        # Ubuntu (GitHub Actions / Railway)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        # macOS
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return None


def generate_image(output_path: Path) -> None:
    """
    4ボタンのリッチメニュー画像を Pillow で生成する（2500×843px コンパクト）。
    各セルにラベルのみを描画する。
    """
    from PIL import Image, ImageDraw, ImageFont

    W, H = 2500, 843
    img  = Image.new("RGB", (W, H), "#0D1B2A")
    draw = ImageDraw.Draw(img)

    # セル定義: (x, y, w, h, label, sub_label, bg_color)
    cells = [
        (0,    0,   1250, 421,  "今日のメインレース", "AI予想 & 推奨買い目",   "#B71C1C"),
        (1250, 0,   1250, 421,  "開催スケジュール",   "netkeiba へ移動",       "#0D47A1"),
        (0,    421, 1250, 422,  "今日の成績",         "的中率 & 収支グラフ",   "#1B5E20"),
        (1250, 421, 1250, 422,  "お問い合わせ / SNS", "X (Twitter) へ移動",   "#4A148C"),
    ]

    font_lg = _find_font(90)   # メインラベル
    font_sm = _find_font(55)   # サブラベル
    fallback = ImageFont.load_default()
    if font_lg is None: font_lg = fallback
    if font_sm is None: font_sm = fallback

    gap = 6
    for x, y, w, h, label, sub, color in cells:
        cx = x + w // 2
        cy = y + h // 2

        # セル背景
        draw.rectangle([x+gap, y+gap, x+w-gap, y+h-gap], fill=color)

        # メインラベル（中央より少し上）
        draw.text((cx, cy - 30), label,
                  anchor="mm", fill="#ffffff", font=font_lg)

        # サブラベル（中央より少し下）
        draw.text((cx, cy + 55), sub,
                  anchor="mm", fill="rgba(255,255,255,180)", font=font_sm)

        # 枠線
        draw.rectangle([x+gap, y+gap, x+w-gap, y+h-gap],
                       outline="rgba(255,255,255,60)", width=5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    logger.info(f"Image saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI エントリーポイント
# ---------------------------------------------------------------------------

def setup() -> None:
    """リッチメニューを作成・画像アップロード・デフォルト設定する。"""
    image_path = Path("data/rich_menu.png")
    if not image_path.exists():
        logger.info("Generating rich menu image...")
        generate_image(image_path)

    rich_menu_id = create_rich_menu()
    upload_image(rich_menu_id, image_path)
    set_default(rich_menu_id)

    Path("data/rich_menu_id.txt").write_text(rich_menu_id)
    logger.info("✅ Rich menu setup complete!")
    print("\nArea JSON:")
    print(json.dumps(RICH_MENU_DEF, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "setup"
    if cmd == "setup":
        setup()
    elif cmd == "delete":
        delete_all()
    else:
        print("Usage: python -m src.line.rich_menu [setup|delete]")
