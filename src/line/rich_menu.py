"""
LINE リッチメニュー セットアップスクリプト。

実行:
    python -m src.line.rich_menu setup   # 作成・画像アップロード・デフォルト設定
    python -m src.line.rich_menu delete  # 既存メニューを全削除

レイアウト (2500 × 1686 px):
    ┌─────────────────┬─────────────────┐  ← y=0
    │   1250×843      │   1250×843      │
    │ ① メインレース  │ ② スケジュール  │
    ├─────────────────┼─────────────────┤  ← y=843
    │   1250×843      │   1250×843      │
    │ ③ 今日の成績    │ ④ SNS          │
    └─────────────────┴─────────────────┘  ← y=1686
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
    "size": {"width": 2500, "height": 1686},
    "selected": True,
    "name": "競馬予想メニュー",
    "chatBarText": "🏇 メニューを開く",
    "areas": [
        # ① 今日のメインレース（左上）
        {
            "bounds": {"x": 0, "y": 0, "width": 1250, "height": 843},
            "action": {
                "type": "message",
                "label": "今日のメインレース",
                "text": "main_race",
            },
        },
        # ② 開催スケジュール（右上 / URL）
        {
            "bounds": {"x": 1250, "y": 0, "width": 1250, "height": 843},
            "action": {
                "type": "uri",
                "label": "開催スケジュール",
                "uri": "https://race.netkeiba.com/top/race_list.html",
            },
        },
        # ③ 今日の成績（左下）
        {
            "bounds": {"x": 0, "y": 843, "width": 1250, "height": 843},
            "action": {
                "type": "message",
                "label": "今日の成績",
                "text": "today_result",
            },
        },
        # ④ お問い合わせ/SNS（右下 / URL）
        {
            "bounds": {"x": 1250, "y": 843, "width": 1250, "height": 843},
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

def generate_image(output_path: Path) -> None:
    """
    5ボタンのリッチメニュー画像を Pillow で生成する。
    日本語フォントが見つからない場合は ASCII フォールバック。
    """
    from PIL import Image, ImageDraw, ImageFont

    W, H = 2500, 1686
    img  = Image.new("RGB", (W, H), "#0D1B2A")
    draw = ImageDraw.Draw(img)

    # セル定義: (x, y, w, h, emoji, label_ja, bg_color)
    cells = [
        (0,    0,   1250, 843,  "🏇", "今日のメインレース", "#C62828"),
        (1250, 0,   1250, 843,  "📅", "開催スケジュール",   "#1565C0"),
        (0,    843, 1250, 843,  "📊", "今日の成績",         "#2E7D32"),
        (1250, 843, 1250, 843,  "🐦", "お問い合わせ / SNS", "#880E4F"),
    ]

    # フォント（日本語対応フォントを試みる）
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    label_font = None
    for fp in font_paths:
        try:
            label_font = ImageFont.truetype(fp, 80)
            break
        except (IOError, OSError):
            continue
    if label_font is None:
        label_font = ImageFont.load_default()

    gap = 6
    for x, y, w, h, emoji, label, color in cells:
        # 背景
        draw.rectangle([x+gap, y+gap, x+w-gap, y+h-gap], fill=color)
        # 上部: emoji
        draw.text((x + w // 2, y + h // 2 - 100), emoji,
                  anchor="mm", fill="#ffffff", font=ImageFont.load_default())
        # 下部: ラベル
        draw.text((x + w // 2, y + h // 2 + 60), label,
                  anchor="mm", fill="#ffffff", font=label_font)
        # 枠線
        draw.rectangle([x+gap, y+gap, x+w-gap, y+h-gap],
                       outline="#ffffff", width=4)

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
