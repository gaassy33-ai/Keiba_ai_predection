"""
LINE リッチメニュー セットアップスクリプト。

実行:
    python -m src.line.rich_menu setup   # 作成・画像アップロード・デフォルト設定
    python -m src.line.rich_menu delete  # 既存メニューを全削除

レイアウト (2500 × 843 px コンパクト):
    ┌─────────────────┬─────────────────┐  ← y=0
    │   1250×421      │   1250×421      │
    │ ① 今日の予想一覧 │ ② AIの成績・回収率│
    ├─────────────────┼─────────────────┤  ← y=421
    │   1250×422      │   1250×422      │
    │ ③ 開催スケジュール│ ④ 公式X        │
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
    "name": "競馬予想メニュー v2",
    "chatBarText": "🏇 メニューを開く",
    "areas": [
        # ① 今日の予想一覧（左上）→ message アクション「今日の予想」を送信
        {
            "bounds": {"x": 0, "y": 0, "width": 1250, "height": 421},
            "action": {
                "type": "message",
                "label": "今日の予想一覧",
                "text": "今日の予想",
            },
        },
        # ② AIの成績・回収率（右上）→ LIFF 統計ページ
        {
            "bounds": {"x": 1250, "y": 0, "width": 1250, "height": 421},
            "action": {
                "type": "uri",
                "label": "AIの成績・回収率",
                "uri": "https://gaassy33-ai.github.io/Keiba_ai_predection/stats.html",
            },
        },
        # ③ 開催スケジュール（左下）→ netkeiba トップ
        {
            "bounds": {"x": 0, "y": 421, "width": 1250, "height": 422},
            "action": {
                "type": "uri",
                "label": "開催スケジュール",
                "uri": "https://race.netkeiba.com/top/",
            },
        },
        # ④ 公式X（右下）→ X アカウント
        {
            "bounds": {"x": 1250, "y": 421, "width": 1250, "height": 422},
            "action": {
                "type": "uri",
                "label": "公式X",
                "uri": "https://x.com/ataru_keiba_ai",
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
        (0,    0,   1250, 421,  "今日の予想一覧",      "単勝・馬連 全レース",   "#B71C1C"),
        (1250, 0,   1250, 421,  "AIの成績・回収率",    "的中率・収支グラフ",    "#1B5E20"),
        (0,    421, 1250, 422,  "開催スケジュール",    "netkeiba へ移動",       "#0D47A1"),
        (1250, 421, 1250, 422,  "公式X",               "@ataru_keiba_ai",      "#4A148C"),
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

def clear_user_richmenu(user_id: str) -> None:
    """ユーザー個別に設定されたリッチメニューを解除する。"""
    resp = requests.delete(
        f"{_BASE}/user/{user_id}/richmenu", headers=_HEADERS
    )
    if resp.status_code == 200:
        logger.info(f"Cleared per-user rich menu: {user_id}")
    else:
        logger.warning(f"clear_user_richmenu [{resp.status_code}]: {resp.text}")


def setup(user_id: str | None = None) -> None:
    """
    リッチメニューを作成・画像アップロード・デフォルト設定する。

    手順:
      1. 既存メニューを全削除（旧メニューが残らないようにする）
      2. 画像を強制再生成（キャッシュを使わない）
      3. 新メニューを作成・画像アップロード・デフォルト設定
      4. ユーザー個別割当がある場合は解除（user_id 指定時）
      5. 設定確認
    """
    # ── 1. 既存メニュー全削除 ────────────────────────────────────────
    logger.info("Step 1: 既存リッチメニューを全削除...")
    delete_all()

    # ── 2. 画像を強制再生成 ──────────────────────────────────────────
    image_path = Path("data/rich_menu.png")
    if image_path.exists():
        image_path.unlink()
        logger.info(f"古い画像を削除: {image_path}")
    logger.info("Step 2: リッチメニュー画像を生成...")
    generate_image(image_path)

    # ── 3. メニュー作成・アップロード・デフォルト設定 ────────────────
    logger.info("Step 3: メニュー作成・アップロード・デフォルト設定...")
    rich_menu_id = create_rich_menu()
    upload_image(rich_menu_id, image_path)
    set_default(rich_menu_id)
    Path("data/rich_menu_id.txt").write_text(rich_menu_id)

    # ── 4. ユーザー個別割当の解除（指定時）─────────────────────────
    if user_id:
        logger.info(f"Step 4: ユーザー個別割当を解除: {user_id}")
        clear_user_richmenu(user_id)

    # ── 5. 設定確認 ──────────────────────────────────────────────────
    logger.info("Step 5: 設定確認...")
    resp = requests.get(f"{_BASE}/richmenu/list", headers=_HEADERS)
    menus = resp.json().get("richmenus", []) if resp.ok else []
    logger.info(f"現在のリッチメニュー数: {len(menus)}")
    for m in menus:
        logger.info(f"  {m['richMenuId']}  name={m.get('name')}  selected={m.get('selected')}")

    default_resp = requests.get(f"{_BASE}/user/all/richmenu", headers=_HEADERS)
    if default_resp.ok:
        default_id = default_resp.json().get("richMenuId", "")
        match = "✅ 一致" if default_id == rich_menu_id else "❌ 不一致"
        logger.info(f"デフォルトメニュー: {default_id}  {match}")
    else:
        logger.warning(f"デフォルト確認失敗 [{default_resp.status_code}]: {default_resp.text}")

    logger.info("✅ Rich menu setup complete!")
    print(f"\n新しいリッチメニュー ID: {rich_menu_id}")
    print("\nArea JSON:")
    print(json.dumps(RICH_MENU_DEF, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "setup"
    if cmd == "setup":
        # 省略可: python -m src.line.rich_menu setup <USER_ID>
        uid = sys.argv[2] if len(sys.argv) > 2 else None
        if uid is None:
            # 環境変数からも取得
            import os
            uid = os.environ.get("LINE_TARGET_USER_ID")
        setup(user_id=uid)
    elif cmd == "delete":
        delete_all()
    else:
        print("Usage: python -m src.line.rich_menu [setup [USER_ID]|delete]")
