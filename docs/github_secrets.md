# GitHub Secrets 登録リスト

`Settings > Secrets and variables > Actions > New repository secret` に以下を登録してください。

## 必須（LINE 通知）

| Secret 名 | 説明 | 取得場所 |
|-----------|------|----------|
| `LINE_CHANNEL_ACCESS_TOKEN` | LINE Messaging API のチャネルアクセストークン | LINE Developers > チャネル > Messaging API > チャネルアクセストークン |
| `LINE_CHANNEL_SECRET` | LINE チャネルシークレット | LINE Developers > チャネル > チャネル基本設定 > チャネルシークレット |
| `LINE_TARGET_USER_ID` | 通知先ユーザーの LINE User ID（`U` から始まる文字列） | LINE Developers > チャネル > Messaging API > あなたのユーザーID |

## 必須（netkeiba スクレイピング）

| Secret 名 | 説明 |
|-----------|------|
| `NETKEIBA_EMAIL` | netkeiba 会員登録メールアドレス |
| `NETKEIBA_PASSWORD` | netkeiba パスワード |

## `train_weekly.yml` 専用（週次再学習）

| Secret 名 | 説明 | 取得場所 |
|-----------|------|----------|
| `GH_PAT` | GitHub Personal Access Token（`repo` スコープ） | GitHub > Settings > Developer settings > Personal access tokens > Tokens (classic) |

> `GH_PAT` はモデルファイルをリポジトリに push するために必要。
> `keiba_notify.yml`（通知のみ）では不要。

## 動作確認手順

1. `Actions` タブを開く
2. `競馬予想 LINE 通知` ワークフローを選択
3. `Run workflow` → `workflow_dispatch` で手動実行
4. ログを確認し、LINE に通知が届くことを確認
