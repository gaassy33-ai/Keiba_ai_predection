/**
 * Cloudflare Worker: 競馬レース20分前 GitHub Actions ディスパッチャー
 *
 * 動作フロー:
 *   1. 毎分 cron で起動
 *   2. GitHub Pages から docs/race_schedule.json を fetch
 *   3. notify_at が「今から0〜60秒以内」のレースを検出
 *   4. KV でベキ等性を保証（同一レースの多重 dispatch を防止）
 *   5. GitHub API で keiba_dispatch.yml の workflow_dispatch を発火
 *
 * 必要な Cloudflare 設定（wrangler.toml または Workers ダッシュボード）:
 *   Secrets:
 *     GH_TOKEN          GitHub Personal Access Token（repo + workflow スコープ）
 *   Environment Variables:
 *     GH_OWNER          GitHubユーザー名  例: gaassy33-ai
 *     GH_REPO           リポジトリ名      例: Keiba_ai_predection
 *     PAGES_BASE_URL    GitHub Pages URL  例: https://gaassy33-ai.github.io/Keiba_ai_predection
 *   KV Binding:
 *     DISPATCH_KV       KV Namespace（dispatched レース ID の記録用）
 */

export default {
  /**
   * Cron トリガー（毎分）
   */
  async scheduled(event, env, ctx) {
    ctx.waitUntil(runDispatcher(env));
  },

  /**
   * HTTP トリガー（手動テスト・デバッグ用）
   * GET /dispatch?dry_run=1  で dispatch せず一覧を返す
   */
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const dryRun = url.searchParams.get("dry_run") === "1";
    const result = await runDispatcher(env, dryRun);
    return new Response(JSON.stringify(result, null, 2), {
      headers: { "Content-Type": "application/json" },
    });
  },
};

// ──────────────────────────────────────────────────────────────────────
// コア処理
// ──────────────────────────────────────────────────────────────────────

async function runDispatcher(env, dryRun = false) {
  const now = new Date();
  const log = [];

  // ── レーススケジュール JSON を GitHub Pages から取得 ──────────────
  const scheduleUrl = `${env.PAGES_BASE_URL}/race_schedule.json`;
  let schedule;
  try {
    const resp = await fetch(scheduleUrl, {
      headers: { "User-Agent": "Cloudflare-Worker-KeibaDispatcher/1.0" },
      // CF CDN キャッシュを 60 秒以内に更新（朝バッチ後の反映に対応）
      cf: { cacheTtl: 55 },
    });
    if (!resp.ok) {
      return { error: `schedule fetch failed: ${resp.status}`, url: scheduleUrl };
    }
    schedule = await resp.json();
  } catch (e) {
    return { error: `schedule fetch error: ${e.message}` };
  }

  log.push(`schedule loaded: ${schedule.length} races`);

  // ── 今日（JST）の日付文字列 ─────────────────────────────────────
  const jstDate = new Date(now.getTime() + 9 * 60 * 60 * 1000);
  const todayJst = jstDate.toISOString().slice(0, 10); // "YYYY-MM-DD"

  // ── 通知対象レースを抽出 ─────────────────────────────────────────
  // notify_at が「今から 0〜60 秒以内（次の cron 実行前）」のレースのみ dispatch
  const toDispatch = [];
  for (const race of schedule) {
    if (race.date !== todayJst) continue;

    const notifyAt = new Date(race.notify_at);
    const diffMs = notifyAt.getTime() - now.getTime();

    // 0ms ≤ diff < 60,000ms → 今の 1 分枠で通知すべきレース
    if (diffMs >= 0 && diffMs < 60_000) {
      toDispatch.push(race);
    }
  }

  log.push(`to dispatch: ${toDispatch.length} races`);

  // ── KV チェック & GitHub Actions dispatch ───────────────────────
  const dispatched = [];
  const skipped = [];

  for (const race of toDispatch) {
    const kvKey = `dispatched:${race.race_id}`;

    // 冪等性チェック: 既に dispatch 済みなら skip
    if (env.DISPATCH_KV) {
      const existing = await env.DISPATCH_KV.get(kvKey);
      if (existing) {
        skipped.push(race.race_id);
        log.push(`skip (already dispatched): ${race.race_id}`);
        continue;
      }
    }

    if (!dryRun) {
      const ok = await triggerGitHubActions(race.race_id, env);
      if (ok) {
        // dispatch 済みとして記録（TTL: 4 時間）
        if (env.DISPATCH_KV) {
          await env.DISPATCH_KV.put(kvKey, "1", { expirationTtl: 14400 });
        }
        dispatched.push(race.race_id);
        log.push(`dispatched: ${race.race_id} (${race.race_name})`);
      } else {
        log.push(`dispatch failed: ${race.race_id}`);
      }
    } else {
      dispatched.push(`[dry_run] ${race.race_id}`);
      log.push(`[dry_run] would dispatch: ${race.race_id} (${race.race_name})`);
    }
  }

  return {
    now: now.toISOString(),
    today_jst: todayJst,
    schedule_count: schedule.length,
    dispatched,
    skipped,
    log,
  };
}

// ──────────────────────────────────────────────────────────────────────
// GitHub Actions workflow_dispatch を発火する
// ──────────────────────────────────────────────────────────────────────

async function triggerGitHubActions(raceId, env) {
  const url = `https://api.github.com/repos/${env.GH_OWNER}/${env.GH_REPO}/actions/workflows/keiba_dispatch.yml/dispatches`;

  const resp = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GH_TOKEN}`,
      Accept: "application/vnd.github.v3+json",
      "Content-Type": "application/json",
      "User-Agent": "Cloudflare-Worker-KeibaDispatcher/1.0",
    },
    body: JSON.stringify({
      ref: "main",
      inputs: { race_id: raceId },
    }),
  });

  // 204 No Content = 成功
  if (resp.status === 204) return true;

  const body = await resp.text().catch(() => "(no body)");
  console.error(`triggerGitHubActions failed: ${resp.status} ${body}`);
  return false;
}
