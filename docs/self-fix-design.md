# Self-fix loop for broken scrapers

When an outlet's front page changes and its XPath stops matching, CI should
notice, have an agent repair that one outlet's extractor offline, and open a
draft PR that shows the fix working. A human always merges.

Status: design agreed (2026-09-25); groundwork in progress.

## Scope

Only one kind of failure is repaired automatically: **the XPath no longer
matches a page that was fetched fine**. Everything else — the site blocking
us (401/403), timeouts, DNS failures, bot-challenge pages — is reported in the
run summary and left alone.

That is deliberately narrow. The task is always "restore an extractor that
used to work", so success has a crisp, checkable definition, and the change
is confined to one outlet's entry in `toxic_news/newspapers.py`.

At the time of writing (run of 2026-09-24), 11 of 17 outlets failed: 6 blocked
(NYT, AP, The Hill, Washington Times with 403; Reuters, WSJ with 401), 1
timeout (Washington Post), and 4 fetched fine but yielded no headlines (BBC,
Newsweek, Guardian US, Epoch Times). NBC and Washington Examiner were thin,
Fox over-counted. Every one of those runs was green: nothing today treats a
broken outlet as a failure.

## Overview

```
Update workflow (existing)
  scrape → score → publish
  + health.json per day (committed to `data`)
  + raw HTML of every outlet (artifact, 14 days)
        │ workflow_run
        ▼
Selfheal workflow (new)
  triage  ── no LLM ──  classify outlets, manage issues, pick outlets to fix
  fix     ── LLM ─────  one job per outlet: record the page, agent writes a patch
  pr      ── no LLM ──  verify the patch, build the evidence, open a draft PR
        │
        ▼
Daniel reviews the PR, approves CI, merges
```

The LLM does exactly one thing: write a patch for one outlet. Deciding
whether to spend a session, verifying the result and every claim in the PR
are the job of deterministic code.

## Health report

The `update` job records, per outlet:

- HTTP status, or the network exception;
- the number of headlines extracted, against `expected_headlines`;
- whether the extractor raised (parse errors are caught per outlet, so one
  broken outlet cannot crash the run);
- a verdict: `ok`, `xpath` or `other`.

An outlet is **`xpath`** when all of these hold:

1. the fetch returned HTTP 200, with no network error;
2. the body looks like the real page: at least 20 KB, at least 30% of the
   size of the outlet's last `ok` page in the previous 30 days' reports, and
   none of the usual bot-wall markers (Cloudflare's "Just a moment", Akamai's
   "Access Denied", PerimeterX and DataDome captchas — a bare "captcha" is
   not one, since sign-up forms on ordinary front pages carry it);
3. the extractor raised, or its count falls outside 0.6–1.4× of
   `expected_headlines` (the tolerance `test_parse_live` already uses) — this
   covers empty, thin and over-counting extractors alike.

A failure that is not `xpath` is `other`.

The report is committed to the `data` branch as `health/YYYY/MM/DD.json`,
written to the step summary, and the raw HTML of every outlet is uploaded as
an artifact with 14-day retention. An `aiohttp.ClientTimeout` bounds how long
one hung outlet can hold up the run.

## Triage

Runs after every Update run that produced a health report; no LLM, no
secrets, `issues: write`.

- Keeps **one issue per outlet** in state `xpath`, labelled `selfheal` and
  `selfheal:<outlet>`. It comments rather than re-filing, and closes the
  issue once the outlet has been `ok` for two runs.
- Sends an outlet to the fix job when:
  - it was `xpath` in the last **two consecutive** runs (a one-day layout
    experiment is not worth a fix);
  - no selfheal PR for it is open;
  - it had **no attempt in the last 7 days** — whatever the outcome, merged
    fixes included. A site that changes daily loses data between fixes rather
    than generating a PR a day.
- Attempts are recorded as a hidden marker in an issue comment,
  `<!-- selfheal-attempt YYYY-MM-DD -->`; triage reads the latest one across
  open and closed issues for that outlet. No other state is kept.

## Fix job

One job per outlet, fully independent, `fail-fast: false`,
`timeout-minutes: 20`. Permissions: `contents: read` only. The only secret is
`DEEPSEEK_API_KEY`, held in a `selfheal` environment so no other job can read
it; the checkout uses `persist-credentials: false`.

Inputs, under `selfheal/<outlet>/`:

- `today.html` — the bytes aiohttp received, the page to fix against;
- `task.json` — outlet, URL, `expected_headlines`, current extractor, dates of
  the last good and first bad runs;
- `last_good.json` — the last good day's headlines, from the `data` branch,
  so the agent knows what a real headline on this site looks like;
- a [canary](https://github.com/0xnyn/canary) recording of the live page in
  a real browser (`--headless`), as reference only.

The agent is [omp](https://github.com/can1357/oh-my-pi) on DeepSeek V4.1
Flash:

```
omp -p --mode json --no-session --tools read,edit,bash \
    --model deepseek/deepseek-flash
```

with a fixed prompt from the repo. It works offline on the saved page: it
adds a `DatedXpath` to that outlet's entry, starting 00:00 UTC on the first
bad day, and a new fixture for today with its snapshot. It never edits an
existing fixture or `expected_headlines`.

It iterates against `poe selfheal-check <outlet>`, which passes when:

1. on today's fixture, the count is within 0.6–1.4× of `expected_headlines`;
2. headlines are non-empty, unique, at most 300 characters, not navigation
   text, and link to the outlet's domain;
3. every existing fixture still reproduces its existing snapshot exactly, so
   historical and Wayback parsing are unchanged;
4. the diff stays within that outlet (see below);
5. `expected_headlines` is unchanged;
6. lint and type checks pass.

Output: `patch.diff` and `result.json` (`fixed` or `gave_up`, old and new
XPath, `from_date`, an explanation of at most five sentences). The agent's
own account of the headlines is not used anywhere.

## PR job

One per outlet, after its fix job. No LLM, no DeepSeek key; `contents`,
`pull-requests` and `issues: write`.

1. Apply the patch to a fresh checkout of `main`; reject it if it doesn't
   apply.
2. Check scope by line range: parse `newspapers.py` with `ast`, and require
   every hunk to fall inside this outlet's `Newspaper(...)` call or its own
   helper function. The only other files allowed are this outlet's new
   fixture and snapshots.
3. Re-run `poe selfheal-check <outlet>`. This run is the authoritative one.
4. Re-extract the headlines itself and build the evidence.
5. Screenshot: canary loads the saved `today.html` (with a `<base href>` so
   CSS resolves, JavaScript off so the DOM is what lxml parsed), evaluates the
   new XPath with `document.evaluate`, outlines every match and takes a
   full-page screenshot. Both sides implement XPath 1.0, so the outlined count
   should match the table.
6. Push the screenshot to the orphan `selfheal-evidence` branch and embed it
   through `raw.githubusercontent.com`. It never goes on the PR branch.
7. Open a draft PR from `selfheal/<outlet>-<date>`, labelled `selfheal` and
   `selfheal:<outlet>`, closing the outlet's issue; post the attempt marker.

If the agent gave up, there is no PR: the job comments on the issue with the
agent's explanation, the counts and the evidence, plus the attempt marker.

### What the PR shows

The PR has to prove the fix works without the reviewer running anything:

```
## selfheal(bbc): headline XPath drifted              closes #NN
Broken since 2026-09-20 · last good 2026-09-19

| run                          | headlines | expected |
|------------------------------|-----------|----------|
| last good (09-19, old XPath) | 45        | 47       |
| today, old XPath             | 0         | 47       |
| today, new XPath             | 44        | 47       |

First 10 headlines, with links
▸ All 44 extracted headlines, in page order, with links
▸ Last good day's 45 headlines

<screenshot: the front page with every new-XPath match outlined>

XPath change:  - //h3[@class='media__title' and a]
               + //h2[@data-testid='card-headline']
Why: <the agent's explanation>

Checks: today's fixture · old fixtures unchanged · scope · lint · types
Evidence: canary report · trace · HAR · saved HTML (artifacts)
Session: deepseek-flash · 23 turns · $0.04
```

Every number and headline in it is computed by the PR job. The session cost
comes from omp, which accounts for DeepSeek's cache and matches the wallet.

### CI on selfheal PRs

Pushes made with `GITHUB_TOKEN` trigger no workflows, but a PR opened with it
starts its `pull_request` workflows in an approval-required state. So
`backend.yaml` gains a `pull_request` trigger, and approving the CI run
becomes part of the review.

## Storage

| What | Where | Kept |
|---|---|---|
| Health report | `data` branch, `health/YYYY/MM/DD.json` | forever |
| Raw HTML of every outlet | Actions artifact | 14 days |
| canary report, trace, HAR, video | Actions artifact | 90 days |
| PR screenshot | `selfheal-evidence` branch | forever |
| New fixture | the PR itself | forever |

A day of raw HTML is about 18 MB (3 MB zipped); canary sessions happen at
most a few times a week. This stays well inside Actions' limits (90-day
maximum retention on public repos, 500 MB of artifact storage on the Free
plan).

## Wiring

- `update.yml` keeps publishing exactly as today, plus the health report.
- `selfheal.yml` is a separate workflow on `workflow_run` of Update, so a
  broken self-heal can never block publishing. It never writes to `data` or
  `public`, so it runs in its own `selfheal` concurrency group. It also takes
  `workflow_dispatch` with:
  - `outlet` — force one outlet, bypassing the two-run rule and the weekly
    cap;
  - `dry_run` — stop after the fix job: patch and evidence as artifacts, no
    PR, no attempt recorded.
- Runs land around 15:30–18:00 UTC, which is DeepSeek off-peak.

## Safety

The agent only reads the bot's own data and the newspapers' pages, so prompt
injection is a hygiene concern rather than a design driver. The controls that
cost nothing stay: the job running the agent holds no GitHub write access and
only a dedicated, low-balance DeepSeek key; it works offline on saved HTML;
its output is a patch that deterministic code checks; merging is human.

## Cost

A session is a few hundred thousand input tokens, mostly cache hits, and
about 20K output, which on `deepseek-flash` comes to a few cents. With at most
one attempt per outlet per week, this stays well under $5 a month. Runners
are free on a public repo.

## Rollout

Each step has to prove itself before the next starts.

0. **Groundwork**, no LLM:
   1. health report, per-outlet error isolation, client timeout, raw HTML
      artifact, step summary;
   2. fixtures per date (`tests/assets/html/<outlet>/<date>.html`), with
      `test_parse` running each at its own date — today every fixture is read
      as of 2023-05-20, so a new `DatedXpath` would never be exercised;
   3. `poe selfheal-check <outlet>`;
   4. `pull_request` trigger on `backend.yaml`;
   5. separately, the `heal` job's naive/aware datetime `TypeError`.

   Done when a week of health reports exists and the saved pages of the
   current candidates confirm the real-page heuristic.
1. **Triage only**: issues, no agent. Done after a week of correct classes,
   no duplicates, and issues closing on recovery.
2. **Runner spike**: canary headless, the saved-HTML screenshot, installing
   omp, downloading the Update run's artifacts. Done when the screenshot's
   count matches lxml's.
3. **Fix job, dry run**: Daniel creates the `selfheal` environment and key,
   then dispatches every current `xpath` outlet with `dry_run`. Done when the
   patches are judged and the prompt is tuned.
4. **PR job**: by hand first, then chained to Update. Done after two weeks of
   real PRs.
5. **Optional**: backfill an outlet's broken days from the Wayback Machine
   after its fix merges (`heal` only refetches days with no data at all), and
   branch protection on `main`.

## Designs not chosen

- **Agent-written issues, then an agent turning issues into PRs.** Two
  sessions per fix, and issue wording would drift, breaking deduplication.
  The shape survives with the issue written by code.
- **A single LLM call with no tool loop.** Cheapest, but only fits simple
  XPath swaps, not custom extractors like `nytimes_fn`.
- **LLM extraction at runtime when the XPath fails.** Keeps data flowing, but
  hallucinated headlines would be scored and published as real data.
