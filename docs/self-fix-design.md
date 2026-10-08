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
  triage  ── no LLM ──  classify outlets, keep one issue per failure
        │ workflow_run
        ▼
Self-fix workflow
  prepare ── no LLM ──  pick the outlets to fix, write each one's inputs
  fix     ── LLM ─────  one job per outlet: the agent writes a patch
  check   ── no LLM ──  gate and judge the patch, screenshot the page
  pr      ── no LLM ──  open a draft PR with the evidence, or record the attempt
        │
        ▼
Daniel reviews the PR, marks it ready (which runs CI), merges
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
- whether scoring failed (a model that fails to load or to score is caught
  too, and makes an outlet whose extractor worked `other`);
- a verdict: `ok`, `xpath` or `other`.

A page is decoded as its declared charset, then UTF-8, then UTF-8 with bad
bytes replaced, so an encoding slip never passes for an extractor error.

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

The report is committed to the `data` branch as `health/YYYY/MM/DD.json`
(alone, if the run failed after writing it),
written to the step summary, and the raw HTML of every outlet is uploaded as
an artifact with 14-day retention. An `aiohttp.ClientTimeout` bounds how long
one hung outlet can hold up the run.

## Triage

Runs after every Update run that produced a health report; no LLM, no
secrets, `issues: write`.

- Keeps **one issue per failure**: an outlet in state `xpath` has one open
  issue, labelled `selfheal` and `selfheal:<outlet>` (the outlet's fixture
  slug, e.g. `selfheal:bbc.com`). Each run refreshes that issue's body rather
  than re-filing, and the issue is closed once the outlet has been `ok` for
  two runs. A closed issue is never reopened: an outlet that breaks again, or
  whose issue was closed while it was still broken, is a new failure with a
  new issue. The body ends in a `<!-- selfheal-failure -->` JSON block (outlet,
  url, first and last day seen, headline counts) for the fix job to read.
- The Self-fix workflow's `prepare` job sends an outlet to the fix job when:
  - it was `xpath` in the last **two consecutive** runs (a one-day layout
    experiment is not worth a fix);
  - no selfheal PR for it is open;
  - it had **no attempt in the last 7 days** — whatever the outcome, merged
    fixes included. A site that changes daily loses data between fixes rather
    than generating a PR a day.

  A run by hand naming an outlet skips the first and last rule, never the
  open PR one.
- Attempts are recorded as a hidden marker in an issue comment,
  `<!-- selfheal-attempt YYYY-MM-DD -->`; `prepare` reads the latest one
  across open and closed issues for that outlet, counting only the bot's own
  comments (anyone can comment on a public repository). No other state is
  kept. A failure of DeepSeek or the harness is no attempt: it is recorded
  with `<!-- selfheal-infra-error YYYY-MM-DD -->`, which `prepare` ignores.

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
   text, at most 5% promos (newsletter, app, account or subscription links,
   calls to action), and link to the outlet's domain;
3. every existing fixture still reproduces its existing snapshot exactly, so
   historical and Wayback parsing are unchanged;
4. the diff stays within that outlet (see below);
5. `expected_headlines` is unchanged;
6. lint and type checks pass;
7. the test suite passes (less the network and model tests). The agent can't
   edit a test, so a fix that breaks one fails until a person fixes the test.

Output: `patch.diff` and `result.json` (`fixed`, `gave_up` or `infra_error`, old and new
XPath, `from_date`, an explanation of at most five sentences, an `excluded`
note on what looks like a headline but was left out). The agent's
own account of the headlines is not used anywhere.

`infra_error` is not the agent's verdict: DeepSeek or the harness failed. It
is a session that left no `result.json` after a provider error (such as
`402 Insufficient Balance`) or a failing omp exit, or that DeepSeek's balance
endpoint says it will refuse (asked before omp starts; only a clear no stops a
run). Its patch is empty, so nothing is judged, and `agent` exits 1: the fix
leg is red, with an error annotation naming what failed.

## PR job

The patch's code runs only in the `check` job, never where there is anything
to write with.

`check`, per outlet, with `contents: read` and no secrets:

1. Gate the patch (`poe selfheal-fix gate`): the patch is read with
   `git apply --numstat --summary` and `newspapers.py` parsed with `ast`; it
   may only change this outlet's `Newspaper(...)` call or its own helpers and
   add today's page and snapshot, with one `DatedXpath` from the task's
   `from_date`. Nothing of the patch runs before this passes.
2. Apply it to a copy of the checkout and run `poe selfheal-check --json` on
   that copy, as the sandbox user (see Safety). Its verdict and headlines are
   evidence for the reviewer, not proof: the patch's code could make them lie.
3. Screenshot, from the untouched checkout: the XPath to outline is read from
   the patched `newspapers.py` as text (`poe selfheal-pr xpath`), never
   imported; canary loads the saved `today.html` with a `<base href>` so CSS
   resolves and a CSP that turns off scripts, frames, media and requests, so
   the DOM is the one lxml parsed. It evaluates the XPath with
   `document.evaluate`, outlines every match and takes a full-page
   screenshot. Both sides implement XPath 1.0, so the outlined count must
   match lxml's; then every outline is also looked for in the pixels, since a
   full-page capture can cut outlines off while the counts still agree. A fix
   whose extractor is not a plain XPath gets no screenshot.
4. Keep everything as the `selfheal-evidence-<outlet>` artifact, and write the
   PR body it would open into the run's summary.

`pr` needs the repository setting "Allow GitHub Actions to create and approve
pull requests" (Settings > Actions > General, or `can_approve_pull_request_reviews`
in the API): without it, `GITHUB_TOKEN` cannot open a PR, and the job fails
after pushing its branch, recording no attempt.

`pr`, per outlet and one at a time, with `contents`, `pull-requests` and
`issues: write`, only when the run is not a dry run. It runs none of the
patch's code: main's own code reads the evidence as data.

1. Gate the patch again, and decide what came of the attempt: `pr` for a
   passing patch within scope, `none` when no session ran (deferred to dodge
   DeepSeek's peak), `infra_error` when DeepSeek or the harness failed, else
   gave up, refused, failed or no patch.
2. For `pr`: apply the patch in a separate worktree, commit it to
   `selfheal/<outlet>-<date>-<run id>-<run attempt>` and push. The checkout the job's code runs from
   never holds the patch.
3. Push the screenshot to the orphan `selfheal-evidence` branch and embed it
   through `raw.githubusercontent.com`, pinned to that commit. It never goes
   on the PR branch.
4. Open a draft PR, labelled `selfheal` and `selfheal:<outlet>`, closing the
   outlet's open issue.
5. Comment on the outlet's issue, with the attempt marker: the PR, or how the
   attempt failed, quoting the agent's explanation and the checks. For an
   `infra_error` the comment says what failed, carries the infra-error marker
   instead, and the job then fails, so the run is red and the outlet is tried
   again by the next run.

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

Checks: today's fixture · old fixtures unchanged · scope · lint · types · tests
Evidence: patch, check report, screenshot, agent session (artifacts)
Session: deepseek-flash · 23 turns · $0.04
```

Every number and headline in it is computed by the workflow, not taken from
the agent: today's count with the old extractor is the health report's, the
last good day's comes from the `data` branch, and the new extractor's comes
from `selfheal-check`. Scraped and agent-written text is escaped so it cannot
mention anyone, link, or embed HTML; the diff sits in a fence it cannot
close. The session cost
comes from omp, which accounts for DeepSeek's cache and matches the wallet.

### CI on selfheal PRs

A PR opened with `GITHUB_TOKEN` starts no workflow, so the checks a PR would
get are the ones above. `backend.yaml` also runs on `ready_for_review`: a
person marking the draft ready runs CI on it.

## Storage

| What | Where | Kept |
|---|---|---|
| Health report | `data` branch, `health/YYYY/MM/DD.json` | forever |
| Raw HTML of every outlet | Actions artifact | 14 days |
| Patch, agent session, evidence (check report, screenshot, counts) | Actions artifact | 14 days |
| PR screenshot | `selfheal-evidence` branch | forever |
| New fixture | the PR itself | forever |

A day of raw HTML is about 18 MB (3 MB zipped). canary's own session
(trace, video, about 115 MB) is not kept: the page is a saved one, so the
screenshot is all it adds. This stays well inside Actions' limits (500 MB of
artifact storage on the Free plan).

## Wiring

- `update.yml` keeps publishing exactly as today, plus the health report.
- `selfheal.yml` is a separate workflow on `workflow_run` of Update on
  main, so a broken self-heal can never block publishing. It never writes to
  `data` or `public`, so it runs in its own `selfheal` concurrency group. The
  repository variable `SELFHEAL_AUTO` decides what an automatic run does:
  `dry-run` stops after `check`, `on` opens PRs, anything else (unset
  included) skips it. A run of Update that saved no pages (the weekly heal)
  has nothing to fix. It also takes `workflow_dispatch` with:
  - `outlet` — force one outlet, bypassing the two-run rule and the weekly
    cap;
  - `dry_run` (default true) — stop after `check`: patch and evidence as
    artifacts, no PR, no attempt recorded. Only a dry run may come from a
    branch other than main.
- DeepSeek is only ever called off-peak. Its peak hours are 01:00–04:00 and
  06:00–10:00 UTC on weekdays (rates double then; weekends and Chinese public
  holidays are off-peak all day, though the guard treats holidays as weekdays). A run by hand started too close to them is refused, an automatic one
  waits for the next Update, and a fix session that could reach one is
  deferred. Update's 12:00 UTC run leaves about 13 hours of
  off-peak for the fix job to follow it, which a test holds it to.

## Safety

The agent only reads the bot's own data and the newspapers' pages, so prompt
injection is a hygiene concern rather than a design driver. The controls that
cost nothing stay: the job running the agent holds no GitHub write access and
only a dedicated, low-balance DeepSeek key; it works offline on saved HTML;
its output is a patch that deterministic code checks; merging is human.

What the agent writes is code, and it does run: the agent has `bash`, and
`selfheal-check` imports the patched `newspapers.py`. So the check job first
runs `poe selfheal-fix gate`, which reads the patch with `git apply --summary`
and refuses anything beyond the outlet's entry and today's page and snapshot,
before any of it runs. The checker's verdict is still advisory: a patch's code
could make it lie. Artifacts are public, so the fix job masks the DeepSeek
key in everything it keeps.

The agent and the checker run as `selfheal`, a user made for the job with
no sudo, no docker, no access to the runner's home, and nothing of the
runner's to write (`.github/selfheal-sandbox.sh`, which checks this on every
run). Run as the runner's own user, that code could rewrite a JavaScript
action a later step runs, which is handed the job's `ACTIONS_RUNTIME_TOKEN`,
and so write an Actions cache entry on `main` that Update, which restores
caches and can push, would pick up. As `selfheal` it reaches neither the
actions nor the token, and its outputs are copied out file by file as it
reads them, so a symlink it leaves leaks nothing. No job of the workflow
writes or restores a cache, and `tests/test_selfheal_pr.py` holds the jobs
to these rules: write access only in `pr`, the key only in `fix`, the
agent and the checker only through the sandbox, no checkout running trusted
code with the patch in it.

The screenshot still loads the outlet's CSS, images and fonts from its
servers, as a browser would; everything else a page could load is off.

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
   current candidates confirm the real-page heuristic. (Checked on the pages
   of 2026-09-29 to 10-04: all seven `xpath` outlets served their real front
   page every day; the Guardian's ~90 "captcha" hits are reCAPTCHA settings.)
1. **Triage only**: issues, no agent. Done after a week of correct classes,
   no duplicates, and issues closing on recovery.
2. **Runner spike**: canary headless, the saved-HTML screenshot, installing
   omp, downloading the Update run's artifacts. Done when the screenshot's
   count matches lxml's.
3. **Fix job, dry run**: Daniel creates the `selfheal` environment and key,
   then dispatches every current `xpath` outlet with `dry_run`. Done when the
   patches are judged and the prompt is tuned.
4. **PR job**: once Actions may create pull requests (see PR job), by hand
   first (a dispatch with `dry_run` off), then chained
   to Update: `SELFHEAL_AUTO=dry-run`, then `on`. Done after two weeks of
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
