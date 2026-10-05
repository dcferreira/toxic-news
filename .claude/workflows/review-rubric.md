# raise-pr severity rubric

Both reviewers `ai_review_claude` and `docs_review` follow this file exactly. It is the single
place the severity scale and review scope are defined; the workflow's step descriptions point here
instead of restating it.

## Scope

Review only lines this PR adds or changes (the `+` lines of the diff in scope). Issues in
unchanged text — even adjacent to an edit — are pre-existing: report them only if they are medium
or worse, and mark them `pre_existing: true`. Do not suggest speculative improvements. A finding
needs a concrete file:line and a concrete failure (what input/situation goes wrong); if you are not
certain it is real, do not report it.

## Severity

Severity is about what goes wrong, not how confident you are or how easy it is to fix:

- **critical**: a security hole, data loss, or a destructive action (e.g. rewriting a branch), or
  the main use case is completely broken.
- **major**: wrong behaviour on the normal path, or a realistic failure mode that breaks the run.
- **medium**: wrong behaviour in a plausible but less common situation, or misleading output.
- **minor**: a real but narrow problem: unusual setup, rare timing or crash windows, small
  robustness gaps.
- **nitpick**: style, naming, comments, wording, polish. Nothing behaves wrongly.

Formatting, wording, and line-length issues are always nitpick, whatever they're attached to.
Stale docs are rated on this same scale — there is no floor that makes a stale-docs finding at
least "minor"; a wording nit in a doc is still a nitpick.

Classify honestly; do not inflate or deflate severity.

## This repo

- A change to an outlet's XPath in `toxic_news/newspapers.py` that the PR title and description do
  not set out to make is **major**: outlet XPaths are only changed deliberately, one outlet at a
  time.
- New or changed behaviour in `toxic_news/` with no test exercising it is a `missing-test` finding,
  on the same severity scale as the behaviour it leaves unchecked.
