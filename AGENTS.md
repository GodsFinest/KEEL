# KEEL — Agent Operating Rules (stable)

These rules are stable for the life of the project. They condense the founding
mandate; when in doubt, the mandate text in the project brief wins.

## Scope contract
- Authorized scope (superseded 2026-11-01 by the full-runtime mandate): the
  **complete CPU runtime, Phases 2-9** of docs/ROADMAP.md, building on the
  verified Phase 1 model. Continue automatically through ordinary milestones;
  verification gates, not permission gates. Do not push to remote, rewrite
  history, weaken tests, or fabricate evidence.

## The central claim KEEL must earn or falsify
A tensor that is safe to discard now may be impossible to give up later under a
smaller memory grant. KEEL must plan a legal memory-grant transition from a live
execution state, accompanied by an independently checked continuation proving
the original computation still finishes. A tensor being destroyed is NOT
sufficient evidence that memory was returned; a planning timeout is NOT proof of
infeasibility; a simulated ownership change is NOT a real handover.

## Phase 1 truth discipline
- The exact oracle is ground truth. Heuristics/baselines never report proven
  infeasibility; at most REJECT or UNKNOWN.
- Every accepted plan (oracle or policy) is validated by an independently
  written checker before being counted. Accepted-but-invalid is a defect and a
  regression test.
- Hypotheses, evaluation grid, metrics, and go/no-go criteria are pre-declared
  in docs/EXPERIMENTS.md BEFORE held-out results are generated. Success criteria
  are never modified after seeing results. Negative findings are preserved and
  reported.
- Development instances (seeds 1–20) are for iteration; held-out instances
  (seeds 21–70) are run once against the frozen grid and never used to tune.
- Report full distributions (min/median/p90/max), not cherry-picked examples.
  Include planner and checker overhead in cost accounting.

## Engineering discipline
- Small, reviewable, tested changes; build + tests green at every commit.
- Every discovered defect gets a permanent regression test; never weaken a test
  to pass.
- Checked arithmetic for all byte/size math; no raw pointers as identities;
  explicit statuses over exceptions on planning paths; no hidden global state.
- Keep Phase 1 minimal: no stub libraries, no speculative architecture.
  The checker must not reuse the planner's transition logic (independence).
- Repository is durable memory: update docs/STATUS.md and docs/DECISIONS.md
  with factual evidence at each milestone; commit at each meaningful gate.

## Environment notes
- Repo lives in iCloud Drive; expect slower fsync. Keep build dirs in `build/`
  (gitignored). macOS arm64, AppleClang 17, C++20, CMake, doctest (vendored).
- Do not commit without the user's standing authorization for milestones;
  committing local checkpoints on the work branch is authorized for this project.
