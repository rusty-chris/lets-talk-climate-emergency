# ORCHESTRATION.md — autonomous build methodology

> Recorded 2026-08-16 at the owner's direction. This governs how the software is built until it is ready for live deployment. Companion to IMPLEMENTATION.md (the TDD design) and CONTRIBUTING.md (the VC protocol). If this file and an agent's instincts disagree, this file wins.

## Roles

| Role | Model tier | Rule |
|---|---|---|
| **Orchestrator** | main session | Assigns issues, sequences the dependency graph, merges PRs, triages review findings. Does not write feature code itself. |
| **Test author** | **Fable — always** | Writes the failing tests for an issue *before* any implementation exists (the red phase). No other tier ever authors behavioural tests. |
| **Implementer** | by complexity (table below) | Makes the failing tests pass, then refactors. Never weakens, skips, or deletes a test to get to green — if a test looks wrong, it reports back to the orchestrator instead. |
| **Adversarial reviewer** | **Fable — always** | Reviews every merged feature trying to break it; files correction/improvement issues on GitHub. |

## Implementer tier assignment

- **haiku** — mechanical work: config, boilerplate, doc moves, dependency bumps.
- **sonnet** — standard, well-specified modules with clear test contracts (e.g. manifest schema enforcement, fetchers, classifier wiring, static pages).
- **opus** — complex integration or subtle logic: parsing pipelines, retrieval/rerank/refusal, native-citations generation path, renderer, FastAPI service.
- **fable** — the hardest or most novel work (flagship chart spike, gold-set authoring with severity annotations, test infrastructure), **plus any issue whose complexity is uncertain — when in doubt, Fable.**

Interpretation note: pure configuration checks (e.g. "CI goes green", "docker compose up works") are not behavioural tests and do not require the Fable test author; any test asserting *behaviour* does.

## The per-issue loop (TDD, per IMPLEMENTATION.md)

1. Orchestrator picks the next issue whose dependencies are merged; creates branch `issue-<n>-<slug>` off current `main`.
2. **Fable test author** writes the issue's TDD-plan tests, failing, and commits them first (`Add failing tests for … (#n)`).
3. **Implementer** (assigned tier) implements to green, refactors, commits per CONTRIBUTING.md conventions.
4. PR to `main` referencing the issue. CI must be green. Direct pushes to `main` are blocked by a repo-side hook — **always** branch + PR; orchestrator merges with a merge commit (no squash — granular history is kept).
5. **After merge, the Fable adversarial reviewer** examines all new code and functionality from that feature (the full merged diff, plus how it composes with what exists), actively trying to break it: correctness, licensing invariants, security/injection, chart integrity, cost, and design-conformance (DESIGN.md/DECISIONS.md are the contract).
6. The reviewer **files its findings as GitHub issues** (label `review-finding`, severity label `blocker`/`major`/`minor`), each specifying **(a) the new or updated failing test(s)** that capture the defect and **(b) the implementation change** required. Findings that are working-as-designed are closed with the reasoning, not silently dropped.
7. Orchestrator triages: `blocker` findings are fixed before dependent work builds on the feature; others join the queue. Fixes follow this same loop (Fable writes the tests first).
8. Repeat until the issue queue and the review-finding queue are both dry and the release gates (DESIGN §10) pass.

## Parallelism

Independent issues may run concurrently, each agent in an **isolated worktree** on its own branch. Never two agents on one branch. The orchestrator serialises merges and resolves conflicts by rebasing the later branch.

## Stop-and-ask points (the only reasons to interrupt the owner)

- Sending the permission letters (they go out under the owner's name — drafts are prepared, sending is his act).
- Creating external accounts, registering domains, ICO registration, or any spend beyond Anthropic API usage.
- Live/public deployment, or making the repo public.
- Final approval of voices-layer content (first-party prose about real people) before launch.
- Credentials of any kind.

Everything else — including test failures, review findings, and design ambiguities resolvable from DESIGN.md/DECISIONS.md — is handled autonomously and reported after the fact.

## Definition of "ready for live deployment"

- All 24 build issues and all `review-finding` issues closed (or explicitly deferred with recorded reasons).
- Full eval suite passes the DESIGN §10 release gates.
- Staging deployment verified per issue #22, including the fail-closed budget cut-off test.
- A deployment runbook exists, and a compiled list of the owner-action items (accounts, keys, domain, ICO, letter sending, voices sign-off) is presented.

## Recurring logic-efficiency audit (owner instruction 2026-08-17)

At each phase gate, a logic-efficiency audit of all production code merged since the previous gate recurs: review-only (no production code changes on the audit branch), findings recorded in `reviews/refactor-audit-<YYYY-MM>.md`, and fixes scheduled as GitHub issues labelled `refactor` — actioned by implementer agents at merge gates like any other issue. Logic economy only; verbose human-readable naming is preferred and never a finding, and licensing/safety/§3.4 invariant checks are never candidates for removal.
