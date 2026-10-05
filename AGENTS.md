---
doc_status: active
default_read: yes
owner: user
purpose: Start-here orchestration contract, router, and top-level repo safety rules.
do_not_use_for: Detailed workflows, research history, generated reports, runtime procedures, or evidence dumps.
last_rebuilt: 2026-10-05
---

# AGENTS.md - Start Here

## 1. Orchestration model

The root agent is the control plane for substantive work. Its main job is to understand the goal, load the minimum relevant guidance, resolve important ambiguity, split work into bounded tasks, integrate results, review evidence, and make the final acceptance decision.

The root agent should preserve its context for judgement. It should not perform substantial implementation, broad mechanical repository exploration, bulk data collection, repetitive test execution, or fixes arising from its own review when those tasks can be delegated.

Tiny, tightly dependent, or genuinely trivial steps may remain in the root agent when delegation would cost more than the work. This exception must not expand into substantive implementation.

### 1.1 Use fresh leaf agents for bounded work

Sub-agents should normally be fresh, disposable leaf workers.

Do not let sub-agents recursively create their own agent trees unless a routed rule explicitly requires it.

Use only the roles that add value to the current task:

- **Explorer / Researcher** - maps code, files, behaviour, documentation, or evidence when discovery is needed. Normally read-only.
- **Builder / Worker** - owns one coherent bounded implementation or remediation task.
- **Tester** - reproduces behaviour and executes targeted verification when separate validation adds value.
- **Reviewer / QA** - independently challenges the integrated candidate in a fresh read-only context. It reports findings and never fixes them.

Do not instantiate every role automatically. A small explicit task may combine Builder + Tester. A separate Explorer is unnecessary when the relevant area is already known.

### 1.2 Front-load the contract

Before delegating substantive work, the root agent should make the task sufficiently explicit that a fresh worker can act without the parent's full conversation.

Each delegated task should contain only what is needed:

- exact objective,
- relevant paths, symbols, subsystem, or data,
- applicable routed guidance,
- important constraints and preserved interfaces,
- allowed write scope,
- acceptance criteria,
- required verification,
- expected result format,
- explicit exclusions where useful.

Resolve important architectural or requirement ambiguity before dispatch rather than making workers rediscover the project goal.

### 1.3 Keep worker context narrow

Fresh independent workers should not inherit the parent thread by default.

Pass the minimum task-specific context required to do the job correctly. Include a small amount of recent context only when the task genuinely depends on those decisions.

Do not send unrelated project instructions, historical chat, broad archives, large logs, or the complete instruction tree.

Where practical, the root agent resolves instruction routing first and points the worker directly at the relevant guidance.

Prefer search/grep followed by targeted reads over broad repository reading.

Keep detailed evidence in existing task artifacts/files where appropriate and return only decision-relevant summaries to the root agent.

### 1.4 Model and reasoning selection

Roles do not imply fixed model names.

Choose capability dynamically:

- use the cheapest capable execution model/reasoning level for bounded mechanical or well-specified work,
- raise worker capability when the bounded task is genuinely difficult or ambiguous,
- preserve the strongest available reasoning capability for architecture, cross-system judgement, unresolved ambiguity, integration, adjudication, and final acceptance,
- choose independent review strength primarily from the consequence of error, not from how difficult the implementation was.

Do not silently use the parent model and reasoning level for every child merely because it is available.

### 1.5 Standard substantive-work flow

Use this flow unless routed guidance gives a more specific one:

1. **Root understands and scopes** the goal.
2. **Optional Explorer / Researcher** gathers only missing evidence needed to form a clear implementation contract.
3. **Builder / Worker** implements one bounded task and performs its required local checks.
4. **Optional separate Tester** performs independent reproduction or targeted verification when this adds useful evidence.
5. **Root integrates and sanity-checks** the accumulated candidate against surrounding code, interfaces, dependencies, and the original requirement.
6. **Fresh Reviewer / QA** independently challenges meaningful or risky changes.
7. **Root adjudicates** reviewer findings.
8. Accepted findings become bounded remediation tasks for a fresh Worker rather than being fixed by the Reviewer or casually patched by the root.
9. Re-run affected verification after every material edit.
10. Obtain a new fresh review when the changed candidate's risk warrants it.
11. **Root gives final acceptance** only to the same verified revision that was actually reviewed.

For substantial QA, prefer expectations-first review: derive the checks and likely failure modes from the requirement before relying on the implementer's explanation of what changed.

Reviewer evidence informs acceptance; the Reviewer does not make the final project decision.

### 1.6 Fresh review and independence

A Reviewer must be fresh and read-only.

Give it the requirement/acceptance criteria, the integrated candidate or diff, and only the evidence it needs.

Do not give it the implementer's reasoning unless necessary to investigate a specific issue.

The Reviewer:
- looks for missing requirements, regressions, edge cases, integration failures, safety issues, compatibility problems, and unjustified complexity,
- reports PASS / NEEDS WORK / BLOCKED or equivalent with evidence,
- never edits the implementation,
- never fixes its own findings.

Fresh context provides context independence, not guaranteed model-failure independence. The root remains responsible for final judgement.

Independent review may be omitted for genuinely trivial/very-low-risk work when root verification is sufficient. Review depth should increase with consequence of error.

### 1.7 Fix loop

When a review finding is accepted:

1. root defines the correction precisely,
2. fresh Worker performs the correction,
3. affected tests/verification are rerun,
4. root integrates and checks the updated candidate,
5. fresh review is repeated when the materiality/risk warrants it.

Any material edit invalidates acceptance evidence for the earlier revision.

### 1.8 Parallelism and batching

Parallelise only genuinely independent work.

Do not run agents concurrently when they:
- edit the same files or tightly coupled subsystem,
- depend on one another's output,
- share mutable state that cannot be isolated,
- would duplicate the same investigation.

Prefer sensible batches of related files/work over one agent per file.

Sequential work is preferred when the next task depends on the previous result.

Do not increase agent count merely because capacity exists. More agents can increase token use and coordination errors.

### 1.9 Worker return contract

Worker responses to the root should be concise and normally contain:

- **RESULT**
- **FINDINGS / CHANGES**
- **VERIFICATION**
- **UNCERTAINTY / BLOCKERS**

Do not return long conversational histories, raw logs, or large copied files when a concise summary and exact evidence location is enough.

Sub-agents and local compute workers are separate concepts. Runtime/process limits remain governed by routed runtime guidance.

---

## 2. Core routing rule

Start from the user's latest explicit request.

Do not read every guidance file. Route from the request through this file to the minimum relevant subsystem `AGENTS.md` and rule files.

The user's latest explicit request defines the active task. Use programme/objective documents only when routed below or when programme-level context is explicitly required.

---

## 3. Upstream Freqtrade core protection

Treat `freqtrade/**`, upstream-owned `tests/**`, and root dependency/build files as protected and read-only unless the user explicitly approves the affected files and intended behavioural change.

Normal project customisation belongs under `user_data/**`, including strategies, indicators, FreqAI models, launchers, collectors, research tools, custom tests, and research-only dependency manifests. Approved project guidance belongs under `ai_guidance_docs/**`.

Do not add project/research packages to Freqtrade root requirements.

If a task appears to require an upstream/core edit, stop before editing and provide:
- the root cause,
- the preferred solution under `user_data/**`,
- the exact core files and behavioural change that would otherwise be required.

Existing approved maintained-fork exceptions are limited to:
- historical Bybit orderbook integration while it remains compatible with upstream,
- Pandas dtype compatibility adjustment in `strategy_helper.py`,
- user-approved Hyperopt `INITIAL_POINTS = 60`.

Do not expand an existing exception or create another one without explicit user approval.

---

## 4. Task routing

Read the nearest subsystem `AGENTS.md` when one exists, then load only the relevant route.

- Coding implementation/refactoring  
  → `ai_guidance_docs/02_rules/rules_codebase_workflow.md`

- Runtime commands, backtests, dry-run startup, worker environments, process handling, collectors, raw-data locations, long-running jobs, or source-data work  
  → `ai_guidance_docs/02_rules/rules_runtime_environment.md`

- Hyperopt work  
  → `ai_guidance_docs/02_rules/rules_hyperopt_general.md`

- Exit/risk research  
  → `ai_guidance_docs/02_rules/rules_exit_and_risk_research.md`
  → relevant Sieve-stage rules

- Current bounded paper-trial work  
  → `ai_guidance_docs/01_objectives/objective_03_comparative_paper_trial.md`

- Programme roadmap, programme architecture, named parked objectives, or material stage changes  
  → `ai_guidance_docs/00_project_control/objectives_master.md`
  → relevant objective document only when required

- Programme history/traceability  
  → `ai_guidance_docs/05_program_traceability/AGENTS.md`

- FreqAI/context/event research  
  → relevant current objective/subsystem guidance
  → `ai_guidance_docs/02_rules/reference_freqai_event_reaction_research_method.md` only when the routed task requires that method

Do not load a route merely because it exists.

The old Objective 02b batch/branch queue is not routine current context. Read it only when tracing specific research evidence, reopening that research, or when another routed rule explicitly requires it.

---

## 5. Archives and historical context

`ai_guidance_docs/99_archive/**` and long historical logs are no-read by default.

Read historical material only when:
- the user explicitly asks for historical evidence,
- current evidence cannot resolve the task, or
- routed guidance specifically requires it.

---

## 6. Runtime safety

Assume other project processes may already be running.

Before starting, stopping, monitoring, or configuring project processes, worker environments, collectors, backtests, Hyperopts, long-running jobs, or raw-data access, read:

`ai_guidance_docs/02_rules/rules_runtime_environment.md`

Never infer process ownership from process names alone or kill broad Python/Freqtrade processes without the routed runtime rules.

---

## 7. Hyperopt/Sieve invariant

Sieve is the approved Hyperopt system for trading entry/exit candidate discovery unless the user explicitly approves an exception.

For search design, promotion, parameter locking, pair/window coverage, and Sieve-stage requirements, load `rules_hyperopt_general.md` and the relevant Sieve-stage guidance.

Separately approved direct-test/FreqAI/context research is governed by its own routed objective/research guidance and must not be treated as a competing entry/exit Hyperopt system.

---

## 8. Scope changes

Use routed goal-mode rules for bounded goal runs.

Require explicit approval when work would materially:
- change programme stage or requested subsystem,
- reopen or promote parked work,
- change frozen dry-run strategy/config behaviour,
- introduce live or ambiguous trading/order behaviour,
- create new acceptance/pass-fail rules outside the current task,
- require substantial unapproved data acquisition.

Do not stop for ordinary implementation decisions inside the explicit approved task.

If a clearly relevant rule is missing from routing, read only the minimum required rule and flag the routing gap.

---

## 9. Project non-negotiables

1. Preserve behaviour unless the user explicitly requests a change.
2. Do not invent dry-run/live-trading settings, leverage, stake size, pair limits, order behaviour, or safety settings.
3. Do not claim a permanent current best unless the current objective and result evidence support it.
4. Keep indicators and heavy dataframe calculations vectorized unless routed guidance explicitly requires otherwise.
5. Do not treat news/context sources as ready without source-specific readiness evidence.
6. Do not promote FreqAI work without a named trader-readable hypothesis, controls, baselines, and clean-window/source checks.
7. Keep the verified dry-run strategy/config frozen during later research; use separate research variants/configs unless explicitly approved otherwise.
8. Do not delete archives or generated reports without explicit user approval.
