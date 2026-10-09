---
doc_status: active
default_read: routed
owner: user+agent
purpose: Paper-only automated and discretionary trading, with bounded reviews of net results, risk and decision quality.
do_not_use_for: Live trading, FreqAI promotion, Hyperopt, or new exploratory branches.
last_rebuilt: 2026-10-05
---

# Objective 03 - Comparative And Discretionary Paper Trading

## Contract

The 26 September 2026 user instruction supersedes Objective 02b's open-ended research queue. The 28 September extension adds two fully manual paper accounts and discretionary, journalled risk decisions; the 29 September extension adds a matched fast automatic/context-modified pair; the 1 October inverse-signal extension brought the recorded set to twelve. On 5 October the attended, reviewed refresh registered four distinct Sieve3 PAPER candidates and preserved all sixteen identities and databases. At the 2026-10-05 09:02 UTC verification, nine were ACTIVE/entry-enabled, five were DRAINING/PAUSED with thirteen protected open trades, and two were PARKED flat; there were seventeen open trades across all sixteen, including one in `fast_pivot` and three in `leader_inverse`. The four new identities had fresh RUNNING heartbeats but no trades yet; this is startup verification, not evidence of trading quality. Attended restarts caused real process gaps; the run record preserves the restart/restoration events and uncertainty, and no fills or uninterrupted stop operation are imputed during off-process gaps. The run record is authoritative for subsequent process, lifecycle and account facts. Sixteen is not a permanent cap: future PAPER trials must address distinct learning questions and pass reviewed configuration/isolation and measured-capacity checks before launch. Develop and operate actively trading approaches using useful existing discoveries and current, usable information; do not continue old planned research batches by default. The practical question is whether automated rules or combined human/agent judgement can make worthwhile returns after costs without unacceptable losses. Paper results do not prove a durable profitable edge or authorize live trading.

## Long-term goal and review boundaries

Find useful, actively managed PAPER-ONLY trading approaches rather than buy-and-hold
or an endless search for perfect predictions. This includes frozen automated
comparators, four selected Sieve entry mechanisms and two discretionary accounts,
not only a single algorithm.
Use retained Sieve/FreqAI findings as leads, not guarantees. News and global
markets may help describe the background and possible direction; BTC/ETH behaviour
may confirm or contradict it; local levels and clusters across timeframes may
help choose where to enter, exit or invalidate a trade. Volume and order-book
observations can add information only where their coverage supports the claim.
Keep these distinct roles: a market response does not disprove an upstream news
driver, and a useful level alone does not establish direction.

Preserve complementary and competing ideas. Several modest signals may be useful
together under coherent conditions; do not demand that each predicts direction,
volume and profit on its own. Conversely, a plausible story or profitable paper
trade is not evidence that every component helped. Record contrary evidence and
unavailable inputs, and assess repeatability across separate episodes as they
arrive. Historical CPI and other event leads do not justify treating all releases
or all positive/negative headlines alike; expectations, background, overlapping
events and actual market confirmation matter.

The active stage is observation and decision learning, not renewed broad research.
Trade often enough to assess active management over days/weeks when there are
defensible opportunities, without inventing trades merely to fill a quota.
Judge net return after fees/funding, maximum loss from a prior account peak,
exposure, execution reliability and decision quality together. A high win rate
alone is not success: several small wins can be wiped out by one larger loss.
Keep failed trades and account resets visible in cumulative results.

Root reviews operational health every four hours and at approved announcements.
Root owns all account/bot health, recovery, reporting, journals, decisions and
controls. Scheduled news-only work follows
`ai_guidance_docs/02_rules/reference_paper_news_sentinel_trial.md` and must not
read this objective or account/bot context. Bot-quality, comparative strategy,
performance, family-triage and learning-value assessments are user-requested
only; do not run them weekly or on routine wakes.
When requested, assess actual opportunities, filled trades, costs, losses,
decision quality and source/operation coverage. Eight weeks is a reference
observation horizon, not a minimum wait before identifying a real defect or
limited learning value. An insufficient sample remains a limited conclusion,
not proof that an idea cannot work. Do not keep a low-learning version running
solely to satisfy a calendar; make bounded keep/revise/park recommendations only
in the requested assessment.
The long-term aim may continue, but neither profit nor completion is guaranteed
by a calendar date. Live trading remains a separate user decision.

This document is the maintained programme objective; the existing run record
holds changing accounts, process identities and approved event watches. Keep the
goal-mode description aligned with this scope when the app permits replacement;
do not mark an unfinished goal complete merely to rewrite it, and do not claim a
usage-limited goal automatically resumed. Scheduled paper reviews are separate
from active goal execution.

Keep the app goal as a short pointer to this document, not a second detailed plan:
"Pursue the paper-trading objective in
C:/FreqTradeStuff/ai_guidance_docs/01_objectives/objective_03_comparative_paper_trial.md;
use its existing run record and decision journals for progress, observe its review
boundaries, and keep all activity paper-only."
When the user proposes a future change of scope, explain the proposed amendment
and ask whether it should be added here; wait for approval before changing the
objective. Ordinary decisions already authorized by this objective do not require
another scope approval. Maintain this file rather than adding another current-goal
document or scattering objectives among new status files.

## Current accounts and manual-only extension

The exact account identities, launch commands, process ownership and output paths
are recorded in `user_data/research_news_data/context_features/integrated_paper_20260926/run_record.json`.
The initial A-F accounts below were retired; do not restart them. The baseline
record contained 12 identities. The approved attended refresh is complete: four
isolated Sieve PAPER identities were initialized and seven older identities are
in DRAINING/PARKED lifecycle. At the 2026-10-05 09:02 UTC check, all 16 had
initialized databases; nine were ACTIVE, five DRAINING/PAUSED with thirteen open
trades under stored protection, and two PARKED flat. There were seventeen open
trades across the full set, including one in `fast_pivot` and three in
`leader_inverse`. The new Sieve workers had fresh RUNNING heartbeats and no trades
yet. Preserve every existing DB/loss and do not impute fills or uninterrupted
protection during recorded process gaps.
Draining exits/protection continue until each account is flat and orderless; only
then may its exact approved worker be stopped and lifecycle marked PARKED. Never
relabel an uninitialized new DB as zero history.

Before the refresh, the active automatic accounts were IntegratedPaper auto,
IntegratedPaper manual (automatic entries **with manual adjustments**, not a
manual-only account), V01, V10, PaperFastPivot and PaperLeaderImpulse. The later
`fast_auto`, `fast_context`, `leader_inverse` and `fast_level_inverse` are separate
automatic accounts. These descriptions explain preserved history; they do not
override the reviewed target lifecycle above.

On 28 September the user approved two additional, genuinely manual-only PAPER
accounts: `paper_news_manual` (news-led combined judgement) and `paper_news_lab`
(alternative, potentially conflicting ideas and risk-management comparisons).
Both use `PaperNewsManual`, which emits no entry signals. Only an explicit,
journalled main-agent decision through `paper_trial_control --account news_manual`
or `--account news_lab` may open a position. They have separate virtual balances,
databases, local API ports and decision journals. Luna never places orders.
Approved protective stops and optional price targets execute unattended; these are
execution of a prior manual decision, not independent news-driven entry rules.

The user delegates risk selection and encourages bold paper experiments. Initial
implementation reuses the 10,000 USDT virtual balance and three-position limit;
the discretionary execution envelope is up to 25% of current paper equity per
position and up to 5x leverage, with lower exposure whenever the rationale is weaker.
The existing paper broker's 25% **position** emergency stop remains a backstop;
every entry needs a tighter explicit, side-correct price stop. These limits are
paper-only implementation choices under the user's delegated discretion, not
approved live-trading settings. Changes to the envelope must be deliberate and
recorded before use, not implicit in a Luna opinion.

Each of the four new Sieve accounts is separately isolated, PAPER-only, restricted
to BTC/ETH/SOL, at most three positions, 1x leverage and a 2%-of-equity stake.
Their emergency stop floors match the selected Sieve3 plan hard stops (3% for the
pivot, support-break and LVN plans; 2% for D1 VP/BOS). These paper safety floors
and the two wrappers restoring frozen tested exit-state methods mean historical
Sieve3 aggregate results are evidence for selection, not a direct promise of
identical paper execution or future net returns. The archived Sieve2 parent is
read-only lineage evidence; it is not restarted or promoted.

Normally keep a considered position open, but allow flat periods after safety
stops, unusable information or no defensible trade. Uncertainty alone does not
justify increasing frequency. Local high/low trades need recognizable boundaries,
invalidation and costs considered; four-hour decisions are not continuous scalping.
The second account need not oppose the first: align when justified, contrast only
genuinely useful alternatives, and avoid manufacturing trades to fill a comparison.

For each decision record the account, UTC time, source, main-agent reasoning,
supporting and contradicting facts, unavailable inputs, proposed size/leverage,
explicit price protection, expected path and next review. Use the existing
per-account decision journals, not a new narrative ledger. Entry approvals expire
after five minutes; their protective instructions remain with the filled trade.
Update protection only explicitly and do not silently widen a stored stop. An
uncertain order response requires inspection/user attention, never another order
attempt. The controller verifies dry-run mode, exact account identity and local API
before any intervention. Leverage changes apply to a new position, not an existing
one without a separately planned close/re-entry.

Learn from good and bad decisions, after fees/funding and actual fills. Record
decision-process changes before using them and distinguish their observation
periods; do not treat a few wins as a proven edge. If an account is exhausted,
preserve its database and losses and label any replacement as a new attempt with
a retained lesson and full cumulative history. Never overwrite a failed account
or present a reset balance as recovered performance. No live exchange credentials,
live orders or changes to Freqtrade core are authorized.

### Active manual learning and family triage

Scheduled Luna tasks have only the news/market-input role defined in
`ai_guidance_docs/02_rules/reference_paper_news_sentinel_trial.md`; they have
no account/bot knowledge or health, recovery, table, reporting, journal,
decision, control, or trading duties. Root owns all such duties, including
manual decisions and controller actions. The existing
`brief.manual_candidates` field remains empty; no schema migration is needed.

The corrected aggressive-family premises are intentionally distinct: Vacuum
allows weak directional anticipation before a thin-profile breakout is observed;
Auction separates a weak value-return guess, an observed edge rejection and an
established value escape; Rotation follows the sign of the coin's return excess
versus its cohort, using its own return only to break an exact neutral tie and
daily position only to adjust conviction. Reclaim's
separate sweep-and-recovery premise remains unchanged. These descriptions do not
create scheduled bot-quality grading; assess family quality only when the user
asks. Missing optional feeds can affect the relevant thesis or size, but do not
veto every thesis.

Organize the identities in the reviewed registry by their actual learning role—
not as independent entries:

1. Four distinct Sieve3 entry mechanisms: `sieve_pivot_partial`,
   `sieve_d1_vp_bos_short`, `sieve_d1_support_break_long` and
   `sieve_h4_vp_lvn_long`.
2. `fast_pivot` is an alternate-exit comparator for the pivot entry family, not
   a fifth distinct Sieve entry. Compare only overlapping BTC/ETH/SOL opportunities;
   its broader pair list and different risk/exit implementation are not matched.
3. `leader_inverse` is a retained exploratory leader-response account; its
   `leader_impulse` source account is DRAINING, so new-entry comparisons end at
   that account's preserved drain boundary.
4. `fast_level_inverse` remains a separate reversal hypothesis; `fast_auto` and
   `fast_context` are DRAINING comparators, not an expanding family.
5. `news_manual` is the combined sourced market view; `news_lab` is a plausible
   alternative. Align when justified; do not manufacture opposition.

The four shared-entry accounts `auto`, `manual`, `v01` and `v10` are DRAINING
overlapping modifier comparators, not four entry mechanisms. Preserve and report
their histories; do not add entries or count them as four independent ideas after
their drain boundary. There is no numeric limit on the number of PAPER accounts.
The exact currently registered and approved identities and their lifecycle pins
come from the existing `run_record.json`; reporting and recovery use that recorded
set, not a fixed account-count ceiling. Future additions need distinct questions,
reviewed configs and lifecycle pins, and measured RAM/CPU headroom while preserving
the four-processor user reserve. No further bots are auto-launched by this amendment.

The four selected Sieve3 sources were traced to exact locked parameters and
chronological BTC/ETH/SOL validation. Reported results below are aggregate
held-out/Sieve3 figures from 2024-04-01 through 2026-04-01, not forecasts or
paper net P/L; pair-level and weekly independence must not be inferred. Use the
source result's own documented cost assumptions when quoting them.

| New PAPER account / distinct entry | Frozen selected exit | Sieve3 held-out result and coverage |
|---|---|---|
| `sieve_pivot_partial`: 1h pivot-midrange rejection short | Touch partial; frozen invalidation/remainder plan | 553 trades; +16.762%, 8.525% DD, PF 1.378 |
| `sieve_d1_vp_bos_short`: D1 volume-profile/BOS 4h-retest short | Three-stage ratchet; 2% hard stop | 105 trades; +11.655%, 4.713% DD, PF 1.979; 20/24 months, max observed gap 70.8 days |
| `sieve_d1_support_break_long`: D1 support/H4 higher-low 1h break long | Touch partial; entry stop after partial | 122 trades; +9.536%, 5.236% DD, PF 1.460; 21/24 months, max observed gap 49.5 days |
| `sieve_h4_vp_lvn_long`: H4 volume-profile LVN 1h traverse long | Zone reversal | 136 trades; +4.406%, 6.285% DD, PF 1.191; 24/24 months, max observed gap 23.2 days |

Frequency is only an aggregate over the tested three-pair cohort; sparse pair or
calendar stretches remain possible, especially for D1/BOS and support setups.
The VP/BOS Sieve2 parent had only 16 trades and weak standalone results; the
support source has no comparable positive Sieve2 row. Those facts are lineage
caveats, not grounds to misstate the Sieve3 tests or reactivate archived parents.
The LVN Sieve2 row was stronger than its selected Sieve3 result. Keep all four as
bounded prospective PAPER questions, not a claim that each Sieve stage succeeded.

Automated rules remain frozen. The two new wrappers restore only the exact tested
exit methods where the active source had later exit-state drift; canonical Sieve3
sources remain untouched. After any explicit exit or entry adaptation, report the
historical result as reference evidence, not exact reproduced performance. Preserve
all older databases, fills, losses and source histories; no reset or forced exit is
authorized. A rational interaction is not rejected solely because one component
has weak standalone evidence.

## Matched fast paper pair approved 29 September

The user approved two more isolated paper accounts, bringing the active set to
ten. `fast_auto` uses `PaperFastAuto`; `fast_context` uses `PaperFastContext`.
Both have the same fixed entry and exit code in
`user_data/strategies/paper_fast_reaction.py`, separate 10,000 USDT starting
balances/databases/logs, the existing six-pair universe and no order APIs.
These are new prospective rules inspired by retained reaction-point and activity
research, not an exact winning Sieve strategy rescaled to five minutes or a proven
FreqAI direction forecast. Do not alter any of the previous eight accounts.

Both operate on completed 5-minute candles. A calculated-level bounce/rejection
or a range break followed by a held retest supplies the entry. Single levels and
clusters are both valid. Fifteen-minute coin confirmation, observed BTC/ETH
opposition, volume, known one-hour/four-hour profile/range levels and previous-day
boundaries provide context. Higher-timeframe candles must close before use.
Four-hour profile calculations are cached until their source candle changes;
canonical indicator definitions are not edited. The bot does not browse news.
Normal volume must meet the previous 36-bar median; a held-break retest needs
1.25 times that median. The 3-ATR target must exceed twice the 0.14% estimated
round-trip fee/slippage allowance; this is an execution screen, not an edge claim.

The automatic account always requests 3x leverage. The contextual account accepts
only the main agent's journalled controls through
`python -B -m user_data.Custom_Launcher.research.paper_fast_control publish`.
Luna provides facts and an opinion but never publishes these controls or orders.
Use the same report root's `fast_context.json` and
`fast_context_decisions.jsonl`; do not create another decision/history system.

Three controls are available:

1. **Direction preference:** bias -2/-1/0/+1/+2 selects long/short leverage
   blocked:5x, 2x:4x, 3x:3x, 4x:2x, or 5x:blocked respectively. Strong one-way
   settings need defensible interacting evidence and actual market confirmation,
   not just positive/negative headline words.
2. **Exposure:** normal or reduced halves both the margin cap and the per-trade
   planned-loss budget. Calculate leverage and margin together: do not accidentally
   neutralize or double-apply the exposure adjustment.
3. **Entry permission:** normal, strong setups only, or pause new entries. A strong
   setup means volume at least 1.5x its prior median and candle body at least 0.5 ATR;
   it does not require multiple levels. Bounded scheduled-event blackouts are also
   supported. Entry and exit thresholds otherwise remain identical/frozen.

Controls require reasons, fresh Luna source URLs, an ID, observed time and expiry
no later than the supporting Luna brief or four hours. Missing, expired or malformed
controls explicitly block new contextual entries, never existing price protection.
Unknown source coverage is not a neutral market vote. New controls affect new
positions only; do not change existing leverage or widen their stored stops.

Each account permits at most three positions, 25% current-equity margin per
position, 2% planned loss including the sizing cost allowance per position and
4% combined planned loss across all positions. Opposite positions in different
coins are not assumed to offset risk. Price gaps, funding, latency and imperfect
paper fills can exceed planned loss: this is not a guaranteed maximum loss.
Fill-frozen normal protection is 1.5 entry ATR for the stop and 3 entry ATR for the
target. Other exits are completed-candle trigger failure, one-hour stall or four
hours' maximum hold. No averaging down, automatic adds or loss-history reset.

At each existing four-hour main review, consider fresh Luna news/expectations,
political/economic overlaps, US/EU/UK/Asia indices, volatility, yields/bonds, USD,
oil/gold, BTC/ETH 1h/4h/daily background, broad-coin and meme participation,
dominance where usable, local levels/clusters/volume and fresh covered order book.
Use these conditional roles together without pretending all sources are present
or that every input independently predicts direction. Record contrary facts,
missing observations, the selected controls and expected path. Renew an unchanged
assessment explicitly rather than silently keeping an expired aggressive setting.
The main agent may publish these controls in a scheduled review, but cannot force
orders or rewrite either strategy/config there.

Compare matched candidate timestamps/directions and vetoes where feasible. Different
position durations/capacity after controls can change later opportunities; account
return alone is not a clean causal comparison. Report costs, exposure and losses
as well as wins. Assess this pair only in a user-requested quality review; eight
weeks is a reference horizon, not a minimum wait to report a real defect or
limited evidence. Keep strategy-quality comparisons separate from routine control
decisions.

## Inverse-signal paper pair approved 1 October

The user approved two additional, isolated PAPER comparisons after weak results
in the corresponding original signal families. `leader_inverse` uses
`PaperLeaderInverse`: at the same completed BTC-impulse/local-break signal it
enters in the opposite direction to `PaperLeaderImpulse`. It retains the
original 2x leverage, ATR stop/target and time-exit framework. This tests
whether the original signal was pointing the wrong way; it does not assume that
reversal will be profitable after fees and funding.

`fast_level_inverse` uses `PaperFastLevelInverse`: it reverses only the
calculated-level bounce/rejection entries from `PaperFastAuto`, excluding its
held-break/retest entries. For this opposite-side trade, crossing the signal
candle's adverse extreme invalidates the idea; the inherited ATR sizing,
stop/target, stall and maximum-hold rules remain. Because its entry subset and
level-failure exit differ from the original, compare matched level-bounce
opportunities and actual costs, not account totals as a clean one-variable test.

Both variants have their own 10,000 USDT virtual balances, databases and logs,
no order APIs and no live trading authority. The original accounts and their
loss histories stay intact and frozen. The inverse accounts are prospective
counter-hypotheses, not fixes or proven edges; record abstentions, filled trades,
fees/funding and differences in later opportunity availability. Do not flip or
retune another account from a short losing streak. Include them in quality
comparisons only when the user requests an assessment; report operational defects
promptly, and treat eight weeks as a reference horizon rather than a minimum wait.

## Retired initial A-F definitions (reference, not execution queue)

| Account | Rule/question | Comparison |
|---|---|---|
| A | Completed one-hour break of the preceding 24-hour range | Plain technical reference |
| B | Completed one-hour rejection at prior range/4h/volume-profile levels | Single or clustered level reaction |
| C | Exact B rule plus above-median completed-candle volume | Does volume add to B? |
| D | Scheduled release clock, observed first BTC 15-minute response, ETH non-contradiction, then limited follower entry | Event-leader transmission; no entry before observed response |
| E | Exact B automatic rule plus separately journalled, sourced human paper interventions | Does explicit contextual judgement add to B? |
| F | Exact B automatic level rule plus live Binance futures order-book pressure risk sizing on the five covered pairs; PEPE remains unobserved | Does reducing exposure when book pressure contradicts a level rejection help? Compare matched B opportunities on source-covered pairs. |

Freeze the rule files and per-account configs before observation. Do not tune thresholds after seeing paper results. If a defect requires repair, record the change time and assess pre- and post-repair periods separately. Each account has its own dry-run DB and log; no live exchange keys. All six may run simultaneously only if system load and the four-processor user reserve allow it.

The F modifier is deliberately a **new live proxy**, not a claimed implementation of historical Bybit FreqAI's support-removal feature. It reads only completed, fresh, continuous, adequately covered Binance USD-M order-book bars. Sustained opposing pressure halves the normal stake; favorable pressure leaves stake unchanged. Missing/stale/low-coverage book data is explicitly logged as unobserved and makes no size change. F keeps B's six-pair universe, but 1000PEPE has no collected live book and receives no order-book modifier. Interpret comparisons with B on the same eligible pairs and time, including fees and risk-normalized outcomes; do not declare order-book causation from one good trade.

## Fair review and stop points - current accounts

1. First validate dry-run configuration, source clocks, exchange availability, and basic entry/exit behavior. Keep an exact run record: command, PID, worker, log, DB, check interval, and stop conditions.
2. Perform bot-quality, comparative-performance and family assessments only when the user requests them; do not turn routine health checks into strategy scoring. Eight weeks is a reference horizon, not a minimum wait to report a defect or limited evidence. Current registered identities and each lifecycle come from the existing `run_record.json`; do not assume a fixed ACTIVE/DRAINING/PARKED composition from the migration snapshot. Its learning-role grouping (four selected Sieve accounts, `fast_pivot`, two inverse accounts and two discretionary accounts) is historical context only. Rules remain frozen except for the reviewed refresh wrappers and explicit defect repairs; a requested assessment does not authorize optimization. The two discretionary accounts may change trades and rational decision procedures within their approved remit, recording changes before use. A few wins/losses do not prove a new method.
3. The shared-entry `auto`/`manual`/V01/V10 histories overlap and stop receiving entries at their drain boundary. `fast_pivot` is an alternate-exit comparator for the new pivot mechanism, not a distinct entry idea; compare only matched BTC/ETH/SOL opportunities and disclose its different risk/exits. Compare `leader_inverse` with `leader_impulse` only over their overlapping pre-drain history, and `fast_level_inverse` with the fast comparators only before their drain boundary. Compare the two discretionary accounts by idea, opportunity time, exposure and actual costs; opposite positions do not prove that an input caused an outcome. Retired A-F labels remain historical reference only.
4. Report entry opportunities, actual filled trades, size differences, exits, net paper return and drawdown, but also abstentions, no-data decisions, and errors. Separate BTC/ETH, established coins, and DOGE; never infer meme-wide results from one coin.
5. If bots crash, source timestamps are unsafe, account isolation fails, dry-run cannot be verified, or a control could affect live orders, stop only the exact affected trial process and report. Do not modify upstream Freqtrade core. A source outage is unknown data, not a neutral order book or quiet news state.
6. In a user-requested quality assessment, decide keep/revise/park per rule family. Any live-trading proposal requires a new explicit user decision and a separate risk assessment.

Use `paper_trial_snapshot --learning-review` only when the user explicitly
requests a quality/accounting assessment, never on a schedule or routine wake. It reads existing account databases and
journals only; matched pair/side/open-minute entries are learning proxies, not
proof of identical candidates or historical uptime.

## Four-hour market brief and main-agent review

The scheduled news-only task follows the four-hour mode in
`ai_guidance_docs/02_rules/reference_paper_news_sentinel_trial.md`; it does not
read this objective or perform account/bot operations. Root owns the separate
four-hour review: operational health, recovery, account projections, tables on
user request, every ACTIVE news-input decision, journals, and control renewal.
Preserve verified facts, contrary evidence, unknowns, and source/observation
clocks. Do not equate article count or sentiment words with market causation.
Routine wakes do not assess bot quality or strategy performance.

### Local-first coverage and token budget

Root's numeric market projection uses only the existing flags below. The
scheduled news-only task follows the filtered pipeline in its isolated
instruction and must not see raw collector headlines:

```text
C:\FreqTradeStuff\.venv\Scripts\python.exe -B -m user_data.Custom_Launcher.research.paper_trial_snapshot --sources --crypto --higher
```

For the news-only task, remove `sources.recent_unique_headlines` and unneeded
`sources.news`/`sources.web` collector details before model-visible output.
Retain only compact numeric market/source fields and clocks for root's review.
The news-only task reads the five direct publishers in its isolated instruction,
not bundled headline feeds.

Root reports an account table only when the user explicitly asks. No scheduled
Luna task posts an account table or routine account report. The table contains
`ACTIVE` accounts only, sorted by longest
known elapsed runtime first with unknown runtimes last, and one TOTAL row computed
only from those displayed accounts. If no accounts are `ACTIVE`, it says so and
shows zero count/P&L totals with PF undefined. Do not include DRAINING, PARKED,
unknown-lifecycle, retired, or historical accounts in a table; provide their
account facts only on explicit user request, except a genuine current mechanical
fault. Keep the UTC snapshot line and current bounded facts; do not carry forward
an old fault as current without fresh evidence or add generic table footers.

The table is factual reporting, not a bot-quality review. Do not run
`--learning-review` or create a file for it; the separate current bounded market
brief remains required. For any explicit
account detail request, use one row per requested registered identity and label
its lifecycle. Runtime is elapsed span including downtime, not measured uptime;
use a valid recorded account start, or only when `started_at_utc` is absent, the
earliest valid recorded successful `running` event for that exact account. A `†`
on the runtime marks that first-recorded observation, which is not guaranteed to
be the original start. An explicitly invalid or future account start remains
unknown and is not masked by an event. Never substitute restart, first trade,
database mtime, or global trial start. Every future account's first successful
initialization must persist its original `started_at_utc` once, after successful
RUNNING startup and database initialization are established. Root maintains
this reporting contract when adding accounts. For DRAINING accounts, the runtime
endpoint is the first recorded `draining_paused` confirmation at/after the
lifecycle boundary (observed confirmation, not exact pause onset); for PARKED
accounts use the verified `parked_flat` stop. Unknown or invalid clocks/rates
remain unknown. The ratio is profit factor computed from recorded positive and
negative `close_profit_abs` for closed trades only; it is not win probability,
does not include open/partial realized P/L, and is not a claim of complete costs.
No-loss cases are undefined rather than infinite; pooled totals recompute from
underlying closed-profit sums, never average account ratios. Keep total runtime
and trades/day blank across different trial spans. Mark uninitialized DBs and
unknown prices/P&L explicitly; do not infer them.

`--review-context` extracts the exact identities currently recorded and their
ACTIVE/DRAINING/PARKED lifecycle, effective schema-1 fast controls and approved
check clocks from the existing run record. Use it as the source of truth for the
registered set; there is no fixed expected identity count. Root reads this
packet instead of the whole record; expand the record only for an exact fault,
watch approval/update or unresolved decision. Due-watch parsing
supports both check lists and the existing two-times-in-one-field format. It
shows checks from the previous four hours and next eight hours, and flags invalid
approved clocks rather than silently calling them absent.

`--crypto` supplies descriptive Binance open-interest contract-count changes,
aggressive buying/selling, funding/premium and account positioning; participation
in the 16 most-traded eligible crypto perpetuals over one completed hour and 40
over the rolling 24-hour window; Bybit funding/premium and BTC/ETH Deribit
volatility context. Fixed-window completeness, actual observation clocks and
fetch clocks remain separate. Stale/partial/unavailable inputs are explicit,
not neutral votes. Heuristic regime labels are descriptions, not validated
direction predictions or new trading rules. Public reads use at most four
network workers; there is no new daemon, database or LLM stage.

The verified local API supplies one-hour analyzed candles and embedded four-hour
levels, not actual four-hour candles. `--higher` uses the existing four bounded
public BTC/ETH requests for completed four-hour/daily candles, adding factual
recent changes and range position from the same responses, with no extra calls
or invented trend verdict. Retain this until the local API genuinely provides
equivalent coverage. The news-only task uses only its current brief and compact
memory; it must not read prior shared context or old briefs that may contain
account data. Root may inspect exact older evidence when needed for a review.
Fetching stays outside strategy callbacks.

Cover news/events/expectations, politics, regional equities, volatility,
bonds/yields/USD/oil/gold, exchange incidents, BTC/ETH 1h/4h/daily context,
available other coins/memes, local levels/volume and book/liquidity/funding.
Give each category a brief observed, previous-session, stale, conflicting or
unavailable status with its actual observation time and source. Collector fetch
time is not the underlying observation time. A small traded-coin sample is not
whole-market breadth; missing funding, dominance or regional feeds stay unknown.
Book metrics disabled in one collector table do not prove that its completed
minute-bar history is absent: use the existing coverage-aware pressure reader.

The isolated news-only brief uses its stated two-query/four-page limit and fixed
publisher allowlist; do not repeat blocked requests or do historical research.
If coverage is unavailable, say so rather than inventing consensus. Keep the
brief at 400 words or fewer, plus compact coverage/watch fields. Deduplicate
underlying stories and distinguish expected announcements, observed surprises,
price confirmation and interacting background conditions. The news-only task
may express market opinion and conditional cases, but no account-level action,
hold, trade, or control.

The news-only task publishes only the existing schema-1 `luna_context.json`
through the publisher contract in the isolated reference. Preserve its schema
and existing control fields; only news/market observations, the brief, source-
only `main_review`, and `watch_proposals` are in scope. Never write account
health, tables, journals, decisions, or controls.
The routine news brief validity is at most four hours; choose a shorter expiry
for a time-sensitive observation. The existing validator permits a longer
maximum, but the scheduled news-only task must stay within four hours.
This brief-validity envelope does not extend the freshness of any individual
input: keep each metric's actual observation time and its own freshness
assessment. Main
controls retain their existing four-hour maximum; do not alter their schema or
let an expired control become normal permission. The brief should explain the market background, main drivers,
conditional positive/negative cases, contrary evidence, missing sources and an
opinion with its limits. Distinguish sourced consensus expectations from prior
release values; unknown expectations stay unknown. Describe what was already
priced in, what actually changed and whether BTC/ETH or other markets confirmed
it. Include US/EU/UK/Asia indices, rates/bonds, dollar, oil/gold, exchange incidents
and geopolitical influences where current sources permit. Do not turn absent
data into a neutral vote or infer cause from article counts. A downstream volume
or price response can transmit a news effect, not disprove that the news mattered.

Luna proposes additional announcement watches but cannot create, change or remove
schedules or order anything. Each proposal needs an event ID, direct source URL,
exact UTC event clock, expected outcome (or unknown), conditional crypto effects,
reason for monitoring, proposed before/after check clocks and a review condition.
The main agent decides whether the proposal is useful, schedules approved checks
with the automation tool and records them in the existing run record. The user
pre-authorizes occasional new ideas worth monitoring: no need to ask again for
every bounded watch within this paper objective. Avoid redundant wakes near a
regular review; prefer one useful pre-event check and one or two post-event checks
over constant polling. During-release capture is only observational; no guaranteed
instant reaction to breaking news is claimed.

Review an extra monitor after its usable occurrences. If it repeatedly adds no
noticeable crypto reaction information or trading relevance, retire those extra
wakes and record the reason. Missing/overlapping/subdued releases are not proof
that the event can never matter: distinguish no usable evidence, no incremental
value in the observed conditions, and a discarded universal claim. Keep rational
conditional leads parked rather than accumulating permanent schedules.

### Unattended Luna-to-main handoff

Root owns four-hour account-health, recovery, reporting, decision, journal, and
control duties. The schema-1 `luna_context.json` is only the current news/market
handoff; do not read prior content or create a second news report.
`brief.main_review` retains its compact schema (`required`, `reasons`,
`health`, `changed_since_previous` and `decision_options`) and flags only
market/source exceptions. Its `health` describes news/source coverage, never
bot/account health. The news-only task cannot report positions, protection,
account or collector health, or publish controls.

When a scheduled news task escalates a scan still unusable after the shared
recovery ladder, root promptly checks news personally and owns a bounded
resolution of the underlying coverage failure; acknowledgement alone is not
resolution. Until usable coverage is restored, keep the result explicitly
unknown. Do not restart, pause, or reconfigure collectors, or mutate process
or collector state as part of this handoff.

At each four-hour root wake, run the approved runtime helper exactly once,
obtain a separate compact health/account projection, consider all ACTIVE
news-input accounts, and journal each hold/no-trade/action and any control
renewal with exact source clocks. Use `paper_trial_snapshot --review-context`
and the existing manual review projection to determine the current account set;
do not infer it from the news brief. Preserve account-specific authority, risk
limits, stored protection, and source rules.
Root owns every ACTIVE news-input account on each fresh handoff, using
`paper_trial_snapshot --review-context`'s `news_input_reviews` and the existing
manual review projection. Those projections determine the account list; the
news-only `main_review` does not. Root records the considered hold/no-trade/action and any
control renewal through the existing journal/controller with the exact source
clocks. Scheduled news-only tasks never publish account tables. Root uses the
ACTIVE-only table contract above only on user request. A `fast_context`
control renewal is relevant only while that account is ACTIVE.

The main-agent four-hour heartbeat owns the bounded health, recovery, account
review and decision work described here. Read this handoff contract, only a
current newly published news-only observation, and
`paper_trial_snapshot --review-context` for current approved watches/account
identities/effective fast controls, not the whole run record. This lightweight
record projection performs no health/network scan. Read the record only for an
exact exceptional decision or authorized write.
On every fresh handoff, root considers every currently ACTIVE account in
`news_input_reviews`, not only conditions flagged in the news-only brief. Journal each hold/no-trade/
action and publish or explicitly renew any contextual judgement through its
existing controller. Those controllers record the exact `luna_observed_at_utc`
used, so each handoff needs a new consideration; an uncertain submission does
not count as completed. Consideration does not force a trade or direction change.
Preserve the account-specific authority, risk limits, stored protection and
source rules. When `main_review.required` is false, only after the brief is fresh
and validated, **all ACTIVE news-input accounts have been considered for that
handoff**, and there is no already-approved timed watch due may root take the
quiet/no-change shortcut, perform no duplicate process, log, account, market or
web inspection, and stay quiet. A
missing/stale/malformed brief, missing `main_review`, or unverified health is an
exception to flag for attention, not permission to assume that all is well or
to place a trade. For a flagged decision, inspect only the relevant evidence;
accept sensible sourced Luna analysis unless consequentially contradicted.
The main agent alone chooses any holds, trades and controls within the existing
account-specific authority, journals decisions, and verifies any write read-only
before another action. It retains the original account's narrower
fresh-consequential-news override procedure. No new account authority is implied.
No code/config/core edit, new research or invented recovery is authorized by a
routine wake. Stay quiet unless a material decision, failure or user attention
is warranted; a user-requested routine status uses the ACTIVE-only table and
ACTIVE-only totals. Other lifecycles appear only on explicit request.

For the matched fast pair, the main review may publish/renew the three approved
context controls through `paper_fast_control` only while `fast_context` is ACTIVE;
no renewal is needed after it becomes DRAINING/PARKED. Neither fast account has an
order API and the baseline stays unmodified. No scheduled agent may use the runtime
helper's main-agent initial-startup repair-acknowledgement option or amend pins.

The approved 4 October reporting/data improvements do not adopt V2's replacement
fast trading policy: schema-1 bias, exposure, entry permissions and freshness
requirements remain unchanged while `fast_context` is ACTIVE. An expired control
still blocks new contextual entries while ACTIVE; do not silently convert it to
normal risk. Once DRAINING/PARKED, the account needs no control renewal and its
persisted position protection remains the operative safety state.

## Windows restart and bounded paper-process recovery

Reclaim's boundary repair and attended reset were completed earlier; preserve its
results and account state unchanged. The user-approved 8 October correction also
authorizes attended repair, root review and reset for Vacuum, Auction and Rotation
only. Their resets remain pending until exact closeout is verified. Retain one
offline recovery copy per reset, exclude each copy from current totals, and restart
each runtime clock only after verified RUNNING initialization. This does not
authorize scheduled/general resets or affect any other account. Other bots'
results remain intact; weak results alone are not an implementation fault.

On 28 September the user approved Codex startup at Windows sign-in and allowing
scheduled agents, including Luna, to restore the recorded paper bots when they
wake. `Codex_OnLogon` starts the installed desktop package 30 seconds after this
user signs in; it does not bypass sign-in, change power settings, resume a paused
goal, or start trading processes by itself. The existing collector startup/hourly
watchdog remains separate and unchanged. The 9 October role change assigns all
current recovery to root; the news-only Luna task must not read this objective
or perform recovery.

At the beginning of each routine four-hour root review, execute exactly once:

```text
C:\FreqTradeStuff\.venv\Scripts\python.exe -B -m user_data.Custom_Launcher.research.paper_trial_runtime --apply
```

The helper is the only approved automatic recovery path. It can start missing
processes only for identities currently registered and explicitly approved in the
existing run record whose lifecycle is ACTIVE, using the pinned controller,
base/overlay configuration and existing account database/log. It does not execute shell
commands from the run record. A shared Windows lock and exact process-tree checks
prevent concurrent/duplicate launches; normal venv parent/child processes are one
job. Routine recovery does not restart or retune processes. Its sole stop exception
is mechanical closeout only for exact approved DRAINING identities in the current
recorded recovery set, after a fresh PAUSED heartbeat from the matched tree and
repeated DB checks confirm zero open positions and orders; it never force-closes a
trade or force-kills a process.
An attended, exact-list `--restart-draining` can move only a reviewed DRAINING
identity onto its pinned PAUSED overlay, preserving its DB and verifying stored
protection continuity. The attended `--initialize-new` path is limited to exact
reviewed new identities supported by the helper and registered in the existing
run record; it validates approved source parameters before first launch. Routine
scheduled agents never invoke either attended option.

The helper validates config fingerprints, dry-run/no-exchange-credential settings,
isolated account/API/database identities and persisted protection on open
positions, and refuses environment overrides or an unresolved manual order.
Missing databases must not silently create reset balances. A new Sieve identity's
first database creation is attended-only; routine `--apply` refuses an unattempted,
uninitialized new identity. After a recorded creation attempt, recovery cannot
initialize a replacement balance. Resource checks retain the four-processor user
reserve and one-thread worker settings.

Recovery writes only process/recovery facts in the existing run record and uses
existing bot logs. DRAINING accounts remain PAUSED while existing exits and stored
protection operate; PARKED accounts stay unavailable and are never auto-restarted.
Read each identity's lifecycle from the existing run record; never infer a new state
from elapsed time alone. Retired A-F accounts and unrelated parked variants are never
eligible.
Failed or unresolved starts require main-agent attention, not repeated retries,
interpreter guessing, config/strategy fixes, risk retuning or changed fingerprints
by root. The helper must not place/force orders. Restoring an automatic account
resumes its already-approved automated rules; restoring a manual-only account
does not create an entry. Trade authority remains unchanged.

Root then checks actual worker logs/state, distinct databases, open paper
orders and retained position protection before treating the account as healthy;
`already_running` reports process presence, not successful trading logic. Record
the reboot/execution gap and do not invent fills or assume stops executed while
the PC/bot was off. Report a restoration or blocker when material, otherwise stay
quiet. Recovery happens on the root-owned scheduled review, potentially up to
four hours after sign-in; the PC and app still need to be awake, available and
online.
