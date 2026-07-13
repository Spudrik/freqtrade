# Goal Status Tracker

Use this file as a checkpoint for reviews and handoffs.

## Goal Start Baseline - handoff anchor (pre-goal reference)

- Goal id at capture: `019e3a41-0015-7310-bc13-aa8acd7f31fd`
- Goal objective: `Advance objective set 2026-06-07 by running the existing production-alpha and crash-confluence FreqAI objective queue as far as possible...`
- Snapshot captured: `2026-06-07 21:33:43 +01:00` (BST) / `2026-06-07 20:33:51 +00:00` (UTC).
- State before goal continuation (pre-goal reference point):
  - FreqAI queues:
    - `user_data\research_news_data\context_features\freqai_queue`
    - `queue_20260527_231751_233080`: `84 total, 15 completed, 69 pending` (active remainder)
    - `queue_20260526_223846`: `16 total, 2 completed, 13 pending, 1 failed`
    - `queue_20260605_020338_402028`: `16 total, 14 completed, 0 pending, 2 failed`
    - `queue_20260605_020348_711597`: `16 total, 14 completed, 0 pending, 2 failed`
    - `queue_20260605_020454_865507`: `18 total, 18 completed`
    - `queue_20260605_020346_348724`: `24 total, 18 completed, 0 pending, 6 failed`
    - `queue_20260531_214826_680862`: `12 total, 12 completed`
    - `queue_20260531_210705_354735`: `12 total, 12 completed`
    - `queue_20260528_004256_909465`: `84 total, 84 completed`
    - `queue_20260528_014316_660551`: `21 total, 15 completed, 6 failed`
    - `queue_20260527_232436_720772`: `84 total, 84 completed`
    - `queue_20260527_065856`: `96 total, 96 completed`
    - `queue_20260527_055758`: `80 total, 80 completed`
    - `queue_20260527_042809`: `120 total, 120 completed`
    - `queue_20260527_015359`: `168 total, 168 completed`
    - `queue_20260526_224438`: `14 total, 14 completed`
    - `queue_20260605_020342_609646`: `18 total, 18 completed`
    - `queue_20260526_223209`: `2 total, 0 completed, 2 pending`
    - `queue_20260526_223559`: `2 total, 0 completed, 2 pending`
  - Runner/process state:
    - `run_freqai_experiment_queue_until_idle.py`: no active process
    - `--freqaimodel`: no active process
  - Ledger/state snapshots:
    - `user_data\research_news_data\context_features\reports\freqai_research_results_ledger.csv`: `4599` rows
    - `user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h.parquet`: `294.4 MB` (last modified `2026-06-05 00:58:08`)
    - `user_data\research_news_data\context_features\structural_cache\btc_structural_features_1h.parquet`: `2.7 MB` (last modified `2026-05-26 22:20:05`)
    - `user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h.parquet`: `10.8 MB` (last modified `2026-05-27 01:44:51`)
    - `user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_bybit_linear.parquet`: `33.5 MB` (last modified `2026-05-28 11:14:24`)
    - `user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_bybit_inverse.parquet`: `32.5 MB` (last modified `2026-05-28 11:18:58`)
    - `user_data\research_news_data\context_features\context_features.sqlite`: `105.8 MB` (last modified `2026-05-24 00:34:17`)

## Goal Start Baseline (pre-goal checkpoint for this objective)

- Goal id: `019e3a41-0015-7310-bc13-aa8acd7f31fd`
- Goal objective: advance objective set via multi-path trader-confluence research without changing active collectors unless explicitly required.
- Snapshot captured: `2026-06-07 21:26:11 +01:00` (BST) / `2026-06-07 20:26:11 +00:00` (UTC).
- State at goal start:
  - Active queue root: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue`
  - Queue status summary by folder:
    - `queue_20260526_223209`: `2 total, 0 completed, 2 pending`
    - `queue_20260526_223559`: `2 total, 0 completed, 2 pending`
    - `queue_20260526_223846`: `16 total, 2 completed, 13 pending, 0 failed`
    - `queue_20260527_231751_233080`: `84 total, 4 completed, 80 pending`
    - `queue_20260605_020454_865507`: `18 total, 18 completed, 0 pending` (latest fully completed queue at this checkpoint)
  - Latest queue file used as anchor: `queue_20260605_020454_865507/freqai_experiment_queue.json`
  - FreqAI runner state: no active `run_freqai_experiment_queue_until_idle.py` or `--freqaimodel` process observed.
  - Ledger baseline:
    - `freqai_research_results_ledger.csv`: `4539` total rows
  - Data cache baseline:
    - `context_features\confluence_cache\trader_confluence_1h.parquet`
      - rows: `55971`
      - columns: `2576`
      - date range: `2020-01-01 01:00:00+00:00` to `2026-05-22 11:00:00+00:00`
    - `context_features\structural_cache\btc_structural_features_1h.parquet`
      - rows: `9073`
      - date range: `2025-04-29 00:00:00+00:00` to `2026-05-12 00:00:00+00:00`

## Goal Start Baseline - 019e3a41-0015-7310-bc13-aa8acd7f31fd

- Goal objective: `Advance objective set 2026-06-07 by running the existing production-alpha and crash-confluence FreqAI objective queue...`
- Goal started: `2026-06-07 19:47:44 +00:00` (`2026-06-07 20:47:44` BST).
- State when goal started (high-level):
  - Active queue directory: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020454_865507`
  - Queue file: `freqai_experiment_queue.json`
  - Queue experiment status snapshot:
    - total experiments: `18`
    - completed: `18`
    - pending/running/failed/invalid: `0`
  - FreqAI runner status:
    - no active `run_freqai_experiment_queue_until_idle.py` or `--freqaimodel` process was observed at checkpoint.
  - Ledger snapshot:
    - `freqai_research_results_ledger.csv`: `4539` total rows
    - rows matching this goal queue experiments (by `experiment_id`): `90`
    - statuses for those rows: `54 scored`, `36 control_comparison`
  - Running collectors at checkpoint (for continuity; not paused):
    - web/news/global/orderbook live collectors were already running.
  - Data availability snapshot used in queue preflight:
    - confluence cache rows: `55971`
    - cache range: `2020-01-01T01:00:00+00:00` to `2026-05-22T11:00:00+00:00`
    - window coverage checked/confirmed: `spot_q4_2025`, `spot_q1_2026`

## Usage

- This tracker is for review orientation only.
- Do not replace objective or progress semantics here.
- When creating new goals, add a new subsection at the top with the same fields so logic decisions can be audited against a concrete starting state.
