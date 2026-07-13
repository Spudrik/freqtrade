---
doc_status: active
default_read: routed
owner: user+agent
purpose: Compact Sieve3-to-Sieve2 baseline lookup for exit-stage comparison.
do_not_use_for: Full run history or promotion decisions without checking current result files.
last_rebuilt: 2026-06-17
---

# Sieve3 / Sieve2 Baseline Lookup

Use this when a request asks how a Sieve3 exit result compares with the Sieve2 entry foundation that justified the branch.

Columns:

1. `sieve2_source`: parsed source foundation for active top-level `sieve3*.py` files.
2. `s3_files`: count of active top-level Sieve3 files currently mapped to that source.
3. `tp/sl`: baseline fixed-exit row used for comparison, usually the best matched `2/2` Sieve2 row when available.
4. `ret%`: `profit_total * 100`, normalized to the configured test wallet/exposure for that source row.
5. `wr%`: win rate percentage.
6. `dd%`: max drawdown percentage.
7. `pf`: profit factor.

Notes:

1. This is a quick lookup, not a full evidence ledger.
2. Baselines are selected from existing Sieve2 JSONL rows by preferring matched `2/2` rows and then highest winrate, return, and trade count.
3. If pair scope, windows, wallet, stake, or leverage changed, state that before comparing a Sieve3 exit row to this table.
4. Sieve3 pass/fail is user-discretionary; this table only supplies the source baseline context.
5. `sieve3_exit_level_zone_reversal_from_<source>` files map to the same `sieve2_source` row as their source stem; they add one branch count per non-pattern source but do not change the baseline metrics.

Active top-level Sieve3 files scanned: `1669` mapped, `8` unmapped.

| sieve2_source | s3_files | tp/sl | ret% | wr% | trades | dd% | pf | baseline_window | baseline_job |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `sieve2_bos_bear_continuation_short_1h` | 17 | `2/2` | 0.56 | 55.56 | 36 | 1.27 | 1.17 | `pattern_continuation_1h_2021_22` | `20260521T012740_entry_all_resume` |
| `sieve2_bos_bear_continuation_short_8h` | 17 | `2/2` | -1.36 | 48.91 | 137 | 2.81 | 0.90 | `pattern_continuation_8h_2020_23` | `20260521T012740_entry_all_resume` |
| `sieve2_choch_bull_reversal_long_4h` | 16 | `2/2` | 0.70 | 55.88 | 34 | 0.74 | 1.24 | `pattern_head_shoulders_4h_2020_21` | `20260521T012740_entry_all_resume` |
| `sieve2_choch_bull_reversal_long_8h` | 16 | `2/2` | 1.09 | 51.96 | 331 | 5.67 | 1.03 | `pattern_reversal_8h_2020_23` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_avwap_reject_short` | 16 | `2/2` | 1.36 | 60.53 | 38 | 0.62 | 1.45 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_continuation_flag_present_long_1h` | 16 | `2/2` | 0.86 | 61.90 | 21 | 0.43 | 1.52 | `pattern_continuation_1h_2021_22` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_geometry_ascending_channel_lower_bounce_long_4h` | 16 | `2/2` | 2.04 | 59.15 | 71 | 1.63 | 1.34 | `pattern_geometry_4h_2020_21` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_geometry_ascending_channel_lower_bounce_long_8h` | 16 | `2/2` | 1.87 | 66.67 | 30 | 0.45 | 1.89 | `pattern_geometry_8h_2020_23` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_geometry_descending_channel_lower_breakdown_short_1h` | 16 | `2/2` | 0.84 | 61.54 | 26 | 0.52 | 1.42 | `pattern_geometry_1h_2023_24_compression` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_geometry_descending_channel_lower_breakdown_short_4h` | 16 | `2/2` | 1.54 | 70.00 | 20 | 0.65 | 2.23 | `pattern_geometry_4h_2020_21` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_geometry_rectangle_breakdown_short_1h` | 16 | `2/2` | 2.11 | 77.27 | 22 | 0.60 | 3.07 | `pattern_geometry_1h_2023_24_compression` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_geometry_triangle_squeeze_breakdown_short_4h` | 16 | `2/2` | 2.79 | 58.70 | 92 | 1.04 | 1.36 | `pattern_geometry_4h_2020_21` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_geometry_triangle_squeeze_breakout_long_1h` | 16 | `2/2` | 5.25 | 66.27 | 83 | 0.88 | 1.94 | `pattern_geometry_1h_2023_24_compression` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_geometry_wedge_breakdown_short_4h` | 16 | `2/2` | 1.47 | 65.38 | 26 | 1.04 | 1.78 | `pattern_geometry_4h_2020_21` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_geometry_wedge_breakout_long_1h` | 16 | `2/2` | 2.52 | 72.41 | 29 | 1.02 | 2.52 | `pattern_geometry_1h_2023_24_compression` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_geometry_wedge_breakout_long_8h` | 16 | `2/2` | 1.25 | 62.96 | 27 | 0.46 | 1.60 | `pattern_geometry_8h_2020_23` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_mtf_h4_supply_reject_short_1h_local_break` | 16 | `2/2` | 1.72 | 63.89 | 36 | 0.65 | 1.67 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_multi2_tlv2_boschoch_fall_res_ride_bos_bear_short_1h` | 16 | `2/2` | 2.03 | 67.65 | 34 | 0.60 | 1.89 | `pattern_reversal_1h_2024_25` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_multi2_tlv2_boschoch_res_break_bos_bull_long_1h` | 16 | `2/2` | 1.36 | 58.49 | 53 | 0.99 | 1.30 | `pattern_reversal_1h_2024_25` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_multi2_tlv2_boschoch_res_break_bos_bull_long_8h` | 16 | `2/2` | 1.89 | 64.86 | 37 | 0.62 | 1.69 | `pattern_reversal_8h_2020_23` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_multi2_tlv2_boschoch_res_prox_reject_choch_bear_short_4h` | 16 | `2/2` | 2.00 | 75.00 | 20 | 0.40 | 2.98 | `pattern_head_shoulders_4h_2020_21` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_multi2_tlv2_boschoch_sup_break_bos_bear_short_1h` | 16 | `2/2` | 0.81 | 60.00 | 20 | 0.71 | 1.53 | `pattern_reversal_1h_2024_25` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_multi2_tlv2_vp_res_break_vp_node_long_4h` | 16 | `2/2` | 1.81 | 70.83 | 24 | 1.05 | 2.21 | `auto_generic_4h_2020_21_cycle` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_multi2_vp_prior_equal_highs_reject_vp_vah_short` | 16 | `2/2` | 0.91 | 61.90 | 21 | 0.61 | 1.55 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_reversal_double_bottom_confirmed_long_4h` | 16 | `2/2` | 1.17 | 60.00 | 35 | 0.53 | 1.40 | `pattern_head_shoulders_4h_2020_21` | `20260521T012740_entry_all_resume` |
| `sieve2_complete_pattern_reversal_double_bottom_present_long_4h` | 16 | `2/2` | 1.22 | 65.22 | 23 | 0.71 | 1.72 | `pattern_head_shoulders_4h_2020_21` | `20260521T012740_entry_all_resume` |
| `sieve2_crash_flush_reclaim_long_1h` | 17 | `3/3` | -1.91 | 55.92 | 152 | 6.09 | 0.93 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_crash_liquidity_sweep_snapback_long_1h` | 17 | `3/3` | -1.46 | 54.17 | 48 | 2.45 | 0.84 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_crash_rsi_extreme_reclaim_long_4h` | 17 | `3/3` | -0.17 | 55.56 | 9 | 0.98 | 0.89 | `auto_generic_4h_2020_21_cycle` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_ladder_short_res_fail` | 17 | `2/2` | 1.10 | 55.36 | 56 | 1.47 | 1.22 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_loosen_continuation_flag_confirmed_long_1h_broader_trigger` | 16 | `4/2` | 2.94 | 75.00 | 12 | 0.41 | 5.85 | `pattern_continuation_1h_2021_22` | `20260602T061300_entry_sieve2_high_performer_4_2_check` |
| `sieve2_loosen_geometry_ascending_channel_upper_breakout_long_1d_broader_trigger` | 17 | `3/3` | 1.19 | 100.00 | 4 | 0.00 | 0.00 | `pattern_geometry_1d_2020_22` | `20260602T092945_entry_sieve2_tune_fix_next` |
| `sieve2_loosen_geometry_rectangle_breakout_long_1d_broader_trigger` | 16 | `3/3` | 2.07 | 81.82 | 11 | 0.33 | 4.43 | `pattern_geometry_1d_2020_22` | `20260611T234019_entry_sieve_top_level_backlog6_cores14_20260611_01` |
| `sieve2_loosen_geometry_triangle_squeeze_breakdown_short_1h_broader_trigger` | 16 | `3/3` | 4.49 | 63.93 | 61 | 1.40 | 1.66 | `pattern_geometry_1h_2023_24_compression` | `20260529T213457_entry_all` |
| `sieve2_loosen_geometry_triangle_squeeze_breakout_long_4h_broader_trigger` | 16 | `3/3` | 0.44 | 55.56 | 18 | 1.28 | 1.18 | `pattern_geometry_4h_2020_21` | `20260529T213457_entry_all` |
| `sieve2_loosen_geometry_triangle_upper_reject_short_8h_broader_trigger` | 16 | `3/3` | 1.48 | 85.71 | 7 | 0.30 | 5.84 | `pattern_geometry_8h_2020_23` | `20260529T213457_entry_all` |
| `sieve2_loosen_ladder_long_sup_hold_broader_trigger` | 18 | `3/3` | 0.85 | 100.00 | 3 | 0.00 | 0.00 | `auto_generic_1h_2020_q2_q3` | `20260611T234019_entry_sieve_top_level_backlog6_cores14_20260611_01` |
| `sieve2_loosen_mtf_d1_vah_reject_short_4h_reject_broader_trigger` | 17 | `3/3` | 2.23 | 68.18 | 22 | 0.61 | 2.06 | `auto_generic_4h_2020_21_cycle` | `20260529T213457_entry_all` |
| `sieve2_loosen_multi2_tlv2_boschoch_res_reject_choch_bear_short_4h_broader_trigger` | 18 | `3/3` | 1.19 | 75.00 | 8 | 0.33 | 2.92 | `pattern_head_shoulders_4h_2020_21` | `20260529T213457_entry_all` |
| `sieve2_loosen_multi2_tlv2_boschoch_ris_sup_ride_bos_bull_long_8h_broader_trigger` | 18 | `3/3` | 0.86 | 71.43 | 7 | 0.61 | 2.39 | `pattern_reversal_8h_2020_23` | `20260611T234019_entry_sieve_top_level_backlog6_cores14_20260611_01` |
| `sieve2_loosen_multi2_tlv2_vp_res_reject_vp_node_short_8h_broader_trigger` | 18 | `3/3` | 1.61 | 75.00 | 12 | 0.66 | 2.64 | `auto_generic_8h_2020_22_cycle` | `20260529T213457_entry_all` |
| `sieve2_loosen_multi2_tlv2_vp_sup_bounce_vp_node_long_8h_broader_trigger` | 1 | `3/3` | 1.49 | 100.00 | 5 | 0.00 | 0.00 | `auto_generic_8h_2020_22_cycle` | `20260602T092945_entry_sieve2_tune_fix_next` |
| `sieve2_loosen_pivot_long_trend_pullback_broader_trigger` | 18 | `3/3` | 1.15 | 60.00 | 25 | 0.87 | 1.38 | `auto_generic_1h_2020_q2_q3` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_loosen_reversal_double_bottom_confirmed_long_1h_broader_trigger` | 16 | `3/3` | 3.50 | 70.00 | 30 | 1.24 | 2.22 | `pattern_reversal_1h_2024_25` | `20260611T234019_entry_sieve_top_level_backlog6_cores14_20260611_01` |
| `sieve2_loosen_reversal_double_bottom_present_long_1h_broader_trigger` | 16 | `3/3` | 3.50 | 70.00 | 30 | 1.24 | 2.22 | `pattern_reversal_1h_2024_25` | `20260611T234019_entry_sieve_top_level_backlog6_cores14_20260611_01` |
| `sieve2_loosen_triple_bottom_present_long_4h_broader_trigger` | 16 | `3/3` | 1.58 | 59.38 | 32 | 1.28 | 1.39 | `pattern_multi_peak_4h_2023_25` | `20260529T213457_entry_all` |
| `sieve2_loosen_triple_top_confirmation_pressure_short_1h_broader_trigger` | 16 | `3/3` | 1.93 | 68.42 | 19 | 0.58 | 2.03 | `pattern_multi_peak_1h_2021_22` | `20260529T213457_entry_all` |
| `sieve2_mtf_confluence_d1_bos_h4_vp_node_long` | 17 | `3/3` | 1.09 | 54.41 | 68 | 1.44 | 1.11 | `auto_generic_4h_2020_21_cycle` | `20260611T234019_entry_sieve_top_level_backlog6_cores14_20260611_01` |
| `sieve2_mtf_confluence_d1_resistance_reject_4h_short` | 17 | `3/3` | 4.81 | 54.82 | 197 | 2.62 | 1.17 | `auto_generic_4h_2020_21_cycle` | `20260611T234019_entry_sieve_top_level_backlog6_cores14_20260611_01` |
| `sieve2_mtf_confluence_d1_resistance_sweep_h4_bos_short` | 1 | `3/3` | 2.08 | 65.22 | 23 | 0.63 | 1.88 | `auto_generic_4h_2020_21_cycle` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_mtf_confluence_d1_support_sweep_h4_bos_long` | 1 | `3/3` | 5.53 | 53.48 | 402 | 9.22 | 1.10 | `auto_generic_4h_2020_21_cycle` | `20260529T213457_entry_all` |
| `sieve2_mtf_confluence_d1_vp_bos_4h_retest_long` | 17 | `3/3` | 2.57 | 58.33 | 60 | 1.51 | 1.33 | `auto_generic_4h_2020_21_cycle` | `20260529T213457_entry_all` |
| `sieve2_mtf_confluence_d1_vp_bos_4h_retest_short` | 1 | `3/3` | 0.78 | 62.50 | 16 | 1.59 | 1.35 | `auto_generic_4h_2020_21_cycle` | `20260602T092945_entry_sieve2_tune_fix_next` |
| `sieve2_mtf_confluence_h4_bos_prior_1h_breakdown_short` | 1 | `3/3` | 1.50 | 53.57 | 112 | 2.19 | 1.09 | `auto_generic_1h_2020_q2_q3` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_mtf_confluence_h4_demand_reclaim_1h_long` | 1 | `3/3` | 6.85 | 51.71 | 1787 | 13.52 | 1.02 | `auto_generic_1h_2020_q2_q3` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_mtf_std_daily_ema_bb_reject_short_1h` | 17 | `3/3` | 31.68 | 62.77 | 838 | 6.99 | 1.22 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtf_std_daily_ema_bb_retest_long_1h` | 17 | `3/3` | -5.32 | 57.92 | 4004 | 33.62 | 0.99 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtf_std_daily_ema_bb_volume_breakdown_short_1h` | 17 | `3/3` | 25.87 | 59.78 | 1651 | 14.14 | 1.09 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtf_std_daily_ema_bb_volume_breakout_long_1h` | 17 | `3/3` | 42.72 | 59.55 | 3211 | 30.84 | 1.06 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtf_std_daily_h4_trend_pullback_reclaim_long_1h` | 17 | `3/3` | 23.17 | 59.70 | 2206 | 19.79 | 1.05 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtf_std_daily_h4_trend_pullback_reject_short_1h` | 17 | `3/3` | 3.55 | 59.57 | 282 | 7.15 | 1.08 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtf_std_daily_macd_volume_breakdown_short_1h` | 17 | `3/3` | -5.34 | 56.85 | 825 | 13.54 | 0.96 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtf_std_daily_macd_volume_breakout_long_1h` | 17 | `3/3` | 47.68 | 60.63 | 2192 | 16.87 | 1.10 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtf_std_daily_prior_high_breakout_long_1h` | 17 | `3/3` | 28.47 | 65.29 | 507 | 5.32 | 1.35 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtf_std_daily_prior_low_breakdown_short_1h` | 17 | `3/3` | 7.07 | 58.86 | 1038 | 7.99 | 1.04 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtf_std_daily_rsi_pullback_reclaim_long_1h` | 17 | `3/3` | 39.40 | 60.29 | 2113 | 15.92 | 1.09 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtf_std_daily_rsi_pullback_reject_short_1h` | 17 | `3/3` | 0.35 | 58.42 | 101 | 5.97 | 1.02 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtfx_d1_avwap_reclaim_long_4h_retest` | 17 | `3/3` | 4.12 | 52.66 | 357 | 6.93 | 1.08 | `auto_generic_4h_2020_21_cycle` | `20260602T092945_entry_sieve2_tune_fix_next` |
| `sieve2_mtfx_d1_avwap_reject_short_4h_retest` | 17 | `4/2` | 1.28 | 80.00 | 5 | 0.18 | 8.26 | `auto_generic_4h_2020_21_cycle` | `20260602T062802_entry_sieve2_high_performer_4_2_check` |
| `sieve2_mtfx_d1_hs_short_4h_breakdown` | 16 | `3/3` | 0.57 | 100.00 | 2 | 0.00 | 0.00 | `auto_generic_4h_2020_21_cycle` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtfx_d1_liq_equal_highs_short_4h_choch` | 17 | `3/3` | 1.42 | 77.78 | 9 | 0.31 | 3.44 | `pattern_head_shoulders_4h_2020_21` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtfx_d1_rectangle_long_4h_breakout` | 16 | `3/3` | 1.97 | 62.96 | 27 | 0.92 | 1.64 | `pattern_geometry_4h_2020_21` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtfx_d1_rectangle_long_4h_retest` | 16 | `3/3` | 2.27 | 63.33 | 30 | 1.47 | 1.68 | `pattern_geometry_4h_2020_21` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtfx_d1_rectangle_short_4h_breakdown` | 16 | `3/3` | 1.28 | 60.00 | 25 | 0.93 | 1.42 | `pattern_geometry_4h_2020_21` | `20260602T092945_entry_sieve2_tune_fix_next` |
| `sieve2_mtfx_d1_rectangle_short_4h_retest` | 1 | `3/3` | 0.27 | 55.56 | 9 | 0.61 | 1.22 | `pattern_geometry_4h_2020_21` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_mtfx_d1_tlv2_res_break_long_4h_bos` | 17 | `3/3` | 0.51 | 55.56 | 18 | 1.25 | 1.21 | `auto_generic_4h_2020_21_cycle` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtfx_d1_vp_poc_reclaim_long_4h_retest` | 17 | `3/3` | 4.38 | 56.20 | 121 | 4.45 | 1.27 | `auto_generic_4h_2020_21_cycle` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtfx_d1_vp_vah_reject_short_4h_retest` | 18 | `3/3` | 1.25 | 83.33 | 6 | 0.25 | 5.99 | `auto_generic_4h_2020_21_cycle` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtfx_h4_avwap_reject_short_1h_retest` | 1 | `3/3` | 0.92 | 56.52 | 23 | 1.30 | 1.33 | `auto_generic_1h_2020_q2_q3` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_mtfx_h4_ihs_long_1h_breakout` | 16 | `3/3` | 1.10 | 57.69 | 26 | 0.62 | 1.33 | `auto_generic_1h_2020_q2_q3` | `20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01` |
| `sieve2_mtfx_h4_liq_prior_high_short_1h_choch` | 18 | `3/3` | 1.52 | 77.78 | 9 | 0.31 | 3.78 | `pattern_reversal_1h_2024_25` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_mtfx_h4_tlv2_sup_break_short_1h_choch` | 17 | `3/3` | 0.59 | 75.00 | 4 | 0.29 | 3.03 | `pattern_reversal_1h_2024_25` | `20260602T092945_entry_sieve2_tune_fix_next` |
| `sieve2_mtfx_h4_vp_lvn_traverse_long_1h_breakout` | 17 | `3/3` | 15.79 | 54.45 | 775 | 3.43 | 1.14 | `auto_generic_1h_2020_q2_q3` | `20260529T213457_entry_all` |
| `sieve2_mtfx_h4_vp_poc_reject_short_1h_retest` | 18 | `3/3` | 2.97 | 57.35 | 68 | 2.46 | 1.35 | `auto_generic_1h_2020_q2_q3` | `20260612T130425_entry_sieve2_top_level_clear100_cores8_20260612_01_failed32_retry02` |
| `sieve2_mtfx_h4_wedge_long_1h_retest` | 16 | `3/3` | 0.24 | 56.25 | 16 | 1.42 | 1.12 | `pattern_geometry_1h_2023_24_compression` | `20260612T130425_entry_sieve2_top_level_clear100_cores8_20260612_01_failed32_retry02` |
| `sieve2_multi2_prior_vp_breakout_long` | 17 | `2/2` | 1.60 | 56.92 | 65 | 0.87 | 1.28 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_tlv2_vp_res_break_vp_bullctx_long_1h` | 17 | `2/2` | 6.82 | 58.90 | 219 | 1.93 | 1.37 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_tlv2_vp_res_break_vp_val_long_1d` | 17 | `2/2` | -0.66 | 50.00 | 140 | 2.84 | 0.95 | `auto_generic_1d_2020_22_cycle` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h` | 17 | `2/2` | 8.37 | 57.63 | 354 | 3.00 | 1.26 | `auto_generic_4h_2020_21_cycle` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_tlv2_vp_res_reject_vp_node_short_1h` | 17 | `2/2` | 0.63 | 52.24 | 245 | 3.06 | 1.03 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_tlv2_vp_res_reject_vp_node_short_4h` | 17 | `2/2` | 0.94 | 54.72 | 53 | 0.72 | 1.20 | `auto_generic_4h_2020_21_cycle` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_tlv2_vp_res_reject_vp_vah_short_4h` | 17 | `2/2` | 0.73 | 55.17 | 58 | 1.47 | 1.14 | `auto_generic_4h_2020_21_cycle` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_tlv2_vp_sup_break_vp_bearctx_short_1h` | 17 | `2/2` | 1.59 | 57.81 | 64 | 1.18 | 1.29 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_vp_prior_month_high_break_vp_bullctx_long` | 17 | `2/2` | 2.84 | 55.17 | 174 | 3.23 | 1.17 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_vp_prior_month_high_break_vp_node_long` | 17 | `2/2` | 3.25 | 53.50 | 357 | 4.43 | 1.09 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_vp_prior_month_high_break_vp_val_long` | 17 | `2/2` | 5.23 | 54.43 | 406 | 2.79 | 1.13 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_vp_prior_week_high_break_vp_bullctx_long` | 17 | `2/2` | 5.13 | 58.79 | 165 | 1.58 | 1.36 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_vp_prior_week_high_break_vp_node_long` | 17 | `2/2` | 3.31 | 54.82 | 228 | 2.84 | 1.16 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_multi2_vp_prior_week_high_break_vp_val_long` | 17 | `2/2` | 4.34 | 55.69 | 255 | 2.71 | 1.19 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_multi3_prior_avwap_vp_breakout_long` | 17 | `2/2` | 1.37 | 54.55 | 99 | 1.26 | 1.15 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_multi4_prior_range_avwap_vp_breakout_long` | 17 | `2/2` | -0.38 | 50.71 | 282 | 4.78 | 0.99 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_novel` | 6 |  |  |  |  |  |  | missing | missing |
| `sieve2_novel_mtf_d1_midline_reject_1h_reject_short` | 17 | `3/3` | 2.91 | 72.73 | 22 | 0.91 | 2.68 | `auto_generic_1h_2020_q2_q3` | `20260612T130425_entry_sieve2_top_level_clear100_cores8_20260612_01_failed32_retry02` |
| `sieve2_overtrade_bos_bull_continuation_long_1h_vp_market_guard` | 2 | `3/3` | 5.82 | 66.67 | 63 | 1.86 | 1.87 | `pattern_continuation_1h_2021_22` | `20260529T213457_entry_all` |
| `sieve2_overtrade_demand_zone_breakdown_short_vp_market_guard` | 1 | `3/3` | 11.30 | 53.79 | 647 | 7.92 | 1.12 | `auto_generic_1h_2020_q2_q3` | `20260529T213457_entry_all` |
| `sieve2_overtrade_mtf_h4_prior_high_break_long_1h_retest_vp_market_guard` | 1 | `3/3` | 0.01 | 51.47 | 375 | 5.24 | 1.00 | `auto_generic_1h_2020_q2_q3` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_overtrade_mtf_h4_supply_breakout_long_1h_local_break_vp_market_guard` | 1 | `3/3` | 0.86 | 51.26 | 517 | 7.51 | 1.01 | `auto_generic_1h_2020_q2_q3` | `20260529T213457_entry_all` |
| `sieve2_overtrade_multi2_tlv2_vp_sup_break_vp_node_short_1h_vp_market_guard` | 1 | `3/3` | 4.74 | 52.99 | 368 | 5.05 | 1.09 | `auto_generic_1h_2020_q2_q3` | `20260605T004320_entry_sieve2_nonprofitable_fundamental_rework` |
| `sieve2_overtrade_multi2_tlv2_vp_sup_break_vp_node_short_8h_vp_market_guard` | 2 | `3/3` | 4.80 | 66.67 | 51 | 0.93 | 1.96 | `auto_generic_8h_2020_22_cycle` | `20260602T092945_entry_sieve2_tune_fix_next` |
| `sieve2_overtrade_multi2_tlv2_vp_sup_break_vp_vah_short_8h_vp_market_guard` | 2 | `3/3` | 0.97 | 60.00 | 15 | 0.62 | 1.61 | `auto_generic_8h_2020_22_cycle` | `20260529T213457_entry_all` |
| `sieve2_overtrade_multi2_tlv2_vp_sup_reclaim_vp_node_long_1h_vp_market_guard` | 1 | `3/3` | 11.32 | 63.31 | 139 | 2.59 | 1.73 | `auto_generic_1h_2020_q2_q3` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_overtrade_multi2_vp_prior_day_high_break_vp_bullctx_long_vp_market_guard` | 1 | `3/3` | 7.80 | 52.91 | 669 | 11.71 | 1.07 | `auto_generic_1h_2020_q2_q3` | `20260602T092945_entry_sieve2_tune_fix_next` |
| `sieve2_overtrade_multi2_vp_prior_day_high_break_vp_node_long_vp_market_guard` | 2 | `3/3` | 12.07 | 53.60 | 709 | 8.22 | 1.11 | `auto_generic_1h_2020_q2_q3` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_overtrade_multi2_vp_prior_day_high_break_vp_val_long_vp_market_guard` | 2 | `3/3` | 11.40 | 58.15 | 270 | 5.62 | 1.31 | `auto_generic_1h_2020_q2_q3` | `20260529T213457_entry_all` |
| `sieve2_overtrade_multi2_vp_prior_day_low_break_vp_vah_short_vp_market_guard` | 2 | `3/3` | 17.86 | 53.01 | 1213 | 11.69 | 1.09 | `auto_generic_1h_2020_q2_q3` | `20260529T213457_entry_all` |
| `sieve2_overtrade_pivot_long_resistance_breakout_vp_market_guard` | 1 | `3/3` | 2.01 | 52.04 | 319 | 5.36 | 1.04 | `auto_generic_1h_2020_q2_q3` | `20260605T004320_entry_sieve2_nonprofitable_fundamental_rework` |
| `sieve2_overtrade_supply_zone_breakout_long_vp_market_guard` | 2 | `3/3` | 18.38 | 54.73 | 824 | 5.20 | 1.15 | `auto_generic_1h_2020_q2_q3` | `20260529T213457_entry_all` |
| `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard` | 2 | `3/3` | 15.27 | 55.32 | 564 | 5.03 | 1.19 | `auto_generic_1h_2020_q2_q3` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_overtrade_tlv2_support_breakdown_short_1h_vp_market_guard` | 2 | `3/3` | 14.11 | 54.05 | 753 | 8.27 | 1.12 | `auto_generic_1h_2020_q2_q3` | `20260529T213457_entry_all` |
| `sieve2_overtrade_tlv2_support_breakdown_short_8h_vp_market_guard` | 2 | `3/3` | 11.23 | 63.40 | 153 | 1.26 | 1.64 | `auto_generic_8h_2020_22_cycle` | `20260529T213457_entry_all` |
| `sieve2_overtrade_tlv2_support_reclaim_long_1h_vp_market_guard` | 1 | `3/3` | 0.74 | 51.15 | 2780 | 22.33 | 1.00 | `auto_generic_1h_2020_q2_q3` | `20260605T004320_entry_sieve2_nonprofitable_fundamental_rework` |
| `sieve2_pivot_higher_low_sequence_long_4h` | 1 | `3/3` | -7.27 | 56.49 | 609 | 12.39 | 0.93 | `auto_generic_4h_2020_21_cycle` | `20260612T211448_entry_sieve2_top_level_remaining120_cores8_20260612_failed15_retry01` |
| `sieve2_pivot_lower_high_sequence_short_4h` | 1 | `3/3` | 1.16 | 58.63 | 278 | 6.58 | 1.02 | `auto_generic_4h_2020_21_cycle` | `20260612T211448_entry_sieve2_top_level_remaining120_cores8_20260612_failed15_retry01` |
| `sieve2_pivot_midrange_reclaim_long_1h` | 1 | `3/3` | -51.31 | 55.16 | 3849 | 53.56 | 0.89 | `auto_generic_1h_2020_q2_q3` | `20260612T211448_entry_sieve2_top_level_remaining120_cores8_20260612_failed15_retry01` |
| `sieve2_pivot_midrange_reject_short_1h` | 1 | `3/3` | -10.28 | 57.08 | 2875 | 19.82 | 0.98 | `auto_generic_1h_2020_q2_q3` | `20260612T211448_entry_sieve2_top_level_remaining120_cores8_20260612_failed15_retry01` |
| `sieve2_pivot_prominence_breakdown_short_1h` | 1 | `3/3` | -18.49 | 57.09 | 3757 | 33.29 | 0.97 | `auto_generic_1h_2020_q2_q3` | `20260612T211448_entry_sieve2_top_level_remaining120_cores8_20260612_failed15_retry01` |
| `sieve2_pivot_prominence_breakout_long_1h` | 1 | `3/3` | -11.07 | 57.78 | 5374 | 40.24 | 0.99 | `auto_generic_1h_2020_q2_q3` | `20260612T211448_entry_sieve2_top_level_remaining120_cores8_20260612_failed15_retry01` |
| `sieve2_pivot_reclaim_after_confirmation_long_1h` | 1 | `3/3` | -21.94 | 56.49 | 2319 | 30.08 | 0.94 | `auto_generic_1h_2020_q2_q3` | `20260612T211448_entry_sieve2_top_level_remaining120_cores8_20260612_failed15_retry01` |
| `sieve2_pivot_score_flip_long_1h` | 1 | `3/3` | -60.56 | 56.43 | 8589 | 68.62 | 0.95 | `auto_generic_1h_2020_q2_q3` | `20260612T211448_entry_sieve2_top_level_remaining120_cores8_20260612_failed15_retry01` |
| `sieve2_pivot_score_flip_short_1h` | 1 | `3/3` | -34.07 | 56.74 | 5541 | 47.23 | 0.95 | `auto_generic_1h_2020_q2_q3` | `20260612T211448_entry_sieve2_top_level_remaining120_cores8_20260612_failed15_retry01` |
| `sieve2_prior_month_high_breakout_long` | 17 | `2/2` | 4.05 | 53.59 | 418 | 4.07 | 1.10 | `auto_generic_1h_2020_q2_q3` | `20260521T012740_entry_all_resume` |
| `sieve2_reframed_capitulation_vp_reclaim_long_1h` | 1 | `3/3` | 0.57 | 100.00 | 2 | 0.00 | 0.00 | `pattern_reversal_1h_2024_25` | `20260605T004320_entry_sieve2_nonprofitable_fundamental_rework` |
| `sieve2_reframed_liquidity_equal_highs_reject_short_1h` | 2 | `3/3` | 5.44 | 62.67 | 75 | 2.34 | 1.64 | `auto_generic_1h_2020_q2_q3` | `20260612T143424_entry_sieve2_top_level_remaining120_cores8_20260612` |
| `sieve2_reframed_mtf_4h_flag_long_1h_retest` | 2 | `3/3` | 2.96 | 64.29 | 42 | 1.78 | 1.64 | `pattern_continuation_1h_2021_22` | `20260529T213457_entry_all` |
| `sieve2_reframed_mtf_4h_triangle_long_1h_breakout` | 2 | `3/3` | 4.67 | 67.39 | 46 | 1.22 | 2.04 | `pattern_geometry_1h_2023_24_compression` | `20260529T213457_entry_all` |
| `sieve2_reframed_mtf_4h_triangle_long_1h_retest` | 2 | `3/3` | 2.33 | 72.22 | 18 | 0.60 | 2.54 | `pattern_geometry_1h_2023_24_compression` | `20260529T213457_entry_all` |
| `sieve2_reframed_mtf_4h_wolfe_bear_short_1h_breakdown` | 2 | `3/3` | 1.18 | 83.33 | 6 | 0.31 | 4.79 | `pattern_wolfe_1h_2020_21` | `20260612T143424_entry_sieve2_top_level_remaining120_cores8_20260612` |
| `sieve2_reframed_tlv2_res_break_retest_vp_node_long_1h` | 2 | `3/3` | 12.75 | 55.06 | 474 | 6.76 | 1.19 | `auto_generic_1h_2020_q2_q3` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_reframed_tlv2_res_break_vp_node_long_1h` | 2 | `3/3` | 1.56 | 72.73 | 11 | 0.37 | 2.77 | `auto_generic_1h_2020_q2_q3` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_reframed_tlv2_support_break_retest_vp_node_short_1h` | 1 | `3/3` | 5.77 | 56.21 | 169 | 4.09 | 1.26 | `auto_generic_1h_2020_q2_q3` | `20260603T101611_entry_sieve2_guard_revised_weak_unrun` |
| `sieve2_reframed_tlv2_support_break_vp_node_short_1h` | 1 | `3/3` | 7.53 | 54.31 | 348 | 4.33 | 1.15 | `auto_generic_1h_2020_q2_q3` | `20260605T004320_entry_sieve2_nonprofitable_fundamental_rework` |
| `sieve2_reframed_vp_poc_reclaim_long_1h` | 1 | `3/3` | 0.90 | 71.43 | 7 | 0.61 | 2.46 | `auto_generic_1h_2020_q2_q3` | `20260529T213457_entry_all` |
| `sieve2_reversal_double_top_present_short_1h` | 16 | `2/2` | 3.55 | 53.64 | 453 | 3.86 | 1.08 | `pattern_reversal_1h_2024_25` | `20260521T012740_entry_all_resume` |

## Unmapped Active Sieve3 Files

These files did not expose a source that this lookup could parse from the filename or metadata. Check `SOURCE_STRATEGY` / `ENTRY_TAG` before using them for baseline comparison.

- `sieve3_exit_novel_mtf_d1_h4_dual_break_1h_bos_long.py`
- `sieve3_exit_novel_mtf_d1_support_hold_h4_break_1h_higher_low_break_long.py`
- `sieve3_exit_novel_mtf_h4_compression_break_1h_retest_long.py`
- `sieve3_exit_novel_mtf_h4_prior_high_break_1h_bos_long.py`
- `sieve3_exit_novel_mtf_h4_range_expansion_1h_breakout_long.py`
- `sieve3_exit_novel_mtf_h4focus_compression_break_1h_breakout_long.py`
- `sieve3_exit_novel_mtf_h4focus_failed_high_reject_1h_retest_short.py`
- `sieve3_exit_novel_mtf_h4focus_prior_high_break_1h_retest_long.py`
