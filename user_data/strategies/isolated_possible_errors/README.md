# Isolated Possible Errors

These nine Sieve3 V2 target-family strategies are intentionally excluded from
active batches. They are preserved unchanged so their entry logic and original
exit implementation remain available as evidence.

They do not count toward active Sieve3 file coverage, queue coverage, positive-exit coverage, or completion. An isolated file means the user rejected or withheld that implementation. A stale manifest reference to one of these files does not restore it.

## 7 August 2026 Review Outcome

The original quarantine contained 50 files. A source-specific static review
restored 41 files after finding that most recorded concerns were not executable
errors:

1. A higher-timeframe target followed by a base-timeframe reversal check is an
   intentional target/trigger split, not a timeframe mismatch by itself.
2. TLV2 target candidates were already filtered by the strategy's line-score,
   active-bar, and distance settings before rank-zero output was exposed.
3. `complex_volume_profile` already qualifies `hvn_above` as a high-volume node;
   the separate strength column is metadata about an already qualified target.
4. `prior_low_1d_1d` is the expected suffix after the strategy's own
   `prior_low_1d` feature is merged from the 1d informative dataframe.
5. The full-target D1-support/H4-break strategy had one real inconsistency: it
   exposed invalidation Hyperparameters but omitted the matching 1h range-mid
   invalidation used by its partial sibling. That declaration was corrected
   before restoration.

All 41 restored files compile and import independently, and their locked entry
parameters/signatures match their active siblings.

## Remaining Isolation

The remaining nine files come from five sources whose target is only a nearest
generic volume-profile obstacle. That model is superseded by the approved
scored-level families, which assess VA edge, strength-qualified HVN, prior POC,
and confluence rather than accepting the nearest obstacle blindly:

- `mtf_confluence_d1_bos_h4_vp_node_long`
- `mtf_confluence_d1_vp_bos_4h_retest_long`
- `mtfx_d1_tlv2_res_break_long_4h_bos`
- `mtfx_h4_tlv2_sup_break_short_1h_choch`
- `overtrade_bos_bull_continuation_long_1h_vp_market_guard`

Do not restore these nine files mechanically. Replace their generic-nearest
target theory through the current scored-level regeneration contract.
