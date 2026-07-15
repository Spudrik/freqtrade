from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TraderHypothesis:
    hypothesis_id: str
    theory: str
    direction: str
    source_families: tuple[str, ...]
    setup_column: str
    trigger_column: str
    score_column: str
    target_columns: tuple[str, ...]
    controls: tuple[str, ...]
    ablations: tuple[str, ...]
    minimum_rows: int = 100
    status: str = "designed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


COMMON_CONTROLS = (
    "random_eligible_rows",
    "same_regime_without_trigger",
    "opposite_direction_setup",
    "shuffled_labels",
    "price_structure_baseline",
)

COMMON_ABLATIONS = (
    "minus_orderbook",
    "minus_context",
    "minus_structure",
    "minus_volume_pressure",
    "minus_multi_timeframe_confirmation",
    "minus_each_orderbook_venue",
)


HYPOTHESES: tuple[TraderHypothesis, ...] = (
    TraderHypothesis(
        hypothesis_id="bearish_event_breakdown_confluence",
        theory=(
            "Downside continuation is more likely when higher-timeframe resistance, negative "
            "orderbook transition, rising event context, and volume-confirmed structure break align."
        ),
        direction="bearish",
        source_families=("price", "structure", "orderbook", "context"),
        setup_column="conf_bearish_event_breakdown_confluence_setup",
        trigger_column="conf_bearish_event_breakdown_confluence_trigger",
        score_column="conf_bearish_event_breakdown_confluence_score",
        target_columns=("breakdown_success_next_6h", "large_drawdown_next_6h", "large_drawdown_next_24h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="bullish_event_breakout_confluence",
        theory=(
            "Upside continuation is more likely when resistance retreats, context attention rises, "
            "and price breaks resistance with volume and orderbook agreement."
        ),
        direction="bullish",
        source_families=("price", "structure", "orderbook", "context"),
        setup_column="conf_bullish_event_breakout_confluence_setup",
        trigger_column="conf_bullish_event_breakout_confluence_trigger",
        score_column="conf_bullish_event_breakout_confluence_score",
        target_columns=("breakout_success_next_6h", "future_return_6h", "future_max_upside_24h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="quiet_technical_breakout_confluence",
        theory=(
            "When context is quiet and the orderbook is balanced, clean technical resistance breaks "
            "should travel toward the next structure or liquidity target."
        ),
        direction="bullish",
        source_families=("price", "structure", "orderbook", "context"),
        setup_column="conf_quiet_technical_breakout_confluence_setup",
        trigger_column="conf_quiet_technical_breakout_confluence_trigger",
        score_column="conf_quiet_technical_breakout_confluence_score",
        target_columns=("breakout_success_next_6h", "future_return_6h", "future_max_upside_24h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="quiet_technical_breakdown_confluence",
        theory=(
            "When context is quiet, support breaks with weak book defense and volume expansion can "
            "continue cleanly to the next lower technical or liquidity target."
        ),
        direction="bearish",
        source_families=("price", "structure", "orderbook", "context"),
        setup_column="conf_quiet_technical_breakdown_confluence_setup",
        trigger_column="conf_quiet_technical_breakdown_confluence_trigger",
        score_column="conf_quiet_technical_breakdown_confluence_score",
        target_columns=("breakdown_success_next_6h", "future_max_drawdown_6h", "future_max_drawdown_24h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="multi_tf_resistance_rejection",
        theory=(
            "Rejection is stronger when multiple independent resistance definitions align across "
            "timeframes and price fails to accept above them."
        ),
        direction="bearish",
        source_families=("price", "structure", "orderbook"),
        setup_column="conf_multi_tf_resistance_rejection_setup",
        trigger_column="conf_multi_tf_resistance_rejection_trigger",
        score_column="conf_multi_tf_resistance_rejection_score",
        target_columns=("breakout_failure_next_6h", "failed_breakout_next_24h", "future_max_drawdown_6h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="multi_tf_support_bounce",
        theory=(
            "Bounce probability improves when VP, trendline, pattern, and orderbook support align "
            "and price reclaims the support area."
        ),
        direction="bullish",
        source_families=("price", "structure", "orderbook"),
        setup_column="conf_multi_tf_support_bounce_setup",
        trigger_column="conf_multi_tf_support_bounce_trigger",
        score_column="conf_multi_tf_support_bounce_score",
        target_columns=("breakdown_failure_next_6h", "failed_breakdown_next_24h", "future_return_6h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="failed_breakout_exhaustion",
        theory=(
            "Breakout attempts fail when price reaches resistance but volume, orderbook, and context "
            "do not confirm acceptance."
        ),
        direction="bearish",
        source_families=("price", "structure", "orderbook", "context"),
        setup_column="conf_failed_breakout_exhaustion_setup",
        trigger_column="conf_failed_breakout_exhaustion_trigger",
        score_column="conf_failed_breakout_exhaustion_score",
        target_columns=("breakout_failure_next_6h", "failed_breakout_next_24h", "fakeout_next_24h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="failed_breakdown_exhaustion",
        theory=(
            "Breakdown attempts fail when price breaks support but sell pressure fades and support "
            "rebuilds quickly."
        ),
        direction="bullish",
        source_families=("price", "structure", "orderbook"),
        setup_column="conf_failed_breakdown_exhaustion_setup",
        trigger_column="conf_failed_breakdown_exhaustion_trigger",
        score_column="conf_failed_breakdown_exhaustion_score",
        target_columns=("breakdown_failure_next_6h", "failed_breakdown_next_24h", "fakeout_next_24h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="lvn_fast_travel_up",
        theory=(
            "Price can move quickly through low-volume or thin-liquidity areas when upside liquidity "
            "is sparse and volume confirms."
        ),
        direction="bullish",
        source_families=("price", "structure", "orderbook"),
        setup_column="conf_lvn_fast_travel_up_setup",
        trigger_column="conf_lvn_fast_travel_up_trigger",
        score_column="conf_lvn_fast_travel_up_score",
        target_columns=("future_max_upside_6h", "future_return_6h", "time_to_plus_2pct"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="lvn_fast_travel_down",
        theory=(
            "Downside can accelerate through low-volume or thin-liquidity areas when support is sparse "
            "and sell volume confirms."
        ),
        direction="bearish",
        source_families=("price", "structure", "orderbook"),
        setup_column="conf_lvn_fast_travel_down_setup",
        trigger_column="conf_lvn_fast_travel_down_trigger",
        score_column="conf_lvn_fast_travel_down_score",
        target_columns=("future_max_drawdown_6h", "future_return_6h", "time_to_minus_2pct"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="poc_reclaim_continuation",
        theory="Reclaiming POC after trading below value can imply acceptance back into value and continuation toward VAH.",
        direction="bullish",
        source_families=("price", "structure", "orderbook"),
        setup_column="conf_poc_reclaim_continuation_setup",
        trigger_column="conf_poc_reclaim_continuation_trigger",
        score_column="conf_poc_reclaim_continuation_score",
        target_columns=("future_return_6h", "future_max_upside_24h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="designed",
    ),
    TraderHypothesis(
        hypothesis_id="poc_rejection_continuation",
        theory="Rejection from POC after trading below value can confirm bearish acceptance and continuation toward VAL or support.",
        direction="bearish",
        source_families=("price", "structure", "orderbook"),
        setup_column="conf_poc_rejection_continuation_setup",
        trigger_column="conf_poc_rejection_continuation_trigger",
        score_column="conf_poc_rejection_continuation_score",
        target_columns=("future_return_6h", "future_max_drawdown_24h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="designed",
    ),
    TraderHypothesis(
        hypothesis_id="orderbook_resistance_evaporation_breakout",
        theory="Breakout odds improve when ask walls above price evaporate before or during a resistance break.",
        direction="bullish",
        source_families=("price", "structure", "orderbook"),
        setup_column="conf_orderbook_resistance_evaporation_breakout_setup",
        trigger_column="conf_orderbook_resistance_evaporation_breakout_trigger",
        score_column="conf_orderbook_resistance_evaporation_breakout_score",
        target_columns=("breakout_success_next_6h", "future_max_upside_6h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="orderbook_support_removal_breakdown",
        theory="Breakdown odds increase when bid support vanishes and new sell liquidity appears lower.",
        direction="bearish",
        source_families=("price", "structure", "orderbook"),
        setup_column="conf_orderbook_support_removal_breakdown_setup",
        trigger_column="conf_orderbook_support_removal_breakdown_trigger",
        score_column="conf_orderbook_support_removal_breakdown_score",
        target_columns=("breakdown_success_next_6h", "future_max_drawdown_6h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="wall_relocation_trend_confirmation",
        theory="Walls shifting in the direction of price can confirm trend acceptance; walls shifting against price can warn of rejection.",
        direction="bidirectional",
        source_families=("price", "structure", "orderbook"),
        setup_column="conf_wall_relocation_trend_confirmation_setup",
        trigger_column="conf_wall_relocation_trend_confirmation_trigger",
        score_column="conf_wall_relocation_trend_confirmation_score",
        target_columns=("future_return_6h", "future_return_24h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="designed",
    ),
    TraderHypothesis(
        hypothesis_id="orderbook_divergence_reversal",
        theory="If price trends but orderbook pressure diverges and liquidity rebuilds against the move, reversal or failure risk rises.",
        direction="bidirectional",
        source_families=("price", "orderbook", "structure"),
        setup_column="conf_orderbook_divergence_reversal_setup",
        trigger_column="conf_orderbook_divergence_reversal_trigger",
        score_column="conf_orderbook_divergence_reversal_score",
        target_columns=("fakeout_next_24h", "future_return_6h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="designed",
    ),
    TraderHypothesis(
        hypothesis_id="context_pressure_plus_structure_break",
        theory="News and macro context pressure matters most when it coincides with a technical break, not in isolation.",
        direction="bidirectional",
        source_families=("price", "structure", "context"),
        setup_column="conf_context_pressure_plus_structure_break_setup",
        trigger_column="conf_context_pressure_plus_structure_break_trigger",
        score_column="conf_context_pressure_plus_structure_break_score",
        target_columns=("future_return_6h", "breakout_success_next_6h", "breakdown_success_next_6h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="quiet_context_range_reversion",
        theory="Quiet context, balanced book, and price inside established range or value can favour mean reversion over continuation.",
        direction="reversion",
        source_families=("price", "structure", "orderbook", "context"),
        setup_column="conf_quiet_context_range_reversion_setup",
        trigger_column="conf_quiet_context_range_reversion_trigger",
        score_column="conf_quiet_context_range_reversion_score",
        target_columns=("future_return_6h", "fakeout_next_24h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="designed",
    ),
    TraderHypothesis(
        hypothesis_id="macro_context_liquidity_stress_breakdown",
        theory="Macro, geopolitical, or liquidity stress plus thinning orderbook and support break increases crash-risk probability.",
        direction="bearish",
        source_families=("price", "structure", "orderbook", "context"),
        setup_column="conf_macro_context_liquidity_stress_breakdown_setup",
        trigger_column="conf_macro_context_liquidity_stress_breakdown_trigger",
        score_column="conf_macro_context_liquidity_stress_breakdown_score",
        target_columns=("large_drawdown_next_6h", "large_drawdown_next_24h", "hit_minus_3pct_before_plus_2pct"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
    TraderHypothesis(
        hypothesis_id="multi_venue_orderbook_confluence",
        theory="Orderbook signals are more trustworthy when spot, linear, and inverse agree; divergence may identify noise or spoofing.",
        direction="bidirectional",
        source_families=("price", "orderbook", "structure"),
        setup_column="conf_multi_venue_orderbook_confluence_setup",
        trigger_column="conf_multi_venue_orderbook_confluence_trigger",
        score_column="conf_multi_venue_orderbook_confluence_score",
        target_columns=("future_return_6h", "breakout_success_next_6h", "breakdown_success_next_6h"),
        controls=COMMON_CONTROLS,
        ablations=COMMON_ABLATIONS,
        status="initial_implementation",
    ),
)


BETA_HYPOTHESIS_SPECS: tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("beta_structure_resistance_breakout_continuation", "Resistance stack plus volume-confirmed break should favour upside continuation.", "bullish", ("price", "structure"), ("breakout_success_next_6h", "future_max_upside_24h")),
    ("beta_structure_support_breakdown_continuation", "Support stack breaking with sell volume should favour downside continuation.", "bearish", ("price", "structure"), ("breakdown_success_next_6h", "large_drawdown_next_6h")),
    ("beta_structure_resistance_rejection_failure", "Price near stacked resistance with bearish volume and no acceptance should favour breakout failure.", "bearish", ("price", "structure"), ("breakout_failure_next_6h", "failed_breakout_next_24h")),
    ("beta_structure_support_bounce_failure", "Price near stacked support with bullish volume and no clean breakdown should favour breakdown failure or bounce.", "bullish", ("price", "structure"), ("breakdown_failure_next_6h", "failed_breakdown_next_24h")),
    ("beta_structure_lvn_up_continuation", "Thin volume above price plus breakout pressure should favour fast upside travel.", "bullish", ("price", "structure"), ("future_max_upside_6h", "hit_plus_3pct_before_minus_2pct")),
    ("beta_structure_lvn_down_continuation", "Thin volume below price plus breakdown pressure should favour fast downside travel.", "bearish", ("price", "structure"), ("future_max_drawdown_6h", "hit_minus_3pct_before_plus_2pct")),
    ("beta_structure_range_high_fakeout", "Range-high breaks without volume confirmation should favour fakeout or failed breakout.", "bearish", ("price", "structure"), ("breakout_failure_next_6h", "fakeout_next_24h")),
    ("beta_structure_range_low_fakeout", "Range-low breaks without volume confirmation should favour fakeout or failed breakdown.", "bullish", ("price", "structure"), ("breakdown_failure_next_6h", "fakeout_next_24h")),
    ("beta_ob_resistance_removed_breakout", "Resistance removal plus bullish orderbook pressure during a break should favour upside acceptance.", "bullish", ("price", "structure", "orderbook"), ("breakout_success_next_6h", "future_max_upside_6h")),
    ("beta_ob_support_removed_breakdown", "Support removal plus bearish orderbook pressure during a break should favour downside acceptance.", "bearish", ("price", "structure", "orderbook"), ("breakdown_success_next_6h", "large_drawdown_next_6h")),
    ("beta_ob_ask_absorption_rejection", "Ask absorption near resistance should favour stalled upside or breakout failure.", "bearish", ("price", "structure", "orderbook"), ("breakout_failure_next_6h", "failed_breakout_next_24h")),
    ("beta_ob_bid_absorption_bounce", "Bid absorption near support should favour bounce or failed breakdown.", "bullish", ("price", "structure", "orderbook"), ("breakdown_failure_next_6h", "failed_breakdown_next_24h")),
    ("beta_ob_upside_vacuum_breakout", "Upside liquidity vacuum plus breakout pressure should favour fast upside path.", "bullish", ("price", "structure", "orderbook"), ("breakout_success_next_6h", "future_max_upside_6h")),
    ("beta_ob_downside_vacuum_breakdown", "Downside liquidity vacuum plus breakdown pressure should favour fast downside path.", "bearish", ("price", "structure", "orderbook"), ("breakdown_success_next_6h", "large_drawdown_next_6h")),
    ("beta_ob_pressure_disagreement_fakeout", "Orderbook venue disagreement around key levels should increase fakeout risk.", "reversion", ("price", "structure", "orderbook"), ("fakeout_next_24h", "breakout_failure_next_6h", "breakdown_failure_next_6h")),
    ("beta_ob_multi_venue_pressure_continuation", "Multi-venue orderbook pressure agreement should improve continuation odds when price breaks structure.", "bidirectional", ("price", "structure", "orderbook"), ("breakout_success_next_6h", "breakdown_success_next_6h")),
    ("beta_context_risk_support_breakdown", "Macro or risk-event pressure at support should matter most when support breaks with sell volume.", "bearish", ("price", "structure", "context"), ("breakdown_success_next_6h", "large_drawdown_next_6h", "large_drawdown_next_24h")),
    ("beta_context_risk_resistance_rejection", "Macro or risk-event pressure at resistance should favour rejection more than breakout.", "bearish", ("price", "structure", "context"), ("breakout_failure_next_6h", "failed_breakout_next_24h")),
    ("beta_context_attention_breakout", "Rising context attention near resistance can matter if price confirms with breakout volume.", "bullish", ("price", "structure", "context"), ("breakout_success_next_6h", "future_max_upside_24h")),
    ("beta_context_attention_breakdown", "Rising context attention near support can matter if price confirms with breakdown volume.", "bearish", ("price", "structure", "context"), ("breakdown_success_next_6h", "large_drawdown_next_24h")),
    ("beta_context_persistent_topic_trend", "Persistent source/topic attention should matter more when structure breaks in either direction.", "bidirectional", ("price", "structure", "context"), ("breakout_success_next_6h", "breakdown_success_next_6h")),
    ("beta_quiet_context_structure_breakout", "Quiet context should let technical resistance breakout behaviour dominate when volume confirms.", "bullish", ("price", "structure", "context"), ("breakout_success_next_6h", "future_return_6h")),
    ("beta_quiet_context_structure_breakdown", "Quiet context should let technical support breakdown behaviour dominate when volume confirms.", "bearish", ("price", "structure", "context"), ("breakdown_success_next_6h", "future_max_drawdown_6h")),
    ("beta_macro_ob_support_evaporation_crash", "Macro risk plus evaporating bid support and bearish book pressure should flag crash risk.", "bearish", ("price", "structure", "orderbook", "context"), ("large_drawdown_next_6h", "large_drawdown_next_24h")),
    ("beta_macro_ob_resistance_evaporation_squeeze", "High-severity context plus removed ask resistance and bullish book pressure should flag upside squeeze risk.", "bullish", ("price", "structure", "orderbook", "context"), ("breakout_success_next_6h", "hit_plus_3pct_before_minus_2pct")),
    ("beta_macro_ob_pressure_flip_down", "Macro risk near a key level plus bearish orderbook flip should flag downside continuation risk.", "bearish", ("price", "structure", "orderbook", "context"), ("breakdown_success_next_6h", "large_drawdown_next_6h")),
    ("beta_macro_ob_pressure_flip_up", "High-severity context near a key level plus bullish orderbook flip should flag upside continuation risk.", "bullish", ("price", "structure", "orderbook", "context"), ("breakout_success_next_6h", "future_max_upside_24h")),
    ("beta_event_volume_expansion_down", "Event attention plus bearish volume expansion should matter most when breakdown appears.", "bearish", ("price", "structure", "orderbook", "context"), ("breakdown_success_next_6h", "large_drawdown_next_6h")),
    ("beta_event_volume_expansion_up", "Event attention plus bullish volume expansion should matter most when breakout appears.", "bullish", ("price", "structure", "orderbook", "context"), ("breakout_success_next_6h", "future_max_upside_24h")),
    ("beta_failed_breakdown_reclaim_context", "A failed breakdown under risk context may mark forced selling exhaustion and reclaim/bounce risk.", "bullish", ("price", "structure", "orderbook", "context"), ("breakdown_failure_next_6h", "failed_breakdown_next_24h")),
    ("beta_failed_breakout_reject_context", "A failed breakout under severe context may mark rejection and fakeout risk.", "bearish", ("price", "structure", "orderbook", "context"), ("breakout_failure_next_6h", "failed_breakout_next_24h")),
)


GAMMA_BULL_HYPOTHESIS_SPECS: tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("gamma_bull_structure_breakout_regime", "Bullish market regime plus resistance break and volume impulse should favour upside continuation.", "bullish", ("price", "structure"), ("breakout_success_next_6h", "future_return_6h", "future_max_upside_24h", "hit_plus_3pct_before_minus_2pct")),
    ("gamma_bull_mtf_structure_alignment", "Multi-timeframe bullish structure alignment should improve breakout follow-through when the current break confirms.", "bullish", ("price", "structure"), ("breakout_success_next_6h", "future_max_upside_6h", "future_max_upside_24h")),
    ("gamma_bull_vah_acceptance", "Acceptance above range/value-area high in a bullish regime should favour continued travel rather than rejection.", "bullish", ("price", "structure"), ("breakout_success_next_6h", "future_return_6h", "future_max_upside_24h")),
    ("gamma_bull_lvn_fast_travel", "A bullish breakout into thin volume above price should favour fast upside travel.", "bullish", ("price", "structure"), ("future_max_upside_6h", "future_max_upside_24h", "hit_plus_3pct_before_minus_2pct")),
    ("gamma_bull_compression_breakout", "Breakout from compressed price action with bullish volume should favour range expansion upward.", "bullish", ("price", "structure"), ("breakout_success_next_6h", "future_return_6h", "future_max_upside_24h")),
    ("gamma_bull_pullback_higher_low_resume", "In a bullish regime, support hold or higher-low pullback followed by fresh breakout should favour trend resumption.", "bullish", ("price", "structure"), ("future_return_6h", "future_max_upside_24h", "hit_plus_3pct_before_minus_2pct")),
    ("gamma_bull_tlv2_resistance_break", "Breaking a meaningful TLV2 resistance zone with bullish structure and volume should favour upside acceptance.", "bullish", ("price", "structure"), ("breakout_success_next_6h", "future_return_6h", "future_max_upside_24h")),
    ("gamma_bull_range_expansion_volume", "Range-high breakout after compression plus volume expansion should favour upward range expansion.", "bullish", ("price", "structure"), ("breakout_success_next_6h", "future_max_upside_6h", "future_max_upside_24h")),
    ("gamma_bull_structure_only_no_context", "Structure-only bullish breakout should be treated as valid even without news or orderbook confluence.", "bullish", ("price", "structure"), ("breakout_success_next_6h", "future_return_6h", "future_max_upside_24h")),
    ("gamma_bull_quiet_technical_breakout", "Quiet context and balanced orderbook can let clean technical bullish breaks dominate.", "bullish", ("price", "structure", "orderbook", "context"), ("breakout_success_next_6h", "future_return_6h", "future_max_upside_24h")),
    ("gamma_bull_ob_resistance_removed", "Bullish regime plus ask resistance removal during a break should favour upside acceptance.", "bullish", ("price", "structure", "orderbook"), ("breakout_success_next_6h", "future_max_upside_6h", "future_max_upside_24h")),
    ("gamma_bull_ob_upside_vacuum", "Upside liquidity vacuum during bullish breakout should favour fast upside travel.", "bullish", ("price", "structure", "orderbook"), ("future_max_upside_6h", "future_max_upside_24h", "hit_plus_3pct_before_minus_2pct")),
    ("gamma_bull_ob_pressure_confirmation", "Bullish orderbook pressure should improve follow-through odds when structure is already breaking upward.", "bullish", ("price", "structure", "orderbook"), ("breakout_success_next_6h", "future_return_6h", "future_max_upside_24h")),
    ("gamma_bull_reclaim_after_support_defense", "Support defense in a bullish regime followed by volume recovery should favour bounce/resumption.", "bullish", ("price", "structure", "orderbook"), ("future_return_6h", "future_max_upside_24h", "hit_plus_3pct_before_minus_2pct")),
    ("gamma_bull_context_attention_breakout", "Rising context attention should help bullish breakouts only when price and volume already confirm.", "bullish", ("price", "structure", "context"), ("breakout_success_next_6h", "future_max_upside_24h", "hit_plus_3pct_before_minus_2pct")),
    ("gamma_bull_event_squeeze_breakout", "Event attention plus resistance removal and bullish book pressure should flag upside squeeze risk.", "bullish", ("price", "structure", "orderbook", "context"), ("breakout_success_next_6h", "future_max_upside_24h", "hit_plus_3pct_before_minus_2pct")),
)


DELTA_DISCOVERY_HYPOTHESIS_SPECS: tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "delta_range_high_breakout_success",
        "Discovery screen showed recent range-high position is a stable breakout-success clue; test whether range-high location plus bullish volume confirms upward follow-through.",
        "bullish",
        ("price", "structure"),
        ("breakout_success_next_6h", "future_max_upside_6h", "future_max_upside_24h"),
    ),
    (
        "delta_range_low_breakdown_success",
        "Mirror the discovered range-position behaviour on the downside: recent range-low location plus bearish volume should favour breakdown follow-through.",
        "bearish",
        ("price", "structure"),
        ("breakdown_success_next_6h", "large_drawdown_next_6h", "future_max_drawdown_24h"),
    ),
    (
        "delta_expansion_drawdown_risk",
        "Discovery showed low compression/high expansion was associated with drawdown risk; test whether expansion plus bearish pressure flags risk rather than ordinary chop.",
        "bearish",
        ("price", "structure", "orderbook"),
        ("large_drawdown_next_6h", "large_drawdown_next_24h", "hit_minus_3pct_before_plus_2pct"),
    ),
    (
        "delta_compression_breakout_release",
        "After compression, a range-high break with bullish volume should release upward if the trader compression-breakout story is valid.",
        "bullish",
        ("price", "structure"),
        ("breakout_success_next_6h", "future_max_upside_6h", "hit_plus_3pct_before_minus_2pct"),
    ),
    (
        "delta_compression_breakdown_release",
        "After compression, a range-low break with bearish volume should release downward if the trader breakdown story is valid.",
        "bearish",
        ("price", "structure"),
        ("breakdown_success_next_6h", "large_drawdown_next_6h", "hit_minus_3pct_before_plus_2pct"),
    ),
    (
        "delta_orderbook_support_weak_drawdown",
        "Discovery found Bybit persistence/distance clues; test whether weak persistent support, stretched walls, and bearish book pressure increase drawdown risk.",
        "bearish",
        ("price", "orderbook"),
        ("large_drawdown_next_6h", "large_drawdown_next_24h", "future_max_drawdown_6h"),
    ),
    (
        "delta_orderbook_resistance_weak_breakout",
        "Mirror the orderbook weakness idea on the upside: weak persistent resistance, stretched walls, and bullish pressure should favour upside travel.",
        "bullish",
        ("price", "orderbook"),
        ("breakout_success_next_6h", "future_max_upside_6h", "future_max_upside_24h"),
    ),
    (
        "delta_structure_orderbook_drawdown_confluence",
        "Test whether discovered price range-low weakness becomes more useful when orderbook support is also weak or thin.",
        "bearish",
        ("price", "structure", "orderbook"),
        ("breakdown_success_next_6h", "large_drawdown_next_6h", "large_drawdown_next_24h"),
    ),
    (
        "delta_structure_orderbook_breakout_confluence",
        "Test whether discovered range-high breakout behaviour becomes more useful when orderbook resistance is weak or thin.",
        "bullish",
        ("price", "structure", "orderbook"),
        ("breakout_success_next_6h", "future_max_upside_6h", "future_max_upside_24h"),
    ),
)


EPSILON_DOWNSIDE_AFTER_BREAK_SPECS: tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "epsilon_downside_first_break_continuation",
        "After the first downside break is already visible, falling price should keep going when bearish volume stays elevated, support weakens, and the orderbook shifts into a panic/fragile state.",
        "bearish",
        ("price", "structure", "orderbook"),
        ("downside_continuation_next_3h", "downside_continuation_next_6h", "large_drawdown_next_6h", "hit_minus_3pct_before_plus_2pct"),
    ),
    (
        "epsilon_downside_first_break_exhaustion",
        "After the first downside break is visible, the drop should slow or fail when sell pressure fades and support or bid liquidity rebuilds.",
        "bullish",
        ("price", "structure", "orderbook"),
        ("downside_exhaustion_next_6h", "breakdown_failure_next_6h", "support_reclaim_next_6h", "future_return_6h"),
    ),
    (
        "epsilon_support_reclaim_after_break",
        "A downside break becomes less dangerous when broken support is reclaimed with support rebuild and bearish volume fading.",
        "bullish",
        ("price", "structure", "orderbook"),
        ("support_reclaim_next_6h", "breakdown_failure_next_6h", "future_return_6h"),
    ),
    (
        "epsilon_orderbook_panic_after_break",
        "An early downside break becomes more dangerous when support has been removed and the book shows bearish pressure plus downside liquidity vacuum.",
        "bearish",
        ("price", "structure", "orderbook"),
        ("downside_continuation_next_3h", "downside_continuation_next_6h", "large_drawdown_next_6h"),
    ),
)


HYPOTHESES = (
    *HYPOTHESES,
    *(
        TraderHypothesis(
            hypothesis_id=hypothesis_id,
            theory=theory,
            direction=direction,
            source_families=source_families,
            setup_column=f"conf_{hypothesis_id}_setup",
            trigger_column=f"conf_{hypothesis_id}_trigger",
            score_column=f"conf_{hypothesis_id}_score",
            target_columns=target_columns,
            controls=COMMON_CONTROLS,
            ablations=COMMON_ABLATIONS,
            status="initial_implementation",
        )
        for hypothesis_id, theory, direction, source_families, target_columns in BETA_HYPOTHESIS_SPECS
    ),
    *(
        TraderHypothesis(
            hypothesis_id=hypothesis_id,
            theory=theory,
            direction=direction,
            source_families=source_families,
            setup_column=f"conf_{hypothesis_id}_setup",
            trigger_column=f"conf_{hypothesis_id}_trigger",
            score_column=f"conf_{hypothesis_id}_score",
            target_columns=target_columns,
            controls=COMMON_CONTROLS,
            ablations=COMMON_ABLATIONS,
            status="initial_implementation",
        )
        for hypothesis_id, theory, direction, source_families, target_columns in GAMMA_BULL_HYPOTHESIS_SPECS
    ),
    *(
        TraderHypothesis(
            hypothesis_id=hypothesis_id,
            theory=theory,
            direction=direction,
            source_families=source_families,
            setup_column=f"conf_{hypothesis_id}_setup",
            trigger_column=f"conf_{hypothesis_id}_trigger",
            score_column=f"conf_{hypothesis_id}_score",
            target_columns=target_columns,
            controls=COMMON_CONTROLS,
            ablations=COMMON_ABLATIONS,
            status="initial_implementation",
        )
        for hypothesis_id, theory, direction, source_families, target_columns in DELTA_DISCOVERY_HYPOTHESIS_SPECS
    ),
    *(
        TraderHypothesis(
            hypothesis_id=hypothesis_id,
            theory=theory,
            direction=direction,
            source_families=source_families,
            setup_column=f"conf_{hypothesis_id}_setup",
            trigger_column=f"conf_{hypothesis_id}_trigger",
            score_column=f"conf_{hypothesis_id}_score",
            target_columns=target_columns,
            controls=COMMON_CONTROLS,
            ablations=COMMON_ABLATIONS,
            status="initial_implementation",
        )
        for hypothesis_id, theory, direction, source_families, target_columns in EPSILON_DOWNSIDE_AFTER_BREAK_SPECS
    ),
)


def hypothesis_by_id() -> dict[str, TraderHypothesis]:
    return {hypothesis.hypothesis_id: hypothesis for hypothesis in HYPOTHESES}


def implemented_hypotheses() -> tuple[TraderHypothesis, ...]:
    return tuple(h for h in HYPOTHESES if h.status == "initial_implementation")
