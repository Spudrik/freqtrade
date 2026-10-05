from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPO_ROOT = USER_DATA_DIR.parent


@dataclass(frozen=True)
class FeatureProfile:
    profile_id: str
    family: str
    description: str
    strategy: str
    freqaimodel: str
    model_training_parameters: dict[str, Any]
    required_files: tuple[Path, ...]
    control_profile_id: str | None
    event_id: str
    actual_column: str
    prediction_column: str
    objective: str
    pass_rule: str
    default_train_days: int | None = None
    default_backtest_days: int | None = None
    allowed_windows: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_files"] = [str(path) for path in self.required_files]
        return payload


LIGHTGBM_SMALL = {
    "n_estimators": 90,
    "learning_rate": 0.04,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 18,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 1,
    "verbosity": -1,
}
RIDGE_DEFAULT = {}


STRUCTURAL_FEATURES = USER_DATA_DIR / "research_news_data" / "context_features" / "structural_cache" / "btc_structural_features_1h_latest.parquet"
ORDERBOOK_SPOT_FEATURES = USER_DATA_DIR / "orderbook_data" / "historical_bybit" / "features" / "orderbook_trader_state_1h_latest.parquet"
CONTEXT_FEATURES_DB = USER_DATA_DIR / "research_news_data" / "context_features" / "context_features.sqlite"
CONTEXT_FEATURES_PARQUET = USER_DATA_DIR / "research_news_data" / "context_features" / "exports" / "context_features_1h_latest.parquet"
TRADER_CONFLUENCE_FEATURES = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
GENERAL_EXIT_CACHE_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "general_exit_level_cache"
GENERAL_EXIT_FEATURES = (
    GENERAL_EXIT_CACHE_DIR / "btc_general_exit_levels_1h.parquet",
    GENERAL_EXIT_CACHE_DIR / "eth_general_exit_levels_1h.parquet",
    GENERAL_EXIT_CACHE_DIR / "sol_general_exit_levels_1h.parquet",
)
GENERAL_EXIT_REACTION_CACHE_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "general_exit_level_reaction_cache"
)
GENERAL_EXIT_REACTION_FEATURES = (
    GENERAL_EXIT_REACTION_CACHE_DIR / "btc_general_exit_levels_5tf.parquet",
    GENERAL_EXIT_REACTION_CACHE_DIR / "eth_general_exit_levels_5tf.parquet",
    GENERAL_EXIT_REACTION_CACHE_DIR / "sol_general_exit_levels_5tf.parquet",
)

TARGET_SPECS = {
    "future_return_6h": {
        "event_id": "all_rows",
        "actual_column": "future_return_6h",
        "prediction_column": "&-future_return_6h",
        "objective": "Rank ordinary 6h forward return from the selected feature family.",
        "pass_rule": "Positive prediction/actual correlation and stable top-vs-bottom bucket separation across windows.",
    },
    "future_return_24h": {
        "event_id": "all_rows",
        "actual_column": "future_return_24h",
        "prediction_column": "&-future_return_24h",
        "objective": "Rank 24h forward return from the selected feature family.",
        "pass_rule": "Positive prediction/actual correlation and stable top-vs-bottom bucket separation across windows.",
    },
    "future_drawdown_24h": {
        "event_id": "all_rows",
        "actual_column": "future_max_drawdown_24h",
        "prediction_column": "&-future_max_drawdown_24h",
        "objective": "Rank next-24h adverse path risk from the selected feature family.",
        "pass_rule": "Negative drawdown should rank cleanly by prediction bucket with stable correlation across windows.",
    },
    "breakout_success_6h": {
        "event_id": "breakout_acceptance",
        "actual_column": "breakout_success_next_6h",
        "prediction_column": "&-breakout_success_next_6h",
        "objective": "Test whether the feature family identifies successful breakout/acceptance behaviour.",
        "pass_rule": "AUC above 0.55 on event rows and positive lift over the matching control.",
    },
    "breakout_success_setup_6h": {
        "event_id": "breakout_acceptance",
        "actual_column": "breakout_success_from_current_setup_6h",
        "prediction_column": "&-breakout_success_from_current_setup_6h",
        "objective": "Given a current resistance/breakout setup, test whether the feature family identifies successful breakout/acceptance behaviour.",
        "pass_rule": "AUC above 0.55 on current setup rows and positive lift over the matching control.",
    },
    "breakout_failure_6h": {
        "event_id": "vah_rejection",
        "actual_column": "breakout_failure_next_6h",
        "prediction_column": "&-breakout_failure_next_6h",
        "objective": "Test whether the feature family identifies failed breakouts/rejections near resistance.",
        "pass_rule": "AUC above 0.55 on event rows and positive lift over the matching control.",
    },
    "breakout_failure_setup_6h": {
        "event_id": "vah_rejection",
        "actual_column": "breakout_failure_from_current_setup_6h",
        "prediction_column": "&-breakout_failure_from_current_setup_6h",
        "objective": "Given a current resistance/rejection setup, test whether the feature family identifies failed breakouts/rejections.",
        "pass_rule": "AUC above 0.55 on current setup rows and positive lift over the matching control.",
    },
    "breakdown_success_6h": {
        "event_id": "crash_detection",
        "actual_column": "breakdown_success_next_6h",
        "prediction_column": "&-breakdown_success_next_6h",
        "objective": "Test whether the feature family identifies breakdown/crash continuation risk.",
        "pass_rule": "AUC above 0.55 on event rows and positive lift over the matching control.",
    },
    "breakdown_success_setup_6h": {
        "event_id": "crash_detection",
        "actual_column": "breakdown_success_from_current_setup_6h",
        "prediction_column": "&-breakdown_success_from_current_setup_6h",
        "objective": "Given a current support/breakdown setup, test whether the feature family identifies breakdown/crash continuation risk.",
        "pass_rule": "AUC above 0.55 on current setup rows and positive lift over the matching control.",
    },
    "downside_continuation_3h": {
        "event_id": "crash_detection",
        "actual_column": "downside_continuation_next_3h",
        "prediction_column": "&-downside_continuation_next_3h",
        "objective": "After an early downside break is visible, test whether the feature family identifies drops that keep going over the next 3h.",
        "pass_rule": "AUC above 0.55 on early-break/orderbook-present rows and positive lift over price controls.",
    },
    "downside_continuation_6h": {
        "event_id": "crash_detection",
        "actual_column": "downside_continuation_next_6h",
        "prediction_column": "&-downside_continuation_next_6h",
        "objective": "After an early downside break is visible, test whether the feature family identifies drops that keep going over the next 6h.",
        "pass_rule": "AUC above 0.55 on early-break/orderbook-present rows and positive lift over price controls.",
    },
    "downside_exhaustion_6h": {
        "event_id": "crash_detection",
        "actual_column": "downside_exhaustion_next_6h",
        "prediction_column": "&-downside_exhaustion_next_6h",
        "objective": "After an early downside break is visible, test whether the feature family identifies drops that slow, stall, or bounce.",
        "pass_rule": "AUC above 0.55 on early-break/orderbook-present rows and positive lift over price controls.",
    },
    "support_reclaim_6h": {
        "event_id": "crash_detection",
        "actual_column": "support_reclaim_next_6h",
        "prediction_column": "&-support_reclaim_next_6h",
        "objective": "After a support break, test whether the feature family identifies rows where broken support is reclaimed within 6h.",
        "pass_rule": "AUC above 0.55 on support-break rows and positive lift over price controls.",
    },
    "fakeout_24h": {
        "event_id": "breakout_failure",
        "actual_column": "fakeout_next_24h",
        "prediction_column": "&-fakeout_next_24h",
        "objective": "Test whether the feature family identifies fakeout behaviour after breakout or breakdown attempts.",
        "pass_rule": "AUC above 0.55 on event rows and positive lift over the matching control.",
    },
    "fakeout_setup_24h": {
        "event_id": "breakout_failure",
        "actual_column": "fakeout_from_current_setup_24h",
        "prediction_column": "&-fakeout_from_current_setup_24h",
        "objective": "Given a current support/resistance setup, test whether the feature family identifies fakeout behaviour.",
        "pass_rule": "AUC above 0.55 on current setup rows and positive lift over the matching control.",
    },
}

TARGET_SPECS["structure_vah_breakout_failure_setup_6h"] = {
    "event_id": "vah_rejection",
    "actual_column": "breakout_failure_from_current_setup_6h",
    "prediction_column": "&-so_vah_rejection_breakout_failure_from_current_setup_6h",
    "objective": "Given a current VAH/resistance rejection setup, test whether custom structure/orderbook confluence identifies failed breakout behaviour.",
    "pass_rule": "AUC above 0.55 on current setup rows and positive lift over the matching control.",
}

MODEL_SPECS = {
    "tree": ("LightGBMRegressorMultiTarget", LIGHTGBM_SMALL),
    "ridge": ("SKLearnRidgeRegressorMultiTarget", RIDGE_DEFAULT),
}

FAMILY_SPECS = {
    "price": {
        "description": "Price/volume baseline.",
        "strategy": "ContextBybitOrderbookTraderEventPriceOnlyFreqAIResearchStrategy",
        "required_files": (),
        "control_family": None,
    },
    "structure_vp": {
        "description": "Custom indicator structure and volume-profile features.",
        "strategy": "ContextStructureVahRejectionFreqAIResearchStrategy",
        "required_files": (STRUCTURAL_FEATURES,),
        "control_family": "price",
    },
    "orderbook": {
        "description": "Bybit spot trader-state orderbook features plus price baseline.",
        "strategy": "ContextBybitOrderbookTraderStateParquetFreqAIResearchStrategy",
        "required_files": (ORDERBOOK_SPOT_FEATURES,),
        "control_family": "price",
    },
    "structure_vp_orderbook": {
        "description": "Custom structure/VP plus Bybit spot trader-state orderbook features.",
        "strategy": "ContextStructureOrderbookVahRejectionFreqAIResearchStrategy",
        "required_files": (STRUCTURAL_FEATURES, ORDERBOOK_SPOT_FEATURES),
        "control_family": "structure_vp",
    },
    "context": {
        "description": "External context/news/global/macro parquet feature cache.",
        "strategy": "ContextFreqAIResearchContextEventStrategy",
        "required_files": (CONTEXT_FEATURES_PARQUET,),
        "control_family": "price",
    },
    "price_context": {
        "description": "Price/volume baseline plus external context/news/global/macro parquet feature cache.",
        "strategy": "ContextFreqAIResearchPriceContextEventStrategy",
        "required_files": (CONTEXT_FEATURES_PARQUET,),
        "control_family": "price",
    },
    "trader_confluence_delta": {
        "description": "Discovery-derived trader-confluence delta features plus price baseline.",
        "strategy": "ContextTraderConfluenceDeltaFreqAIResearchStrategy",
        "required_files": (TRADER_CONFLUENCE_FEATURES,),
        "control_family": "price",
    },
    "trader_confluence_delta_components": {
        "description": "Discovery-derived trader-confluence delta component evidence only, excluding pre-baked setup/trigger/score flags.",
        "strategy": "ContextTraderConfluenceDeltaComponentsFreqAIResearchStrategy",
        "required_files": (TRADER_CONFLUENCE_FEATURES,),
        "control_family": "price",
    },
    "trader_confluence_epsilon": {
        "description": "Early-downside trader-state features after the first visible break, including support weakness, pressure persistence, and exhaustion/reclaim evidence.",
        "strategy": "ContextTraderConfluenceEpsilonFreqAIResearchStrategy",
        "required_files": (TRADER_CONFLUENCE_FEATURES,),
        "control_family": "price",
    },
    "trader_confluence_epsilon_components": {
        "description": "Early-downside component evidence only, excluding pre-baked setup/trigger/score flags.",
        "strategy": "ContextTraderConfluenceEpsilonComponentsFreqAIResearchStrategy",
        "required_files": (TRADER_CONFLUENCE_FEATURES,),
        "control_family": "price",
    },
}


def default_profiles() -> dict[str, FeatureProfile]:
    profiles = [
        FeatureProfile(
            profile_id="structure_vp_tree",
            family="structure",
            description="Custom indicator structure and volume-profile baseline.",
            strategy="ContextStructureVahRejectionFreqAIResearchStrategy",
            freqaimodel="LightGBMRegressorMultiTarget",
            model_training_parameters=LIGHTGBM_SMALL,
            required_files=(STRUCTURAL_FEATURES,),
            control_profile_id=None,
            event_id="vah_rejection",
            actual_column="breakout_failure_next_6h",
            prediction_column="&-so_vah_rejection_breakout_failure_next_6h",
            objective="Check whether custom structure/VP can rank VAH rejection failure risk.",
            pass_rule="AUC above 0.55 on event rows and stable month-by-month before adding extra sources.",
            allowed_windows=("broad", "jul_aug", "oct", "jan_feb"),
        ),
        FeatureProfile(
            profile_id="structure_vp_orderbook_tree",
            family="structure_orderbook",
            description="Structure/VP plus Bybit trader-state orderbook features.",
            strategy="ContextStructureOrderbookVahRejectionFreqAIResearchStrategy",
            freqaimodel="LightGBMRegressorMultiTarget",
            model_training_parameters=LIGHTGBM_SMALL,
            required_files=(STRUCTURAL_FEATURES, ORDERBOOK_SPOT_FEATURES),
            control_profile_id="structure_vp_tree",
            event_id="vah_rejection",
            actual_column="breakout_failure_next_6h",
            prediction_column="&-so_vah_rejection_breakout_failure_next_6h",
            objective="Test whether orderbook state improves structure/VP VAH rejection failure prediction on orderbook-present rows.",
            pass_rule="Candidate should beat the control on event+orderbook-present rows, with positive lift in most monthly windows.",
            default_train_days=14,
            default_backtest_days=7,
            allowed_windows=("ob_jul_aug", "ob_oct", "ob_jan_feb"),
        ),
        FeatureProfile(
            profile_id="structure_vp_ridge",
            family="structure",
            description="Custom indicator structure and volume-profile linear baseline.",
            strategy="ContextStructureVahRejectionFreqAIResearchStrategy",
            freqaimodel="SKLearnRidgeRegressorMultiTarget",
            model_training_parameters=RIDGE_DEFAULT,
            required_files=(STRUCTURAL_FEATURES,),
            control_profile_id=None,
            event_id="vah_rejection",
            actual_column="breakout_failure_next_6h",
            prediction_column="&-so_vah_rejection_breakout_failure_next_6h",
            objective="Check whether a simple linear model can rank VAH rejection failure risk.",
            pass_rule="AUC above 0.55 on event rows and stable month-by-month before adding extra sources.",
            allowed_windows=("broad", "jul_aug", "oct", "jan_feb"),
        ),
        FeatureProfile(
            profile_id="structure_vp_orderbook_ridge",
            family="structure_orderbook",
            description="Structure/VP plus Bybit trader-state orderbook features with a linear model.",
            strategy="ContextStructureOrderbookVahRejectionFreqAIResearchStrategy",
            freqaimodel="SKLearnRidgeRegressorMultiTarget",
            model_training_parameters=RIDGE_DEFAULT,
            required_files=(STRUCTURAL_FEATURES, ORDERBOOK_SPOT_FEATURES),
            control_profile_id="structure_vp_ridge",
            event_id="vah_rejection",
            actual_column="breakout_failure_next_6h",
            prediction_column="&-so_vah_rejection_breakout_failure_next_6h",
            objective="Test whether orderbook state adds linear lift over structure/VP on orderbook-present rows.",
            pass_rule="Candidate should beat the matching Ridge control on event+orderbook-present rows.",
            default_train_days=14,
            default_backtest_days=7,
            allowed_windows=("ob_jul_aug", "ob_oct", "ob_jan_feb"),
        ),
    ]
    profiles.extend(_structure_vah_setup_profiles())
    profiles.extend(_generated_research_profiles())
    profiles.extend(_general_exit_target_quality_profiles())
    profiles.extend(_general_exit_level_reaction_profiles())
    profiles.extend(_general_exit_readiness_profiles())
    return {profile.profile_id: profile for profile in profiles}


def _general_exit_target_quality_profiles() -> list[FeatureProfile]:
    common = {
        "family": "general_exit_target_quality",
        "required_files": GENERAL_EXIT_FEATURES,
        "event_id": "all_rows",
        "actual_column": "down_vs_up_advantage_24h",
        "prediction_column": "&-down_vs_up_advantage_24h",
        "objective": (
            "Test whether general VP/TLV2 level quality, timeframe alignment, and BTC context "
            "rank whether downside or upside dominates after a possible exit decision."
        ),
        "pass_rule": (
            "Keep a feature block only when chronological out-of-sample lift is directionally "
            "consistent across ETH and SOL, survives level-event and held-out-entry scoring, and "
            "beats its nested control rather than merely producing one higher headline metric."
        ),
        "default_train_days": 365,
        "default_backtest_days": 90,
        "allowed_windows": ("exit_dev", "exit_holdout"),
    }
    specs = (
        (
            "general_exit_price_tree",
            "Price/volume-only nonlinear control.",
            "GeneralExitPriceOnlyFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            None,
        ),
        (
            "general_exit_level_1h_tree",
            "Price plus base-timeframe VP/TLV2 target quality.",
            "GeneralExitLevel1hFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_price_tree",
        ),
        (
            "general_exit_level_mtf_tree",
            "Price plus 1h/4h/1d target levels and alignment.",
            "GeneralExitLevelMtfFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_level_1h_tree",
        ),
        (
            "general_exit_level_mtf_raw_tree",
            "Price plus 1h/4h/1d target levels without cross-timeframe alignment features.",
            "GeneralExitLevelMtfRawFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_level_1h_tree",
        ),
        (
            "general_exit_va_mtf_tree",
            "Price plus multi-timeframe value-area edge levels and alignment.",
            "GeneralExitVaMtfFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_price_tree",
        ),
        (
            "general_exit_poc_mtf_tree",
            "Price plus multi-timeframe prior POC levels, persistence, and alignment.",
            "GeneralExitPocMtfFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_price_tree",
        ),
        (
            "general_exit_hvn_mtf_tree",
            "Price plus multi-timeframe HVN levels, strength, and alignment.",
            "GeneralExitHvnMtfFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_price_tree",
        ),
        (
            "general_exit_tlv2_mtf_tree",
            "Price plus multi-timeframe TLV2 levels, quality, and alignment.",
            "GeneralExitTlv2MtfFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_price_tree",
        ),
        (
            "general_exit_va_tlv2_mtf_tree",
            "Price plus interacting multi-timeframe value-area and TLV2 levels.",
            "GeneralExitVaTlv2MtfFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_va_mtf_tree",
        ),
        (
            "general_exit_level_mtf_btc_price_tree",
            "MTF target levels plus BTC price/volume context.",
            "GeneralExitLevelMtfBtcPriceFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_level_mtf_tree",
        ),
        (
            "general_exit_level_mtf_btc_levels_tree",
            "MTF target levels plus BTC price and analogous BTC target levels.",
            "GeneralExitLevelMtfBtcLevelsFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_level_mtf_btc_price_tree",
        ),
        (
            "general_exit_level_mtf_btc_poc_confluence_tree",
            "MTF target levels plus BTC POC persistence and MTF confluence, without BTC price context.",
            "GeneralExitLevelMtfBtcPocConfluenceFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_level_mtf_tree",
        ),
        (
            "general_exit_level_mtf_btc_hvn_confluence_tree",
            "MTF target levels plus BTC HVN strength and MTF confluence, without BTC price context.",
            "GeneralExitLevelMtfBtcHvnConfluenceFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_level_mtf_tree",
        ),
        (
            "general_exit_level_mtf_btc_poc_hvn_confluence_tree",
            "MTF target levels plus interacting BTC POC/HVN quality and MTF confluence.",
            "GeneralExitLevelMtfBtcPocHvnConfluenceFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_level_mtf_tree",
        ),
        (
            "general_exit_level_mtf_btc_poc_hvn_va_confluence_tree",
            "MTF target levels plus BTC POC/HVN confluence and VA-edge continuation context.",
            "GeneralExitLevelMtfBtcPocHvnVaConfluenceFreqAIResearchStrategy",
            "LightGBMRegressorMultiTarget",
            LIGHTGBM_SMALL,
            "general_exit_level_mtf_btc_poc_hvn_confluence_tree",
        ),
        (
            "general_exit_level_mtf_btc_levels_ridge",
            "Linear control using the complete target/context feature surface.",
            "GeneralExitLevelMtfBtcLevelsFreqAIResearchStrategy",
            "SKLearnRidgeRegressorMultiTarget",
            RIDGE_DEFAULT,
            None,
        ),
    )
    return [
        FeatureProfile(
            profile_id=profile_id,
            description=description,
            strategy=strategy,
            freqaimodel=model,
            model_training_parameters=dict(parameters),
            control_profile_id=control,
            **common,
        )
        for profile_id, description, strategy, model, parameters, control in specs
    ]


def _general_exit_level_reaction_profiles() -> list[FeatureProfile]:
    common = {
        "family": "general_exit_level_reaction",
        "required_files": GENERAL_EXIT_REACTION_FEATURES,
        "event_id": "named_level_reaction",
        "actual_column": "down_vs_up_advantage_4h",
        "prediction_column": "&-down_vs_up_advantage_4h",
        "objective": (
            "Predict the direction, timing, volume, and directional-pressure response over "
            "the next 1-4 candles after explicitly named value-area edge, prior POC, HVN, "
            "and trendline levels or exact level clusters are approached or touched."
        ),
        "pass_rule": (
            "A feature layer is useful only when its incremental lift over the immediately "
            "simpler control repeats in development and holdout, remains visible by pair and "
            "side, and agrees with event-study and placebo evidence. A volume response alone "
            "does not prove reversal, and association is not causation."
        ),
        "default_train_days": 365,
        "default_backtest_days": 90,
        "allowed_windows": ("exit_dev", "exit_holdout"),
    }
    specs = (
        (
            "general_exit_reaction_price_control_tree",
            "Price, volatility, volume, and directional-pressure control with no level data.",
            "GeneralExitReactionPriceControlFreqAIResearchStrategy",
            None,
        ),
        (
            "general_exit_reaction_named_levels_tree",
            (
                "Control plus each value-area edge, prior POC, HVN, and trendline distance and "
                "quality field named separately for 1h, 4h, 8h, 1d, and 3d."
            ),
            "GeneralExitReactionExplicitSingleLevelsFreqAIResearchStrategy",
            "general_exit_reaction_price_control_tree",
        ),
        (
            "general_exit_reaction_same_asset_cluster_compositions_tree",
            (
                "Named levels plus explicit same-timeframe cross-type, same-type "
                "cross-timeframe, and mixed same-asset cluster composition features."
            ),
            "GeneralExitReactionSameAssetClustersFreqAIResearchStrategy",
            "general_exit_reaction_named_levels_tree",
        ),
        (
            "general_exit_reaction_same_asset_candle_state_control_tree",
            (
                "Same-asset cluster composition plus prior cluster state and current closed-"
                "candle volume/pressure, but no exact named-level hit identities."
            ),
            "GeneralExitReactionSameAssetCandleStateControlFreqAIResearchStrategy",
            "general_exit_reaction_same_asset_cluster_compositions_tree",
        ),
        (
            "general_exit_reaction_same_asset_hit_response_tree",
            (
                "Same-asset cluster composition plus exact level identities touched on the "
                "closed candle, hit-candle volume, and directional-pressure response."
            ),
            "GeneralExitReactionSameAssetClustersHitResponseFreqAIResearchStrategy",
            "general_exit_reaction_same_asset_candle_state_control_tree",
        ),
        (
            "general_exit_reaction_target_coin_btc_hit_response_tree",
            (
                "Target-coin hit response plus BTC's separately named level identities, "
                "cluster composition, hit response, and same-side target-coin/BTC alignment."
            ),
            "GeneralExitReactionSameAssetBtcClustersHitResponseFreqAIResearchStrategy",
            "general_exit_reaction_same_asset_hit_response_tree",
        ),
    )
    return [
        FeatureProfile(
            profile_id=profile_id,
            description=description,
            strategy=strategy,
            freqaimodel="LightGBMRegressorMultiTarget",
            model_training_parameters=dict(LIGHTGBM_SMALL),
            control_profile_id=control,
            **common,
        )
        for profile_id, description, strategy, control in specs
    ]


def _general_exit_readiness_profiles() -> list[FeatureProfile]:
    common = {
        "family": "general_exit_readiness",
        "required_files": GENERAL_EXIT_REACTION_FEATURES,
        "event_id": "named_level_exit_readiness",
        "actual_column": "future_reaction_magnitude_4h",
        "prediction_column": "&-future_reaction_magnitude_4h",
        "objective": (
            "Learn a continuous importance score for nearby value-area edges, prior POCs, "
            "HVNs, and trendlines, then test whether closed-candle weakening distinguishes "
            "exit-worthy reversal from continuation over the next 1-4 candles."
        ),
        "pass_rule": (
            "Keep a layer only when its incremental ranking lift over the immediately simpler "
            "control repeats in development and holdout, by pair and side, and is stronger on "
            "contemporaneous level events than stale-level placebos. Level importance alone "
            "must not be interpreted as an exit direction."
        ),
        "default_train_days": 365,
        "default_backtest_days": 90,
        "allowed_windows": ("exit_dev", "exit_holdout"),
    }
    specs = (
        (
            "general_exit_readiness_price_control_tree",
            "Price, volatility, volume, and pressure control without level data.",
            "GeneralExitReadinessPriceControlFreqAIResearchStrategy",
            None,
        ),
        (
            "general_exit_readiness_level_quality_tree",
            (
                "Control plus each exact level type, timeframe, distance, and available "
                "quality metadata."
            ),
            "GeneralExitReadinessLevelQualityFreqAIResearchStrategy",
            "general_exit_readiness_price_control_tree",
        ),
        (
            "general_exit_readiness_proximity_alignment_tree",
            (
                "Exact levels plus ATR-normalized proximity and candidate-centred same-type, "
                "same-timeframe, and mixed multi-timeframe alignment."
            ),
            "GeneralExitReadinessProximityAlignmentFreqAIResearchStrategy",
            "general_exit_readiness_level_quality_tree",
        ),
        (
            "general_exit_readiness_weakening_tree",
            (
                "Level importance plus touch identity, rejection, wick, momentum, volume, "
                "and directional-pressure weakening on closed candles."
            ),
            "GeneralExitReadinessWeakeningFreqAIResearchStrategy",
            "general_exit_readiness_proximity_alignment_tree",
        ),
        (
            "general_exit_readiness_btc_context_tree",
            (
                "Target-coin exit readiness plus BTC level importance, alignment, touch, and "
                "closed-candle weakening context."
            ),
            "GeneralExitReadinessBtcContextFreqAIResearchStrategy",
            "general_exit_readiness_weakening_tree",
        ),
    )
    return [
        FeatureProfile(
            profile_id=profile_id,
            description=description,
            strategy=strategy,
            freqaimodel="LightGBMRegressorMultiTarget",
            model_training_parameters=dict(LIGHTGBM_SMALL),
            control_profile_id=control,
            **common,
        )
        for profile_id, description, strategy, control in specs
    ]


def _structure_vah_setup_profiles() -> list[FeatureProfile]:
    return [
        FeatureProfile(
            profile_id="structure_vp_tree_vah_breakout_failure_setup_6h",
            family="structure",
            description="Custom indicator structure and volume-profile current-setup VAH failure target.",
            strategy="ContextStructureVahRejectionFreqAIResearchStrategy",
            freqaimodel="LightGBMRegressorMultiTarget",
            model_training_parameters=LIGHTGBM_SMALL,
            required_files=(STRUCTURAL_FEATURES,),
            control_profile_id=None,
            event_id="vah_rejection",
            actual_column="breakout_failure_from_current_setup_6h",
            prediction_column="&-so_vah_rejection_breakout_failure_from_current_setup_6h",
            objective="Given a current VAH/resistance rejection setup, test whether custom structure/VP ranks failed breakout risk.",
            pass_rule="AUC above 0.55 on current setup rows and stable month-by-month before adding extra sources.",
            allowed_windows=("broad", "jul_aug", "oct", "jan_feb"),
        ),
        FeatureProfile(
            profile_id="structure_vp_orderbook_tree_vah_breakout_failure_setup_6h",
            family="structure_orderbook",
            description="Structure/VP plus Bybit trader-state orderbook features on current-setup VAH failure target.",
            strategy="ContextStructureOrderbookVahRejectionFreqAIResearchStrategy",
            freqaimodel="LightGBMRegressorMultiTarget",
            model_training_parameters=LIGHTGBM_SMALL,
            required_files=(STRUCTURAL_FEATURES, ORDERBOOK_SPOT_FEATURES),
            control_profile_id="structure_vp_tree_vah_breakout_failure_setup_6h",
            event_id="vah_rejection",
            actual_column="breakout_failure_from_current_setup_6h",
            prediction_column="&-so_vah_rejection_breakout_failure_from_current_setup_6h",
            objective="Given a current VAH/resistance rejection setup, test whether orderbook state improves structure/VP failed breakout prediction.",
            pass_rule="Candidate should beat the control on current setup plus orderbook-present rows in most monthly windows.",
            default_train_days=14,
            default_backtest_days=7,
            allowed_windows=("ob_jul_aug", "ob_oct", "ob_jan_feb"),
        ),
    ]


def _generated_research_profiles() -> list[FeatureProfile]:
    profiles: list[FeatureProfile] = []
    for target_id, target in TARGET_SPECS.items():
        for model_id, (freqaimodel, training_parameters) in MODEL_SPECS.items():
            for family_id, family in FAMILY_SPECS.items():
                control_family = family["control_family"]
                control_profile_id = f"{control_family}_{model_id}_{target_id}" if control_family else None
                profile_id = f"{family_id}_{model_id}_{target_id}"
                profiles.append(
                    FeatureProfile(
                        profile_id=profile_id,
                        family=family_id,
                        description=f"{family['description']} Target: {target_id}.",
                        strategy=str(family["strategy"]),
                        freqaimodel=freqaimodel,
                        model_training_parameters=dict(training_parameters),
                        required_files=tuple(family["required_files"]),
                        control_profile_id=control_profile_id,
                        event_id=str(target["event_id"]),
                        actual_column=str(target["actual_column"]),
                        prediction_column=str(target["prediction_column"]),
                        objective=str(target["objective"]),
                        pass_rule=str(target["pass_rule"]),
                        default_train_days=60,
                        default_backtest_days=14,
                        allowed_windows=(
                            "spot_full",
                            "spot_q4_2025",
                            "spot_q1_2026",
                            "spot_recent",
                            "spot_aug_2025",
                            "spot_sep_2025",
                            "spot_oct_2025",
                            "spot_nov_2025",
                            "spot_dec_2025",
                            "spot_jan_2026",
                            "spot_feb_2026",
                            "spot_mar_2026",
                            "spot_apr_2026",
                        ),
                    )
                )
    return profiles


def available_profiles() -> dict[str, FeatureProfile]:
    profiles = {}
    for profile_id, profile in default_profiles().items():
        if all(path.exists() for path in profile.required_files):
            profiles[profile_id] = profile
    return profiles


def missing_profile_requirements() -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for profile_id, profile in default_profiles().items():
        missing_files = [str(path) for path in profile.required_files if not path.exists()]
        if missing_files:
            missing[profile_id] = missing_files
    return missing
