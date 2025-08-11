# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import logging
import time
from freqtrade.user_data.strategies.hyperparams_mixin import HyperoptParamsMixin
import numpy as np  # noqa
import pandas as pd  # noqa
import os

from pandas import DataFrame
from datetime import datetime, timedelta, timezone
from functools import reduce
from pathlib import Path
from ruamel.yaml import YAML

from typing import Any, Dict, List, Optional, Tuple, Union
import re
from technical import qtpylib

from Backups.Nov2024.freqtrade_back_up.freqai.freqai_interface import IFreqaiModel
from freqtrade.persistence import Trade, Order, PairLocks
from freqtrade.strategy import (
    IStrategy,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)  # noqaimport os


from freqtrade.user_data.strategies.helper_functions import (
    define_info_intervals,
    save_dca_trade_data,
)
from freqtrade.user_data.strategies.apply_indicators import apply_indicators
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

# Turns off the fragmented dataframe warning.
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)

# Define colours for print statements
# Text colors
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Background colors
BG_BLACK = "\033[40m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_WHITE = "\033[47m"

# Reset color
RESET = "\033[0m"


class MyStrategy(HyperoptParamsMixin, IStrategy):
    def __init__(self, config):
        super().__init__(config)
        # Only store basic data types on self to keep the strategy picklable
        self._params_loaded = False

    def _load_parameters_from_yaml(self) -> None:
        """Load parameter overrides from the bundled yaml file lazily."""
        if self._params_loaded:
            return
        params_path = Path(__file__).with_name("hyperopt_tracking_results.yaml")
        if params_path.is_file():
            yaml_loader = YAML(typ="safe")
            with params_path.open() as file:
                data = yaml_loader.load(file) or {}
            if isinstance(data, dict):
                items = data.items()
            elif isinstance(data, list):
                items = (
                    (entry.get("parameter"), entry.get("value"))
                    for entry in data
                    if isinstance(entry, dict)
                )
            else:
                items = []
            # Assign only basic data types to self
            for key, value in items:
                if key and isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    setattr(self, key, value)
        self._params_loaded = True

    def _create_freqai_model(self) -> IFreqaiModel:
        """Create a new instance of IFreqaiModel on demand."""
        return IFreqaiModel()

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    class HyperOpt:
        # Define a custom stoploss space.
        # def stoploss_space() -> List[SKDecimal]:
        #     return [SKDecimal(-0.9, -0.5, decimals=1, name='stoploss')]

        def generate_estimator(dimensions: List["Dimension"], **kwargs):
            from skopt.learning import ExtraTreesRegressor

            # Corresponds to "ET" - but allows additional parameters.
            return ExtraTreesRegressor(
                n_estimators=200,  # Increased number of trees to capture more patterns and interactions
                # criterion='squared_error',      # Using default criterion for regression
                # # max_depth=None,               # Default Value
                # max_depth=10,                   # Limit tree depth to prevent overfitting while capturing complex patterns
                # min_samples_split=4,          # Default Value
                # # min_samples_split=10,           # Require more samples to split a node, reducing overfitting from small sample splits
                # # min_samples_leaf=1,           # Default Value
                # min_samples_leaf=7,             # Ensure leaves have enough samples for reliable predictions, reducing overfitting
                # min_weight_fraction_leaf=0.0,   # No change needed; default setting suitable for non-weighted samples
                # # max_features=None,            # Default Value
                # max_features='sqrt',            # Consider a subset of features at each split to introduce randomness and handle high-dimensional data
                # max_leaf_nodes=None,            # Allow unlimited leaf nodes to capture all necessary patterns; default setting
                # min_impurity_decrease=0.0,      # Default setting; splits allowed if they decrease impurity
                # bootstrap=False,              # Default Value
                # # bootstrap=True,                 # Use bootstrapped samples to increase diversity among trees and improve generalization
                # oob_score=False,                # Out-of-bag score not needed; keep default of False
                # random_state=None,            # Default Value
                # # random_state=42,                # Set seed for reproducibility of results
                # verbose=0,                      # Set to 0 for no additional logging output
                # warm_start=False,               # Do not reuse previous fits; start fresh each time
                # min_variance=0.0,                # Default setting; splits allowed regardless of variance reduction
            )

        """
        n_estimators: This decides how many individual "decision-maker" trees we want in our model. Think of each
         tree as a separate person giving an opinion. More trees (higher n_estimators) mean the model can combine more
         opinions, making its overall prediction stronger and more accurate, as it smooths out mistakes any single
         tree might make. However, having more trees also means the model takes longer to train because it has to
         ask more "people" before making a decision. This differs from the n_initial_points in freqtrade.hyperopt file.
         n_initial_points controls the number of random params before we start the optimizer.
         so even with 1000 estimators, and 50 epochs we would still get consensus about those 50 tests results from 1000 different perspectives
         """

    ########################## Track Open Trades Manually ###########################

    trade_data = pd.DataFrame()
    trade_counter = 0
    ################################ timeframe settings ##################################

    # Defines base timeframe of the code
    timeframe = "30m"
    # Due to the ability to change self.timeframe through various methods which affects the populate
    # indicator section. We use self.candle to determine what the current candle actually is. This allows
    # for quicker modifications to the base tf when switching between plotting, backtesting, debugging etc.
    candle = timeframe

    coin_pair_intervals = [
        "1w",
        "3d",
        "1d",
        # '12h',
        # '8h',
        "4h",
        # '2h',
        "1h",
        "30m",
        # '15m',
        # '5m',
    ]
    compatible_intervals = define_info_intervals(timeframe, coin_pair_intervals)
    coin_pairs_used_with_tf = []

    # Can this strategy go short?
    can_short: bool = True

    stoploss = -0.9

    # Enable position adjustment for Grid Bots and DCA
    position_adjustment_enable = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {"0": 4}
    # Stoploss is multiplied by leverage. So lev 20x, with SL @ 0.5 would stop at -2.5% from entry price.
    # SL does not update for DCA safety orders. It's absolute from the entry.
    use_custom_stoploss = True

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.1
    trailing_stop_positive_offset = 0.1  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair, so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, "1h") for pair in pairs]
        informative_pairs += ((pair, "4h") for pair in pairs)
        informative_pairs += ((pair, "1d") for pair in pairs)
        informative_pairs += ((pair, "3d") for pair in pairs)
        # informative_pairs += ((pair, '1w') for pair in pairs)

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Lazily load heavy resources required by this method
        self._load_parameters_from_yaml()
        # Call apply_indicators to get indicators on the target timeframe/pair
        # filter out symbols Like ' and [
        input_string = f"{[metadata['pair']]}"
        # Check if target is BTC to help apply indicators decide whether or not to append a timeframe/coinpair data to end of df column names
        if metadata["pair"] == "BTC/USDT:USDT":
            target_btc = True
        else:
            target_btc = False
        cleaned_string = input_string.replace("[", "").replace("]", "").replace("'", "")
        # Remove the /usdt:usdt part
        cleaned_string = re.sub(r"/usdt:usdt", "", cleaned_string, flags=re.IGNORECASE)
        default_tf = f"{cleaned_string}_{self.timeframe}"

        apply_indicators(self, dataframe, pair_tf=default_tf, target_btc=target_btc, full_set=1)

        ############### INFORMATIVE PAIRS ###########################
        # Ensure DataProvider is available
        if not self.dp:
            return dataframe

        # If the coin being checked is BTC then we need to avoid appending BTC twice.
        if metadata["pair"] == "BTC/USDT:USDT":
            coin_pairs = [
                metadata["pair"],
            ]
        else:
            coin_pairs = [metadata["pair"], "BTC/USDT:USDT"]

        # Define the list of informative timeframes you are interested in
        informative_timeframes = ["3d", "1d", "4h", "1h", "30m"]

        # Re-assign dataframe to base_dataframe to avoid informative and dataframe overlap errors when pasting functions.
        base_dataframe = dataframe.copy()

        for inf_tf in informative_timeframes:
            for pair in coin_pairs:
                # Fetch informative dataframe for BTC/USDT for each informative timeframe

                dataframe = self.dp.get_pair_dataframe(pair=pair, timeframe=inf_tf)
                # skip btc if the inf_tf matches the strategy timeframe, otherwise we end up with auto named columns with x/y at the end to differentiate them.
                if metadata["pair"] == "BTC/USDT:USDT" and inf_tf == self.timeframe:
                    continue
                # Check if the informative DataFrame is emptY
                if dataframe.empty:
                    print(
                        f"Warning: No data available for pair {pair} in timeframe {inf_tf}. Exiting loop."
                    )
                    break  # This will only break out of the inner loop (btc_and_target_pair loop)
                else:
                    ##################### END OF MY INFORMATIVE INDICATORS ##############################
                    pair_tf = f"{pair}_{inf_tf}"
                    # filter out symbols Like ' and [
                    input_string = pair_tf
                    cleaned_string = input_string.replace("[", "").replace("]", "").replace("'", "")
                    # Remove the /usdt:usdt part
                    cleaned_string = re.sub(r"/usdt:usdt", "", cleaned_string, flags=re.IGNORECASE)

                    apply_indicators(
                        self, dataframe, pair_tf=cleaned_string, target_btc=False, full_set=1
                    )

                    ##################### END OF MY INFORMATIVE INDICATORS ##############################

                    # Use the helper function to merge the informative dataframe with the main dataframe
                    # The function is expected to rename columns to prevent conflicts and forward fill data
                    # Note use append tf false and use custom suffix to resolve column naming issues from old system
                    # Remove "/USDT:USDT" from the pair string
                    cleaned_pair = pair.replace("/USDT:USDT", "")

                    base_dataframe = merge_informative_pair(
                        base_dataframe,
                        dataframe,
                        self.timeframe,
                        inf_tf,
                        append_timeframe=False,
                        suffix=f"{cleaned_pair}_{inf_tf}",
                        ffill=True,
                    )

        return base_dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        pair = pair.replace("/USDT:USDT", "")
        coin1w = f"{pair}_1w"
        coin3d = f"{pair}_3d"
        coin1d = f"{pair}_1d"
        coin4h = f"{pair}_4h"
        coin1h = f"{pair}_1h"
        coin30m = f"{pair}_30m"

        btc1w = f"BTC_1w"
        btc3d = f"BTC_3d"
        btc1d = f"BTC_1d"
        btc4h = f"BTC_4h"
        btc1h = f"BTC_1h"
        btc30m = f"BTC_30m"

        # Initialize conditions
        enter_long_conditions = []
        enter_short_conditions = []
        enter_long_ranging_conditions = []
        enter_short_ranging_conditions = []

        # Ranging conditions loop
        indicators = ["rsi_14", "mfi", "uo"]
        timeframes = {"30m": coin30m, "1h": coin1h, "4h": coin4h, "1d": coin1d, "3d": coin3d}

        # ----------------------------
        # Handle Enter Long Ranging
        # ----------------------------
        if self.enter_long_ranging.value:
            # rsi_14 Indicator - 30m Timeframe - Coin
            if self.enable_long_ranging_coin_rsi_14_30m.value:
                coin_rsi_low_30m = self.coin_rsi_14_low_30m_long_ranging.value * 5
                condition = dataframe["rsi_14_coin30m"] <= coin_rsi_low_30m
                enter_long_ranging_conditions.append(condition)

            # rsi_14 Indicator - 30m Timeframe - BTC
            if self.enable_long_ranging_btc_rsi_14_30m.value:
                btc_rsi_low_30m = self.btc_rsi_14_low_30m_long_ranging.value * 5
                condition = dataframe["rsi_14_BTC_30m"] <= btc_rsi_low_30m
                enter_long_ranging_conditions.append(condition)

            # rsi_14 Indicator - 1h Timeframe - Coin
            if self.enable_long_ranging_coin_rsi_14_1h.value:
                coin_rsi_low_1h = self.coin_rsi_14_low_1h_long_ranging.value * 5
                condition = dataframe["rsi_14_coin1h"] <= coin_rsi_low_1h
                enter_long_ranging_conditions.append(condition)

            # rsi_14 Indicator - 1h Timeframe - BTC
            if self.enable_long_ranging_btc_rsi_14_1h.value:
                btc_rsi_low_1h = self.btc_rsi_14_low_1h_long_ranging.value * 5
                condition = dataframe["rsi_14_BTC_1h"] <= btc_rsi_low_1h
                enter_long_ranging_conditions.append(condition)

            # rsi_14 Indicator - 4h Timeframe - Coin
            if self.enable_long_ranging_coin_rsi_14_4h.value:
                coin_rsi_low_4h = self.coin_rsi_14_low_4h_long_ranging.value * 5
                condition = dataframe["rsi_14_coin4h"] <= coin_rsi_low_4h
                enter_long_ranging_conditions.append(condition)

            # rsi_14 Indicator - 4h Timeframe - BTC
            if self.enable_long_ranging_btc_rsi_14_4h.value:
                btc_rsi_low_4h = self.btc_rsi_14_low_4h_long_ranging.value * 5
                condition = dataframe["rsi_14_BTC_4h"] <= btc_rsi_low_4h
                enter_long_ranging_conditions.append(condition)

            # rsi_14 Indicator - 1d Timeframe - Coin
            if self.enable_long_ranging_coin_rsi_14_1d.value:
                coin_rsi_low_1d = self.coin_rsi_14_low_1d_long_ranging.value * 5
                condition = dataframe["rsi_14_coin1d"] <= coin_rsi_low_1d
                enter_long_ranging_conditions.append(condition)

            # rsi_14 Indicator - 1d Timeframe - BTC
            if self.enable_long_ranging_btc_rsi_14_1d.value:
                btc_rsi_low_1d = self.btc_rsi_14_low_1d_long_ranging.value * 5
                condition = dataframe["rsi_14_BTC_1d"] <= btc_rsi_low_1d
                enter_long_ranging_conditions.append(condition)

            # rsi_14 Indicator - 3d Timeframe - Coin
            if self.enable_long_ranging_coin_rsi_14_3d.value:
                coin_rsi_low_3d = self.coin_rsi_14_low_3d_long_ranging.value * 5
                condition = dataframe["rsi_14_coin3d"] <= coin_rsi_low_3d
                enter_long_ranging_conditions.append(condition)

            # rsi_14 Indicator - 3d Timeframe - BTC
            if self.enable_long_ranging_btc_rsi_14_3d.value:
                btc_rsi_low_3d = self.btc_rsi_14_low_3d_long_ranging.value * 5
                condition = dataframe["rsi_14_BTC_3d"] <= btc_rsi_low_3d
                enter_long_ranging_conditions.append(condition)

            # mfi Indicator - 30m Timeframe - Coin
            if self.enable_long_ranging_coin_mfi_30m.value:
                coin_mfi_low_30m = self.coin_mfi_low_30m_long_ranging.value * 5
                condition = dataframe["mfi_coin30m"] <= coin_mfi_low_30m
                enter_long_ranging_conditions.append(condition)

            # mfi Indicator - 30m Timeframe - BTC
            if self.enable_long_ranging_btc_mfi_30m.value:
                btc_mfi_low_30m = self.btc_mfi_low_30m_long_ranging.value * 5
                condition = dataframe["mfi_BTC_30m"] <= btc_mfi_low_30m
                enter_long_ranging_conditions.append(condition)

            # mfi Indicator - 1h Timeframe - Coin
            if self.enable_long_ranging_coin_mfi_1h.value:
                coin_mfi_low_1h = self.coin_mfi_low_1h_long_ranging.value * 5
                condition = dataframe["mfi_coin1h"] <= coin_mfi_low_1h
                enter_long_ranging_conditions.append(condition)

            # mfi Indicator - 1h Timeframe - BTC
            if self.enable_long_ranging_btc_mfi_1h.value:
                btc_mfi_low_1h = self.btc_mfi_low_1h_long_ranging.value * 5
                condition = dataframe["mfi_BTC_1h"] <= btc_mfi_low_1h
                enter_long_ranging_conditions.append(condition)

            # mfi Indicator - 4h Timeframe - Coin
            if self.enable_long_ranging_coin_mfi_4h.value:
                coin_mfi_low_4h = self.coin_mfi_low_4h_long_ranging.value * 5
                condition = dataframe["mfi_coin4h"] <= coin_mfi_low_4h
                enter_long_ranging_conditions.append(condition)

            # mfi Indicator - 4h Timeframe - BTC
            if self.enable_long_ranging_btc_mfi_4h.value:
                btc_mfi_low_4h = self.btc_mfi_low_4h_long_ranging.value * 5
                condition = dataframe["mfi_BTC_4h"] <= btc_mfi_low_4h
                enter_long_ranging_conditions.append(condition)

            # mfi Indicator - 1d Timeframe - Coin
            if self.enable_long_ranging_coin_mfi_1d.value:
                coin_mfi_low_1d = self.coin_mfi_low_1d_long_ranging.value * 5
                condition = dataframe["mfi_coin1d"] <= coin_mfi_low_1d
                enter_long_ranging_conditions.append(condition)

            # mfi Indicator - 1d Timeframe - BTC
            if self.enable_long_ranging_btc_mfi_1d.value:
                btc_mfi_low_1d = self.btc_mfi_low_1d_long_ranging.value * 5
                condition = dataframe["mfi_BTC_1d"] <= btc_mfi_low_1d
                enter_long_ranging_conditions.append(condition)

            # mfi Indicator - 3d Timeframe - Coin
            if self.enable_long_ranging_coin_mfi_3d.value:
                coin_mfi_low_3d = self.coin_mfi_low_3d_long_ranging.value * 5
                condition = dataframe["mfi_coin3d"] <= coin_mfi_low_3d
                enter_long_ranging_conditions.append(condition)

            # mfi Indicator - 3d Timeframe - BTC
            if self.enable_long_ranging_btc_mfi_3d.value:
                btc_mfi_low_3d = self.btc_mfi_low_3d_long_ranging.value * 5
                condition = dataframe["mfi_BTC_3d"] <= btc_mfi_low_3d
                enter_long_ranging_conditions.append(condition)

            # uo Indicator - 30m Timeframe - Coin
            if self.enable_long_ranging_coin_uo_30m.value:
                coin_uo_low_30m = self.coin_uo_low_30m_long_ranging.value * 5
                condition = dataframe["uo_coin30m"] <= coin_uo_low_30m
                enter_long_ranging_conditions.append(condition)

            # uo Indicator - 30m Timeframe - BTC
            if self.enable_long_ranging_btc_uo_30m.value:
                btc_uo_low_30m = self.btc_uo_low_30m_long_ranging.value * 5
                condition = dataframe["uo_BTC_30m"] <= btc_uo_low_30m
                enter_long_ranging_conditions.append(condition)

            # uo Indicator - 1h Timeframe - Coin
            if self.enable_long_ranging_coin_uo_1h.value:
                coin_uo_low_1h = self.coin_uo_low_1h_long_ranging.value * 5
                condition = dataframe["uo_coin1h"] <= coin_uo_low_1h
                enter_long_ranging_conditions.append(condition)

            # uo Indicator - 1h Timeframe - BTC
            if self.enable_long_ranging_btc_uo_1h.value:
                btc_uo_low_1h = self.btc_uo_low_1h_long_ranging.value * 5
                condition = dataframe["uo_BTC_1h"] <= btc_uo_low_1h
                enter_long_ranging_conditions.append(condition)

            # uo Indicator - 4h Timeframe - Coin
            if self.enable_long_ranging_coin_uo_4h.value:
                coin_uo_low_4h = self.coin_uo_low_4h_long_ranging.value * 5
                condition = dataframe["uo_coin4h"] <= coin_uo_low_4h
                enter_long_ranging_conditions.append(condition)

            # uo Indicator - 4h Timeframe - BTC
            if self.enable_long_ranging_btc_uo_4h.value:
                btc_uo_low_4h = self.btc_uo_low_4h_long_ranging.value * 5
                condition = dataframe["uo_BTC_4h"] <= btc_uo_low_4h
                enter_long_ranging_conditions.append(condition)

            # uo Indicator - 1d Timeframe - Coin
            if self.enable_long_ranging_coin_uo_1d.value:
                coin_uo_low_1d = self.coin_uo_low_1d_long_ranging.value * 5
                condition = dataframe["uo_coin1d"] <= coin_uo_low_1d
                enter_long_ranging_conditions.append(condition)

            # uo Indicator - 1d Timeframe - BTC
            if self.enable_long_ranging_btc_uo_1d.value:
                btc_uo_low_1d = self.btc_uo_low_1d_long_ranging.value * 5
                condition = dataframe["uo_BTC_1d"] <= btc_uo_low_1d
                enter_long_ranging_conditions.append(condition)

            # uo Indicator - 3d Timeframe - Coin
            if self.enable_long_ranging_coin_uo_3d.value:
                coin_uo_low_3d = self.coin_uo_low_3d_long_ranging.value * 5
                condition = dataframe["uo_coin3d"] <= coin_uo_low_3d
                enter_long_ranging_conditions.append(condition)

            # uo Indicator - 3d Timeframe - BTC
            if self.enable_long_ranging_btc_uo_3d.value:
                btc_uo_low_3d = self.btc_uo_low_3d_long_ranging.value * 5
                condition = dataframe["uo_BTC_3d"] <= btc_uo_low_3d
                enter_long_ranging_conditions.append(condition)

            # Trigger Conditions for Long Ranging (Low Reversals)
            tf_trigger_long = self.set_trigger_long_ranging.value
            period_trigger_long = (
                self.set_trigger_period_long_ranging.value
            )  # Assuming there's a period trigger for long
            current_rsi_long = dataframe[f"rsi_{period_trigger_long}_{pair}_{tf_trigger_long}"]

            # Condition 1: Current RSI >= Previous RSI
            condition_current_long = current_rsi_long >= current_rsi_long.shift(1)
            enter_long_ranging_conditions.append(condition_current_long)

            # Condition 2: Previous RSI <= RSI before that
            condition_previous_long = current_rsi_long.shift(1) <= current_rsi_long.shift(2)
            enter_long_ranging_conditions.append(condition_previous_long)

            # Ensure no NaNs to prevent failures
            enter_long_ranging_conditions = [
                (x if isinstance(x, bool) else False) or pd.isnull(x)
                for x in enter_long_ranging_conditions
            ]

            # Combine all conditions using logical AND
            if enter_long_ranging_conditions:
                combined_long_ranging_condition = reduce(
                    lambda x, y: x & y, enter_long_ranging_conditions
                )
                dataframe.loc[combined_long_ranging_condition, ["enter_long", "enter_tag"]] = (
                    1,
                    "long_ranging",
                )

        # -----------------------------
        # Handle Enter Short Ranging
        # -----------------------------
        if self.enter_short_ranging.value:
            # rsi_14 Indicator - 30m Timeframe - Coin
            if self.enable_short_ranging_coin_rsi_14_30m.value:
                coin_rsi_high_30m = self.coin_rsi_14_high_30m_short_ranging.value * 5
                condition = dataframe["rsi_14_coin30m"] >= coin_rsi_high_30m
                enter_short_ranging_conditions.append(condition)

            # rsi_14 Indicator - 30m Timeframe - BTC
            if self.enable_short_ranging_btc_rsi_14_30m.value:
                btc_rsi_high_30m = self.btc_rsi_14_high_30m_short_ranging.value * 5
                condition = dataframe["rsi_14_BTC_30m"] >= btc_rsi_high_30m
                enter_short_ranging_conditions.append(condition)

            # rsi_14 Indicator - 1h Timeframe - Coin
            if self.enable_short_ranging_coin_rsi_14_1h.value:
                coin_rsi_high_1h = self.coin_rsi_14_high_1h_short_ranging.value * 5
                condition = dataframe["rsi_14_coin1h"] >= coin_rsi_high_1h
                enter_short_ranging_conditions.append(condition)

            # rsi_14 Indicator - 1h Timeframe - BTC
            if self.enable_short_ranging_btc_rsi_14_1h.value:
                btc_rsi_high_1h = self.btc_rsi_14_high_1h_short_ranging.value * 5
                condition = dataframe["rsi_14_BTC_1h"] >= btc_rsi_high_1h
                enter_short_ranging_conditions.append(condition)

            # rsi_14 Indicator - 4h Timeframe - Coin
            if self.enable_short_ranging_coin_rsi_14_4h.value:
                coin_rsi_high_4h = self.coin_rsi_14_high_4h_short_ranging.value * 5
                condition = dataframe["rsi_14_coin4h"] >= coin_rsi_high_4h
                enter_short_ranging_conditions.append(condition)

            # rsi_14 Indicator - 4h Timeframe - BTC
            if self.enable_short_ranging_btc_rsi_14_4h.value:
                btc_rsi_high_4h = self.btc_rsi_14_high_4h_short_ranging.value * 5
                condition = dataframe["rsi_14_BTC_4h"] >= btc_rsi_high_4h
                enter_short_ranging_conditions.append(condition)

            # rsi_14 Indicator - 1d Timeframe - Coin
            if self.enable_short_ranging_coin_rsi_14_1d.value:
                coin_rsi_high_1d = self.coin_rsi_14_high_1d_short_ranging.value * 5
                condition = dataframe["rsi_14_coin1d"] >= coin_rsi_high_1d
                enter_short_ranging_conditions.append(condition)

            # rsi_14 Indicator - 1d Timeframe - BTC
            if self.enable_short_ranging_btc_rsi_14_1d.value:
                btc_rsi_high_1d = self.btc_rsi_14_high_1d_short_ranging.value * 5
                condition = dataframe["rsi_14_BTC_1d"] >= btc_rsi_high_1d
                enter_short_ranging_conditions.append(condition)

            # rsi_14 Indicator - 3d Timeframe - Coin
            if self.enable_short_ranging_coin_rsi_14_3d.value:
                coin_rsi_high_3d = self.coin_rsi_14_high_3d_short_ranging.value * 5
                condition = dataframe["rsi_14_coin3d"] >= coin_rsi_high_3d
                enter_short_ranging_conditions.append(condition)

            # rsi_14 Indicator - 3d Timeframe - BTC
            if self.enable_short_ranging_btc_rsi_14_3d.value:
                btc_rsi_high_3d = self.btc_rsi_14_high_3d_short_ranging.value * 5
                condition = dataframe["rsi_14_BTC_3d"] >= btc_rsi_high_3d
                enter_short_ranging_conditions.append(condition)

            # mfi Indicator - 30m Timeframe - Coin
            if self.enable_short_ranging_coin_mfi_30m.value:
                coin_mfi_high_30m = self.coin_mfi_high_30m_short_ranging.value * 5
                condition = dataframe["mfi_coin30m"] >= coin_mfi_high_30m
                enter_short_ranging_conditions.append(condition)

            # mfi Indicator - 30m Timeframe - BTC
            if self.enable_short_ranging_btc_mfi_30m.value:
                btc_mfi_high_30m = self.btc_mfi_high_30m_short_ranging.value * 5
                condition = dataframe["mfi_BTC_30m"] >= btc_mfi_high_30m
                enter_short_ranging_conditions.append(condition)

            # mfi Indicator - 1h Timeframe - Coin
            if self.enable_short_ranging_coin_mfi_1h.value:
                coin_mfi_high_1h = self.coin_mfi_high_1h_short_ranging.value * 5
                condition = dataframe["mfi_coin1h"] >= coin_mfi_high_1h
                enter_short_ranging_conditions.append(condition)

            # mfi Indicator - 1h Timeframe - BTC
            if self.enable_short_ranging_btc_mfi_1h.value:
                btc_mfi_high_1h = self.btc_mfi_high_1h_short_ranging.value * 5
                condition = dataframe["mfi_BTC_1h"] >= btc_mfi_high_1h
                enter_short_ranging_conditions.append(condition)

            # mfi Indicator - 4h Timeframe - Coin
            if self.enable_short_ranging_coin_mfi_4h.value:
                coin_mfi_high_4h = self.coin_mfi_high_4h_short_ranging.value * 5
                condition = dataframe["mfi_coin4h"] >= coin_mfi_high_4h
                enter_short_ranging_conditions.append(condition)

            # mfi Indicator - 4h Timeframe - BTC
            if self.enable_short_ranging_btc_mfi_4h.value:
                btc_mfi_high_4h = self.btc_mfi_high_4h_short_ranging.value * 5
                condition = dataframe["mfi_BTC_4h"] >= btc_mfi_high_4h
                enter_short_ranging_conditions.append(condition)

            # mfi Indicator - 1d Timeframe - Coin
            if self.enable_short_ranging_coin_mfi_1d.value:
                coin_mfi_high_1d = self.coin_mfi_high_1d_short_ranging.value * 5
                condition = dataframe["mfi_coin1d"] >= coin_mfi_high_1d
                enter_short_ranging_conditions.append(condition)

            # mfi Indicator - 1d Timeframe - BTC
            if self.enable_short_ranging_btc_mfi_1d.value:
                btc_mfi_high_1d = self.btc_mfi_high_1d_short_ranging.value * 5
                condition = dataframe["mfi_BTC_1d"] >= btc_mfi_high_1d
                enter_short_ranging_conditions.append(condition)

            # mfi Indicator - 3d Timeframe - Coin
            if self.enable_short_ranging_coin_mfi_3d.value:
                coin_mfi_high_3d = self.coin_mfi_high_3d_short_ranging.value * 5
                condition = dataframe["mfi_coin3d"] >= coin_mfi_high_3d
                enter_short_ranging_conditions.append(condition)

            # mfi Indicator - 3d Timeframe - BTC
            if self.enable_short_ranging_btc_mfi_3d.value:
                btc_mfi_high_3d = self.btc_mfi_high_3d_short_ranging.value * 5
                condition = dataframe["mfi_BTC_3d"] >= btc_mfi_high_3d
                enter_short_ranging_conditions.append(condition)

            # uo Indicator - 30m Timeframe - Coin
            if self.enable_short_ranging_coin_uo_30m.value:
                coin_uo_high_30m = self.coin_uo_high_30m_short_ranging.value * 5
                condition = dataframe["uo_coin30m"] >= coin_uo_high_30m
                enter_short_ranging_conditions.append(condition)

            # uo Indicator - 30m Timeframe - BTC
            if self.enable_short_ranging_btc_uo_30m.value:
                btc_uo_high_30m = self.btc_uo_high_30m_short_ranging.value * 5
                condition = dataframe["uo_BTC_30m"] >= btc_uo_high_30m
                enter_short_ranging_conditions.append(condition)

            # uo Indicator - 1h Timeframe - Coin
            if self.enable_short_ranging_coin_uo_1h.value:
                coin_uo_high_1h = self.coin_uo_high_1h_short_ranging.value * 5
                condition = dataframe["uo_coin1h"] >= coin_uo_high_1h
                enter_short_ranging_conditions.append(condition)

            # uo Indicator - 1h Timeframe - BTC
            if self.enable_short_ranging_btc_uo_1h.value:
                btc_uo_high_1h = self.btc_uo_high_1h_short_ranging.value * 5
                condition = dataframe["uo_BTC_1h"] >= btc_uo_high_1h
                enter_short_ranging_conditions.append(condition)

            # uo Indicator - 4h Timeframe - Coin
            if self.enable_short_ranging_coin_uo_4h.value:
                coin_uo_high_4h = self.coin_uo_high_4h_short_ranging.value * 5
                condition = dataframe["uo_coin4h"] >= coin_uo_high_4h
                enter_short_ranging_conditions.append(condition)

            # uo Indicator - 4h Timeframe - BTC
            if self.enable_short_ranging_btc_uo_4h.value:
                btc_uo_high_4h = self.btc_uo_high_4h_short_ranging.value * 5
                condition = dataframe["uo_BTC_4h"] >= btc_uo_high_4h
                enter_short_ranging_conditions.append(condition)

            # uo Indicator - 1d Timeframe - Coin
            if self.enable_short_ranging_coin_uo_1d.value:
                coin_uo_high_1d = self.coin_uo_high_1d_short_ranging.value * 5
                condition = dataframe["uo_coin1d"] >= coin_uo_high_1d
                enter_short_ranging_conditions.append(condition)

            # uo Indicator - 1d Timeframe - BTC
            if self.enable_short_ranging_btc_uo_1d.value:
                btc_uo_high_1d = self.btc_uo_high_1d_short_ranging.value * 5
                condition = dataframe["uo_BTC_1d"] >= btc_uo_high_1d
                enter_short_ranging_conditions.append(condition)

            # uo Indicator - 3d Timeframe - Coin
            if self.enable_short_ranging_coin_uo_3d.value:
                coin_uo_high_3d = self.coin_uo_high_3d_short_ranging.value * 5
                condition = dataframe["uo_coin3d"] >= coin_uo_high_3d
                enter_short_ranging_conditions.append(condition)

            # uo Indicator - 3d Timeframe - BTC
            if self.enable_short_ranging_btc_uo_3d.value:
                btc_uo_high_3d = self.btc_uo_high_3d_short_ranging.value * 5
                condition = dataframe["uo_BTC_3d"] >= btc_uo_high_3d
                enter_short_ranging_conditions.append(condition)

            # Trigger Conditions for Short Ranging
            tf_trigger_short = self.set_trigger_tf_short_ranging.value
            period_trigger_short = self.set_trigger_period_short_ranging.value
            current_rsi_short = dataframe[f"rsi_{period_trigger_short}_{pair}_{tf_trigger_short}"]

            # Condition 1: Current RSI <= Previous RSI
            condition_current_short = current_rsi_short <= current_rsi_short.shift(1)
            enter_short_ranging_conditions.append(condition_current_short)

            # Condition 2: Previous RSI >= RSI before that
            condition_previous_short = current_rsi_short.shift(1) >= current_rsi_short.shift(2)
            enter_short_ranging_conditions.append(condition_previous_short)

            # Ensure no NaNs to prevent failures
            enter_short_ranging_conditions = [
                (x if isinstance(x, bool) else False) or pd.isnull(x)
                for x in enter_short_ranging_conditions
            ]

            # Combine all conditions using logical AND
            if enter_short_ranging_conditions:
                combined_short_ranging_condition = reduce(
                    lambda x, y: x & y, enter_short_ranging_conditions
                )
                dataframe.loc[combined_short_ranging_condition, ["enter_short", "enter_tag"]] = (
                    1,
                    "short_ranging",
                )

        # # master flag for enabling bull conditions. repeated at the final enter long check
        # # if True:
        # if self.enter_bull.value:
        #     # Extract timeframe and flag type from the trigger value
        #     trig, flag_type, tf = self.set_trigger_bull.value.split("_", 2)
        #     # Construct the full column name
        #     if trig == "rsi" and flag_type == "vol":
        #         enter_long_conditions.append(qtpylib.crossed_above(dataframe[f'{trig}_vol_{self.rsi_window_bull.value}_{pair}_{tf}'], getattr(self, f"rsi_vol_coin_{tf}_bull").value))
        #     elif flag_type == "rsi":
        #         trigger_col = f"{trig}_{flag_type}_{self.rsi_window_bull.value}_{pair}_{tf}"
        #         enter_long_conditions.append(dataframe[trigger_col] == 1)
        #
        #     else:
        #         trigger_col = f"{trig}_{flag_type}_bull_{pair}_{tf}"
        #         # Use the trigger column in your buy condition
        #         enter_long_conditions.append(dataframe[trigger_col] == 1)
        #
        #     # Guards: Check if the other flags are = 1
        #     if self.enable_short_coin_1h_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_short_flg_{coin1h}'] == 1)
        #     if self.enable_short_coin_4h_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_short_flg_{coin4h}'] == 1)
        #     if self.enable_short_coin_1d_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_short_flg_{coin1d}'] == 1)
        #     if self.enable_short_coin_3d_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_short_flg_{coin3d}'] == 1)
        #     # Stop the check for BTC_1h when the coin is BTC as this would fail due to
        #     # BTC1h being the default coin/tf defined as coin
        #     if self.enable_short_btc_1h_bull.value:
        #             enter_long_conditions.append(dataframe[f'bull_short_flg_{btc1h}'] == 1)
        #     if self.enable_short_btc_4h_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_short_flg_{btc4h}'] == 1)
        #     if self.enable_short_btc_1d_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_short_flg_{btc1d}'] == 1)
        #     if self.enable_short_btc_3d_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_short_flg_{btc3d}'] == 1)
        #
        #     # Guards for bull_med_flg
        #     if self.enable_med_coin_1h_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_med_flg_{coin1h}'] == 1)
        #     if self.enable_med_coin_4h_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_med_flg_{coin4h}'] == 1)
        #     if self.enable_med_coin_1d_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_med_flg_{coin1d}'] == 1)
        #     if self.enable_med_coin_3d_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_med_flg_{coin3d}'] == 1)
        #     if self.enable_med_btc_1h_bull.value:
        #             enter_long_conditions.append(dataframe[f'bull_med_flg_{btc1h}'] == 1)
        #     if self.enable_med_btc_4h_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_med_flg_{btc4h}'] == 1)
        #     if self.enable_med_btc_1d_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_med_flg_{btc1d}'] == 1)
        #     if self.enable_med_btc_3d_bull.value:
        #         enter_long_conditions.append(dataframe[f'bull_med_flg_{btc3d}'] == 1)
        #
        #     # Volume RSI  Checks
        #     # coin30m - enabled and volume check
        #     if self.enable_vol_rsi_coin_30m_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bull.value}_{coin30m}"] > (5 * self.rsi_vol_coin_30m_bull.value))
        #     # coin1h - enabled and volume check
        #     if self.enable_vol_rsi_coin_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bull.value}_{coin1h}"] > (5 * self.rsi_vol_coin_1h_bull.value))
        #     # coin4h
        #     if self.enable_vol_rsi_coin_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bull.value}_{coin4h}"] > (5 * self.rsi_vol_coin_4h_bull.value))
        #     # coin1d
        #     if self.enable_vol_rsi_coin_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bull.value}_{coin1d}"] > (5 * self.rsi_vol_coin_1d_bull.value))
        #     # coin3d
        #     if self.enable_vol_rsi_coin_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bull.value}_{coin3d}"] > (5 * self.rsi_vol_coin_3d_bull.value))
        #
        #     # # btc1h
        #     if self.enable_vol_rsi_btc_1h_bull.value:
        #             enter_long_conditions.append(
        #                 dataframe[f"rsi_vol_{self.rsi_window_bull.value}_{btc1h}"] > (5 * self.rsi_vol_btc_1h_bull.value))
        #     # btc4h
        #     if self.enable_vol_rsi_btc_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bull.value}_{btc4h}"] > (5 * self.rsi_vol_btc_4h_bull.value))
        #     # btc1d
        #     if self.enable_vol_rsi_btc_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bull.value}_{btc1d}"] > (5 * self.rsi_vol_btc_1d_bull.value))
        #     # btc3d
        #     if self.enable_vol_rsi_btc_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bull.value}_{btc3d}"] > (5 * self.rsi_vol_btc_3d_bull.value))
        #
        #     # Ultimate Oscillator (UO) Checks
        #     # coin30m
        #     if self.enable_uo_coin_30m_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"uo_{coin30m}"] > (5 * self.uo_coin_30m_bull.value))
        #     # coin1h
        #     if self.enable_uo_coin_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"uo_{coin1h}"] > (5 * self.uo_coin_1h_bull.value))
        #     # coin4h
        #     if self.enable_uo_coin_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"uo_{coin4h}"] > (5 * self.uo_coin_4h_bull.value))
        #     # coin1d
        #     if self.enable_uo_coin_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"uo_{coin1d}"] > (5 * self.uo_coin_1d_bull.value))
        #     # coin3d
        #     if self.enable_uo_coin_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"uo_{coin3d}"] > (5 * self.uo_coin_3d_bull.value))
        #
        #     # btc1h
        #     if self.enable_uo_btc_1h_bull.value:
        #             enter_long_conditions.append(
        #                 dataframe[f"uo_{btc1h}"] > (5 * self.uo_btc_1h_bull.value))
        #     # btc4h
        #     if self.enable_uo_btc_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"uo_{btc4h}"] > (5 * self.uo_btc_4h_bull.value))
        #     # btc1d
        #     if self.enable_uo_btc_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"uo_{btc1d}"] > (5 * self.uo_btc_1d_bull.value))
        #     # btc3d
        #     if self.enable_uo_btc_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"uo_{btc3d}"] > (5 * self.uo_btc_3d_bull.value))
        #
        #     # Money Flow Index (MFI) Checks
        #     # coin30m
        #     if self.enable_mfi_coin_30m_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"mfi_{coin30m}"] > (5 * self.mfi_coin_30m_bull.value))
        #     # coin1h
        #     if self.enable_mfi_coin_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"mfi_{coin1h}"] > (5 * self.mfi_coin_1h_bull.value))
        #     # coin4h
        #     if self.enable_mfi_coin_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"mfi_{coin4h}"] > (5 * self.mfi_coin_4h_bull.value))
        #     # coin1d
        #     if self.enable_mfi_coin_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"mfi_{coin1d}"] > (5 * self.mfi_coin_1d_bull.value))
        #     # coin3d
        #     if self.enable_mfi_coin_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"mfi_{coin3d}"] > (5 * self.mfi_coin_3d_bull.value))
        #
        #     # btc1h
        #     if self.enable_mfi_btc_1h_bull.value:
        #             enter_long_conditions.append(
        #                 dataframe[f"mfi_{btc1h}"] > (5 * self.mfi_btc_1h_bull.value))
        #     # btc4h
        #     if self.enable_mfi_btc_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"mfi_{btc4h}"] > (5 * self.mfi_btc_4h_bull.value))
        #     # btc1d
        #     if self.enable_mfi_btc_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"mfi_{btc1d}"] > (5 * self.mfi_btc_1d_bull.value))
        #     # btc3d
        #     if self.enable_mfi_btc_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"mfi_{btc3d}"] > (5 * self.mfi_btc_3d_bull.value))
        #
        #     # Check if we are at the range min/max/mid point etc or in breakout
        #
        #     # Range 1
        #     if self.enable_range1_coin_30m_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range1_{coin30m}"] == self.range1_value_coin_30m_bull.value)
        #
        #     if self.enable_range1_coin_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range1_{coin1h}"] == self.range1_value_coin_1h_bull.value)
        #
        #     if self.enable_range1_coin_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range1_{coin4h}"] == self.range1_value_coin_4h_bull.value)
        #
        #     if self.enable_range1_coin_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range1_{coin1d}"] == self.range1_value_coin_1d_bull.value)
        #
        #     if self.enable_range1_coin_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range1_{coin3d}"] == self.range1_value_coin_3d_bull.value)
        #
        #
        #     if self.enable_range1_btc_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range1_{btc1h}"] == self.range1_value_btc_1h_bull.value)
        #
        #     if self.enable_range1_btc_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range1_{btc4h}"] == self.range1_value_btc_4h_bull.value)
        #
        #     if self.enable_range1_btc_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range1_{btc1d}"] == self.range1_value_btc_1d_bull.value)
        #
        #     if self.enable_range1_btc_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range1_{btc3d}"] == self.range1_value_btc_3d_bull.value)
        #
        #     # Range 2
        #     if self.enable_range2_coin_30m_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range2_{coin30m}"] == self.range2_value_coin_30m_bull.value)
        #
        #     if self.enable_range2_coin_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range2_{coin1h}"] == self.range2_value_coin_1h_bull.value)
        #
        #     if self.enable_range2_coin_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range2_{coin4h}"] == self.range2_value_coin_4h_bull.value)
        #
        #     if self.enable_range2_coin_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range2_{coin1d}"] == self.range2_value_coin_1d_bull.value)
        #
        #     if self.enable_range2_coin_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range2_{coin3d}"] == self.range2_value_coin_3d_bull.value)
        #
        #     if self.enable_range2_btc_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range2_{btc1h}"] == self.range2_value_btc_1h_bull.value)
        #
        #     if self.enable_range2_btc_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range2_{btc4h}"] == self.range2_value_btc_4h_bull.value)
        #
        #     if self.enable_range2_btc_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range2_{btc1d}"] == self.range2_value_btc_1d_bull.value)
        #
        #     if self.enable_range2_btc_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range2_{btc3d}"] == self.range2_value_btc_3d_bull.value)
        #
        #     # Range 3
        #     if self.enable_range3_coin_30m_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range3_{coin30m}"] == self.range3_value_coin_30m_bull.value)
        #
        #     if self.enable_range3_coin_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range3_{coin1h}"] == self.range3_value_coin_1h_bull.value)
        #
        #     if self.enable_range3_coin_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range3_{coin4h}"] == self.range3_value_coin_4h_bull.value)
        #
        #     if self.enable_range3_coin_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range3_{coin1d}"] == self.range3_value_coin_1d_bull.value)
        #
        #     if self.enable_range3_coin_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range3_{coin3d}"] == self.range3_value_coin_3d_bull.value)
        #
        #     if self.enable_range3_btc_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range3_{btc1h}"] == self.range3_value_btc_1h_bull.value)
        #
        #     if self.enable_range3_btc_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range3_{btc4h}"] == self.range3_value_btc_4h_bull.value)
        #
        #     if self.enable_range3_btc_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range3_{btc1d}"] == self.range3_value_btc_1d_bull.value)
        #
        #     if self.enable_range3_btc_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range3_{btc3d}"] == self.range3_value_btc_3d_bull.value)
        #
        #     # Range 4
        #     if self.enable_range4_coin_30m_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range4_{coin30m}"] == self.range4_value_coin_30m_bull.value)
        #
        #     if self.enable_range4_coin_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range4_{coin1h}"] == self.range4_value_coin_1h_bull.value)
        #
        #     if self.enable_range4_coin_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range4_{coin4h}"] == self.range4_value_coin_4h_bull.value)
        #
        #     if self.enable_range4_coin_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range4_{coin1d}"] == self.range4_value_coin_1d_bull.value)
        #
        #     if self.enable_range4_coin_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range4_{coin3d}"] == self.range4_value_coin_3d_bull.value)
        #
        #     if self.enable_range4_btc_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range4_{btc1h}"] == self.range4_value_btc_1h_bull.value)
        #
        #     if self.enable_range4_btc_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range4_{btc4h}"] == self.range4_value_btc_4h_bull.value)
        #
        #     if self.enable_range4_btc_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range4_{btc1d}"] == self.range4_value_btc_1d_bull.value)
        #
        #     if self.enable_range4_btc_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range4_{btc3d}"] == self.range4_value_btc_3d_bull.value)
        #
        #     # Range 5
        #     if self.enable_range5_coin_30m_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range5_{coin30m}"] == self.range5_value_coin_30m_bull.value)
        #
        #     if self.enable_range5_coin_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range5_{coin1h}"] == self.range5_value_coin_1h_bull.value)
        #
        #     if self.enable_range5_coin_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range5_{coin4h}"] == self.range5_value_coin_4h_bull.value)
        #
        #     if self.enable_range5_coin_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range5_{coin1d}"] == self.range5_value_coin_1d_bull.value)
        #
        #     if self.enable_range5_coin_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range5_{coin3d}"] == self.range5_value_coin_3d_bull.value)
        #
        #     if self.enable_range5_btc_1h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range5_{btc1h}"] == self.range5_value_btc_1h_bull.value)
        #
        #     if self.enable_range5_btc_4h_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range5_{btc4h}"] == self.range5_value_btc_4h_bull.value)
        #
        #     if self.enable_range5_btc_1d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range5_{btc1d}"] == self.range5_value_btc_1d_bull.value)
        #
        #     if self.enable_range5_btc_3d_bull.value:
        #         enter_long_conditions.append(
        #             dataframe[f"range5_{btc3d}"] == self.range5_value_btc_3d_bull.value)
        #
        # # ################### Start of Short Conditions ###################
        # #if True:
        # if self.enter_bear.value:
        #
        #     # Extract timeframe and flag type from the trigger value
        #     trig, flag_type, tf = self.set_trigger_bear.value.split("_", 2)
        #     # Construct the full column name
        #     if trig == "rsi" and flag_type == "vol":
        #         enter_short_conditions.append(qtpylib.crossed_above(dataframe[f'{trig}_vol_{self.rsi_window_bear.value}_{pair}_{tf}'], getattr(self, f"rsi_vol_coin_{tf}_bear").value))
        #     elif flag_type == "rsi":
        #         trigger_col = f"{trig}_{flag_type}_{self.rsi_window_bear.value}_{pair}_{tf}"
        #         enter_short_conditions.append(dataframe[trigger_col] == 1)
        #
        #     else:
        #         trigger_col = f"{trig}_{flag_type}_bear_{pair}_{tf}"
        #         # Use the trigger column in your buy condition
        #         enter_short_conditions.append(dataframe[trigger_col] == 1)
        #
        #     # Guards: Check if the other flags are = 1
        #     if self.enable_short_coin_1h_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_short_flg_{coin1h}'] == 1)
        #     if self.enable_short_coin_4h_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_short_flg_{coin4h}'] == 1)
        #     if self.enable_short_coin_1d_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_short_flg_{coin1d}'] == 1)
        #     if self.enable_short_coin_3d_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_short_flg_{coin3d}'] == 1)
        #     # Stop the check for BTC_1h when the coin is BTC as this would fail due to
        #     # BTC1h being the default coin/tf defined as coin
        #     if self.enable_short_btc_1h_bear.value:
        #             enter_short_conditions.append(dataframe[f'bear_short_flg_{btc1h}'] == 1)
        #     if self.enable_short_btc_4h_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_short_flg_{btc4h}'] == 1)
        #     if self.enable_short_btc_1d_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_short_flg_{btc1d}'] == 1)
        #     if self.enable_short_btc_3d_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_short_flg_{btc3d}'] == 1)
        #
        #     # Guards for bear_med_flg
        #     if self.enable_med_coin_1h_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_med_flg_{coin1h}'] == 1)
        #     if self.enable_med_coin_4h_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_med_flg_{coin4h}'] == 1)
        #     if self.enable_med_coin_1d_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_med_flg_{coin1d}'] == 1)
        #     if self.enable_med_coin_3d_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_med_flg_{coin3d}'] == 1)
        #     if self.enable_med_btc_1h_bear.value:
        #             enter_short_conditions.append(dataframe[f'bear_med_flg_{btc1h}'] == 1)
        #     if self.enable_med_btc_4h_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_med_flg_{btc4h}'] == 1)
        #     if self.enable_med_btc_1d_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_med_flg_{btc1d}'] == 1)
        #     if self.enable_med_btc_3d_bear.value:
        #         enter_short_conditions.append(dataframe[f'bear_med_flg_{btc3d}'] == 1)
        #
        #     # Volume RSI  Checks
        #     # coin30m - enabled and volume check
        #     if self.enable_vol_rsi_coin_30m_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bear.value}_{coin30m}"] > (5 * self.rsi_vol_coin_30m_bear.value))
        #     # coin1h - enabled and volume check
        #     if self.enable_vol_rsi_coin_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bear.value}_{coin1h}"] > (5 * self.rsi_vol_coin_1h_bear.value))
        #     # coin4h
        #     if self.enable_vol_rsi_coin_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bear.value}_{coin4h}"] > (5 * self.rsi_vol_coin_4h_bear.value))
        #     # coin1d
        #     if self.enable_vol_rsi_coin_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bear.value}_{coin1d}"] > (5 * self.rsi_vol_coin_1d_bear.value))
        #     # coin3d
        #     if self.enable_vol_rsi_coin_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bear.value}_{coin3d}"] > (5 * self.rsi_vol_coin_3d_bear.value))
        #
        #     # # btc1h
        #     if self.enable_vol_rsi_btc_1h_bear.value:
        #             enter_short_conditions.append(
        #                 dataframe[f"rsi_vol_{self.rsi_window_bear.value}_{btc1h}"] > (5 * self.rsi_vol_btc_1h_bear.value))
        #     # btc4h
        #     if self.enable_vol_rsi_btc_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bear.value}_{btc4h}"] > (5 * self.rsi_vol_btc_4h_bear.value))
        #     # btc1d
        #     if self.enable_vol_rsi_btc_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bear.value}_{btc1d}"] > (5 * self.rsi_vol_btc_1d_bear.value))
        #     # btc3d
        #     if self.enable_vol_rsi_btc_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"rsi_vol_{self.rsi_window_bear.value}_{btc3d}"] > (5 * self.rsi_vol_btc_3d_bear.value))
        #
        #     # Ultimate Oscillator (UO) Checks
        #     # coin30m
        #     if self.enable_uo_coin_30m_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"uo_{coin30m}"] > (5 * self.uo_coin_30m_bear.value))
        #     # coin1h
        #     if self.enable_uo_coin_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"uo_{coin1h}"] < (5 * self.uo_coin_1h_bear.value))
        #     # coin4h
        #     if self.enable_uo_coin_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"uo_{coin4h}"] < (5 * self.uo_coin_4h_bear.value))
        #     # coin1d
        #     if self.enable_uo_coin_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"uo_{coin1d}"] < (5 * self.uo_coin_1d_bear.value))
        #     # coin3d
        #     if self.enable_uo_coin_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"uo_{coin3d}"] < (5 * self.uo_coin_3d_bear.value))
        #
        #     # btc1h
        #     if self.enable_uo_btc_1h_bear.value:
        #             enter_short_conditions.append(
        #                 dataframe[f"uo_{btc1h}"] < (5 * self.uo_btc_1h_bear.value))
        #     # btc4h
        #     if self.enable_uo_btc_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"uo_{btc4h}"] < (5 * self.uo_btc_4h_bear.value))
        #     # btc1d
        #     if self.enable_uo_btc_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"uo_{btc1d}"] < (5 * self.uo_btc_1d_bear.value))
        #     # btc3d
        #     if self.enable_uo_btc_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"uo_{btc3d}"] < (5 * self.uo_btc_3d_bear.value))
        #
        #     # Money Flow Index (MFI) Checks
        #     # coin30m
        #     if self.enable_mfi_coin_30m_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"mfi_{coin30m}"] < (5 * self.mfi_coin_30m_bear.value))
        #     # coin1h
        #     if self.enable_mfi_coin_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"mfi_{coin1h}"] < (5 * self.mfi_coin_1h_bear.value))
        #     # coin4h
        #     if self.enable_mfi_coin_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"mfi_{coin4h}"] < (5 * self.mfi_coin_4h_bear.value))
        #     # coin1d
        #     if self.enable_mfi_coin_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"mfi_{coin1d}"] < (5 * self.mfi_coin_1d_bear.value))
        #     # coin3d
        #     if self.enable_mfi_coin_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"mfi_{coin3d}"] < (5 * self.mfi_coin_3d_bear.value))
        #
        #     # btc1h
        #     if self.enable_mfi_btc_1h_bear.value:
        #             enter_short_conditions.append(
        #                 dataframe[f"mfi_{btc1h}"] < (5 * self.mfi_btc_1h_bear.value))
        #     # btc4h
        #     if self.enable_mfi_btc_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"mfi_{btc4h}"] < (5 * self.mfi_btc_4h_bear.value))
        #     # btc1d
        #     if self.enable_mfi_btc_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"mfi_{btc1d}"] < (5 * self.mfi_btc_1d_bear.value))
        #     # btc3d
        #     if self.enable_mfi_btc_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"mfi_{btc3d}"] < (5 * self.mfi_btc_3d_bear.value))
        #
        #     # Check if we are at the range min/max/mid point etc or in breakout
        #
        #     # Range 1
        #     if self.enable_range1_coin_30m_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range1_{coin30m}"] == self.range1_value_coin_30m_bear.value)
        #
        #     if self.enable_range1_coin_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range1_{coin1h}"] == self.range1_value_coin_1h_bear.value)
        #
        #     if self.enable_range1_coin_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range1_{coin4h}"] == self.range1_value_coin_4h_bear.value)
        #
        #     if self.enable_range1_coin_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range1_{coin1d}"] == self.range1_value_coin_1d_bear.value)
        #
        #     if self.enable_range1_coin_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range1_{coin3d}"] == self.range1_value_coin_3d_bear.value)
        #
        #
        #     if self.enable_range1_btc_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range1_{btc1h}"] == self.range1_value_btc_1h_bear.value)
        #
        #     if self.enable_range1_btc_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range1_{btc4h}"] == self.range1_value_btc_4h_bear.value)
        #
        #     if self.enable_range1_btc_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range1_{btc1d}"] == self.range1_value_btc_1d_bear.value)
        #
        #     if self.enable_range1_btc_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range1_{btc3d}"] == self.range1_value_btc_3d_bear.value)
        #
        #     # Range 2
        #     if self.enable_range2_coin_30m_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range2_{coin30m}"] == self.range2_value_coin_30m_bear.value)
        #
        #     if self.enable_range2_coin_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range2_{coin1h}"] == self.range2_value_coin_1h_bear.value)
        #
        #     if self.enable_range2_coin_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range2_{coin4h}"] == self.range2_value_coin_4h_bear.value)
        #
        #     if self.enable_range2_coin_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range2_{coin1d}"] == self.range2_value_coin_1d_bear.value)
        #
        #     if self.enable_range2_coin_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range2_{coin3d}"] == self.range2_value_coin_3d_bear.value)
        #
        #     if self.enable_range2_btc_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range2_{btc1h}"] == self.range2_value_btc_1h_bear.value)
        #
        #     if self.enable_range2_btc_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range2_{btc4h}"] == self.range2_value_btc_4h_bear.value)
        #
        #     if self.enable_range2_btc_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range2_{btc1d}"] == self.range2_value_btc_1d_bear.value)
        #
        #     if self.enable_range2_btc_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range2_{btc3d}"] == self.range2_value_btc_3d_bear.value)
        #
        #     # Range 3
        #     if self.enable_range3_coin_30m_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range3_{coin30m}"] == self.range3_value_coin_30m_bear.value)
        #
        #     if self.enable_range3_coin_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range3_{coin1h}"] == self.range3_value_coin_1h_bear.value)
        #
        #     if self.enable_range3_coin_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range3_{coin4h}"] == self.range3_value_coin_4h_bear.value)
        #
        #     if self.enable_range3_coin_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range3_{coin1d}"] == self.range3_value_coin_1d_bear.value)
        #
        #     if self.enable_range3_coin_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range3_{coin3d}"] == self.range3_value_coin_3d_bear.value)
        #
        #     if self.enable_range3_btc_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range3_{btc1h}"] == self.range3_value_btc_1h_bear.value)
        #
        #     if self.enable_range3_btc_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range3_{btc4h}"] == self.range3_value_btc_4h_bear.value)
        #
        #     if self.enable_range3_btc_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range3_{btc1d}"] == self.range3_value_btc_1d_bear.value)
        #
        #     if self.enable_range3_btc_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range3_{btc3d}"] == self.range3_value_btc_3d_bear.value)
        #
        #     # Range 4
        #     if self.enable_range4_coin_30m_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range4_{coin30m}"] == self.range4_value_coin_30m_bear.value)
        #
        #     if self.enable_range4_coin_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range4_{coin1h}"] == self.range4_value_coin_1h_bear.value)
        #
        #     if self.enable_range4_coin_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range4_{coin4h}"] == self.range4_value_coin_4h_bear.value)
        #
        #     if self.enable_range4_coin_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range4_{coin1d}"] == self.range4_value_coin_1d_bear.value)
        #
        #     if self.enable_range4_coin_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range4_{coin3d}"] == self.range4_value_coin_3d_bear.value)
        #
        #     if self.enable_range4_btc_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range4_{btc1h}"] == self.range4_value_btc_1h_bear.value)
        #
        #     if self.enable_range4_btc_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range4_{btc4h}"] == self.range4_value_btc_4h_bear.value)
        #
        #     if self.enable_range4_btc_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range4_{btc1d}"] == self.range4_value_btc_1d_bear.value)
        #
        #     if self.enable_range4_btc_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range4_{btc3d}"] == self.range4_value_btc_3d_bear.value)
        #
        #     # Range 5
        #     if self.enable_range5_coin_30m_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range5_{coin30m}"] == self.range5_value_coin_30m_bear.value)
        #
        #     if self.enable_range5_coin_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range5_{coin1h}"] == self.range5_value_coin_1h_bear.value)
        #
        #     if self.enable_range5_coin_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range5_{coin4h}"] == self.range5_value_coin_4h_bear.value)
        #
        #     if self.enable_range5_coin_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range5_{coin1d}"] == self.range5_value_coin_1d_bear.value)
        #
        #     if self.enable_range5_coin_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range5_{coin3d}"] == self.range5_value_coin_3d_bear.value)
        #
        #     if self.enable_range5_btc_1h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range5_{btc1h}"] == self.range5_value_btc_1h_bear.value)
        #
        #     if self.enable_range5_btc_4h_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range5_{btc4h}"] == self.range5_value_btc_4h_bear.value)
        #
        #     if self.enable_range5_btc_1d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range5_{btc1d}"] == self.range5_value_btc_1d_bear.value)
        #
        #     if self.enable_range5_btc_3d_bear.value:
        #         enter_short_conditions.append(
        #             dataframe[f"range5_{btc3d}"] == self.range5_value_btc_3d_bear.value)
        #
        # ################# ENTER POSITION FINAL CHECK ##################
        # # if self.enter_bull.value:
        # # Combine all conditions for entering a long position
        # if enter_long_conditions:
        #     dataframe.loc[
        #         reduce(lambda x, y: x & y, enter_long_conditions),
        #         ["enter_long", "enter_tag"]
        #     ] = (1, "bull")
        #
        # # Combine all conditions for entering a short position
        # # if self.enter_bear.value:
        # if enter_short_conditions:
        #     dataframe.loc[
        #         reduce(lambda x, y: x & y, enter_short_conditions),
        #         ["enter_short", "enter_tag"]
        #     ] = (1, "bear")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = dataframe
        pair = metadata["pair"]
        pair = pair.replace("/USDT:USDT", "")
        coin1w = f"{pair}_1w"
        coin1d = f"{pair}_1d"
        coin4h = f"{pair}_4h"
        coin1h = f"{pair}_1h"
        btc1w = f"BTC_1w"
        btc1d = f"BTC_1d"
        btc4h = f"BTC_4h"
        btc1h = f"BTC_1h"

        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        # Get dataframe and current candle data in case we want to analyse anything from it
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        current_candle_index = dataframe.index[-1]

        # To access the enter tag reason to decide what rules to apply we use the following data
        entry_tag = current_candle.enter_tag
        leverage_return = (getattr(self, f"leverage_{entry_tag}").value) * 3
        return leverage_return

    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        Determines the custom stake amount for a trade.

        This function calculates the amount to be used for an initial order, considering the available balance and entry strategy.
        It adjusts the stake dynamically based on the entry tag and available funds, especially when using unlimited stake configuration.

        :param pair: The trading pair.
        :param current_time: The current datetime.
        :param current_rate: Current market rate for the trading pair.
        :param proposed_stake: The initially proposed stake.
        :param min_stake: The minimum stake amount.
        :param max_stake: The maximum stake amount.
        :param leverage: Leverage used for the trade.
        :param entry_tag: Optional entry tag to determine trading rules.
        :param side: The trade side, "long" or "short".
        :return: The calculated stake amount to be used for the trade.
        """

        # Retrieve the analyzed dataframe and extract the latest candle data
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        # Obtain the entry tag to decide applicable rules
        entry_tag = current_candle.enter_tag
        risk = Trade.total_open_trades_stakes()

        try:
            # Filter trades with entries >= 1 and exits < 1
            if hasattr(Trade, "trades_open"):
                # Relevant trades are defined as those trades that have not yet been successfully placed but
                # have not yet hit a TP level which would protect them from ever going negative.
                relevant_trades = [
                    t
                    for t in Trade.trades_open
                    if t.nr_of_successful_entries >= 1 and t.nr_of_successful_exits < 1
                ]

                long_count = 0
                short_count = 0
                for t in relevant_trades:
                    if t.is_short:
                        short_count += 1
                    else:
                        long_count += 1

                # Check against the maximum allowed open trades
                max_open_trades = getattr(self, f"max_open_trades_{entry_tag}").value
                if long_count - short_count >= max_open_trades:
                    return None  # Prevent entering a new trade

            else:
                # Set trade size as a percentage of available funds based on entry tag
                trade_size_pct = getattr(self, f"set_initial_trade_size_{entry_tag}").value

                # If the stake amount is unlimited, calculate the stake based on available USDT
                if self.config["stake_amount"] == "unlimited":
                    usdt_available = self.wallets.get_free("USDT")
                    proposed_stake = usdt_available * trade_size_pct

                return proposed_stake

        except Exception as e:
            logger.error(f"Error in custom_entry: {e}")
            return None  # Prevent entering a new trade in case of error

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Union[Optional[float], Tuple[Optional[float], Optional[str]]]:
        """
        Adjusts the trade position by evaluating current trade conditions and determining
        whether to place additional DCA entries or take profits based on custom logic.

        Key Features:
        - Uses `custom_dca` to store TP and DCA data, avoiding recalculations.
        - Tracks whether price levels have been crossed (price_hit).
        - Updates profit targets dynamically based on `trade.open_rate`.
        - Processes one TP or DCA entry at a time.
        - Relies on `trade.nr_of_successful_entries` and `trade.nr_of_successful_exits` to confirm actions.
        """
        try:
            # Retrieve Dynamic Parameters
            entry_tag: str = trade.enter_tag
            number_of_dca_entries: int = getattr(
                self, f"number_of_additional_entries_{entry_tag}"
            ).value
            number_of_tp: int = getattr(self, f"number_of_tp_{entry_tag}").value
            leverage: int = (
                getattr(self, f"leverage_{entry_tag}").value * 3
            )  # Adjusted for granularity

            # Early Exit Conditions to limit run time.
            if (
                (
                    (not trade.is_open)  # If trade is not open, no action to be taken
                    or trade.has_open_orders  # If orders are already open we should let them finish
                    or trade.nr_of_successful_entries
                    < 1  # No entry has been placed so no action can be taken
                    or trade.nr_of_successful_exits >= number_of_tp
                )  # We have reach max adjust position logic
                or current_profit == 0  # No action can be taken if profit is 0
            ):
                return None

            # Determine Trade Direction
            direction: int = -1 if trade.is_short else 1
            # if direction == 1:
            #     print(f"Long trade detected for {trade.pair}.")
            # Abbreviate Pair Name
            pair: str = trade.pair.replace("/USDT:USDT", "")

            # Retrieve and Process Dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
            current_candle: Dict[str, Any] = dataframe.iloc[-1].squeeze().to_dict()
            previous_candle: Dict[str, Any] = dataframe.iloc[-2].squeeze().to_dict()

            # Extract RSI Values
            tp_tf_value: str = getattr(self, f"dca_tp_tf_{entry_tag}").value
            current_rsi: float = current_candle.get(f"rsi_7_{pair}_{tp_tf_value}", 0.0)
            prev_rsi: float = previous_candle.get(f"rsi_7_{pair}_{tp_tf_value}", 0.0)

            # Debugging Data Points
            debug = 0
            if debug == 1:
                open_trades = trade.get_open_trades()
                all_trades = trade.get_trades_proxy()
                all_custom_data = trade.get_all_custom_data()

            # If no custom_dca data exists, we need to generate it.
            if (
                trade.nr_of_successful_entries == 1
                and trade.nr_of_successful_exits == 0
                and not trade.get_custom_data(key="custom_dca")
            ):
                # Retrieve or Set Custom DCA Data
                # RIf first iteration, generate tp and dca values. Else, we use get custom data to retrieve them. etrieve tp_data and dca_data
                tp_data = self.set_tp(trade, number_of_tp, entry_tag, direction, leverage)
                dca_data = self.calculate_scaled_dca_entries(entry_tag=entry_tag, trade=trade)

                # Initialize custom_dca with necessary data
                custom_dca = {
                    "tp_data": tp_data,
                    "dca_data": dca_data,
                    "last_nr_of_entries": trade.nr_of_successful_entries,
                }
                # Store custom_dca in trade
                trade.set_custom_data(key="custom_dca", value=custom_dca)
            # if custom data exists extract values.
            else:
                custom_dca = trade.get_custom_data(key="custom_dca")
                tp_data = custom_dca.get("tp_data")
                dca_data = custom_dca.get("dca_data")
                # Update tp targets if a new entry has been made
                if custom_dca.get("last_nr_of_entries") != trade.nr_of_successful_entries:
                    custom_dca["tp_data"] = tp_data
                    custom_dca["last_nr_of_entries"] = trade.nr_of_successful_entries
                    trade.set_custom_data(key="custom_dca", value=custom_dca)

            # Handle DCA Entries
            if (
                trade.nr_of_successful_exits < 1  # no entries made if tp has been hit
                and trade.nr_of_successful_entries
                < number_of_dca_entries  # Limit number of entries
                and current_profit < 0  # Ensure profit is negative
            ):
                for dca_entry in dca_data:
                    if trade.nr_of_successful_entries == dca_entry["entry_num"] - 1:
                        if not dca_entry["price_hit"]:
                            if (direction == 1 and current_rate <= dca_entry["price"]) or (
                                direction == -1 and current_rate >= dca_entry["price"]
                            ):
                                # Update price_hit status
                                dca_entry["price_hit"] = True
                                # Update custom_dca in trade
                                trade.set_custom_data(key="custom_dca", value=custom_dca)
                        # After price is hit, check RSI condition
                        if dca_entry["price_hit"]:
                            rsi_condition_met = False
                            if trade.is_short:
                                if current_rsi < prev_rsi:
                                    rsi_condition_met = True
                            else:
                                if current_rsi > prev_rsi:
                                    rsi_condition_met = True
                            if rsi_condition_met:
                                # Place DCA entry
                                safety_order_stake = max(dca_entry["size"], min_stake * 1.01)
                                return safety_order_stake, dca_entry["tag"]

            # Handle Take Profits
            # Limit TP logic to only run if we are in profit.
            if current_profit > 0:
                # TP1 - Only runs when in profit and no exits have been made
                if trade.nr_of_successful_exits == 0:
                    # Check if TP1 has not been hit, then verify if the target price is reached
                    if not tp_data[0]["price_hit"]:
                        # Multiply by direction and check for a positive result to determine if target has been reached
                        # This allows for shared logic on long and short positions
                        if direction * (current_rate - tp_data[0]["target_price"]) >= 0:
                            tp_data[0]["price_hit"] = True
                            trade.set_custom_data(key="custom_dca", value=custom_dca)
                    # Once TP1 price is hit, check for an RSI dip before exiting
                    if tp_data[0]["price_hit"]:
                        # Use a check for a negative result to identify an RSI reversal against the trade direction
                        if direction * (current_rsi - prev_rsi) < 0:
                            # TP1 Position Reduction
                            tp_reduction = tp_data[0]["position_reduction_pct"]
                            stake_reduction = -trade.stake_amount * tp_reduction
                            if number_of_tp == 1:
                                # Full exit as the final TP has been hit
                                return -trade.stake_amount, f"Full exit at TP1/TP{number_of_tp}"
                            else:
                                # Partial exit
                                return stake_reduction, "TP1 reduction hit"

                # TP2 - Executes when there are at least 2 TPs and 1 successful exit
                elif number_of_tp >= 2 and trade.nr_of_successful_exits == 1:
                    if len(tp_data) >= 2:
                        # Check if TP2 has not been hit, then verify if the target price is reached
                        if not tp_data[1]["price_hit"]:
                            # Multiply by direction and check for a positive result to determine if target has been reached
                            if direction * (current_rate - tp_data[1]["target_price"]) >= 0:
                                tp_data[1]["price_hit"] = True
                                trade.set_custom_data(key="custom_dca", value=custom_dca)
                        # Once TP2 price is hit, check for an RSI dip before exiting
                        if tp_data[1]["price_hit"]:
                            # Use a check for a negative result to identify an RSI reversal against the trade direction
                            if direction * (current_rsi - prev_rsi) < 0:
                                # TP2 Position Reduction
                                tp_reduction = tp_data[1]["position_reduction_pct"]
                                stake_reduction = -trade.stake_amount * tp_reduction
                                if number_of_tp == 2:
                                    # Full exit as the final TP has been hit
                                    return -trade.stake_amount, f"Full exit at TP2/TP{number_of_tp}"
                                else:
                                    # Partial exit
                                    return stake_reduction, "TP2 reduction hit"

                # TP3 - Executes when there are at least 3 TPs and 2 successful exits
                elif number_of_tp >= 3 and trade.nr_of_successful_exits == 2:
                    if len(tp_data) >= 3:
                        # Check if TP3 has not been hit, then verify if the target price is reached
                        if not tp_data[2]["price_hit"]:
                            # Multiply by direction and check for a positive result to determine if target has been reached
                            if direction * (current_rate - tp_data[2]["target_price"]) >= 0:
                                tp_data[2]["price_hit"] = True
                                trade.set_custom_data(key="custom_dca", value=custom_dca)
                        # Once TP3 price is hit, check for an RSI dip before exiting
                        if tp_data[2]["price_hit"]:
                            # Use a check for a negative result to identify an RSI reversal against the trade direction
                            if direction * (current_rsi - prev_rsi) < 0:
                                # TP3 Position Reduction
                                tp_reduction = tp_data[2]["position_reduction_pct"]
                                stake_reduction = -trade.stake_amount * tp_reduction
                                if number_of_tp == 3:
                                    # Full exit as the final TP has been hit
                                    return -trade.stake_amount, f"Full exit at TP3/TP{number_of_tp}"
                                else:
                                    # Partial exit
                                    return stake_reduction, "TP3 reduction hit"

                # TP4 - Executes when there are at least 4 TPs and 3 successful exits
                elif number_of_tp >= 4 and trade.nr_of_successful_exits == 3:
                    if len(tp_data) >= 4:
                        # Check if TP4 has not been hit, then verify if the target price is reached
                        if not tp_data[3]["price_hit"]:
                            # Multiply by direction and check for a positive result to determine if target has been reached
                            if direction * (current_rate - tp_data[3]["target_price"]) >= 0:
                                tp_data[3]["price_hit"] = True
                                trade.set_custom_data(key="custom_dca", value=custom_dca)
                        # Once TP4 price is hit, check for an RSI dip before exiting
                        if tp_data[3]["price_hit"]:
                            # Use a check for a negative result to identify an RSI reversal against the trade direction
                            if direction * (current_rsi - prev_rsi) < 0:
                                # TP4 Position Reduction
                                tp_reduction = tp_data[3]["position_reduction_pct"]
                                stake_reduction = -trade.stake_amount * tp_reduction
                                if number_of_tp == 4:
                                    # Full exit as the final TP has been hit
                                    return -trade.stake_amount, f"Full exit at TP4/TP{number_of_tp}"
                                else:
                                    # Partial exit
                                    return stake_reduction, "TP4 reduction hit"

                # TP5 - Executes when there are at least 5 TPs and 4 successful exits
                elif number_of_tp >= 5 and trade.nr_of_successful_exits == 4:
                    if len(tp_data) >= 5:
                        # Check if TP5 has not been hit, then verify if the target price is reached
                        if not tp_data[4]["price_hit"]:
                            # Multiply by direction and check for a positive result to determine if target has been reached
                            if direction * (current_rate - tp_data[4]["target_price"]) >= 0:
                                tp_data[4]["price_hit"] = True
                                trade.set_custom_data(key="custom_dca", value=custom_dca)
                        # Once TP5 price is hit, check for an RSI dip before exiting
                        if tp_data[4]["price_hit"]:
                            # Use a check for a negative result to identify an RSI reversal against the trade direction
                            if direction * (current_rsi - prev_rsi) < 0:
                                # TP5 is the final take profit level, trigger a full exit
                                return -trade.stake_amount, f"Full exit at TP5/TP{number_of_tp}"

            # Handle Trailing Exits
            trailing_condition = False
            if trade.nr_of_successful_exits >= 1:
                if trade.nr_of_successful_exits == 1:
                    # For the first TP, use dynamic trade.open_rate to get averaged entry price.
                    trailing_condition = direction * current_rate <= direction * trade.open_rate
                else:
                    # For additional TPs, use the previous TP as the trigger level
                    # todo should this be -2 or -1
                    previous_order: Order = trade.orders[-2]
                    trailing_condition = (
                        direction * current_rate <= direction * previous_order.price
                    )

                if trailing_condition:
                    return (
                        -trade.stake_amount,
                        f"trailing exit @TP{trade.nr_of_successful_exits - 1}",
                    )

            # Default Return
            return None
        except Exception as e:
            print(f"Error in adjust_trade_position: {e}")
            return None

    def set_tp(
        self, trade: Trade, number_of_tp: int, entry_tag: str, direction: int, leverage: int
    ) -> List[Dict[str, Any]]:
        """
        Calculates take profit (TP) levels and returns a list of dictionaries (tp_data), each representing a TP level.

        Key Features:
        - Uses the current `trade.open_rate` to calculate dynamic `target_price`.
        - Scales position percentages to sum up to 100%.
        - Initializes 'price_hit' flag for TP tracking.
        - Stores unique tags for each TP level.
        """
        # Define the open rate of the trade.
        open_rate: float = trade.open_rate

        # Step 1: Define TP Target Profit Percentages
        tp1_price = getattr(self, f"tp1_price_{entry_tag}").value
        tp2_price = tp1_price + getattr(self, f"tp2_price_{entry_tag}").value
        tp3_price = tp2_price + getattr(self, f"tp3_price_{entry_tag}").value
        tp4_price = tp3_price + getattr(self, f"tp4_price_{entry_tag}").value
        tp5_price = tp4_price + getattr(self, f"tp5_price_{entry_tag}").value

        # Step 2: Apply Leverage
        tp_prices = [tp1_price, tp2_price, tp3_price, tp4_price, tp5_price]
        tp_leveraged = [tp * leverage for tp in tp_prices]

        # Step 3: Retrieve and Calculate Position Size Percentages
        tp1_pct_size = getattr(self, f"tp1_pct_size_{entry_tag}").value
        tp2_pct_size = getattr(self, f"tp2_pct_size_{entry_tag}").value
        tp3_pct_size = getattr(self, f"tp3_pct_size_{entry_tag}").value
        tp4_pct_size = getattr(self, f"tp4_pct_size_{entry_tag}").value
        tp5_pct_size = getattr(self, f"tp5_pct_size_{entry_tag}").value

        tp_pct_sizes = [tp1_pct_size, tp2_pct_size, tp3_pct_size, tp4_pct_size, tp5_pct_size]
        total_pos_pct = sum(tp_pct_sizes[:number_of_tp])

        # Step 4: Calculate Scaling Factor
        if total_pos_pct == 0:
            scaling_factor = 0
        else:
            scaling_factor: float = 1 / total_pos_pct

        # Scale the position percentages
        scaled_tp_pcts = [
            tp_pct_size * scaling_factor for tp_pct_size in tp_pct_sizes[:number_of_tp]
        ]

        # Step 5: Build TP Data and Return
        tp_data = []
        for i in range(number_of_tp):
            profit_pct = tp_leveraged[i]
            position_reduction_pct = scaled_tp_pcts[i]
            target_price = open_rate * (1 + profit_pct * direction / leverage)
            tp_level = {
                "profit_pct": profit_pct,
                "position_reduction_pct": position_reduction_pct,
                "price_hit": False,
                "target_price": target_price,
                "tag": f"TP{i + 1}",
            }
            tp_data.append(tp_level)

        return tp_data

    def calculate_scaled_dca_entries(self, entry_tag: str, trade: Trade) -> List[Dict[str, Any]]:
        """
        Calculates scaled DCA entry sizes and their corresponding price thresholds.

        Returns a list of dictionaries (dca_data), each representing a DCA entry.

        Key Features:
        - Uses the current `trade.open_rate` for calculations.
        - Scales DCA entry sizes to sum up to the total DCA trade size.
        - Initializes 'price_hit' flag for DCA tracking.
        - Stores unique tags for each DCA entry.
        """
        # Retrieve Strategy Parameters
        stoploss: float = getattr(self, f"stoploss_{entry_tag}").value
        last_dca_entry_dist_from_SL: float = getattr(
            self, f"last_dca_entry_dist_from_SL_{entry_tag}"
        ).value
        leverage: float = getattr(self, f"leverage_{entry_tag}").value * 3
        number_of_dca_entries: int = getattr(
            self, f"number_of_additional_entries_{entry_tag}"
        ).value
        initial_entry_price: float = trade.open_rate  # Use the current open_rate
        direction: int = -1 if trade.is_short else 1

        # Calculate Maximum Allowable Price Movement
        furthest_dca_pct_away_from_open_rate: float = (
            stoploss - last_dca_entry_dist_from_SL
        ) / leverage
        maximum_dca_entry_rate: float = initial_entry_price * (
            1 - direction * furthest_dca_pct_away_from_open_rate
        )
        total_range: float = abs(initial_entry_price - maximum_dca_entry_rate)

        # Retrieve and Aggregate DCA Entry Prices
        dca_entry_prices: List[float] = []
        for i in range(2, number_of_dca_entries + 2):
            dca_price: float = getattr(self, f"dca_entry{i}_price_{entry_tag}").value
            dca_entry_prices.append(dca_price)

        # Calculate Cumulative DCA Entry Prices
        cumulative_dca_prices: List[float] = []
        cumulative_sum: float = 0.0
        for price_drop in dca_entry_prices:
            cumulative_sum += price_drop
            cumulative_dca_prices.append(cumulative_sum)

        total_dca_price: float = cumulative_sum
        if total_dca_price == 0:
            total_dca_price = 1  # To prevent division by zero

        # Scale Cumulative DCA Entry Prices
        scaled_cumulative_dca_prices: List[float] = [
            (price_drop / total_dca_price) * total_range for price_drop in cumulative_dca_prices
        ]

        # Calculate Scaled DCA Entry Prices
        scaled_dca_entry_prices: List[float] = [
            initial_entry_price - direction * scaled_drop
            for scaled_drop in scaled_cumulative_dca_prices
        ]

        # Sort Scaled DCA Entry Prices
        scaled_dca_entry_prices = sorted(scaled_dca_entry_prices, reverse=(direction == -1))

        # Retrieve and Scale DCA Entry Sizes
        dca_size_multiplier: int = getattr(self, f"set_dca_trade_size_{entry_tag}").value
        set_dca_trade_size: float = dca_size_multiplier * trade.orders[0].stake_amount

        total_dca_size: float = sum(dca_entry_prices)
        if total_dca_size == 0:
            total_dca_size = 1  # To prevent division by zero

        dca_entry_sizes: List[float] = []
        for dca_size in dca_entry_prices:
            scaled_size: float = (dca_size / total_dca_size) * set_dca_trade_size
            dca_entry_sizes.append(scaled_size)

        # Prepare DCA Data
        dca_data = []
        for idx in range(number_of_dca_entries):
            dca_entry = {
                "entry_num": idx + 2,  # Entries start from 2
                "price": scaled_dca_entry_prices[idx],
                "size": dca_entry_sizes[idx],
                "price_hit": False,
                "tag": f"safety_order_{idx + 2}",
            }
            dca_data.append(dca_entry)

        return dca_data

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        if after_fill:
            # Get dataframe and current candle data in case we want to analyse anything from it
            dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            current_candle = dataframe.iloc[-1].squeeze()
            current_candle_index = dataframe.index[-1]
            # To access the enter tag reason to decide what rules to apply we use the following data
            entry_tag = trade.enter_tag

            # After an additional order, start with a stoploss of 10% below the new open rate
            # Open rate its averaged across all entries
            stoploss = getattr(self, f"stoploss_{entry_tag}").value
            SL_Open = stoploss_from_open(
                stoploss, current_profit, is_short=trade.is_short, leverage=trade.leverage
            )
            SL_ABS = stoploss_from_open(
                stoploss, current_profit, is_short=trade.is_short, leverage=trade.leverage
            )
            if SL_Open < SL_ABS:
                print(f"SL_Open: {SL_Open} SL_ABS: {SL_ABS}")
                return SL_Open
        # Make sure you have the longest interval first - these conditions are evaluated from top to bottom.
        if current_time - timedelta(days=20) > trade.open_date_utc:
            # Use the initial stoploss for the first X minutes, after this change to 10% trailing stoploss, and after
            # x days we use a trailing1 trailing stoploss. If an additional order fills, set stoploss to -10%
            # below the new open_rate (Averaged across all entries).
            trailing = -0.05
            return trailing
        elif current_time - timedelta(days=10) > trade.open_date_utc:
            trailing = -0.1
            return trailing
        return None
