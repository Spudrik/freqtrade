# apply_indicators.py

import talib.abstract as ta
from technical import qtpylib
import os
import pandas as pd
import numpy as np#
from freqtrade.user_data.strategies.price_action import analyze_price_action
from freqtrade.user_data.strategies.helper_functions import plotly_graph

"""
This module contains the `apply_indicators` function, which applies various technical indicators to a given dataframe.
The function is used to add additional columns representing indicator values such as RSI, EMA, MACD, ADX, ATR, Bollinger Bands, 
and volume-based metrics to the target coin, informative timeframes, and BTC pairs.
"""

def apply_indicators(self, dataframe, pair_tf, target_btc=False, full_set=0):
    """Applies multiple technical indicators to the given dataframe for a specified pair timeframe.

        Args:
            dataframe (DataFrame): The pandas dataframe containing market data.
            pair_tf (str): The pair and timeframe string (e.g., BTC_1h) used to determine applicable indicators.
            full_set (int, optional): Indicates if the full set of indicators should be applied.

        Returns:
            DataFrame: The updated dataframe containing indicator columns.
     """

    # Check if the pair_tf ends with the same timeframe as the strategy's timeframe
    # and is not BTC_{self.timeframe}. Unless the target coin is BTC itself.
    # If true, append _{pair_tf} to each indicator column.
    append_suffix = (
        (pair_tf.endswith(f'_{self.timeframe}') and target_btc) or
        (pair_tf.endswith(f'_{self.timeframe}') and not pair_tf.startswith(f'BTC_{self.timeframe}'))
    )

    # TODO untested ideas from ChatGPT
    # analyze_price_action(dataframe)
    # Calculate the mean of absolute percentage changes of the close price
    dataframe['pct_change'] = dataframe['close'].pct_change().abs() * 100
    mean_pct_change = dataframe['pct_change'].mean()
    # # Bollinger Bands
    # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=30, stds=2)
    # dataframe['bb_lowerband'] = bollinger['lower']
    # dataframe['bb_middleband'] = bollinger['mid']
    # dataframe['bb_upperband'] = bollinger['upper']
    #
    # # Calculate Bollinger Bandwidth
    # # Calculate rolling average of Bollinger Bandwidth
    # dataframe['bbw'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
    # # Calculate the mean and standard deviation of Bollinger Bandwidth
    # mean_bbw = dataframe['bbw'].mean()
    # std_dev_bbw = dataframe['bbw'].std()
    #
    # # add lowest_contraction for bb width for last 20 and 100 candles
    # dataframe['bbw_lowest_20'] = dataframe['bbw'].rolling(window=20).min()
    # dataframe['bbw_lowest_100'] = dataframe['bbw'].rolling(window=100).min()
    # # add highest expansion for bbw for last 20 and 100 candles
    # dataframe['bbw_highest_20'] = dataframe['bbw'].rolling(window=20).max()
    # dataframe['bbw_highest_100'] = dataframe['bbw'].rolling(window=100).max()
    # # add a flag for when we cross above highest and a seperate flag column for when we cross below
    # dataframe['bbw_highest_below_mean_100'] = (dataframe['bbw_highest_100'] <= mean_bbw).astype(int)
    # dataframe['bbw_cross_below_low_100'] = qtpylib.crossed_below(dataframe['bbw'],
    #                                                                 dataframe['bbw_lowest_100'].shift(1)).astype(int)
    # dataframe['bbw_highest_below_mean_20'] = (dataframe['bbw_highest_20'] <= mean_bbw).astype(int)
    # dataframe['bbw_cross_below_low_20'] = qtpylib.crossed_below(dataframe['bbw'],
    #                                                                 dataframe['bbw_lowest_20'].shift(1)).astype(int)
    # dataframe['bbw_lowest_above_mean_20'] = (dataframe['bbw_lowest_20'] >= mean_bbw).astype(int)
    # dataframe['bbw_lowest_above_mean_100'] = (dataframe['bbw_lowest_100'] >= mean_bbw).astype(int)
    #
    # dataframe['bbw_mean20'] = dataframe['bbw'].rolling(window=20).mean()
    # # Identify consolidation and breakout
    # dataframe['bb_consolidating'] = (dataframe['bbw'] < dataframe['bbw_mean20']).astype(int)
    # dataframe['bb_breaking_out'] = qtpylib.crossed_above(dataframe['bbw'], dataframe['bbw_mean20']).astype(int)

    # plotly_graph(dataframe, pair_tf)
    volatility_calculations(dataframe, pair_tf, window=50, print_to_file=False)

    # Momentum Indicators
    # ------------------------------------
    # RSI Trigger
    rsi5 = ta.RSI(dataframe, timeperiod=5)

    # set trigger flag when a low pivot is detected.
    dataframe.loc[qtpylib.crossed_above(rsi5, rsi5.shift(1)), 'trig_rsi5_bull'] = True
    dataframe.loc[qtpylib.crossed_below(rsi5, rsi5.shift(1)), 'trig_rsi5_bear'] = True


    for window in self.rsi_common.opt_range:
        dataframe[f'rsi_{window}'] = ta.RSI(dataframe, timeperiod=window)
    # Used for volume guards analysis
    for window in self.rsi_common.opt_range:
        dataframe[f"rsi_vol_{window}"] = ta.RSI(dataframe['volume'], timeperiod=window)

    #  # Use moving averages to set market conditions
    support_resistance_bands(self, dataframe)
    # Use moving averages to set market conditions
    market_conditions(self, dataframe)

    # ADX - Calculate ADX to quantify trend strength
    dataframe['adx'] = ta.ADX(dataframe)

    # ATR - Calculate ATR for market volatility
    dataframe['atr'] = ta.ATR(dataframe)

    # Ultimate Oscillator (UO) - Combines short, medium, and long-term price action with volume
    dataframe['uo'] = ta.ULTOSC(dataframe, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # Money Flow Index (MFI) - Volume-weighted RSI-like oscillator
    dataframe['mfi'] = ta.MFI(dataframe)

    if append_suffix:
        exclude_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.columns = [
            f"{col}_{pair_tf}" if col not in exclude_columns else col for col in dataframe.columns
        ]

    # # Debug
    # plotly_graph(dataframe, pair_tf)

    return dataframe

def market_conditions(self, dataframe):
    ema3 = ta.EMA(dataframe, timeperiod=3)
    ema5 = ta.EMA(dataframe, timeperiod=5)
    ema7 = ta.EMA(dataframe, timeperiod=7)
    ema10 = ta.EMA(dataframe, timeperiod=10)
    ema20 = ta.EMA(dataframe, timeperiod=20)
    ema30 = ta.EMA(dataframe, timeperiod=30)



    # Set conditions flags for breakout trends
    ma5_abv_10 = (ema5 > ema10)
    ma10_abv_20 = (ema10 > ema20)
    ma20_abv_30 = (ema20 > ema30)

    ma5_bel_10 = (ema5 < ema10)
    ma10_bel_20 = (ema10 < ema20)
    ma20_bel_30 = (ema20 < ema30)

    # Creating optional bull trigger flags
    dataframe['trig_3-5_bull'] = qtpylib.crossed_above(ema3, ema5).astype(int)
    dataframe['trig_5-7_bull'] = qtpylib.crossed_above(ema5, ema7).astype(int)
    dataframe['trig_3-7_bull'] = qtpylib.crossed_above(ema3, ema7).astype(int)
    # Creating optional bear trigger flags
    dataframe['trig_3-5_bear'] = qtpylib.crossed_below(ema3, ema5).astype(int)
    dataframe['trig_5-7_bear'] = qtpylib.crossed_below(ema5, ema7).astype(int)
    dataframe['trig_3-7_bear'] = qtpylib.crossed_below(ema3, ema7).astype(int)

    # Creating Bull Breakout conditions
    dataframe['bull_short_flg'] = (ma5_abv_10 & ma10_abv_20 & ma20_abv_30).astype(int)
    dataframe['bull_med_flg'] = (ma10_abv_20 & ma20_abv_30).astype(int)

    # Creating Bear Breakout conditions
    dataframe['bear_short_flg'] = (ma5_bel_10 & ma10_bel_20 & ma20_bel_30).astype(int)
    dataframe['bear_med_flg'] = (ma10_bel_20 & ma20_bel_30).astype(int)
    # Detect changes in conditions to trigger flags
    dataframe['trig_short_bull'] = qtpylib.crossed_above(dataframe['bull_short_flg'], dataframe['bull_short_flg'].shift(1)).astype(int)
    dataframe['trig_med_bull'] = qtpylib.crossed_above(dataframe['bull_med_flg'], dataframe['bull_med_flg'].shift(1)).astype(int)
    dataframe['trig_short_bear'] = qtpylib.crossed_below(dataframe['bear_short_flg'], dataframe['bear_short_flg'].shift(1)).astype(int)
    dataframe['trig_med_bear'] = qtpylib.crossed_below(dataframe['bear_med_flg'], dataframe['bear_med_flg'].shift(1)).astype(int)


def support_resistance_bands(self, dataframe, start=3, step=6):
    """
        Determines if each candle's close value is above, within, or below the max and min bands
        for specified ranges of past candles. Adds a single column to the dataframe for each range
        with values indicating the status:
            2  -> Above max band
            1  -> Within max band
            0  -> Below max band and above min band
           -1  -> Within min band
           -2  -> Below min band

        Timeframes covered for different ranges with different start and window sizes:

        | Start = 2, Window = 4 |
        | Timeframe | Range 1       | Range 2       | Range 3       | Range 4       | Range 5       |
        |-----------|---------------|---------------|---------------|---------------|---------------|
        | 30m       | 1HR to 3HR    | 4HR to 6HR    | 7HR to 9HR    | 10HR to 12HR  | 13HR to 15HR  |
        | 1h        | 2HR to 5HR    | 6HR to 9HR    | 10HR to 13HR  | 14HR to 17HR  | 18HR to 21HR  |
        | 4h        | 8HR to 16HR   | 16HR to 1d    | 1d 4HR to 2d  | 2d 4HR to 3d  | 3d 4HR to 4d  |
        | 1d        | 2d to 4d      | 5d to 7d      | 8d to 10d     | 11d to 13d    | 14d to 16d    |
        | 3d        | 6d to 12d     | 13d to 18d    | 19d to 24d    | 25d to 30d    | 31d to 36d    |

        | Start = 2, Window = 6 |
        | Timeframe | Range 1       | Range 2       | Range 3       | Range 4       | Range 5       |
        |-----------|---------------|---------------|---------------|---------------|---------------|
        | 30m       | 1HR to 4HR    | 5HR to 8HR    | 9HR to 12HR   | 13HR to 16HR  | 17HR to 20HR  |
        | 1h        | 2HR to 7HR    | 8HR to 13HR   | 14HR to 19HR  | 20HR to 25HR  | 26HR to 31HR  |
        | 4h        | 8HR to 1d     | 1d 4HR to 2d  | 2d 8HR to 3d  | 3d 12HR to 4d | 4d 16HR to 5d |
        | 1d        | 2d to 5d      | 6d to 9d      | 10d to 13d    | 14d to 17d    | 18d to 21d    |
        | 3d        | 6d to 15d     | 18d to 27d    | 30d to 39d    | 42d to 51d    | 54d to 63d    |

        | Start = 4, Window = 6 |
        | Timeframe | Range 1       | Range 2       | Range 3       | Range 4       | Range 5       |
        |-----------|---------------|---------------|---------------|---------------|---------------|
        | 30m       | 2HR to 5HR    | 6HR to 9HR    | 10HR to 13HR  | 14HR to 17HR  | 18HR to 21HR  |
        | 1h        | 4HR to 9HR    | 10HR to 15HR  | 16HR to 21HR  | 22HR to 27HR  | 28HR to 33HR  |
        | 4h        | 16HR to 1d    | 1d 4HR to 2d  | 2d 4HR to 3d  | 3d 8HR to 4d  | 4d 8HR to 5d  |
        | 1d        | 4d to 7d      | 8d to 11d     | 12d to 15d    | 16d to 19d    | 20d to 23d    |
        | 3d        | 9d to 18d     | 21d to 30d    | 33d to 42d    | 45d to 54d    | 57d to 66d    |

        | Start = 4, Window = 8 |
        | Timeframe | Range 1       | Range 2       | Range 3       | Range 4       | Range 5       |
        |-----------|---------------|---------------|---------------|---------------|---------------|
        | 30m       | 2HR to 6HR    | 7HR to 11HR   | 12HR to 16HR  | 17HR to 21HR  | 22HR to 26HR  |
        | 1h        | 4HR to 11HR   | 12HR to 19HR  | 20HR to 27HR  | 28HR to 35HR  | 36HR to 43HR  |
        | 4h        | 16HR to 1d    | 1d 8HR to 2d  | 2d 8HR to 3d  | 3d 8HR to 4d  | 4d 16HR to 5d |
        | 1d        | 4d to 8d      | 9d to 13d     | 14d to 18d    | 19d to 23d    | 24d to 28d    |
        | 3d        | 9d to 21d     | 24d to 36d    | 39d to 51d    | 54d to 66d    | 69d to 81d    |
    """

    ranges = [
        {'name': 'range1', 'start_shift': start, 'window_size': step},   # Ex: Candles from i-2 to i-14
        {'name': 'range2', 'start_shift': (start+step), 'window_size': step},  # Ex: Candles from i-16 to i-30
        {'name': 'range3', 'start_shift': (start+(step*2)), 'window_size': step},  # Ex: Candles from i-31 to i-45
        {'name': 'range4', 'start_shift': (start+(step*3)), 'window_size': step},  # Ex: Candles from i-46 to i-60
        {'name': 'range5', 'start_shift': (start+(step*4)), 'window_size': step},  # Ex: Candles from i-60 to i-99
    ]

    # Calculate the rolling mean of the difference between 'open' and 'close' columns
    mean_diff = (dataframe['open'] - dataframe['close']).abs().rolling(window=20).mean() / 3
    current_close = dataframe['close']

    for i, r in enumerate(ranges, start=1):
        start_shift = r['start_shift']
        window_size = r['window_size']

        # Get the rolling window for max and min calculations on shifted close data
        shifted_close = dataframe['close'].shift(start_shift)
        rolling_common = shifted_close.rolling(window=window_size)

        # Calculate peak max and min for the range
        pkMAX = rolling_common.max()
        pkMIN = rolling_common.min()

        # Calculate upper and lower bands for pkMAX and pkMIN
        pkMAX_upper_band = pkMAX + mean_diff
        pkMAX_lower_band = pkMAX - mean_diff
        pkMIN_upper_band = pkMIN + mean_diff
        pkMIN_lower_band = pkMIN - mean_diff

        # todo - add a check to make sure range value didn't just change, if it did don't set the value a# it's possibly a false trigger due to shifting range


        # Define conditions and choices for np.select
        conditions = [
            (current_close > pkMAX_upper_band),  # Above max band
            (current_close >= pkMAX_lower_band) & (current_close <= pkMAX_upper_band),  # Within max band
            (current_close < pkMAX_lower_band) & (current_close > pkMIN_upper_band),  # Below max band and above min band
            (current_close >= pkMIN_lower_band) & (current_close <= pkMIN_upper_band),  # Within min band
            (current_close < pkMIN_lower_band)  # Below min band
        ]
        choices = [2, 1, 0, -1, -2]


        # Create the column with np.select
        dataframe[f'range{i}'] = np.select(conditions, choices, default=np.nan)

    return dataframe


def volatility_calculations(dataframe, pair_tf, window=50, print_to_file=False):
    """
    Calculates and adds various volatility metrics to the given DataFrame.

    This function takes a DataFrame with columns 'close', 'high', and 'low' and computes several
    volatility-related metrics over a specified rolling window (default is 50 periods). The following columns are added:

    1. 'volatility': Rolling mean of the percentage change between 'high' and 'low' prices over the past window periods.
       Represents intraday volatility in percentage terms.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing 'close', 'high', and 'low' columns.
        pair_tf (str): A string representing the coin pair and timeframe, e.g., 'BTC_1h'.
        window (int): The rolling window period for calculations. Default is 50.

    Returns:
        pd.DataFrame: The input DataFrame with new columns representing different volatility metrics.
    """
    # 1. Average High-Low Percentage Change (rolling window)
    # This calculates the percentage change between the open/close for each period and then takes
    # the rolling mean over the specified window to gauge intraday volatility in percentage terms.
    dataframe['volatility'] = (abs(dataframe['close'] - dataframe['open']) / dataframe['open']) * 100

    if print_to_file == True:
        # Extract coin and timeframe from pair_tf
        coin, timeframe = pair_tf.split('_')
        coin_pair = f"{coin}/USDT:USDT"

        # Save high-low percentage change results to CSV file

        csv_filename = "C:\\FreqTradeStuff\\freqtrade\\user_data\\data\\binance\\volatility_metrics.csv"
        new_data = {
            'coin_pair': [coin_pair],
            'timeframe': [timeframe],
            'volatility': [dataframe['volatility'].iloc[-1]]  # Store the latest rolling value
        }

        new_data_df = pd.DataFrame(new_data)

        # If the file exists, check if the combination of coin_pair and timeframe already exists
        if os.path.exists(csv_filename):
            existing_df = pd.read_csv(csv_filename)
            if not ((existing_df['coin_pair'] == coin_pair) & (existing_df['timeframe'] == timeframe)).any():
                new_data_df.to_csv(csv_filename, mode='a', header=False, index=False)
        else:
            new_data_df.to_csv(csv_filename, mode='w', header=True, index=False)

    return dataframe
