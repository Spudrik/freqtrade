# helper_functions.py

import numpy as np  # noqa
import pandas as pd  # noqa
import plotly.graph_objects as go
import importlib
from plotly.subplots import make_subplots
# Turns off the fragmented dataframe warning.
from warnings import simplefilter
from freqtrade.persistence import Trade
from datetime import datetime
from typing import Optional, Union, Tuple
import os
from functools import reduce
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def define_info_intervals(timeframe, coin_pair_intervals):
    enumerate(coin_pair_intervals)
    compatible_intervals = []
    # Now we iterate though the list of intervals by enumerating each position in the list.
    # If any of those enumerated list positions match the timeframe then we set
    # the index to the values enumerated position. An enumerated list is
    # like a dictionary where the numerical position of the list data is the key
    index = 0
    for idx, val in enumerate(coin_pair_intervals):
        if timeframe == val:
            index = idx
    for idx, val in enumerate(coin_pair_intervals):
        if index > idx:
            compatible_intervals.append(val)
    return compatible_intervals

def plotly_graph(dataframe, pair_tf='Coin Not Defined'):
    # Define subplots setup
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, subplot_titles=('OHLC and Indicators', 'Flags', 'Oscillators', 'Volume'),
                        row_heights=[0.4, 0.2, 0.2, 0.2])

    # Add candlestick trace to the figure
    fig.add_trace(
        go.Candlestick(
            x=dataframe['date'],
            open=dataframe['open'],
            high=dataframe['high'],
            low=dataframe['low'],
            close=dataframe['close'],
            name='OHLC',
            hoverinfo="text"
        ),
        row=1, col=1
    )
    if 'pivot_high_price' in dataframe.columns:
        fig.add_trace(go.Scatter(x=dataframe['date'],
                                 y=dataframe['pivot_high_price'],
                                 mode='markers',
                                 name='Pivot High',
                                 marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)

    if 'pivot_low_price' in dataframe.columns:
        fig.add_trace(go.Scatter(x=dataframe['date'],
                                 y=dataframe['pivot_low_price'],
                                 mode='markers',
                                 name='Pivot Low',
                                 marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

    # Add indicators to the figure
    flag_columns = [col for col in dataframe.columns if '_flg' in col or 'market_cond' in col or 'bb_consol' in col or 'bb_breaking' in col or 'bollinger_contracting' in col or 'bollinger_volatile' in col or 'bbw_cross' in col or 'bbw_high' in col or 'bbw_low' in col]
    oscillator_columns = [col for col in dataframe.columns if '_osc' in col or 'chop' in col or 'rsi' in col or 'bb_width' in col or 'atr' in col or 'adx' in col or 'vsa' in col or 'uo' in col or 'mfi' in col or 'pvt' in col or 'pct' in col or 'pct' in col]
    volume_columns = [col for col in dataframe.columns if 'volume' in col or '_vol_' in col.lower()]

    # Exclusions from the OHLC plot
    exclude_from_ohlc = set(flag_columns + oscillator_columns + volume_columns)

    # Add flags to the Flags subplot
    if len(flag_columns) > 0:
        for flag in flag_columns:
            fig.add_trace(go.Scatter(x=dataframe['date'], y=dataframe[flag], mode='lines+markers', name=flag), row=2,
                          col=1)
    # Add oscillators to the oscillators subplot
    if len(oscillator_columns) > 0:
        for osc in oscillator_columns:
            fig.add_trace(go.Scatter(x=dataframe['date'], y=dataframe[osc], mode='lines+markers', name=osc), row=3,
                          col=1)
    # Add Volume-related data to the Volume subplot
    if len(volume_columns) > 0:
        for vol in volume_columns:
            if vol == 'volume':
                # For the explicit 'volume' column, use a bar chart
                fig.add_trace(go.Bar(
                    x=dataframe['date'],
                    y=dataframe[vol],
                    name=vol
                ), row=4, col=1)
            else:
                # For all other volume columns, use line plots as before
                fig.add_trace(go.Scatter(
                    x=dataframe['date'],
                    y=dataframe[vol],
                    mode='lines',
                    name=vol
                ), row=4, col=1)

    # Add remaining non-flag, non-oscillator, and non-volume numerical columns to the OHLC plot for better context
    other_columns = [col for col in dataframe.columns if
                     col not in exclude_from_ohlc and col not in ['date', 'open', 'high', 'low', 'close']]
    for col in other_columns:
        fig.add_trace(go.Scatter(x=dataframe['date'], y=dataframe[col], mode='lines', name=col), row=1, col=1)

    # Update layout to handle multiple subplots and controls
    fig.update_layout(title=f'{pair_tf}',
                      xaxis_rangeslider_visible=False,
                      xaxis2_rangeslider_visible=False,
                      xaxis3_rangeslider_visible=False,
                      xaxis4_rangeslider_visible=False,
                      height=800, width=1200)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)

    # Display the plot
    fig.show()

def normalize(series, min_value=None, max_value=None):
    if min_value is None:
        min_value = series.min()
    if max_value is None:
        max_value = series.max()
    normalized = (series - min_value) / (max_value - min_value)
    return normalized.clip(0, 1)

def calculate_slope(series, window):
    return series.rolling(window=window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)

def save_dca_trade_data(self, trade, current_time,
                        current_rate, current_profit,
                        min_stake, max_stake,
                        current_entry_rate, current_exit_rate,
                        current_entry_profit, current_exit_profit,
                        print_enabled: False, df_expand: False,
                        **kwargs):
    """
    Processes and logs DCA (Dollar-Cost Averaging) trade data by extracting technical indicators and trade attributes,
    compiling them into a custom DataFrame, and saving the data to a CSV file.

    This function performs the following steps:
    - Extracts the last and previous candle data for the trade's pair from the analyzed DataFrame.
    - Constructs a custom DataFrame (`self.trade_data`) by combining trade attributes and the last candle data.
    - Reorganizes and updates the DataFrame columns according to a user defined, hard coded order.
    - Sorts the DataFrame by 'id' and 'candle_count', and resets the index.
    - This resolves issues with all trades being asynchronous and keeps the resulting dataframe and csv organised.
    - Saves the updated DataFrame to a CSV file in a directory named with the current date and hour.

    Parameters:
    - trade: Trade
        The Trade object containing information about the current trade.
    - current_time: datetime
        The current datetime.
    - current_rate: float
        The current market rate for the trade's pair.
    - current_profit: float
        The current profit of the trade.
    - min_stake: float
        The minimum stake amount.
    - max_stake: float
        The maximum stake amount.
    - current_entry_rate: float
        The current entry rate for the trade.
    - current_exit_rate: float
        The current exit rate for the trade.
    - current_entry_profit: float
        The current profit from entry signals.
    - current_exit_profit: float
        The current profit from exit signals.
    - print_enabled: bool, optional
        Flag to enable or disable printing (default False).
        - Future expansion intends to add an option to print the current
            trade (candle) each iteration, for debugging purposes
    - **kwargs:
        Additional keyword arguments.

    Returns:
    - None

    Notes:
    - Suppresses pandas warnings about fragmented dataframes and future changes.
    - Increments `self.trade_counter` for each call.
    - Assumes `self.trade_data` is a pandas DataFrame initialized elsewhere in the class.
    - The CSV file is saved to a directory:
      `'C:\\FreqTradeStuff\\freqtrade\\user_data\\exported_grid_bot_data\\<YYYY-MM-DD-HH>'`.
    - Handles missing columns in the DataFrame by adding them with `None` values.
    - Organizes DataFrame columns according to a `primary_columns` list to ensure consistent ordering.
    - list(last_candle.index) is appended to the primary_columns of the trade data. This is all the standard freqtrade
        indicator columns like rsi and moving averages. With the resulting CSV containing tick by tick data processed
        by the DCA method for each trade, with all their indicators stored at the end.
    """

    # Turns off the fragmented dataframe warning.
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    ####################### START MY CODE #####################################
    # Abbreviate names for easy access to critical coin pair values
    pair = trade.pair
    pair = pair.replace('/USDT:USDT', '')
    coin1w = f"{pair}_3d"
    coin1d = f"{pair}_1d"
    coin1h = f"{pair}_1h"
    coin15m = f"{pair}_4h"
    btc1w = f"BTC_3d"
    btc1d = f"BTC_1d"
    btc1h = f"BTC_1h"
    btc15m = f"BTC_4h"

    ####################### START EXTRACTING TECHNICAL DATAFRAME CANDLE DATA #####################################

    # Obtain pair dataframe (just to show how to access it)
    dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
    # Ensure dataframe is not empty to avoid errors
    if not dataframe.empty:
        # Access the last candle using tail and extract its index directly
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()  # Make sure there are at least two candles

        # Get the actual index of the last candle
        candle_count = dataframe.tail(1).index.item()
        ####################### END EXTRACTING TECHNICAL DATAFRAME CANDLE DATA #####################################

        ####################### START GENERATE CUSTOM TRADE DATAFRAME FROM FREQTRADE "TRADE" INFORMATION #################
        self.trade_counter += 1

        # Create data dictionary starting with candle count and last candle data
        data = {'candle_count': candle_count}
        data.update(last_candle.to_dict())
        # Extract and filter trade object attributes
        attributes = [attr for attr in dir(trade) if not attr.startswith('__') and not callable(getattr(trade, attr))]
        trade_attrs = {attr: getattr(trade, attr, None) for attr in attributes if
                       isinstance(getattr(trade, attr, None), (str, float, int))}
        data.update(trade_attrs)

        # Add the current price to the data dictionary (from last candle)
        data['current_price'] = last_candle['close']

        ####################### DATAFRAME REORGANIZATION #####################
        # Define the new primary columns list with updated order
        if df_expand == False:
            primary_columns = ['candle_count', 'id', 'id_candle_count', 'base_currency', 'current_price', 'open_rate',
                               'price_diff_pct',
                               'liquidation_price', 'realized_profit', 'open_trade_value',
                               'nr_of_successful_entries', 'nr_of_successful_exits', 'amount',
                               'buy_tag', 'entry_side', 'exit_reason', 'initial_stop_loss',
                               'initial_stop_loss_pct', 'stake_amount', 'stop_loss',
                               'stop_loss_pct', 'stoploss_or_liquidation', 'total_profit'
                               ] + list(last_candle.index)
        else:
            primary_columns = ['candle_count', 'id', 'id_candle_count', 'base_currency', 'current_price', 'open_rate',
                               'price_diff_pct',
                               'liquidation_price', 'realized_profit', 'open_trade_value',
                               'nr_of_successful_entries', 'nr_of_successful_exits', 'amount',
                               'buy_tag', 'entry_side', 'exit_reason', 'initial_stop_loss',
                               'initial_stop_loss_pct', 'stake_amount', 'stop_loss',
                               'stop_loss_pct', 'stoploss_or_liquidation', 'total_profit'
                               ]
            # Add any columns from trade_attrs not already in primary_columns
            additional_attrs = [attr for attr in trade_attrs.keys() if attr not in primary_columns]
            primary_columns += additional_attrs
            primary_columns += list(last_candle.index)

        # Remove any duplicate entries in column order due to overlap between last_candle and trade attributes
        primary_columns = list(dict.fromkeys(primary_columns))

        # Drop all columns that are not in primary_columns from the DataFrame
        columns_to_drop = [col for col in self.trade_data.columns if col not in primary_columns]
        self.trade_data.drop(columns=columns_to_drop, inplace=True)

        # Ensure the DataFrame has columns in the specified order of primary_columns
        # Note: This step will ensure that only the columns listed in primary_columns are in the DataFrame and in the correct order.
        # Missing columns in self.trade_data that are specified in primary_columns will cause an error unless handled.
        # To handle potentially missing columns, you can add a check to add empty columns if they are missing.

        # First turn of pd wanrings about defragmented dataframes
        simplefilter(action="ignore", category=pd.errors.DtypeWarning)
        # Filter warnings from pandas future version changes
        simplefilter(action='ignore', category=FutureWarning)

        missing_columns = [col for col in primary_columns if col not in self.trade_data.columns]
        for col in missing_columns:
            self.trade_data[col] = None  # or np.nan, depending on your requirement for missing data

        self.trade_data = self.trade_data[primary_columns]


        ###### Populate Rows ########
        """
        Adds a new row to the trade_data dataframe, matching the columns
        with the keys from the `data` dictionary.
        """
        # Create a new row dictionary with matching columns
        new_row = {col: data.get(col, None) for col in primary_columns}

        # Convert new_row dictionary to a DataFrame
        new_row_df = pd.DataFrame([new_row])

        # Concatenate the new row to the trade_data DataFrame
        self.trade_data = pd.concat([self.trade_data, new_row_df], ignore_index=True)

        ####################### ADDITIONAL ROW SORTING #####################
        # Sort the DataFrame by 'id' and then by 'candle_count' chronologically
        self.trade_data.sort_values(by=['id', 'candle_count'], inplace=True)

        # Reset the index to ensure it's sequential after sorting
        self.trade_data.reset_index(drop=True, inplace=True)
        ############## END OF GENERATE CUSTOM TRADE DATAFRAME FROM FREQTRADE "TRADE" INFORMATION #################
        ########################## START OF SAVE CUSTOM TRADE DATAFRAME ###########################

        # Obtain today's date in the format 'YYYY-MM-DD'
        today_date = datetime.now().strftime("%Y-%m-%d")

        # Obtain the current hour and rounded-down minutes
        current_hour = datetime.now().strftime("%H")
        current_min = (datetime.now().minute // 10) * 10

        # Create a folder path with the new structure: year/month/day/hour/mins
        folder_path = f'C:\\FreqTradeStuff\\freqtrade\\user_data\\exported_grid_bot_data\\{today_date}\\{current_hour}-hour\\{current_min:02d}-mins'

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Create a filename with a datetime stamp
        filename = "TradeOutput.csv"

        # Complete path to save the file
        file_path = os.path.join(folder_path, filename)

        # Save the DataFrame as a CSV file
        self.trade_data.to_csv(file_path, index=False)

        # print(f"\nFile saved at {file_path}")

        ########################## END OF SAVE CUSTOM TRADE DATAFRAME ############################