# informative_pairs.py

import numpy as np  # noqa
import pandas as pd  # noqa
# Turns off the fragmented dataframe warning.
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def informative_pairs(self):
    # todo - think this is redundant as I load the dataprovided within basic_new_strat populate_indicators and just pull in the pairs as I need them
    """
    Define additional informative pair/timeframe combinations to be cached from the exchange.

    This function retrieves all pairs available in the whitelist and assigns multiple
    timeframes to each pair. These combinations are used by the strategy for analysis
    but are non-tradeable unless they are part of the whitelist.

    :return: List of tuples in the format (pair, timeframe)
    """
    pairs = self.dp.current_whitelist()
    # Assign tf to each pair, so they can be downloaded and cached for strategy.
    informative_pairs = [(pair, '1h') for pair in pairs]
    informative_pairs += ((pair, '4h') for pair in pairs)
    informative_pairs += ((pair, '1d') for pair in pairs)
    informative_pairs += ((pair, '3d') for pair in pairs)
    informative_pairs += ((pair, '1w') for pair in pairs)

    return informative_pairs