# hyperparams_mixin.py
from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, RealParameter

class HyperoptParamsMixin:
    """
    Mixin holding all hyperopt parameters.
    Import and inherit this in your main strategy to keep it tidy.
    """

    # --- RSI / directional params ---
    long_rsi = IntParameter(low=1, high=60, default=30, space='buy', optimize=True, load=True)
    exit_long_rsi = IntParameter(low=40, high=100, default=70, space='sell', optimize=True, load=True)

    short_rsi = IntParameter(low=1, high=100, default=70, space='sell', optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=100, default=30, space='buy', optimize=True, load=True)

    buy_rsi = IntParameter(low=1, high=100, default=30, space='buy', optimize=True, load=True)

    # --- Bollinger (fixed / not optimized here) ---
    # Note: You have 10 listed twice; that's allowed but redundant.
    bollinger_window = CategoricalParameter([5, 10, 10], default=5, space="buy", optimize=False, load=False)
    bollinger_std = CategoricalParameter([1, 1.5, 2], default=2, space="buy", optimize=False, load=False)

    # --- Exits ---
    exit_pct = DecimalParameter(low=0.001, high=0.01, default=0.001, decimals=3, space='buy', optimize=True, load=True)

    # --- Grid config (fixed / not optimized here) ---
    total_grid_steps = IntParameter(low=5, high=200, default=5, space='buy', optimize=False, load=False)
    positive_grid_ratio = CategoricalParameter([0.1, 0.25, 0.5, 0.75, 0.9], default=0.5, space="buy", optimize=False, load=False)
    grid_step_pct = DecimalParameter(low=0.001, high=0.03, default=0.001, decimals=3, space='buy', optimize=False, load=False)

    # --- Volume/MA windows (fixed / not optimized here) ---
    va_profile_window = CategoricalParameter([50, 100, 200, 300], default=200, space="buy", optimize=False, load=False)
    ma_window = CategoricalParameter([5, 7, 10, 20, 30, 50], default=20, space="buy", optimize=False, load=False)
