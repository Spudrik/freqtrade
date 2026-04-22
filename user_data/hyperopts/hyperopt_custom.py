from pandas import DataFrame
from freqtrade.optimize.hyperopt import IHyperOptLoss
from datetime import datetime
from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_underwater

# Adjustable weights for each factor
profit_weight_low = 100.0    # Higher value makes profit more important
profit_weight = 340.0    # Higher value makes profit more important
trade_weight_nonideal_results = 55.0      # Higher value makes more trades better
trade_weight_best_results = 65.0      # Higher value makes more trades better
winrate_weight_high = 300.0    # Higher value makes better win rates more rewarding
winrate_weight_low = 60.0    # Higher value makes better win rates more rewarding
drawdown_weight = 5

# Guard thresholds
min_winrate = 0.60      # Minimum acceptable win rate
max_drawdown_limit = 0.50     # Maximum acceptable drawdown

class CustomHyper(IHyperOptLoss):
    """
    Custom HyperOptLoss function with weighted factors: Profit, Drawdown, Trade count, and Win rate.
    The strategy is penalized if the winrate is below MIN_WINRATE or drawdown exceeds MAX_DRAWDOWN.
    The final score is based on the weighted factors, with penalties added if necessary.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Config,  *args, **kwargs) -> float:

        # get values from results, config etc
        starting_balance = config['dry_run_wallet']
        # Extract the win rate from backtest statistics
        winrate = float(kwargs.get('backtest_stats', {}).get('winrate', 0.0))

        # Calculate the number of days in the backtest period
        num_days = (max_date - min_date).days + 1
        abs_profit = results["profit_abs"].sum()
        total_profit_pct = abs_profit / starting_balance

        # Daily Averaging
        avg_daily_profit = abs_profit / num_days
        daily_profit_pct = total_profit_pct / num_days
        average_trades_per_day = trade_count / num_days
        average_trades_per_day_safe = max(average_trades_per_day, 1e-9)

        # Calculate drawdown values for setting guards
        try:
            drawdown_df = calculate_underwater(
                results,
                value_col='profit_abs',
                starting_balance=config['dry_run_wallet']
            )
            max_drawdow = abs(min(drawdown_df['drawdown']))
            # relative_drawdown is a fractional value from 0 to 1, representing 0 to 100% of the account wallet.
            # However, it's relative to the time, so it's drawdown from the peak. Not the initial wallet size.
            relative_drawdown = max(drawdown_df['drawdown_relative'])
        except (Exception, ValueError):
            relative_drawdown = 0



        # Loss calculation based on different scenarios
        if abs_profit <= 0:
            # Loss calculation for non-positive profit, emphasizing loss by converting it to a positive value
            weighted_daily_profit = profit_weight * abs(daily_profit_pct * 100)
            # Dividing by winrate / avg trade results in a larger denominator for higher winrate or trade rate (as both
            # as decimal). This reduces the loss due to negative profits so that winrate/trades being good still results
            # in a better loss value, which helps the hyperopt learn that winrate/trades being higher is always good

            # This takes the weighted daily profit. Reduces it when winrate is high, increase it when winrate is low.
            # It also decreaes it when trade rate is high, increases it when trade rate is low.
            # This makes losing trades result in smaller loss function if they have higher winrate and trade rate.
            loss = weighted_daily_profit / (1 + winrate) / average_trades_per_day_safe

            # Limit loss to the maximum defined value in freqtrade.optimize.hyperopt
            if loss >= 90000:
                loss = 90000
        # This second condition enhances winrate to trade rate ratio. So that low winrates produce loss dominated by winrate.
        # This drives the winrate higher, which then results in condition 3.
        # Which significantly enhances everythings effect on loss.
        elif abs_profit > 0 and (average_trades_per_day <= 0.3):
            weighted_daily_trades_winrate_scaled = -trade_weight_nonideal_results * (average_trades_per_day * winrate)
            loss = weighted_daily_trades_winrate_scaled
        elif (abs_profit > 0 and winrate < min_winrate) or (abs_profit > 0 and relative_drawdown > max_drawdown_limit)\
                or (average_trades_per_day <= 1):
            # Loss calculation for positive profit and winrate below the minimum
            # As winrate is fractional we can multiply the profit by this to reduce the results weight for lower winrates.
            weighted_daily_profit = -profit_weight_low * (daily_profit_pct * winrate)
            weighted_winrate = -winrate_weight_low * winrate
            weighted_daily_trades = -trade_weight_nonideal_results * (average_trades_per_day * winrate)

            loss = weighted_daily_profit + weighted_winrate + weighted_daily_trades     # Sum of weighted profit and adjusted winrate
            # now reduce loss inverse to relative_drawdown
            loss = loss * (1 - relative_drawdown)
            # Then scale down relative to winrate again.
            # This stops very large numbers of losing trades resulting in good loss values.
            loss = loss * winrate
        elif abs_profit > 0 and winrate >= min_winrate:
            ideal_conditions_offset = -200
            if winrate >= 0.7 and average_trades_per_day >= 2:
                ideal_conditions_offset += 100
            # Loss calculation for positive profit and winrate above or equal to the minimum
            weighted_daily_profit = -profit_weight * (daily_profit_pct * winrate)
            weighted_winrate = -winrate_weight_high * winrate
            # trade_weight_high is a low value as already forced trade rate higher with earlier conditions.
            # Now we just need it to add a small improvement so that winrate/profit take precedence in
            # this ideal final results formula
            weighted_daily_trades = -trade_weight_best_results * (average_trades_per_day)
            loss = ideal_conditions_offset + weighted_daily_profit + weighted_winrate + weighted_daily_trades  # Sum of all weighted components
            # Adjustment based on relative_drawdown
            if relative_drawdown < 0.2:
                adjustment = -int(((0.2 - relative_drawdown) ** 2) * 100)  # Slightly more negative below 0.2
            else:  # relative_drawdown between 0.2 and 0.5
                adjustment = int(((relative_drawdown - 0.2) ** 2) * 100)  # Slightly more positive from 0.2 to 0.5


            # Apply the integer adjustment directly to the loss
            loss = loss + (adjustment * drawdown_weight)

        # Worked examples:
        # 1. relative_drawdown = 0.1:
        # - adjustment = -int(((0.2 - 0.1) ** 2) * 100) = -1
        # - loss = loss + (-1), so loss decreases by 1.

        # 2. relative_drawdown = 0.3:
        # - adjustment = int(((0.3 - 0.2) ** 2) * 100) = 1
        # - loss = loss + 1, so loss increases by 1.

        # 3. relative_drawdown = 0.5:
        # - adjustment = int(((0.5 - 0.2) ** 2) * 100) = 9
        # - loss = loss + 9, so loss increases by 9.

        return loss
