"""Paper C: exact level-rejection signal with a completed-candle volume gate."""

from __future__ import annotations

from pandas import DataFrame

from user_data.strategies.paper_trial_level import PaperTrialLevel


class PaperTrialLevelVolume(PaperTrialLevel):
    def _volume_gate(self, dataframe: DataFrame):
        return (dataframe["volume"] > dataframe["prior_20_volume_median"]) & (dataframe["volume"] > 0)
