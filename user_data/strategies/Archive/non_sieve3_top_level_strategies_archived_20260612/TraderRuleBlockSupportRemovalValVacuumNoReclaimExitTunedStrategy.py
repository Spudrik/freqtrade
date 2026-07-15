from __future__ import annotations

from TraderRuleBlockSupportRemovalValVacuumNoReclaimStrategy import TraderRuleBlockSupportRemovalValVacuumNoReclaimStrategy


class TraderRuleBlockSupportRemovalValVacuumNoReclaimExitTunedStrategy(
    TraderRuleBlockSupportRemovalValVacuumNoReclaimStrategy
):
    """
    Research-only validation for the no-reclaim VAL/vacuum short using the
    exit profile selected by the quick-short exit sweep.
    """

    stoploss = -0.010

    rule_exit_settings: dict[str, dict[str, float]] = {
        "val_break_downside_vacuum_after_support_removed_short": {"take_profit": 0.035, "hold_hours": 8.0},
    }
