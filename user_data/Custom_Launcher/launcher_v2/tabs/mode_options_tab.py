from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import ttk

from ..base_tab import BaseTab
from ..ui_helpers import labeled_entry


class ModeOptionsTab(BaseTab):
    tab_key = "mode_options"
    tab_title = "Run Options"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.timeframe_var = tk.StringVar(value="1h")
        self.timerange_var = tk.StringVar(value="")
        self.max_open_trades_var = tk.StringVar(value="")
        self.stake_amount_var = tk.StringVar(value="")
        self.dry_run_wallet_var = tk.StringVar(value="")
        self.fee_var = tk.StringVar(value="")
        self.export_var = tk.StringVar(value="trades")
        self.breakdown_var = tk.StringVar(value="day")
        self.epochs_var = tk.StringVar(value="100")
        self.spaces_var = tk.StringVar(value="buy sell roi stoploss trailing")
        self.jobs_var = tk.StringVar(value="")
        self.random_state_var = tk.StringVar(value="")
        self.hyperopt_loss_var = tk.StringVar(value="ProfitWinrateModerateRiskHyperOptLoss")
        self.ignore_missing_spaces_var = tk.BooleanVar(value=True)
        self.disable_param_export_var = tk.BooleanVar(value=False)
        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        common = ttk.LabelFrame(self, text="Shared run options")
        common.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        common.grid_columnconfigure(1, weight=1)
        common.grid_columnconfigure(3, weight=1)
        labeled_entry(common, 0, 0, "Timeframe", self.timeframe_var, width=18)
        labeled_entry(common, 0, 2, "Timerange", self.timerange_var, width=24)
        labeled_entry(common, 1, 0, "Max open trades", self.max_open_trades_var, width=18)
        labeled_entry(common, 1, 2, "Stake amount", self.stake_amount_var, width=24)
        labeled_entry(common, 2, 0, "Dry run wallet", self.dry_run_wallet_var, width=18)
        labeled_entry(common, 2, 2, "Fee", self.fee_var, width=24)

        backtest = ttk.LabelFrame(self, text="Backtest")
        backtest.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        backtest.grid_columnconfigure(1, weight=1)
        labeled_entry(backtest, 0, 0, "Export", self.export_var)
        labeled_entry(backtest, 1, 0, "Breakdown", self.breakdown_var)

        hyperopt = ttk.LabelFrame(self, text="Hyperopt")
        hyperopt.grid(row=1, column=1, sticky="nsew", padx=8, pady=(0, 8))
        hyperopt.grid_columnconfigure(1, weight=1)
        labeled_entry(hyperopt, 0, 0, "Epochs", self.epochs_var)
        labeled_entry(hyperopt, 1, 0, "Spaces", self.spaces_var)
        labeled_entry(hyperopt, 2, 0, "Jobs", self.jobs_var)
        labeled_entry(hyperopt, 3, 0, "Random state", self.random_state_var)
        ttk.Label(hyperopt, text="Loss").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(
            hyperopt,
            textvariable=self.hyperopt_loss_var,
            values=[
                "ProfitWinrateModerateRiskHyperOptLoss",
                "SharpeHyperOptLossDaily",
                "SortinoHyperOptLossDaily",
                "CalmarHyperOptLoss",
                "MaxDrawDownHyperOptLoss",
                "OnlyProfitHyperOptLoss",
            ],
        ).grid(row=4, column=1, sticky="ew", padx=8, pady=4)
        ttk.Checkbutton(hyperopt, text="Ignore missing spaces", variable=self.ignore_missing_spaces_var).grid(row=5, column=1, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(hyperopt, text="Disable parameter export", variable=self.disable_param_export_var).grid(row=6, column=1, sticky="w", padx=8, pady=(4, 8))

    def get_state(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe_var.get(),
            "timerange": self.timerange_var.get(),
            "max_open_trades": self.max_open_trades_var.get(),
            "stake_amount": self.stake_amount_var.get(),
            "dry_run_wallet": self.dry_run_wallet_var.get(),
            "fee": self.fee_var.get(),
            "export": self.export_var.get(),
            "breakdown": self.breakdown_var.get(),
            "epochs": self.epochs_var.get(),
            "spaces": self.spaces_var.get(),
            "jobs": self.jobs_var.get(),
            "random_state": self.random_state_var.get(),
            "hyperopt_loss": self.hyperopt_loss_var.get(),
            "ignore_missing_spaces": self.ignore_missing_spaces_var.get(),
            "disable_param_export": self.disable_param_export_var.get(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.timeframe_var.set(str(state.get("timeframe") or "1h"))
        self.timerange_var.set(str(state.get("timerange") or ""))
        self.max_open_trades_var.set(str(state.get("max_open_trades") or ""))
        self.stake_amount_var.set(str(state.get("stake_amount") or ""))
        self.dry_run_wallet_var.set(str(state.get("dry_run_wallet") or ""))
        self.fee_var.set(str(state.get("fee") or ""))
        self.export_var.set(str(state.get("export") or "trades"))
        self.breakdown_var.set(str(state.get("breakdown") or "day"))
        self.epochs_var.set(str(state.get("epochs") or "100"))
        self.spaces_var.set(str(state.get("spaces") or "buy sell roi stoploss trailing"))
        self.jobs_var.set(str(state.get("jobs") or ""))
        self.random_state_var.set(str(state.get("random_state") or ""))
        self.hyperopt_loss_var.set(str(state.get("hyperopt_loss") or "ProfitWinrateModerateRiskHyperOptLoss"))
        self.ignore_missing_spaces_var.set(bool(state.get("ignore_missing_spaces", True)))
        self.disable_param_export_var.set(bool(state.get("disable_param_export", False)))
