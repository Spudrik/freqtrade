from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import ttk

from ..base_tab import BaseTab
from ..ui_helpers import labeled_entry


DEFAULT_HYPEROPT_LOSSES = [
    "ProfitWinrateModerateRiskHyperOptLoss",
    "SharpeHyperOptLossDaily",
    "SortinoHyperOptLossDaily",
    "CalmarHyperOptLoss",
    "MaxDrawDownHyperOptLoss",
    "OnlyProfitHyperOptLoss",
]

FORMULA_NAMES = (
    "loss",
    "profit",
    "drawdown",
    "sharpe",
    "sortino",
    "calmar",
    "winrate",
    "expectancy",
    "factor",
    "trade_count_penalty",
)


@dataclass(frozen=True)
class HyperoptLossInfo:
    name: str
    source: str
    sketch: str


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
        self.hyperopt_loss_sketch_var = tk.StringVar(value="")
        self.hyperopt_loss_infos: dict[str, HyperoptLossInfo] = {}
        self.ignore_missing_spaces_var = tk.BooleanVar(value=True)
        self.disable_param_export_var = tk.BooleanVar(value=False)
        self._build_ui()
        self.refresh_hyperopt_losses()

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
        self.hyperopt_loss_combo = ttk.Combobox(
            hyperopt,
            textvariable=self.hyperopt_loss_var,
            values=DEFAULT_HYPEROPT_LOSSES,
        )
        self.hyperopt_loss_combo.grid(row=4, column=1, sticky="ew", padx=8, pady=4)
        self.hyperopt_loss_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_hyperopt_loss_sketch())
        ttk.Button(hyperopt, text="Refresh losses", command=self.refresh_hyperopt_losses).grid(row=4, column=2, sticky="w", padx=(0, 8), pady=4)
        ttk.Label(hyperopt, text="Sketch").grid(row=5, column=0, sticky="nw", padx=8, pady=4)
        ttk.Label(hyperopt, textvariable=self.hyperopt_loss_sketch_var, wraplength=520, justify="left").grid(row=5, column=1, columnspan=2, sticky="ew", padx=8, pady=4)
        ttk.Checkbutton(hyperopt, text="Ignore missing spaces", variable=self.ignore_missing_spaces_var).grid(row=6, column=1, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(hyperopt, text="Disable parameter export", variable=self.disable_param_export_var).grid(row=7, column=1, sticky="w", padx=8, pady=(4, 8))

    def _hyperopt_loss_dirs(self) -> list[tuple[Path, str]]:
        project_root = Path(str(self.context.shared.project_root.get() or "")).expanduser()
        userdir = Path(str(self.context.shared.userdir.get() or "")).expanduser()
        candidates = [
            (project_root / "freqtrade" / "optimize" / "hyperopt_loss", "built-in"),
            (userdir / "hyperopts", "user"),
            (self.context.app_dir.parent / "hyperopts", "user"),
            (self.context.app_dir / "hyperopts", "custom"),
        ]
        seen: set[Path] = set()
        resolved: list[tuple[Path, str]] = []
        for path, label in candidates:
            try:
                key = path.resolve()
            except Exception:
                key = path
            if key in seen or not path.is_dir():
                continue
            seen.add(key)
            resolved.append((path, label))
        return resolved

    def refresh_hyperopt_losses(self) -> None:
        infos: dict[str, HyperoptLossInfo] = {}
        for directory, label in self._hyperopt_loss_dirs():
            for path in sorted(directory.glob("*.py")):
                for info in self._parse_hyperopt_loss_file(path, label):
                    infos[info.name] = info
        for name in DEFAULT_HYPEROPT_LOSSES:
            infos.setdefault(name, HyperoptLossInfo(name=name, source="default", sketch="Formula not found locally."))
        self.hyperopt_loss_infos = dict(sorted(infos.items()))
        values = list(self.hyperopt_loss_infos)
        self.hyperopt_loss_combo.configure(values=values)
        if self.hyperopt_loss_var.get() not in self.hyperopt_loss_infos and values:
            self.hyperopt_loss_var.set(values[0])
        self._refresh_hyperopt_loss_sketch()

    def _parse_hyperopt_loss_file(self, path: Path, source: str) -> list[HyperoptLossInfo]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except Exception:
            return []
        constants = self._module_constants(tree)
        infos: list[HyperoptLossInfo] = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or not self._looks_like_hyperopt_loss(node):
                continue
            if not self._has_loss_function(node):
                continue
            sketch = self._loss_sketch(node, constants)
            infos.append(HyperoptLossInfo(name=node.name, source=source, sketch=sketch))
        return infos

    @staticmethod
    def _module_constants(tree: ast.Module) -> dict[str, str]:
        constants: dict[str, str] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            value = node.value.value
            if not isinstance(value, (int, float, str, bool)):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants[target.id] = str(value)
        return constants

    @staticmethod
    def _looks_like_hyperopt_loss(node: ast.ClassDef) -> bool:
        if node.name in {"IHyperOptLoss", "DefaultHyperOptLoss"}:
            return False
        if node.name.endswith("HyperOptLoss") or node.name.endswith("HyperOptLossDaily"):
            return True
        for base in node.bases:
            text = ast.unparse(base)
            if text.endswith("IHyperOptLoss"):
                return True
        return False

    @staticmethod
    def _has_loss_function(node: ast.ClassDef) -> bool:
        return any(isinstance(item, ast.FunctionDef) and item.name == "hyperopt_loss_function" for item in node.body)

    def _loss_sketch(self, class_node: ast.ClassDef, constants: dict[str, str]) -> str:
        function = next((node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "hyperopt_loss_function"), None)
        if function is None:
            return "No hyperopt_loss_function found."
        constants_text = self._constants_sketch(constants)
        formula_text = self._formula_sketch(function)
        pieces = [piece for piece in (constants_text, formula_text) if piece]
        return " | ".join(pieces)[:360] if pieces else "No compact formula found."

    @staticmethod
    def _constants_sketch(constants: dict[str, str]) -> str:
        keys = [key for key in constants if any(token.upper() in key for token in ("PROFIT", "WINRATE", "DRAW", "DD", "TRADE", "EXPECT", "PF", "TARGET"))]
        if not keys:
            return ""
        return ", ".join(f"{key}={constants[key]}" for key in keys[:5])

    @staticmethod
    def _formula_sketch(function: ast.FunctionDef) -> str:
        formulas: list[tuple[int, str]] = []
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                for target in targets:
                    lowered = target.lower()
                    if any(name in lowered for name in FORMULA_NAMES):
                        priority = 0 if target == "loss" else 2
                        formulas.append((priority, f"{target}={ast.unparse(node.value)}"))
                        break
            elif isinstance(node, ast.Return) and node.value is not None:
                formulas.append((1, f"return {ast.unparse(node.value)}"))
        compact: list[str] = []
        for _priority, formula in sorted(formulas, key=lambda item: item[0]):
            formula = " ".join(formula.split())
            if formula not in compact:
                compact.append(formula)
        return "; ".join(compact[:3])

    def _refresh_hyperopt_loss_sketch(self) -> None:
        name = self.hyperopt_loss_var.get().strip()
        info = self.hyperopt_loss_infos.get(name)
        if info is None:
            self.hyperopt_loss_sketch_var.set("Not discovered locally.")
            return
        self.hyperopt_loss_sketch_var.set(f"{info.source}: {info.sketch}")

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
        self._refresh_hyperopt_loss_sketch()
        self.ignore_missing_spaces_var.set(bool(state.get("ignore_missing_spaces", True)))
        self.disable_param_export_var.set(bool(state.get("disable_param_export", False)))
