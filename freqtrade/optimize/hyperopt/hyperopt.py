# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement

"""
This module contains the hyperopt logic
"""

import gc
import logging
import random
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any

import rapidjson
from joblib import Parallel, cpu_count
from optuna.trial import FrozenTrial, Trial, TrialState

from freqtrade.constants import FTHYPT_FILEVERSION, LAST_BT_RESULT_FN, Config
from freqtrade.enums import HyperoptState
from freqtrade.misc import file_dump_json, plural
from freqtrade.optimize.hyperopt.hyperopt_optimizer import INITIAL_POINTS, HyperOptimizer
from freqtrade.optimize.hyperopt.hyperopt_output import HyperoptOutput
from freqtrade.optimize.hyperopt_tools import (
    HyperoptStateContainer,
    HyperoptTools,
    hyperopt_serializer,
)
from freqtrade.util import get_progress_tracker


logger = logging.getLogger(__name__)
MAX_HYPEROPT_EPOCHS = 800


class Hyperopt:
    """
    Hyperopt class, this class contains all the logic to run a hyperopt simulation

    To start a hyperopt run:
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """

    def __init__(self, config: Config) -> None:
        self._hyper_out: HyperoptOutput = HyperoptOutput(streaming=True)

        self.config = config

        self.analyze_per_epoch = self.config.get("analyze_per_epoch", False)
        HyperoptStateContainer.set_state(HyperoptState.STARTUP)

        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        strategy = str(self.config["strategy"])
        self.results_file: Path = (
            self.config["user_data_dir"]
            / "hyperopt_results"
            / f"strategy_{strategy}_{time_now}.fthypt"
        )
        self.data_pickle_file = (
            self.config["user_data_dir"] / "hyperopt_results" / "hyperopt_tickerdata.pkl"
        )
        configured_epochs = int(config.get("epochs", 0) or 0)
        if configured_epochs > MAX_HYPEROPT_EPOCHS:
            logger.warning(
                "Configured hyperopt epochs (%s) exceeds max allowed (%s). Capping to %s.",
                configured_epochs,
                MAX_HYPEROPT_EPOCHS,
                MAX_HYPEROPT_EPOCHS,
            )
        self.total_epochs = min(configured_epochs, MAX_HYPEROPT_EPOCHS)
        self.config["epochs"] = self.total_epochs

        self.current_best_loss = 100

        self.clean_hyperopt()

        self.num_epochs_saved = 0
        self.current_best_epoch: dict[str, Any] | None = None

        if HyperoptTools.has_space(self.config, "sell"):
            # Make sure use_exit_signal is enabled
            self.config["use_exit_signal"] = True

        self.print_all = self.config.get("print_all", False)
        self.hyperopt_table_header = 0
        self.print_json = self.config.get("print_json", False)

        self.hyperopter = HyperOptimizer(self.config, self.data_pickle_file)
        self.count_skipped_epochs = 0

    @staticmethod
    def get_lock_filename(config: Config) -> str:
        return str(config["user_data_dir"] / "hyperopt.lock")

    def clean_hyperopt(self) -> None:
        """
        Remove hyperopt pickle files to restart hyperopt.
        """
        for f in [self.data_pickle_file, self.results_file]:
            p = Path(f)
            if p.is_file():
                logger.info(f"Removing `{p}`.")
                p.unlink()

    def _save_result(self, epoch: dict) -> None:
        """
        Save hyperopt results to file
        Store one line per epoch.
        While not a valid json object - this allows appending easily.
        :param epoch: result dictionary for this epoch.
        """
        epoch[FTHYPT_FILEVERSION] = 2
        with self.results_file.open("a") as f:
            rapidjson.dump(
                epoch,
                f,
                default=hyperopt_serializer,
                number_mode=rapidjson.NM_NATIVE | rapidjson.NM_NAN,
            )
            f.write("\n")

        self.num_epochs_saved += 1
        logger.debug(
            f"{self.num_epochs_saved} {plural(self.num_epochs_saved, 'epoch')} "
            f"saved to '{self.results_file}'."
        )
        # Store hyperopt filename
        latest_filename = Path.joinpath(self.results_file.parent, LAST_BT_RESULT_FN)
        file_dump_json(latest_filename, {"latest_hyperopt": str(self.results_file.name)}, log=False)

    def print_results(self, results: dict[str, Any]) -> None:
        """
        Log results if it is better than any previous evaluation
        TODO: this should be moved to HyperoptTools too
        """
        is_best = results["is_best"]

        if self.print_all or is_best:
            self._hyper_out.add_data(
                self.config,
                [results],
                self.total_epochs,
                self.print_all,
            )

    def run_optimizer_parallel(self, parallel: Parallel, asked: list[list]) -> list[dict[str, Any]]:
        """Start optimizer in a parallel way"""

        return parallel(self.hyperopter.generate_optimizer_wrapped(v) for v in asked)

    def _set_random_state(self, random_state: int | None) -> int:
        return random_state or random.randint(1, 2**16 - 1)  # noqa: S311

    def get_optuna_asked_points(self, n_points: int, dimensions: dict) -> list[Any]:
        asked: list[list[Any]] = []
        for i in range(n_points):
            asked.append(self.opt.ask(dimensions))
        return asked

    def duplicate_optuna_asked_points(self, trial: Trial, asked_trials: list[FrozenTrial]) -> bool:
        asked_trials_no_dups: list[FrozenTrial] = []
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        # Check whether we already evaluated the sampled `params`.
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return True
        # Check whether same`params` in one batch (asked_trials). Autosampler is doing this.
        for t in asked_trials:
            if t.params not in asked_trials_no_dups:
                asked_trials_no_dups.append(t)
        if len(asked_trials_no_dups) != len(asked_trials):
            return True
        return False

    def get_asked_points(self, n_points: int, dimensions: dict) -> tuple[list[Any], list[bool]]:
        """
        Enforce points returned from `self.opt.ask` have not been already evaluated

        Steps:
        1. Try to get points using `self.opt.ask` first
        2. Discard the points that have already been evaluated
        3. Retry using `self.opt.ask` up to `n_points` times
        """
        asked_non_tried: list[FrozenTrial] = []
        optuna_asked_trials = self.get_optuna_asked_points(n_points=n_points, dimensions=dimensions)
        asked_non_tried += [
            x
            for x in optuna_asked_trials
            if not self.duplicate_optuna_asked_points(x, optuna_asked_trials)
        ]
        i = 0
        while i < 2 * n_points and len(asked_non_tried) < n_points:
            asked_new = self.get_optuna_asked_points(n_points=1, dimensions=dimensions)[0]
            if not self.duplicate_optuna_asked_points(asked_new, asked_non_tried):
                asked_non_tried.append(asked_new)
            i += 1
        if len(asked_non_tried) < n_points:
            if self.count_skipped_epochs == 0:
                logger.warning("Duplicate params detected. Maybe your search space is too small?")
            self.count_skipped_epochs += n_points - len(asked_non_tried)

        return asked_non_tried, [False for _ in range(len(asked_non_tried))]

    def evaluate_result(self, val: dict[str, Any], current: int, is_random: bool):
        """
        Evaluate results returned from generate_optimizer
        """
        val["current_epoch"] = current
        val["is_initial_point"] = current <= INITIAL_POINTS

        logger.debug("Optimizer epoch evaluated: %s", val)

        is_best = HyperoptTools.is_best_loss(val, self.current_best_loss)
        # This value is assigned here and not in the optimization method
        # to keep proper order in the list of results. That's because
        # evaluations can take different time. Here they are aligned in the
        # order they will be shown to the user.
        val["is_best"] = is_best
        val["is_random"] = is_random
        self.print_results(val)
        self._log_epoch_metrics(val)

        if is_best:
            self.current_best_loss = val["loss"]
            self.current_best_epoch = val

        self._save_result(val)

    @staticmethod
    def _extract_grid_usage_metrics(val: dict[str, Any]) -> dict[str, Any]:
        """
        Extract per-epoch virtual level usage metrics from trade order tags.
        Works best with strategies that use tags like:
        - seed_mean_reversion (EN1)
        - add_L02 ... add_L09
        - rebuy_L02 ... rebuy_L09
        """
        metrics = val.get("results_metrics", {}) or {}
        trades = metrics.get("trades")
        if not isinstance(trades, list):
            return {
                "trade_obs": 0,
                "avg_adds_per_trade": 0.0,
                "avg_rebuys_per_trade": 0.0,
                "avg_peels_per_trade": 0.0,
                "max_adds_single_trade": 0,
                "max_peels_single_trade": 0,
                "max_rebuys_single_trade": 0,
                "level_trade_hits": [0] * 9,
                "rebuy_level_trade_hits": [0] * 9,
                "peel_level_trade_hits": [0] * 9,
            }

        level_trade_hits = [0] * 9
        rebuy_level_trade_hits = [0] * 9
        peel_level_trade_hits = [0] * 9
        adds_per_trade: list[int] = []
        peels_per_trade: list[int] = []
        rebuys_per_trade: list[int] = []

        for trade in trades:
            if not isinstance(trade, dict):
                continue
            orders = trade.get("orders")
            if not isinstance(orders, list):
                continue

            reached_levels: set[int] = set()
            rebought_levels: set[int] = set()
            peeled_levels: set[int] = set()
            add_count = 0
            peel_count = 0
            rebuy_count = 0

            for order in orders:
                if not isinstance(order, dict):
                    continue

                tag = str(order.get("ft_order_tag") or "")
                is_entry = bool(order.get("ft_is_entry", False))
                side = str(order.get("ft_order_side") or "")

                if tag.startswith("seed") and is_entry and side == "buy":
                    reached_levels.add(1)
                    continue

                if tag.startswith("add_L") and is_entry and side == "buy":
                    try:
                        level = int(tag.split("add_L", 1)[1][:2])
                    except (ValueError, IndexError):
                        continue
                    if 1 <= level <= 9:
                        reached_levels.add(level)
                        add_count += 1
                    continue

                if tag.startswith("rebuy_L") and is_entry and side == "buy":
                    try:
                        level = int(tag.split("rebuy_L", 1)[1][:2])
                    except (ValueError, IndexError):
                        continue
                    if 1 <= level <= 9:
                        reached_levels.add(level)
                        rebought_levels.add(level)
                        rebuy_count += 1
                    continue

                if tag.startswith("peel_L") and (not is_entry) and side == "sell":
                    try:
                        level = int(tag.split("peel_L", 1)[1][:2])
                    except (ValueError, IndexError):
                        continue
                    if 1 <= level <= 9:
                        peeled_levels.add(level)
                        peel_count += 1

            if reached_levels:
                for level in reached_levels:
                    level_trade_hits[level - 1] += 1

            if peeled_levels:
                for level in peeled_levels:
                    peel_level_trade_hits[level - 1] += 1

            if rebought_levels:
                for level in rebought_levels:
                    rebuy_level_trade_hits[level - 1] += 1

            adds_per_trade.append(add_count)
            peels_per_trade.append(peel_count)
            rebuys_per_trade.append(rebuy_count)

        trade_obs = len(adds_per_trade)
        if trade_obs == 0:
            return {
                "trade_obs": 0,
                "avg_adds_per_trade": 0.0,
                "avg_rebuys_per_trade": 0.0,
                "avg_peels_per_trade": 0.0,
                "max_adds_single_trade": 0,
                "max_peels_single_trade": 0,
                "max_rebuys_single_trade": 0,
                "level_trade_hits": [0] * 9,
                "rebuy_level_trade_hits": [0] * 9,
                "peel_level_trade_hits": [0] * 9,
            }

        return {
            "trade_obs": trade_obs,
            "avg_adds_per_trade": float(sum(adds_per_trade) / trade_obs),
            "avg_rebuys_per_trade": float(sum(rebuys_per_trade) / trade_obs),
            "avg_peels_per_trade": float(sum(peels_per_trade) / trade_obs),
            "max_adds_single_trade": int(max(adds_per_trade)),
            "max_peels_single_trade": int(max(peels_per_trade)),
            "max_rebuys_single_trade": int(max(rebuys_per_trade)),
            "level_trade_hits": level_trade_hits,
            "rebuy_level_trade_hits": rebuy_level_trade_hits,
            "peel_level_trade_hits": peel_level_trade_hits,
        }

    def _log_epoch_metrics(self, val: dict[str, Any]) -> None:
        """
        Emit one structured summary per epoch for launcher monitoring.
        """
        full_report = 0

        metrics = val.get("results_metrics", {}) or {}
        total_trades = int(metrics.get("total_trades") or 0)
        wins = int(metrics.get("wins") or 0)
        draws = int(metrics.get("draws") or 0)
        losses = int(metrics.get("losses") or 0)
        winrate = float(metrics.get("winrate") or 0.0)
        profit_abs = float(metrics.get("profit_total_abs") or 0.0)
        profit_pct = float(metrics.get("profit_total") or 0.0)
        max_dd_account = float(metrics.get("max_drawdown_account") or 0.0)
        trades_per_day = float(metrics.get("trades_per_day") or 0.0)
        holding_avg = metrics.get("holding_avg")

        epoch_num = int(val.get("current_epoch") or 0)
        total_epochs = int(self.total_epochs)
        is_best = bool(val.get("is_best", False))
        is_random = bool(val.get("is_random", False))
        loss = float(val.get("loss") or 0.0)

        if not full_report:
            logger.info(
                (
                    "\n-Epoch=%d/%d "
                    "-Best=%s "
                    "-Loss=%.6f "
                    "-Trades=%d "
                    "-Winrate=%.2f%% "
                    "-Profit_abs=%.8f "
                    "-Profit_pct=%.3f%% "
                    "-Max_dd=%.3f%%"
                ),
                epoch_num,
                total_epochs,
                is_best,
                loss,
                total_trades,
                winrate * 100.0,
                profit_abs,
                profit_pct * 100.0,
                max_dd_account * 100.0,
            )
            return

        grid = self._extract_grid_usage_metrics(val)
        en_1_9 = " ".join([f"EN{i + 1}={int(grid['level_trade_hits'][i])}" for i in range(9)])
        rebuy_1_9 = " ".join(
            [f"REBUY_L{i + 1}={int(grid['rebuy_level_trade_hits'][i])}" for i in range(9)]
        )
        peel_1_9 = " ".join(
            [f"PEEL_L{i + 1}={int(grid['peel_level_trade_hits'][i])}" for i in range(9)]
        )

        logger.info(
            (
                "epoch_metrics\n"
                "  epoch=%d/%d best=%s random=%s loss=%.6f\n"
                "  trades total=%d wins=%d draws=%d losses=%d winrate=%.2f%% tpd=%.3f hold_avg=%s\n"
                "  pnl profit_abs=%.8f profit_pct=%.3f%% max_dd=%.3f%%\n"
                "  grid avg_adds=%.3f avg_rebuys=%.3f avg_peels=%.3f "
                "  max_adds=%d max_rebuys=%d max_peels=%d\n"
                "  levels_en_1_9 %s\n"
                "  levels_rebuy_1_9 %s\n"
                "  levels_peel_1_9 %s"
            ),
            epoch_num,
            total_epochs,
            is_best,
            is_random,
            loss,
            total_trades,
            wins,
            draws,
            losses,
            winrate * 100.0,
            trades_per_day,
            str(holding_avg),
            profit_abs,
            profit_pct * 100.0,
            max_dd_account * 100.0,
            float(grid["avg_adds_per_trade"]),
            float(grid["avg_rebuys_per_trade"]),
            float(grid["avg_peels_per_trade"]),
            int(grid["max_adds_single_trade"]),
            int(grid["max_rebuys_single_trade"]),
            int(grid["max_peels_single_trade"]),
            en_1_9,
            rebuy_1_9,
            peel_1_9,
        )

    def start(self) -> None:
        self.random_state = self._set_random_state(self.config.get("hyperopt_random_state"))
        logger.info(f"Using optimizer random state: {self.random_state}")
        self.hyperopt_table_header = -1
        self.hyperopter.prepare_hyperopt()

        cpus = cpu_count()
        logger.info(f"Found {cpus} CPU cores. Let's make them scream!")
        config_jobs = self.config.get("hyperopt_jobs", -1)
        logger.info(f"Number of parallel jobs set as: {config_jobs}")

        self.opt = self.hyperopter.get_optimizer(self.random_state)
        try:
            with Parallel(n_jobs=config_jobs) as parallel:
                jobs = parallel._effective_n_jobs()
                logger.info(f"Effective number of parallel workers used: {jobs}")

                # Define progressbar
                with get_progress_tracker(cust_callables=[self._hyper_out]) as pbar:
                    task = pbar.add_task("Epochs", total=self.total_epochs)

                    start = 0

                    if self.analyze_per_epoch:
                        # First analysis not in parallel mode when using --analyze-per-epoch.
                        # This allows dataprovider to load it's informative cache.
                        asked, is_random = self.get_asked_points(
                            n_points=1, dimensions=self.hyperopter.o_dimensions
                        )
                        f_val0 = self.hyperopter.generate_optimizer(asked[0].params)
                        self.opt.tell(asked[0], [f_val0["loss"]])
                        self.evaluate_result(f_val0, 1, is_random[0])
                        pbar.update(task, advance=1)
                        start += 1

                    evals = ceil((self.total_epochs - start) / jobs)
                    for i in range(evals):
                        # Correct the number of epochs to be processed for the last
                        # iteration (should not exceed self.total_epochs in total)
                        n_rest = (i + 1) * jobs - (self.total_epochs - start)
                        current_jobs = jobs - n_rest if n_rest > 0 else jobs

                        asked, is_random = self.get_asked_points(
                            n_points=current_jobs, dimensions=self.hyperopter.o_dimensions
                        )

                        f_val = self.run_optimizer_parallel(
                            parallel,
                            [asked1.params for asked1 in asked],
                        )

                        f_val_loss = [v["loss"] for v in f_val]
                        for o_ask, v in zip(asked, f_val_loss, strict=False):
                            self.opt.tell(o_ask, v)

                        for j, val in enumerate(f_val):
                            # Use human-friendly indexes here (starting from 1)
                            current = i * jobs + j + 1 + start

                            self.evaluate_result(val, current, is_random[j])
                            pbar.update(task, advance=1)
                        self.hyperopter.handle_mp_logging()
                        gc.collect()

                        if (
                            self.hyperopter.es_epochs > 0
                            and self.hyperopter.es_terminator.should_terminate(self.opt)
                        ):
                            logger.info(f"Early stopping after {(i + 1) * jobs} epochs")
                            break

        except KeyboardInterrupt:
            print("User interrupted..")

        if self.count_skipped_epochs > 0:
            logger.info(
                f"{self.count_skipped_epochs} {plural(self.count_skipped_epochs, 'epoch')} "
                f"skipped due to duplicate parameters."
            )

        logger.info(
            f"{self.num_epochs_saved} {plural(self.num_epochs_saved, 'epoch')} "
            f"saved to '{self.results_file}'."
        )

        if self.current_best_epoch:
            HyperoptTools.try_export_params(
                self.config,
                self.hyperopter.get_strategy_name(),
                self.current_best_epoch,
            )

            HyperoptTools.show_epoch_details(
                self.current_best_epoch, self.total_epochs, self.print_json
            )
        elif self.num_epochs_saved > 0:
            print(
                f"No good result found for given optimization function in {self.num_epochs_saved} "
                f"{plural(self.num_epochs_saved, 'epoch')}."
            )
        else:
            # This is printed when Ctrl+C is pressed quickly, before first epochs have
            # a chance to be evaluated.
            print("No epochs evaluated yet, no best result.")
