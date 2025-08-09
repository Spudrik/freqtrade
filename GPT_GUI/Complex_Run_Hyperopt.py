import json
import random
import shutil
from time import sleep
import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from freqtrade.main import main as freqtrade_main
import sys
import tempfile  # Added for handling temporary files
# Custom class to handle read/write of Hyperparameters to/from CSV
from freqtrade.user_data.strategies.best_results.hyperparam_csv_manager import HyperParamCSVManager


# Define colours for print statements
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Background colors

BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_WHITE = "\033[47m"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# General Settings
BUFFER = 40
MAX_EPOCHS_MEDIUM = 250
FIXED_EPOCHS_LARGE = 500
WORKERS = 10
DEBUG = 0
GROUPS_ENABLED = [
    # "group1",
    # "group2",
    # "group3",
    # "group4",
    # "group5",
    "group6",
    "group7",
]
# Batch Optimization Settings
batch_size = 10
batch_increase = 5
batch_loop = 5
max_retries_per_batch = 0

# Random Optimization Settings
random_loops = 1
random_loops_batch_size_divider = 1
optimize_count_limit = batch_loop + 3

timerange = "20230301-20240301"
hyperopt = "CustomHyper"

folder_path = Path(r'C:\FreqTradeStuff\freqtrade\user_data\strategies\hyperopt_params')
pairs_file_path = folder_path.parent.parent / 'data/binance/pairs_volatile.json'
json_results_path = folder_path.parent / 'basic_new_strat.json'
path_for_json_copies = r"C:\FreqTradeStuff\freqtrade\user_data\strategies\best_results\backup_copies"
csv_path = folder_path.parent / r'\best_results\hyperparams.csv'

# Controls all actions related to the CSV file
manager = HyperParamCSVManager(csv_path="path/to/hyperparams.csv")
tracking_df = manager.get_dataframe()

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set a custom formatter to display default log messages in magenta and reset the terminal color to red
# after each message regardless of level
logger.handlers[0].setFormatter(logging.Formatter(f'{RED}%(levelname)s{BLUE} %(message)s{RED}'))

# Define custom colors for each log level
logging.addLevelName(logging.WARNING, f'{YELLOW}WARNING')
logging.addLevelName(logging.ERROR, f'{YELLOW}ERROR')
logging.addLevelName(logging.INFO, f'{BLUE}INFO')
logging.addLevelName(logging.DEBUG, f'{GREEN}DEBUG')

os.environ['COLUMNS'] = '200'


def update_defaults(param_list, current_best_objective):
    """
    After a hyperopt run, update defaults from JSON for parameters in param_list and reset optimize=False.
    """

    with open(json_results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    updated_params = {}
    for space, params_dict in results.get('params', {}).items():
        for param_name, new_value in params_dict.items():
            updated_params[param_name] = new_value

    # Build param index
    param_index = {}
    for g_name, params_ in data.items():
        for i, p in enumerate(params_):
            param_index[p['parameter']] = (g_name, i)

    # Update defaults if changed
    for p_name in param_list:
        if p_name in param_index:
            g_name, i = param_index[p_name]
            if p_name in updated_params:
                old_default = data[g_name][i]['default']
                new_value = updated_params[p_name]
                if str(old_default) != str(new_value):
                    manager.update_parameter_default(p_name, updated_params[p_name])
                    logger.info(f"Parameter '{p_name}' default changed from '{old_default}' to '{new_value}'.")

    logger.info(f"Updated defaults due to new all-time best objective.")
    logger.info(f"Start - Cloning Best Loss Results JSON file to {path_for_json_copies}")

def load_pairs_from_file():
    try:
        with open(pairs_file_path, 'r', encoding='utf-8') as file:
            logger.info(f"Loaded pairs from {pairs_file_path}.")
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading pairs from {pairs_file_path}: {e}")
        exit()

def run_hyperopt_with_capture(epochs=200):
    job_workers = 21 - WORKERS
    pairs = load_pairs_from_file()

    if DEBUG == 1:
        pairs = [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "ADA/USDT:USDT",
            "CHZ/USDT:USDT",
            "CKB/USDT:USDT",
        ]
        timerange_to_use = "20230101-20230401"
        job_workers = 20
        epochs = 20
    else:
        timerange_to_use = timerange

    cli_args = [
        "hyperopt",
        "--strategy-path", str(folder_path.parent),
        "--timerange", timerange_to_use,
        "--strategy", "BasicStrategy",
        "--config", r"C:\FreqTradeStuff\freqtrade\user_data\configs\gui_generated.json",
        "--config", r"C:\Users\engin\OneDrive\Desktop\config_private.json",
        "--epochs", str(epochs),
        "--pairs"
    ] + pairs + [
        "--spaces", "buy", "sell",
        "--hyperopt-loss", hyperopt,
        "--timeframe", "30m",
        "--max-open-trades", "1000",
        "--stake-amount", "unlimited",
        "--dry-run-wallet", "3000",
        "--print-all",
        # "--print-json",
        "--job-workers", str(-job_workers),
        "--min-trades", "2",
        "--ignore-missing-spaces",
    ]

    seed = random.randint(0, 1000000)
    cli_args += ["--random-state", str(seed)]
    logger.info(f"Running hyperopt with random state: {seed}")

    logger.info(f"Ensure optimize flags are set True for batch, in YAML file. Prior to hyperopt...")
    save_hyperopt_params_to_yaml(data)
    sleep(1)

    try:
        freqtrade_main(cli_args)
    except SystemExit:
        logger.info("Ignoring SystemExit raised by freqtrade_main.")
    except Exception as e:
        logger.info(f"Hyperopt run error: {e}")
        exit()

    logger.warning(f"Freqtrade Complete - Sleeping for 20 seconds before updating/resetting YAML file to avoid risk of exit manual exits corrupting the YAML writes...")
    sleep(20) # Allows freqtrade prints to finish
    reset_optimize_flags()




    current_loss_file = 'C:\\FreqTradeStuff\\freqtrade\\user_data\\strategies\\current_hyperopt_loss.log'
    with open(current_loss_file, 'r', encoding='utf-8') as loss_file:
        loss_value = loss_file.read().strip()
        try:
            objective_value = float(loss_value)
            logger.info(f"Extracted Objective Value this run: {objective_value}")
            return objective_value
        except ValueError:
            logger.error(f"Failed to parse objective value: {loss_value}")
            exit()

def increment_optimize_counts(param_list):
    if not data:
        return

    # Increment counts directly in data
    refs = find_params_in_data(data, param_list)
    for g_name, i in refs:
        data[g_name][i]['current_optimize_count'] = data[g_name][i].get('current_optimize_count', 0) + 1
        data[g_name][i]['total_optimize_count'] = data[g_name][i].get('total_optimize_count', 0) + 1

def handle_batch_run(batch, batch_identifier, tracking_df_local, previous_best_objective, retry_counters):
    """
    Now we must rely on `batch` (which is a DataFrame slice) only to get param_list.
    We do not update tracking_df anymore. It's read-only. All updates happen in data.
    """
    try:
        logger.debug(f"Starting batch '{batch_identifier}' with {len(batch)} parameters.")
        param_list = batch['parameter'].tolist()  # We still get param_list from batch df slice.

        # Set specific target list of parameters optimize flags = True and save to csv prior to launching freqtrade
        manager.set_optimize_flag(param_list, True)
        manager.save_to_csv()
        sleep(1)

        param_count = 0
        total_combinations = 1
        boolean_param_count = 0
        categorical_param_count = 0
        for _, row in batch.iterrows():
            param_type = row['param_type']
            if param_type == "BooleanParameter":
                num_values = 2
                boolean_param_count += 1
                param_count += 1
            elif param_type == "CategoricalParameter":
                cat_list = row.get('categories', [])
                cat_list = cat_list if cat_list is not None else []
                num_values = len(cat_list) if cat_list else 1
                categorical_param_count += 1
                categorical_param_count += num_values
                param_count += 1
            elif param_type in ["IntParameter", "DecimalParameter"]:
                num_values = 2
                param_count += 1
            else:
                num_values = 1

            total_combinations *= num_values

        total_bool_cat_params = boolean_param_count + categorical_param_count
        if total_combinations <= 100:
            calculated_epochs = int(total_bool_cat_params) + BUFFER
        elif total_combinations <= 1000:
            calculated_epochs = min(int(total_bool_cat_params), MAX_EPOCHS_MEDIUM) + BUFFER
        else:
            calculated_epochs = FIXED_EPOCHS_LARGE

         # Double the max epochs for high numbers of enable parameters
        if total_bool_cat_params >= 11:
            calculated_epochs += FIXED_EPOCHS_LARGE

        logger.info(f"Batch '{batch_identifier}' combinations: {total_combinations}, calculated epochs: {calculated_epochs}")

        initial_random_epochs = max(int(calculated_epochs * 0.5), 40)
        logger.debug(f"Writing initial random epochs {initial_random_epochs} to file.")
        with open(r"C:\FreqTradeStuff\freqtrade\user_data\strategies\initial_random_epochs.log", 'w') as f:
            f.write(str(initial_random_epochs))

        current_best_objective = run_hyperopt_with_capture(epochs=int(calculated_epochs))
        sleep(1)
        logger.info(f"########### Hyperopt Complete #############")

        if current_best_objective < previous_best_objective:
            logger.info(f"Batch '{batch_identifier}': New all-time best objective (best loss) found! Improved from {previous_best_objective} to {current_best_objective}.")
            update_jfdefaults(param_list, current_best_objective)

            best_objective_file = get_best_objective_file()
            with best_objective_file.open('w', encoding='utf-8') as f:
                f.write(str(current_best_objective))
            previous_best_objective = current_best_objective

            logger.info(f"Updating CSV due to best results improvement. Write to CSV will be repeated in call to Freqtrade. "
                        f"But this is no problem as it ensures the best results are saved.")
            manager.save_to_csv()

            # If we want to regenerate tracking_df for read-only updated view:
            global tracking_df
            tracking_df = convert_data_to_df(data)

            return True, tracking_df_local, previous_best_objective

        else:
            logger.info(f"Batch '{batch_identifier}': No improvement detected. \nCurrent Objective: {current_best_objective}\nPrevious all-time best: {previous_best_objective}")
            if batch_identifier not in retry_counters:
                retry_counters[batch_identifier] = 0
            if retry_counters[batch_identifier] < max_retries_per_batch:
                retry_counters[batch_identifier] += 1
                logger.info(f"Batch '{batch_identifier}': Loss worsened, retrying ({retry_counters[batch_identifier]}/{max_retries_per_batch})")
                # No YAML update since no improvement
                return handle_batch_run(batch, batch_identifier, tracking_df_local, previous_best_objective, retry_counters)
            else:
                logger.info(f"Batch '{batch_identifier}': Max retries exceeded, skipping updates.")
                # No YAML update
                return False, tracking_df_local, previous_best_objective

    except Exception as e:
        logger.error(f"Batch '{batch_identifier}': Exception occurred - {e}: # On exception, no improvement, so no YAML update")
        # On exception, no improvement, so no YAML update
        return False, tracking_df_local, previous_best_objective

def execute_group_runs(tracking_df_local, group_columns, previous_best_objective, retry_counters):
    logger.info(f"Starting group-based optimization runs.")

    # Iterate through group_columns in reverse order
    # for group_col in group_columns:
    for group_col in reversed(group_columns):
        if group_col not in GROUPS_ENABLED:
            logger.warning(f"Group '{group_col}' is disabled. Skipping this group.")
            continue

        group_values = tracking_df_local[group_col].dropna().unique()
        group_values = [gv for gv in group_values if gv is not None]

        for group_value in group_values:
            group_params = tracking_df_local[
                (tracking_df_local['optimize_group_allowed'] == True) &
                (tracking_df_local[group_col].astype(str) == str(group_value))
            ]

            if group_params.empty:
                continue

            batch_identifier = f"{group_col}_{group_value}"
            # Convert dataframe slices to strings to ensure they all print.
            logger.info(f"GROUP DATA BEFORE BATCH: {batch_identifier}\n{group_params[['parameter', 'default'] + group_columns].to_string()}")

            # Run handle_batch_run - no re-read best_objective_file after success
            batch = group_params
            success, tracking_df_local, previous_best_objective = handle_batch_run(
                batch,
                batch_identifier,
                tracking_df_local,
                previous_best_objective,
                retry_counters
            )

    logger.info(f"Completed group-based optimization runs.")
    return tracking_df_local

def execute_batch_loops(tracking_df_local, previous_best_objective, retry_counters):
    logger.info(f"Starting batch-based optimization loops.")
    for loop_num in range(batch_loop):
        batch_size_current = batch_size + batch_increase * loop_num
        eligible_params = tracking_df_local[tracking_df_local['optimize_batch_allowed'] == True]

        batch_indices = list(range(0, len(eligible_params), batch_size_current))
        max_batches = len(batch_indices)
        for batch_start in batch_indices:
            batch_end = batch_start + batch_size_current
            batch = eligible_params.iloc[batch_start:batch_end]
            batch_number = batch_start // batch_size_current + 1
            batch_identifier = f"BatchLoop{loop_num+1}_Batch{batch_number}/{max_batches}"
            logger.info(f"BATCH DATA - {batch_identifier}\n{batch[['parameter', 'default'] + [col for col in tracking_df_local.columns if col.startswith('group')]]}")
            success, tracking_df_local, previous_best_objective = handle_batch_run(
                batch,
                batch_identifier,
                tracking_df_local,
                previous_best_objective,
                retry_counters
            )

    logger.info(f"Completed batch-based optimization loops.")
    return tracking_df_local

def execute_random_loops(tracking_df_local, previous_best_objective, retry_counters):
    logger.info(f"Starting random-based optimization loops.")
    for i in range(random_loops):
        eligible_params = tracking_df_local[
            (tracking_df_local['optimize_random_allowed'] == True) &
            (tracking_df_local['current_optimize_count'] < optimize_count_limit)
        ]
        if eligible_params.empty:
            logger.info(f"No eligible parameters left for random optimization.")
            break
        random_loop_batch_size = max(1, round((len(eligible_params)) / random_loops_batch_size_divider))
        batch = eligible_params.sample(
            min(len(eligible_params), random_loop_batch_size),
            random_state=random.randint(0, 1000)
        )
        batch_identifier = f"RandomLoop{i+1}/{random_loops}_Batch"
        logger.info(f"RANDOM MATCHES - {batch_identifier}\n{batch[['parameter', 'default'] + [col for col in tracking_df_local.columns if col.startswith('group')]]}")
        success, tracking_df_local, previous_best_objective = handle_batch_run(
            batch,
            batch_identifier,
            tracking_df_local,
            previous_best_objective,
            retry_counters
        )
    logger.info(f"Completed random-based optimization loops.")
    return tracking_df_local

def get_best_objective_file():
    """
    Generates the best objective file path based on the YAML groups.
    """
    groups = sorted(data.keys()) if data else ["default"]
    strategies = '_'.join(groups) if groups else "default"
    filename = f'best_objective_{strategies}_{hyperopt}.log'
    return folder_path.parent / filename

def hyperopt_routine( ):
    sleep(0.1)
    logger.info(f"Starting hyperopt routine - Initializing current hyperopt loss to 90000.")
    with Path(r"C:\FreqTradeStuff\freqtrade\user_data\strategies\current_hyperopt_loss.log").open('w', encoding='utf-8') as f:
        f.write('100')
    global tracking_df
    logger.info(f"Ensuring all optimize flags to False at start-up")
    reset_optimize_flags()
    group_columns = [col for col in tracking_df.columns if col.startswith('group')]

    # get_best_objective_file() generates the best objective file path using the various trade types that are active like bull/bear/common etc
    best_objective_file = get_best_objective_file()
    if best_objective_file.exists():
        with best_objective_file.open('r', encoding='utf-8') as f:
            try:
                previous_best_objective = float(f.read().strip())
            except ValueError:
                exit()

    else:
        previous_best_objective = 100

    retry_counters = {}

    # Run group loops
    tracking_df = execute_group_runs(tracking_df, group_columns, previous_best_objective, retry_counters)
    # Run random loops
    tracking_df = execute_random_loops(tracking_df, previous_best_objective, retry_counters)
    # Run batch loops
    tracking_df = execute_batch_loops(tracking_df, previous_best_objective, retry_counters)

    # After all loops complete:
    # If no improvements happened, we never rewrote YAML. If improvements happened, we wrote YAML at time of improvement.
    # So no final rewrite needed.

    logger.info(f"Hyperopt routine completed.")

if __name__ == "__main__":
    debug = 0
    for arg in sys.argv:
        if arg == 'debug':
            DEBUG = 1
            logger.warning(f"##### ############################################################################################################### #####"
                  f"\n################################################## Debug Mode ENABLED #####################################################"
                  f"\n##### ############################################################################################################### #####{WHITE}")


    hyperopt_routine()
