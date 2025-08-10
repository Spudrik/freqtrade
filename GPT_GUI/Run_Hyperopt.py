import logging
import logging.config
from time import sleep
from freqtrade.main import main as freqtrade_main
import json
import csv
import os
from collections import defaultdict
import pandas as pd
# Set the console width to 200 characters
os.environ['COLUMNS'] = '200'


# Function to load pairs from a JSON file
def load_pairs_from_file(filepath):
    try:
        with open(filepath, 'r') as file:
            return json.load(file)  # Directly load the list of pairs
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading pairs from {filepath}: {e}")
        return []  # Return an empty list if file loading fails


def run_hyperopt(job_workers, debug):
    # Load pairs from the JSON file
    # pairs_file_path = r"C:\FreqTradeStuff\freqtrade\user_data\data\binance\pairs_volatile.json"
    # restore pair loading from file by uncmenting line below.
    # pairs = load_pairs_from_file(pairs_file_path)
    pairs = [
                "PEPE/USDT:USDT",
                "WIF/USDT:USDT",
                                 ]
    timerange = "20230101-20240105"


    if debug == 1:
        pairs = [
                "PEPE/USDT:USDT",
                "WIF/USDT:USDT",
                 ]
        timerange = "20230101-20230601"


    # Prepare the list of CLI arguments
    cli_args = [
        "hyperopt",
        "--strategy-path", r"C:\FreqTradeStuff\freqtrade\user_data\strategies",
        "--timerange", timerange,
        "--strategy", "MyStrategy",
        "--config", r"C:\FreqTradeStuff\freqtrade\user_data\configs\gui_generated.json",
        "--config", r"C:\Users\engin\OneDrive\Desktop\config_private.json",
        "--epoch", "2",
        "--pairs",
          ] + pairs + [  # Add pairs list to the CLI arguments
        "--spaces", "buy",
        "sell",
        # "stoploss",
        "--hyperopt-loss", "CustomHyper",
        "--timeframe", "30m",
        "--max-open-trades", "1000",
        "--stake-amount", "unlimited",
        "--dry-run-wallet", "3000",
        "--print-all",
        "--print-json",
        "--job-workers", str(-job_workers),  # Negate job_workers
        "--min-trades", "2",
        # "--disable-param-export",
        "--random-state", "1",
        "--ignore-missing-spaces",
    ]

    # Call the freqtrade CLI function
    freqtrade_main(cli_args)


if __name__ == "__main__":
    import sys  # Import sys to access command-line arguments

    # Set default debug value
    debug = 0

    # Check if command-line argument is provided and set debug accordingly
    if len(sys.argv) > 1 and sys.argv[1] == '1':
        debug = 1

    if debug == 1:
        print("Debug mode enabled.")
        print(f"Starting hyperopt debug single run with job_workers = -20.")
        run_hyperopt(20, debug=debug)
    else:
        # Set job_workers at the top to an integer value of 10
        job_workers = 8 # set number of 1 (max) while 1 becomes 20 (min)
        job_workers = 21 - job_workers# Define number of runs you want overnight
        num_runs = 2 # Set to your desired number of runs
        sleep_between_runs = 2  # Time (in seconds) between each run (optional)

        for i in range(num_runs):
            try:
                print(f"Starting hyperopt run {i + 1} of {num_runs} with job_workers = -{job_workers}.")
                run_hyperopt(job_workers, debug)  # Run the hyperopt

            except SystemExit:
                json_results_file = ('C:\\FreqTradeStuff\\freqtrade\\user_data\\strategies\\basic_new_strat.json')
                csv_results_file = ('C:\\FreqTradeStuff\\freqtrade\\user_data\\strategies\\hyperopt_results.csv')
                print(f"Completed hyperopt run {i + 1}.\n")
                print(f'Extracting results to csv for tracking long term results trends'
                      f'\nFiles saved to {csv_results_file} ')

            # Optionally sleep between runs to avoid overwhelming the system
            if i < num_runs - 1:
                print(f"Sleeping for {sleep_between_runs} seconds before the next run...")
                sleep(sleep_between_runs)


        print(f"All {num_runs} hyperopt runs completed.")
