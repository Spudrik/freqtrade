
import sys
from freqtrade.main import main as freqtrade_main
import json  # Import json to load pairs from file

def load_pairs_from_file(filepath):
    try:
        with open(filepath, 'r') as file:
            return json.load(file)  # Directly load the list of pairs
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading pairs from {filepath}: {e}")
        return []  # Return an empty list if file loading fails

def main():
    # Load pairs from the JSON file
    pairs_file_path = r"/freqtrade/user_data/data/binance/pairs.json"
    pairs = load_pairs_from_file(pairs_file_path)

    # CLI arguments for downloading historical data for futures
    cli_args = [
        "download-data",
        "--exchange", "binance",  # Specify the exchange
        "--trading-mode", "futures", # Specify futures trading mode
        "--timeframes",
                   # "15m",
                   "30m",
                   # "1h",
                   # "4h",
                   # "8h",
                   # "12h",
                   # "1d",
                   # "3d",
                   # "1w",  # Timeframes to download
        "--timerange", "20200101-",  # From 2020 up to the current date
        "--pairs",
        # "BADGER/USDT:USDT",
        # "BEAMX/USDT:USDT",
               ] + pairs + [  # Add pairs list to the CLI arguments
        # "--data-format-ohlcv", "feather",  # Save data in JSON format
    ]

    # Call Freqtrade with the arguments
    freqtrade_main(cli_args)

if __name__ == "__main__":
    main()
