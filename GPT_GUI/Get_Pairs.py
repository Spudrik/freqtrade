import logging
import logging.config
from time import sleep
from freqtrade.main import main as freqtrade_main
import os
import sys  # Import sys to access command-line arguments

# Set the console width to 200 characters
os.environ['COLUMNS'] = '200'

def get_markets():
    # Prepare the list of CLI arguments
    exchange = "binance"
    cli_args = [
        "list-markets",
        "--exchange", f"{exchange}",
        "--trading-mode", "futures",
        "--quote", "USDT",
        #"--all",
    ]

    # Call the freqtrade CLI function
    print(f"Getting markets list from {exchange}...")
    freqtrade_main(cli_args)
def get_exchanges():
    # Prepare the list of CLI arguments
    cli_args = [
        "list-exchanges",
    ]

    # Call the freqtrade CLI function
    print(f"Getting exchanges list...")
    freqtrade_main(cli_args)
def get_pairs():
    exchanges = ['binance', 'kucoinfutures']  # List of exchanges to get pairs from
    for exchange in exchanges:
        # Prepare the list of CLI arguments
        cli_args = [
            "list-pairs",
            "--exchange", "binance",
            "--trading-mode", "futures",  # Default to futures on Binance
            # Additional options (commented out for future use)
            "--quote", "USDT",  # Specify quote currencies if needed
            # "--print-list",  # Print list of pairs
            "--print-json",  # Print list in JSON format
            # "-1",  # Print output in one column
            # "--print-csv",  # Print data in CSV format
            # "-a",  # Print all pairs, including inactive ones
            # "-v",  # Verbose output
            # "--logfile", "path/to/logfile",
            # "-V",  # Show version and exit
            # "-c", "path/to/config",  # Specify config file
            # "-d", "path/to/data",  # Specify data directory
            "--userdir", "C:\\FreqTradeStuff\\freqtrade\\user_data",  # Specify user directory
            # "--trading-mode", "spot",  # Use spot trading mode
            # "--trading-mode", "margin",  # Use margin trading mode
        ]

        # Call the freqtrade CLI function
        print(f"Getting pairs from {exchange}...")
        freqtrade_main(cli_args)

if __name__ == "__main__":
    import sys  # Import sys to access command-line arguments

    # Set default debug value
    debug = 0

    # Check if command-line argument is provided and set debug accordingly
    if len(sys.argv) > 1:
        debug = sys.argv[1]

    if debug == '1':
        get_exchanges()
    elif debug == '2':
        get_markets()
    else:
        get_pairs()


