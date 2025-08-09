import os
import sys
from subprocess import call
# Set the console width to 200 characters
os.environ['COLUMNS'] = '200'

"""
Hyperop-list provides a list of hyperopt files in the given directory, sorted by modification time.
Hyperopt-show provides details for a specific epoch from a specific hyperopt file.
"""

def list_hyperopt_files(directory, count):
    """List hyperopt files in the given directory, sorted by modification time."""
    directory = os.path.join(directory, "hyperopt_results")
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.fthypt')]
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[:count]

def run_hyperopt_command(command, args):
    """Run the specified freqtrade hyperopt command."""
    call(['freqtrade', command] + args)

def show_epoch_details(filepath, epoch_number):
    """Show details for a specific epoch from a specific hyperopt file."""
    user_data_directory = os.path.dirname(os.path.dirname(filepath))
    filename = os.path.basename(filepath)
    run_hyperopt_command('hyperopt-show', [
        '--user-data-dir', user_data_directory,
        '--hyperopt-filename', filename,
        '-n', str(epoch_number),  # specify the epoch number
        '--print-json'  # output in JSON format
    ])

def list_rated_epochs(filepath):
    """Show details for a specific epoch from a specific hyperopt file."""
    user_data_directory = os.path.dirname(os.path.dirname(filepath))
    filename = os.path.basename(filepath)
    run_hyperopt_command('hyperopt-list', [
        '--user-data-dir', user_data_directory,
        #'--best',
        #'--profitable',
        #todo this is not yet fully implemented or called from main
        #'-n', str(epoch_number),  # specify the epoch number
        '--print-json'  # output in JSON format
    ])

def main():
    user_data_directory = "C:\\FreqTradeStuff\\GPT_GUI\\user_data"
    specific = 1 # Specific file to run: 0 for range, 1 for most recent, etc.
    range_count = 2  # Range of files to run through if specific is 0

    # Hardcoded variables for the specific file path and epoch number
    specific_file_path = "C:\\FreqTradeStuff\\GPT_GUI\\User_Data\\hyperopt_results\\strategy_BasicStrategy_2024-11-09_00-39-40.fthypt"
    specific_epoch_number = 1748  # Example epoch number
    check_specific = False


    if check_specific:
        show_epoch_details(specific_file_path, specific_epoch_number)
    elif specific > 0:
        files = list_hyperopt_files(user_data_directory, specific)
        if files:

            run_hyperopt_command('hyperopt-show', [
                '--user-data-dir', user_data_directory,
                #'--hyperopt-filename', filename,
                '--best',
                '-n -1',  # show only the best epoch
                '--print-json',  # output in JSON format
                # '-n 139',
                '--profitable',
                ])
    else:
        files = list_hyperopt_files(user_data_directory, range_count)
        for file in files:
            filename = os.path.basename(file)
            run_hyperopt_command('hyperopt-list', [
                '--user-data-dir', user_data_directory,
                '--hyperopt-filename', filename,
                #'--best', '-n-1',  # list only profitable epochs
                '--print-json'  # output in JSON format
            ])

if __name__ == "__main__":
    main()
