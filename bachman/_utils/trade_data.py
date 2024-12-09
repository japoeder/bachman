"""
Get the trade and stats data from the logs folder.
"""

import os
import glob
import pandas as pd


def trade_data(directory_path: str = "./logs"):
    """
    Get the trade and stats data from the logs folder.
    """
    # Construct the search pattern
    stats_pattern = os.path.join(directory_path, "*_stats.csv")
    trades_pattern = os.path.join(directory_path, "*_trades.csv")

    # Find files matching the pattern
    stat_file = glob.glob(stats_pattern)
    trade_file = glob.glob(trades_pattern)

    output = []
    # Check if any files were found
    if not stat_file:
        print("No stats files matching the pattern were found.")
    else:
        stat_file_path = stat_file[0]
        stats = pd.read_csv(stat_file_path)
        output.append(stats)

    # Check if any files were found
    if not trade_file:
        print("No trades files matching the pattern were found.")
    else:
        trade_file_path = trade_file[0]
        trades = pd.read_csv(trade_file_path)
        output.append(trades)

    return stats, trades
