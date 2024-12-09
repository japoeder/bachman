"""
Print a summary of the backtest results.
"""


def backtest_summary(result, strategy_name):
    """
    Print a summary of the backtest results.
    """

    label_dict = {
        "portfolio_value": "Final Portfolio Value",
        "cash": "Final Cash Balance",
        "max_drawdown": "Max Drawdown",
        "drawdown": "Drawdown",
        "total_return": "Total Return",
        "cagr": "CAGR",
        "sharpe": "Sharpe Ratio",
        "volatility": "Volatility",
        "romad": "RoMaD",
        "date": "Date",
    }

    print()
    print()
    print(f"******** {strategy_name} ********")
    for k, v in result.items():
        if k == "max_drawdown":
            for k2, v2 in v.items():
                if k2 == "drawdown":
                    print(f"{label_dict[k]}")
                    print(f"     {label_dict[k2]}: {round(v2 * 100, 2)}%")
                else:
                    print(f"     {label_dict[k2]}: {v2}")
        elif k == "total_return":
            print(f"{label_dict[k]}: {round(v * 100, 2)}%")
        elif k == "portfolio_value" or k == "cash":
            print(f"{label_dict[k]}: ${v:,.2f}")
        else:
            print(f"{label_dict[k]}: {round(v, 5)}")
