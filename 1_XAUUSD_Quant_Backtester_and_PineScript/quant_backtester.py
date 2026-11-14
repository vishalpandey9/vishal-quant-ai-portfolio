import numpy as np
import pandas as pd

def generate_sample_xauusd_trades(n_trades=120, seed=42):
    np.random.seed(seed)
    sessions = np.random.choice(["London", "New York", "London-NY Overlap"], size=n_trades, p=[0.4, 0.35, 0.25])
    directions = np.random.choice(["LONG", "SHORT"], size=n_trades)
    rr_targets = np.random.choice([1.5, 2.0, 2.5], size=n_trades, p=[0.3, 0.5, 0.2])
    # Realistic 48% win rate with positive expectancy
    outcomes = np.random.choice(["WIN", "LOSS"], size=n_trades, p=[0.48, 0.52])
    r_multiples = np.where(outcomes == "WIN", rr_targets, -1.0)
    
    df = pd.DataFrame({
        "Trade_ID": range(1, n_trades + 1),
        "Asset": "XAUUSD",
        "Session": sessions,
        "Setup": "3m_FVG_OrderBlock",
        "Direction": directions,
        "Risk_Pct": 1.0,
        "Target_RR": rr_targets,
        "Outcome": outcomes,
        "R_Multiple": r_multiples
    })
    
    # Calculate Equity Curve starting at $10,000 prop evaluation account
    equity = [10000.0]
    for r in df["R_Multiple"]:
        next_eq = equity[-1] * (1 + (r * 0.01))
        equity.append(round(next_eq, 2))
    df["Account_Balance_USD"] = equity[1:]
    df["Peak_Balance"] = df["Account_Balance_USD"].cummax()
    df["Drawdown_Pct"] = round(((df["Account_Balance_USD"] - df["Peak_Balance"]) / df["Peak_Balance"]) * 100, 2)
    return df

def evaluate_performance(df):
    win_rate = (df["Outcome"] == "WIN").mean() * 100
    avg_r = df["R_Multiple"].mean()
    total_return_pct = ((df["Account_Balance_USD"].iloc[-1] - 10000.0) / 10000.0) * 100
    max_dd = df["Drawdown_Pct"].min()
    sharpe = (df["R_Multiple"].mean() / df["R_Multiple"].std()) * np.sqrt(252)
    
    print("=== XAUUSD QUANTITATIVE BACKTEST SUMMARY ===")
    print(f"Total Trades Executed : {len(df)}")
    print(f"Win Rate              : {win_rate:.2f}%")
    print(f"Expectancy per Trade  : {avg_r:.2f}R")
    print(f"Net Return ($10k Base): +{total_return_pct:.2f}% (${df['Account_Balance_USD'].iloc[-1]:,.2f})")
    print(f"Maximum Drawdown      : {max_dd:.2f}% (Prop Firm Limit: -10.00%)")
    print(f"Annualized Sharpe     : {sharpe:.2f}")

if __name__ == "__main__":
    trades_df = generate_sample_xauusd_trades()
    trades_df.to_csv("xauusd_backtest_log.csv", index=False)
    evaluate_performance(trades_df)
