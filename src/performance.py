import numpy as np
import pandas as pd


def compute_cumulative_return(daily_returns: pd.Series) -> pd.Series:
    """
    Compute cumulative return from daily returns.
    """
    return (1 + daily_returns).cumprod()


def compute_annualized_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute the annualized return from daily returns.
    """
    cumulative_return = compute_cumulative_return(daily_returns).iloc[-1]
    n_years = len(daily_returns) / periods_per_year
    return cumulative_return ** (1 / n_years) - 1


def compute_annualized_volatility(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute the annualized volatility from daily returns.
    """
    return daily_returns.std() * np.sqrt(periods_per_year)


def compute_sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Compute the Sharpe ratio.
    """
    excess_returns = daily_returns - (risk_free_rate / periods_per_year)
    ann_return = compute_annualized_return(excess_returns, periods_per_year)
    ann_vol = compute_annualized_volatility(excess_returns, periods_per_year)
    return ann_return / ann_vol if ann_vol != 0 else np.nan


def evaluate_strategy(daily_returns: pd.Series, name: str = "Portfolio") -> pd.Series:
    """
    Evaluate and summarize portfolio performance.
    """
    metrics = {
        "Cumulative Return": compute_cumulative_return(daily_returns).iloc[-1] - 1,
        "Annualized Return": compute_annualized_return(daily_returns),
        "Annualized Volatility": compute_annualized_volatility(daily_returns),
        "Sharpe Ratio": compute_sharpe_ratio(daily_returns)
    }
    return pd.Series(metrics, name=name)
