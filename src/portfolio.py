import numpy as np
import pandas as pd
from scipy.optimize import minimize


def equal_weight(n_assets: int) -> np.ndarray:
    """Equal Weight strategy"""
    return np.ones(n_assets) / n_assets


def min_variance(cov_matrix: np.ndarray) -> np.ndarray:
    """Minimum Variance Portfolio using quadratic programming"""
    n = cov_matrix.shape[0]

    def objective(w):
        return w.T @ cov_matrix @ w

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]

    x0 = np.ones(n) / n
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    return res.x if res.success else x0


def max_diversification(cov_matrix: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Maximum Diversification Ratio"""
    n = cov_matrix.shape[0]

    def objective(w):
        weighted_vol = np.dot(w, vol)
        portfolio_vol = np.sqrt(w.T @ cov_matrix @ w)
        return -weighted_vol / portfolio_vol  # maximize DR = sum(w_i * σ_i) / σ_p

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    x0 = np.ones(n) / n

    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    return res.x if res.success else x0


def equal_risk_contribution(cov_matrix: np.ndarray) -> np.ndarray:
    """Equal Risk Contribution Portfolio (Risk Parity)"""
    n = cov_matrix.shape[0]

    def portfolio_risk_contribution(w):
        sigma_p = np.sqrt(w.T @ cov_matrix @ w)
        marginal_contrib = cov_matrix @ w
        total_contrib = w * marginal_contrib
        return total_contrib

    def objective(w):
        trc = portfolio_risk_contribution(w)
        avg_contrib = np.mean(trc)
        return np.sum((trc - avg_contrib) ** 2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    x0 = np.ones(n) / n

    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    return res.x if res.success else x0


def get_portfolio_weights(returns: pd.DataFrame, method: str = "minvar") -> np.ndarray:
    """
    Wrapper to calculate weights using one of four strategies.

    Parameters:
        returns: DataFrame of daily returns
        method: one of ["equal", "minvar", "erc", "maxdiv"]

    Returns:
        np.array of asset weights
    """
    cov = returns.cov()
    vol = returns.std()

    if method == "equal":
        return equal_weight(len(returns.columns))
    elif method == "minvar":
        return min_variance(cov)
    elif method == "erc":
        return equal_risk_contribution(cov)
    elif method == "maxdiv":
        return max_diversification(cov, vol)
    else:
        raise ValueError(f"Unknown method: {method}")
