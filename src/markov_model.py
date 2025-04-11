import numpy as np
import pandas as pd
from scipy.stats import dirichlet


def compute_transition_counts(states: pd.Series, n_states: int = 10) -> np.ndarray:
    """
    Count transitions between market states.

    Parameters:
        states: Pandas Series of state labels (e.g., 0-9)
        n_states: Number of distinct states

    Returns:
        A n_states x n_states matrix of transition counts
    """
    counts = np.zeros((n_states, n_states), dtype=int)

    for t in range(len(states) - 1):
        i = states.iloc[t]
        j = states.iloc[t + 1]
        counts[i, j] += 1

    return counts


def gibbs_sampling_dirichlet(counts: np.ndarray, alpha: float = 1.0, num_iter: int = 1000) -> np.ndarray:
    """
    Run Gibbs sampling to compute posterior transition probabilities.

    Parameters:
        counts: Transition count matrix
        alpha: Concentration parameter for Dirichlet prior
        num_iter: Number of Gibbs iterations

    Returns:
        Posterior mean transition matrix
    """
    n_states = counts.shape[0]
    transition_samples = np.zeros((num_iter, n_states, n_states))

    # Prior: Dirichlet(alpha,...,alpha) for each row
    alpha_matrix = np.full(counts.shape, alpha)

    for it in range(num_iter):
        for i in range(n_states):
            posterior_alpha = alpha_matrix[i] + counts[i]
            transition_samples[it, i] = dirichlet.rvs(posterior_alpha)

    # Return the average over samples
    return transition_samples.mean(axis=0)


def save_transition_matrix(matrix: np.ndarray, out_path: str):
    df = pd.DataFrame(matrix, index=[f"State_{i+1}" for i in range(matrix.shape[0])],
                      columns=[f"State_{j+1}" for j in range(matrix.shape[1])])
    df.to_csv(out_path)
    return df


# --- Example usage ---
if __name__ == "__main__":
    # Load clustered state data
    clustered_df = pd.read_csv("data/clustered_volatility.csv", parse_dates=["Date"], index_col="Date")
    states = clustered_df["State"]

    # Build transition count matrix
    counts = compute_transition_counts(states, n_states=10)

    # Run Gibbs sampling with Dirichlet prior
    transition_matrix = gibbs_sampling_dirichlet(counts, alpha=1.0, num_iter=1000)

    # Save to CSV
    df_tm = save_transition_matrix(transition_matrix, "data/transition_matrix.csv")
    print(df_tm.round(4))
