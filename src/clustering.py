import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def load_volatility_data(filepath: str) -> pd.DataFrame:
    """
    Load the 22-day rolling volatility data from CSV.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    return df


def apply_kmeans(vol_df: pd.DataFrame, n_clusters: int = 10, random_state: int = 42) -> pd.DataFrame:
    """
    Apply K-Means clustering to the rolling volatility data.

    Returns a DataFrame with the same index and a new column 'State'.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    vol_values = vol_df.values.reshape(-1, 1)  # 1D input for KMeans
    states = kmeans.fit_predict(vol_values)

    result = vol_df.copy()
    result["State"] = states
    return result


def plot_clusters(clustered_df: pd.DataFrame, title: str = "Volatility States via K-Means"):
    """
    Plot volatility colored by state clusters.
    """
    plt.figure(figsize=(14, 6))
    palette = sns.color_palette("tab10", n_colors=10)
    sns.scatterplot(x=clustered_df.index, y=clustered_df.iloc[:, 0], hue=clustered_df["State"], palette=palette, legend='full')
    plt.title(title)
    plt.ylabel("22-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()


def save_clustered_data(clustered_df: pd.DataFrame, out_path: str):
    clustered_df.to_csv(out_path)


# --- Example Usage ---
if __name__ == "__main__":
    vol_df = load_volatility_data("data/rolling_volatility.csv")
    clustered = apply_kmeans(vol_df)
    plot_clusters(clustered)
    save_clustered_data(clustered, "data/clustered_volatility.csv")
