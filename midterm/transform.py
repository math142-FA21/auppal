# Script to ETL the graph data to be ready for visualization and final models
# Also runs the regressions

import numpy as np
import pandas as pd
import statsmodels.api as sm
import time
import yfinance as yf

from graphgen import create_slices
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from stargazer.stargazer import Stargazer

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


def _load_csv(gtype: str = "spearman_125_mst"):
    """Loads and processes the feature information for a graph type.
    Returns output as a pandas DataFrame ready to be used in analysis.
    Saves this dataframe to {gtype}/graph_features_transformed.csv
    This only needs to be run once -- once these datasets are produced, it does not need to be run again."""
    path = gtype + "/" if "/" not in gtype else gtype
    path += "graph_features"

    try:
        df = pd.read_csv(path + ".csv", index_col="Unnamed: 0")
    except ValueError:
        df = pd.read_csv(path)

    # Compute times associated with the partition
    with pkg_resources.open_text("diffgeo.data", "adjclose.csv") as f:
        df_times = pd.read_csv(f)
    df_times.set_index("Date", inplace=True)

    partitionOffset = gtype.split("_")[1]
    if "/" in partitionOffset:
        partitionOffset = partitionOffset[:-1]
    partition_times = [
        item[1] for item in create_slices(df_times, T=int(partitionOffset))
    ][1:]

    df["Time"] = partition_times

    # Fix the computation of the curvature term:
    # Multiply the adjustment term by 100 because scipy's Wasserstein distance
    # needs to be calculated by using empirical data (absolute numbers) instead
    # of the true distribution (cols of transition matrix)

    df["fixterm"] = 100 * (1 - df["avg_curvature"])
    df["avg_curvature"] = df["avg_curvature"] - df["fixterm"]
    df.drop(labels="fixterm", axis=1, inplace=True)

    # Add SP500 data to the dataframe

    sp = yf.download(tickers="SPY", period="max", interval="1d", auto_adjust=True)[
        "Close"
    ]
    sp = np.log10(sp).diff()
    sp = sp.loc[partition_times[0] : partition_times[-1]]
    sp = sp.reset_index(drop=True)
    df["sp500"] = sp

    df.to_csv(path + "_transformed.csv")

    return df


def granger_tests(gtype: str = "spearman_125_mst"):
    path = gtype + "/" if "/" not in gtype else gtype
    path += "graph_features_transformed.csv"
    df = pd.read_csv(path)
    mst = "mst" in gtype.lower()  # Helps determine the set of variables to use in tests
    if not mst:
        cols_to_iterate = [
            "density",
            "avg_clustering",
            "average_weight",
            "avg_curvature",
            "num_edges",
        ]
    else:
        cols_to_iterate = [
            "average_weight",
            "avg_curvature",
        ]  # No clustering / constant density and number of edges
    for col in cols_to_iterate:
        print(f"GRANGER CAUSALITY TESTS FOR COLUMN: {col}")
        print("----------------------------------------------------")
        data = df[["sp500", col]]
        grangercausalitytests(data, 5)
        print("\n")


def var_model(gtype: str = "spearnman_125", lags: int = 5):
    """Creates VAR model for the the graph of type gtype. gtype is specified as {metric}_{windowlength}.l
    For example, for a spearman correlation metric and a sliding window length of 25, gtype='spearman_25'.
    Note that this corresponds to a folder with this name that has been generated by graphgen.py and graphanalysis.py.

    NOTE: VAR(p): vector autoregression of lag order p

    Runs VAR(lags) models with the following specifications:

    If the graph type is an MST:
        sp500 ~ avg_curvature + average_weight
    If the graph type is not an MST:
        sp500 ~ density + avg_clustering + average_weight + "avg_curvature

    exports the regression results to gtype/figures using stargazer. Regressions are run using statsmodels.
    """
    mst = "mst" in gtype.lower()
    path = gtype + "/" if "/" not in gtype else gtype
    path += "graph_features_transformed.csv"
    df = pd.read_csv(path)

    # Determine the columns to use in the model:

    if not mst:
        cols_to_iterate = [
            "density",
            "avg_clustering",
            "average_weight",
            "avg_curvature",
            "sp500",
        ]
    else:
        cols_to_iterate = ["average_weight", "avg_curvature", "sp500"]

    # Generate the lagged features:
    for col in cols_to_iterate:
        for lag in range(lags):
            df[f"{col}_lag{lag + 1}"] = df[col].shift(lag + 1)
    df = df.iloc[lags:, :]

    Y = df["sp500"]
    X = df[[f"{col}_lag{lag+1}" for col in cols_to_iterate for lag in range(lags)]]
    mod_unfit = sm.RLM(Y, X, M=sm.robust.norms.HuberT())
    model = sm.RLM(Y, X, M=sm.robust.norms.HuberT()).fit()

    # Estimate the R^2 for the model
    # See https://stackoverflow.com/questions/31655196/how-to-get-r-squared-for-robust-regression-rlm-in-statsmodels

    r2_wls = (
        sm.WLS(mod_unfit.endog, mod_unfit.exog, weights=model.weights).fit().rsquared
    )
    print(f"R^2 for model {gtype}: {r2_wls}")

    # Perform an F-test to see if all the coefs are jointly statistically different from 0
    A = np.identity(len(model.params))
    print(model.f_test(A))

    return model


def main(gtype="spearman_125"):
    s25 = var_model("spearman_25")
    s125 = var_model("spearman_125")
    s25_mst = var_model("spearman_25_mst")
    s125_mst = var_model("spearman_125_mst")

    stargazer = Stargazer([s25, s125, s25_mst, s125_mst])
    stargazer.title("SP500 VAR Results")
    stargazer.custom_columns(
        ["Window 25", "Window 125", "Window 25 (MST)", "Window 125 (MST)"], [1, 1, 1, 1]
    )
    stargazer.show_model_numbers(False)

    return stargazer.render_latex()


def main2():
    s125 = var_model("spearman_125")
    s25_mst = var_model("spearman_25_mst")
    s125_mst = var_model("spearman_125_mst")

    stargazer = Stargazer([s125, s25_mst, s125_mst])
    stargazer.title("SP500 VAR Results")
    stargazer.custom_columns(
        ["Window 125", "Window 25 (MST)", "Window 125 (MST)"], [1, 1, 1]
    )

    return stargazer.render_latex()


if __name__ == "__main__":
    print(main())
