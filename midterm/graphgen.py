import networkx as nx
from networkx.algorithms.components.connected import is_connected
import numpy as np
import pandas as pd
import time

from bidict import bidict

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


def _check_symmetric(matrix, rtol=1e-05, atol=1e-08):
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)


def matrix_sort(matrix: np.array):
    mshape = matrix.shape
    return np.dstack(np.unravel_index(np.argsort(matrix.ravel()), mshape))


def generate_graph(df: pd.DataFrame, metric_matrix, symmetric: bool = None):
    # If metric is symmetric, get rid of the top half of the matrix (plus the diagonal):
    if symmetric is None:
        symmetric = _check_symmetric(metric_matrix)
    if symmetric:
        metric_matrix = np.tril(metric_matrix)

    # Compute the absolute value of the metric matrix
    metric_matrix[np.isclose(metric_matrix, np.zeros(metric_matrix.shape))] = np.nan
    sorted_metric = matrix_sort(metric_matrix)[0]
    print(sorted_metric.shape)

    # Generate the MST
    # Initialize an empty graph
    G = nx.Graph()
    G.add_nodes_from(list(range(len((df.columns)))))
    while not nx.is_connected(G):
        row, col = sorted_metric[0][0], sorted_metric[0][1]
        d = metric_matrix[row, col]
        if not G.has_edge(row, col):
            G.add_edge(row, col, weight=d)
        sorted_metric = np.delete(sorted_metric,obj=0, axis=0)

    return None


if __name__ == "__main__":
    with pkg_resources.open_text("diffgeo.data", "adjclose.csv") as f:
        df = pd.read_csv(f)
    df.set_index("Date", inplace=True)
    # Calculate log returns for each column
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]

    for c in [c for c in df.columns if df[c].dtype in numerics]:
        df[c] = np.log10(df[c])
    df = df.diff().iloc[1:, :]
    # df = df.loc[:, "AAPL":"ADM"]

    # Encode the relationship between the column names and index numbers in the graph
    encoder = bidict(enumerate(list(df.columns)))

    corrmat = np.array(df.corr(method="spearman"))
    metric = np.sqrt(2 * (1 - corrmat))
    t0 = time.time()
    generate_graph(df=df, metric_matrix=metric)
    tf = time.time()

    print(f"Time Elapsed: {tf - t0} seconds")
