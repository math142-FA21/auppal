# NOTE: Graph generation takes approximately 240 / (N-2) minutes, where N = The number of CPU cores you have
# That means, for an 8-core CPU, it will take about 40 minutes to generate all of the graphs

import dcor
import functools
import json
import networkx as nx
from networkx.algorithms.components.connected import is_connected
import matplotlib.pyplot as plt
import multiprocessing as mp
from networkx.algorithms.connectivity.kcomponents import _generate_partition
import numpy as np
import pandas as pd
import time

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

from typing import Callable


def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped


def _check_symmetric(matrix, rtol=1e-05, atol=1e-08):
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)


def matrix_sort(matrix: np.array):
    mshape = matrix.shape
    return np.dstack(np.unravel_index(np.argsort(matrix.ravel()), mshape))


def compute_metric(df: pd.DataFrame, metric: str = "spearman"):
    """Computation of various metrics"""
    if metric == "spearman":
        corrmat = np.array(df.corr(method="spearman"))
        metmat = np.sqrt(2 * (1 - corrmat))
    elif metric == "dcor":  # Takes about 5 min
        mat = df.to_numpy()
        metmat = np.zeros(
            shape=(mat.shape[1], mat.shape[1])
        )  # Want an n x n matrix, where n = number of cols
        idxs = np.triu_indices(n=mat.shape[1])
        for i in range(len(idxs[0])):
            x = idxs[0][i]
            y = idxs[1][i]
            metmat[x][y] = dcor.distance_correlation(mat[:, x], mat[:, y])
    return metmat


def create_slices(df: pd.DataFrame, T: int):
    """create_slices creates the slices for which we will split a dataframe on"""
    partitions = []
    shifted = df.shift(periods=T).iloc[T:, :]

    dfidxlist = list(df.index)
    shiftedidxlist = list(shifted.index)

    for i in range(len(shiftedidxlist)):
        partitions.append([dfidxlist[i], shiftedidxlist[i]])

    return partitions

@counted
def generate_graph(
    df: pd.DataFrame,
    metric_matrix: np.array = None,
    partition: list = None,
    symmetric: bool = None,
    metric: str = "spearman",
    mst: bool = False,
    save: str = None,
):
    """Generates a graph of a dataframe using a matrix representing the metric"""
    if generate_graph.calls % 100 == 0:
        print(f"Iteration {generate_graph.calls}.")
    if partition is not None:
        df = df.loc[partition[0] : partition[1], :]

    if metric_matrix is None:
        metric_matrix = compute_metric(df=df, metric=metric)

    # If metric is symmetric, get rid of the top half of the matrix (plus the diagonal):
    if symmetric is None:
        symmetric = _check_symmetric(metric_matrix)
    if symmetric:
        metric_matrix = np.tril(metric_matrix)

    # Compute the absolute value of the metric matrix
    metric_matrix[np.isclose(metric_matrix, np.zeros(metric_matrix.shape))] = np.nan
    sorted_metric = matrix_sort(metric_matrix)[0]

    # Generate the MST
    # Initialize an empty graph
    gengraph = nx.Graph()
    gengraph.add_nodes_from(list(range(len((df.columns)))))

    # Iteratively add links until G is connected (to turn it into a spanning tree)
    while not nx.is_connected(gengraph):
        row, col = sorted_metric[0][0], sorted_metric[0][1]
        d = metric_matrix[row, col]
        if not mst:
            if not gengraph.has_edge(row, col):
                gengraph.add_edge(row, col, weight=d)
        else:
            if not nx.has_path(gengraph, row, col):
                gengraph.add_edge(row, col, weight=d)
        sorted_metric = np.delete(sorted_metric, obj=0, axis=0)

    if save is not None:
        adj = nx.convert_matrix.to_numpy_array(gengraph)
        np.savez(save, adj)

    return gengraph


def graphgen_sync(
    df: pd.DataFrame, T: int = 25, metric: str = "spearman", folder: str = None
):
    graphs = []
    partitions = create_slices(df, T=T)
    dfs = [df.loc[partition[0] : partition[1]] for partition in partitions]
    for df in dfs:
        graphs.append(generate_graph(df, metric=metric))
    return graphs


class Copier(object):
    """Class that allows a psuedo-lambda function to be passed into graphgen"""

    def __init__(
        self,
        metric_matrix=None,
        partition=None,
        symmetric=None,
        metric="spearman",
        mst=None,
        save=None,
    ):
        self.metric_matrix = metric_matrix
        self.partition = partition
        self.symmetric = symmetric
        self.metric = metric
        self.mst = mst
        self.save = save

    def __call__(self, df):
        return generate_graph(
            df,
            self.metric_matrix,
            self.partition,
            self.symmetric,
            self.metric,
            self.mst,
            self.save,
        )


def graphgen_mp(
    df: pd.DataFrame, T: int = 25, metric: str = "spearman", mst: bool = False, folder: str = None
):
    partitions = create_slices(df, T=T)
    dfs = [df.loc[partition[0] : partition[1]] for partition in partitions]
    with mp.Pool(mp.cpu_count() - 2) as p:
        graphs = p.map(
            Copier(
                metric_matrix=None,
                partition=None,
                symmetric=True,
                metric=metric,
                mst=mst,
                save=None,
            ),
            dfs,
        )

        As = []

        for g in graphs:
            As.append(nx.convert_matrix.to_numpy_array(g))

        return graphs, As


def main():
    with pkg_resources.open_text("diffgeo.data", "adjclose.csv") as f:
        df = pd.read_csv(f)
    df.set_index("Date", inplace=True)
    # Calculate log returns for each column
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]

    for c in [c for c in df.columns if df[c].dtype in numerics]:
        df[c] = np.log10(df[c])
    df = df.diff().iloc[1:, :]
    # df = df.loc["2020-01-01":, :]

    t0 = time.time()
    Gs, As = graphgen_mp(df, T=125, mst=True)
    # testmetric = compute_metric(df, metric="spearman")
    tf = time.time()
    print(f"Time Elapsed: {tf - t0} seconds")
    # print(testmetric)
    fnames = {f"partition_{i}": A for i, A in enumerate(As)}
    np.savez("spearman_125_mst/graphs/adjs", **fnames)


if __name__ == "__main__":
    main()
