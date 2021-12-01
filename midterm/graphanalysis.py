import enum
from os import cpu_count
import matplotlib.pyplot as plt
import multiprocessing as mp
import networkx as nx
from networkx import convert_matrix
import numpy as np
from numpy.core.defchararray import partition
import pandas as pd
import time

from scipy.stats import wasserstein_distance
from typing import List, Union

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


def create_slices(df: pd.DataFrame, T: int):
    """create_slices creates the slices for which we will split a dataframe on"""
    partitions = []
    shifted = df.shift(periods=T).iloc[T:, :]

    dfidxlist = list(df.index)
    shiftedidxlist = list(shifted.index)

    for i in range(len(shiftedidxlist)):
        partitions.append([dfidxlist[i], shiftedidxlist[i]])

    return partitions


def load_graphs(
    path: str = "spearman_25/graphs/adjs.npz", slice: Union[list, tuple] = None
):
    graphs = np.load(path)
    As = (
        [graphs[key] for key in graphs.files]
        if slice is None
        else [graphs[key] for key in graphs.files[slice[0] : slice[1]]]
    )
    Gs = [nx.convert_matrix.from_numpy_array(A) for A in As]

    return Gs, As


def load_curvature_matrices(
    path: str = "spearman_25/graphs/ricci.npz", slice: Union[list, tuple] = None
):
    ks = np.load(path)
    if slice is None:
        ret = [ks[key] for key in ks.files]
    else:
        ret = [ks[key] for key in ks.files[slice[0] : slice[1]]]
    return ret


def ricci_matrix(G: nx.Graph, adj_matrix: np.array):
    """Computes a matrix for the ricci curvature between any two nodes"""
    transition = adj_matrix / adj_matrix.sum(axis=0, keepdims=1)
    shape = adj_matrix.shape
    inds = np.triu_indices(shape[0], k=1)
    wasserstein_matrix = np.zeros(shape)
    sp_matrix = np.zeros(shape)
    sp_lengths = dict(nx.all_pairs_shortest_path_length(G))

    for i in range(len(inds[0])):
        x = inds[0][i]
        y = inds[1][i]
        wasserstein_matrix[x, y] = wasserstein_distance(
            transition[:, x], transition[:, y]
        )
        sp_matrix[x, y] = sp_lengths[x][y]

    ricci = 1 - (wasserstein_matrix / sp_matrix)

    return ricci


def ricci_mp(Gs: List[nx.Graph], As: List[np.array], savepath: str = None):
    partition = list((range(len(Gs))))
    args_to_pass = [(Gs[i], As[i]) for i in range(len(Gs))]

    with mp.Pool(mp.cpu_count() - 2) as p:
        curvature = p.starmap(ricci_matrix, args_to_pass)

    if savepath is not None:
        fnames = {
            f"partition_{i}": R for i, R in enumerate(curvature)
        }  # R = curvature matrix
        np.savez(savepath, **fnames)

    return curvature


class ClusteringCopier:
    def __init__(self, weight="weight"):
        self.weight = weight

    @counted
    def __call__(self, G):
        if self.__call__.calls % 100 == 0:
            print(f"Iteration {self.__call__.calls}")

        return nx.average_clustering(G, weight=self.weight)


def graph_features(Gs: List[nx.Graph], cores: bool = False):
    """Computes features other than the ricci curvature using the graphs"""
    nedges = []
    density = []
    clustering = []
    avgweight = []

    with mp.Pool(2) as p:
        clustering = p.map(ClusteringCopier(), Gs)

    for G in Gs:
        A = nx.linalg.graphmatrix.adjacency_matrix(G)
        nedges.append(G.number_of_edges())
        density.append(nx.density(G))
        avgweight.append(np.mean(A) / 2)

    out = {
        "num_edges": nedges,
        "density": density,
        "avg_clustering": clustering,
        "average_weight": avgweight,
    }

    return out


def construct_df(path: str = "spearman_25/", slice: Union[list, tuple] = None):
    """Constructs a feature dataframe where each row represents a graph.
    NOTE: This function takes a ridculous amount of RAM (~40 GB) to run"""
    path += "/" if "/" not in path else ""
    graphpaths = path + "graphs/adjs.npz"
    ricci_path = path + "graphs/ricci.npz"

    Gs, As = load_graphs(graphpaths, slice=slice)
    Ks = load_curvature_matrices(ricci_path, slice=slice)
    avgK = []

    print("Graphs and curvature matrices are loaded")

    featuredict = graph_features(Gs)
    for i in range(len(Ks)):
        avgK.append(np.nanmean(Ks[i]))

    featuredict["avg_curvature"] = avgK

    df = pd.DataFrame(featuredict)
    df.to_csv(path + "features.csv")

    return df


def main():
    mp.set_start_method("spawn")
    # Gs, As = load_graphs("spearman_125/graphs/adjs.npz", slice=[0, 10])
    # print("Graphs are loaded")
    with pkg_resources.open_text("diffgeo.data", "adjclose.csv") as f:
        df = pd.read_csv(f)
    df.set_index("Date", inplace=True)
    partitions = create_slices(df, T=125)
    timecol = [partition[1] for partition in partitions[1:]]
    t0 = time.time()
    testdf = construct_df(path="spearman_125")
    # ricci = ricci_mp(Gs=Gs, As=As, savepath="spearman_125_mst/graphs/ricci")
    tf = time.time()
    print(testdf)
    print(f"Time elapsed: {tf - t0} seconds.")


# REASONS FOR DIFFERENCES
# 1) I used the Spearman correlation rather than the pearson correlation to better deal with outliers
# 2) I used a weighted graph
# 3) I considered all pairs of nodes; they only considered nodes for which d(x, y) = 1
# 4) Within a community, given how we define the curvature, it should NEVER be negative
# Source for 4) is https://arxiv.org/pdf/1806.00676.pdf

if __name__ == "__main__":
    main()
