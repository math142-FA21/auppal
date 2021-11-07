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


def create_slices(df: pd.DataFrame, T: int):
    """create_slices creates the slices for which we will split a dataframe on"""
    partitions = []
    shifted = df.shift(periods=T).iloc[T:, :]

    dfidxlist = list(df.index)
    shiftedidxlist = list(shifted.index)

    for i in range(len(shiftedidxlist)):
        partitions.append([dfidxlist[i], shiftedidxlist[i]])

    return partitions


def load_graphs(slice: Union[list, tuple] = None):
    graphs = np.load("graphs/spearman_25_difflog/adjs.npz")
    As = (
        [graphs[key] for key in graphs.files]
        if slice is None
        else [graphs[key] for key in graphs.files[slice[0] : slice[1]]]
    )
    Gs = [nx.convert_matrix.from_numpy_array(A) for A in As]

    return Gs, As


def load_curvature_matrices(slice: Union[list, tuple] = None):
    ks = np.load("graphs/spearman_25_difflog/ricci.npz")
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


def ricci_mp(Gs: List[nx.Graph], As: List[np.array]):
    partition = list((range(len(Gs))))
    args_to_pass = [(Gs[i], As[i]) for i in range(len(Gs))]

    with mp.Pool(mp.cpu_count() - 2) as p:
        curvature = p.starmap(ricci_matrix, args_to_pass)

    return curvature


def graph_features(Gs: List[nx.Graph]):
    """Computes features other than the ricci curvature using the graphs"""
    nedges = []
    density = []
    clustering = []

    for G in Gs:
        nedges.append(G.number_of_edges())
        density.append(nx.density(G))
        clustering.append(nx.average_clustering(G, weight="weight"))

    out = {"num_edges": nedges, "density": density, "avg_clustering": clustering}

    return out


def main():
    Gs_start, As_start = load_graphs(slice=[0, 10])
    Gs_rec, As_rec = load_graphs(slice=[1025, 1035])
    print("Graphs are loaded")
    with pkg_resources.open_text("diffgeo.data", "adjclose.csv") as f:
        df = pd.read_csv(f)
    df.set_index("Date", inplace=True)
    partitions = create_slices(df, T=25)
    timecol = [partition[1] for partition in partitions[1:]]
    timedecoder = {d: idx for idx, d in enumerate(timecol)}

    t0 = time.time()
    Ginit, Grec = Gs_start[0], Gs_rec[0]
    print(f"Number of intial edges: {nx.number_of_edges(Ginit)}")
    print(f"Number of recession edges: {nx.number_of_edges(Grec)}")
    nx.draw(Ginit, node_size=50, node_color="#283AF1")
    plt.savefig("figures/initialgraph.png", dpi=400)
    plt.show()
    nx.draw(Grec, node_size=50, node_color="#F31A1A")
    plt.savefig("figures/recessiongraph.png", dpi=400)
    plt.show()
    tf = time.time()
    print(f"Time elapsed: {tf - t0} seconds.")

    print(df.head())
    plt.show()


# REASONS FOR DIFFERENCES
# 1) I used the Spearman correlation rather than the pearson correlation to better deal with outliers
# 2) I used a weighted graph
# 3) I considered all pairs of nodes; they only considered nodes for which d(x, y) = 1
# 4) Within a community, given how we define the curvature, it should NEVER be negative
# Source for 4) is https://arxiv.org/pdf/1806.00676.pdf

if __name__ == "__main__":
    main()
