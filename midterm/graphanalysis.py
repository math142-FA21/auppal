import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

def main():
    Gs = np.load("graphs/spearman_25_difflog/adjs.npz")
    g0 = nx.convert_matrix.from_numpy_array(Gs["partition_0"])

    nx.draw(g0, with_labels=False, node_size=100)
    plt.show()
    

if __name__ == "__main__":
    main()