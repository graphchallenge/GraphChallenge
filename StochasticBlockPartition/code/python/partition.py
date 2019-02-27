"""Parameters for partitioning.
"""

from collections import namedtuple
from typing import List

import numpy as np
from scipy import sparse as sparse

from graph import Graph

class Partition():
    def __init__(self, num_blocks: int, out_neighbors: List[np.ndarray], use_sparse: bool) -> None:
        self.num_blocks = num_blocks
        self.block_assignment = np.array(range(num_blocks))
        self.overall_entropy = np.inf
        self.interblock_edge_count = np.zeros((num_blocks, num_blocks))
        self.block_degrees = np.zeros(num_blocks)
        self.block_degrees_out = np.zeros(num_blocks)
        self.block_degrees_in = np.zeros(num_blocks)
        self.initialize_edge_counts(out_neighbors, use_sparse)

    def initialize_edge_counts(self, out_neighbors: List[np.ndarray], use_sparse: bool):
        """Initialize the edge count matrix and block degrees according to the current partition

        Parameters
        ----------
        out_neighbors : list of ndarray; list length is N, the number of nodes
                    each element of the list is a ndarray of out neighbors, where the first column is the node indices
                    and the second column the corresponding edge weights
        B : int
                    total number of blocks in the current partition
        b : ndarray (int)
                    array of block assignment for each node
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
        d_out : ndarray (int)
                    the current out degree of each block
        d_in : ndarray (int)
                    the current in degree of each block
        d : ndarray (int)
                    the current total degree of each block

        Notes
        -----
        Compute the edge count matrix and the block degrees from scratch
        """
        if use_sparse: # store interblock edge counts as a sparse matrix
            self.interblock_edge_count = sparse.lil_matrix((self.num_blocks, self.num_blocks), dtype=int)
        else:
            self.interblock_edge_count = np.zeros((self.num_blocks, self.num_blocks), dtype=int)
        # compute the initial interblock edge count
        for v in range(len(out_neighbors)):
            if len(out_neighbors[v]) > 0:
                k1 = self.block_assignment[v]
                k2, inverse_idx = np.unique(self.block_assignment[out_neighbors[v][:, 0]], return_inverse=True)
                count = np.bincount(inverse_idx, weights=out_neighbors[v][:, 1]).astype(int)
                self.interblock_edge_count[k1, k2] += count
        # compute initial block degrees
        self.block_degrees_out = np.asarray(self.interblock_edge_count.sum(axis=1)).ravel()
        self.block_degrees_in = np.asarray(self.interblock_edge_count.sum(axis=0)).ravel()
        self.block_degrees = self.block_degrees_out + self.block_degrees_in
    # End of initialize_edge_counts()


class PartitionTriplet():
    def __init__(self) -> None:
        self.block_assignment = [[], [], []]  # partition for the high, best, and low number of blocks so far
        self.interblock_edge_count = [[], [], []]  # edge count matrix for the high, best, and low number of blocks so far
        self.block_degrees = [[], [], []]  # block degrees for the high, best, and low number of blocks so far
        self.block_degrees_out = [[], [], []]  # out block degrees for the high, best, and low number of blocks so far
        self.block_degrees_in = [[], [], []]  # in block degrees for the high, best, and low number of blocks so far
        self.overall_entropy = [np.Inf, np.Inf, np.Inf] # overall entropy for the high, best, and low number of blocks so far
        self.num_blocks = [[], [], []]  # number of blocks for the high, best, and low number of blocks so far
        self.optimal_num_blocks_found = False