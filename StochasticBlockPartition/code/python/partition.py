"""Parameters for partitioning.
"""

from collections import namedtuple
from typing import List
from argparse import Namespace

import numpy as np
from scipy import sparse as sparse

from graph import Graph

class Partition():
    """Stores the current partitioning results.
    """

    def __init__(self, num_blocks: int, out_neighbors: List[np.ndarray], args: Namespace) -> None:
        """Creates a new Partition object.

            Parameters
            ---------
            num_blocks : int
                    the number of blocks in the current partition
            out_neighbors : List[np.ndarray]
                    list of outgoing edges for each node
            args : Namespace
                    the command-line arguments
        """
        self.num_blocks = num_blocks
        self.block_assignment = np.array(range(num_blocks))
        self.overall_entropy = np.inf
        self.interblock_edge_count = np.zeros((num_blocks, num_blocks))
        self.block_degrees = np.zeros(num_blocks)
        self.block_degrees_out = np.zeros(num_blocks)
        self.block_degrees_in = np.zeros(num_blocks)
        self.num_blocks_to_merge = int(self.num_blocks * args.blockReductionRate)
        self.initialize_edge_counts(out_neighbors, args.sparse)
    # End of __init__()

    def initialize_edge_counts(self, out_neighbors: List[np.ndarray], use_sparse: bool):
        """Initialize the edge count matrix and block degrees according to the current partition

            Parameters
            ----------
            out_neighbors : list of ndarray; list length is N, the number of nodes
                        each element of the list is a ndarray of out neighbors, where the first column is the node
                        indices and the second column the corresponding edge weights
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
        # partition for the high, best, and low number of blocks so far
        self.block_assignment = [[], [], []]  # type: List[np.array]
        # edge count matrix for the high, best, and low number of blocks so far
        self.interblock_edge_count = [[], [], []]  # type: List[np.array]
        # block degrees for the high, best, and low number of blocks so far
        self.block_degrees = [[], [], []]  # type: List[np.array]
        # out block degrees for the high, best, and low number of blocks so far
        self.block_degrees_out = [[], [], []]  # type: List[np.array]
        # in block degrees for the high, best, and low number of blocks so far
        self.block_degrees_in = [[], [], []]  # type: List[np.array]
        # overall entropy for the high, best, and low number of blocks so far
        self.overall_entropy = [np.Inf, np.Inf, np.Inf]  # type: List[float]
        # number of blocks for the high, best, and low number of blocks so far
        self.num_blocks = [[], [], []]  # type: List[int]
        self.optimal_num_blocks_found = False
    
    def update(self, partition: Partition):
        """If the entropy of the current partition is the best so far, moves the middle triplet
        values to the left or right depending on the current partition's block number.

        Then, updates the appropriate triplet with the results of the newest partition.

            Parameters
            ---------
            partition : Partition
                    the most recent partitioning results
        """
        if partition.overall_entropy <= self.overall_entropy[1]:  # if the current partition is the best so far
            old_index = 0 if self.num_blocks[1] > partition.num_blocks else 2
            self.block_assignment[old_index] = self.block_assignment[1]
            self.interblock_edge_count[old_index] = self.interblock_edge_count[1]
            self.block_degrees[old_index] = self.block_degrees[1]
            self.block_degrees_out[old_index] = self.block_degrees_out[1]
            self.block_degrees_in[old_index] = self.block_degrees_in[1]
            self.overall_entropy[old_index] = self.overall_entropy[1]
            self.num_blocks[old_index] = self.num_blocks[1]
            index = 1
        else:  # the current partition is not the best so far
            # if the current number of blocks is smaller than the best number of blocks so far
            index = 2 if self.num_blocks[1] > partition.num_blocks else 0

        self.block_assignment[index] = partition.block_assignment
        self.interblock_edge_count[index] = partition.interblock_edge_count
        self.block_degrees[index] = partition.block_degrees
        self.block_degrees_out[index] = partition.block_degrees_out
        self.block_degrees_in[index] = partition.block_degrees_in
        self.overall_entropy[index] = partition.overall_entropy
        self.num_blocks[index] = partition.num_blocks
    # End of update()

    def extract_partition(self, index: int) -> Partition:
        """Extracts a partition from the given triplet indexes.

            Parameters
            ----------
            index : int
                    the triplet index from which to extract the partition
            
            Returns:
            --------
            partition : Partition
                    the extracted partition
        """
        raise NotImplementedError()
    # End of extract_partition()
