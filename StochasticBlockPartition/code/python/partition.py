"""Parameters for partitioning.
"""

from collections import namedtuple
from typing import List, Dict
from argparse import Namespace

import numpy as np
from scipy import sparse as sparse

from graph import Graph

class Partition():
    """Stores the current partitioning results.
    """

    def __init__(self, num_blocks: int, out_neighbors: List[np.ndarray], args: Namespace,
        block_assignment: np.ndarray = None) -> None:
        """Creates a new Partition object.

            Parameters
            ---------
            num_blocks : int
                    the number of blocks in the current partition
            out_neighbors : List[np.ndarray]
                    list of outgoing edges for each node
            args : Namespace
                    the command-line arguments
            block_assignment : np.ndarray [int]
                    the provided block assignment. Default = None
        """
        self.num_blocks = num_blocks
        if block_assignment is None:
            self.block_assignment = np.array(range(num_blocks))
        else:
            self.block_assignment = block_assignment
        self.overall_entropy = np.inf
        self.interblock_edge_count = [[]]  # type: np.array
        self.block_degrees = np.zeros(num_blocks)
        self.block_degrees_out = np.zeros(num_blocks)
        self.block_degrees_in = np.zeros(num_blocks)
        self._args = args
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

    def clone_with_true_block_membership(self, out_neighbors: List[np.ndarray], 
        true_block_membership: np.ndarray) -> 'Partition':
        """Creates a new Partition object for the correctly partitioned graph.

            Parameters
            ----------
            out_neighbors : List[np.ndarray]
                    list of outgoing edges for each node
            true_block_membership : np.ndarray [int]
                    the correct block membership for every vertex
            
            Returns
            ------
            partition : Partition
                    the Partition when the partitioning is 100% accurate
        """
        partition = Partition(len(true_block_membership), out_neighbors, self._args)
        partition.block_assignment = true_block_membership
        partition.num_blocks = len(np.unique(true_block_membership))
        partition.initialize_edge_counts(out_neighbors, self._args.sparse)
        return partition
    # End of clone_with_true_block_membership()

    @staticmethod
    def from_sample(num_blocks: int, out_neighbors: List[np.ndarray],
        sample_block_assignment: np.ndarray, mapping: Dict[int,int], args: 'argparse.Namespace') -> 'Partition':
        """Creates a new Partition object from the block assignment array.

            Parameters
            ----------
            num_blocks : int
                    the number of blocks in the current partition
            out_neighbors : List[np.ndarray]
                    list of outgoing edges for each node
            sample_block_assignment : np.ndarray [int]
                    the partitioning results on the sample
            mapping : Dict[int,int]
                    the mapping of sample vertex indices to full graph vertex indices
            args : argparse.Namespace
                    the command-line args passed to the program

            Returns
            -------
            partition : Partition
                    the partition created from the sample
        """
        block_assignment = np.full(len(out_neighbors), -1)
        for key, value in mapping.items():
            block_assignment[key] = sample_block_assignment[value]
        next_block = num_blocks
        for vertex in range(len(out_neighbors)):
            if block_assignment[vertex] == -1:
                block_assignment[vertex] = next_block
                next_block += 1
        for vertex in range(len(out_neighbors)):
            if block_assignment[vertex] >= num_blocks:
                # count links to each block
                block_counts = np.zeros(num_blocks)
                for neighbor in out_neighbors[vertex]:
                    neighbor_block = block_assignment[neighbor[0]]
                    if neighbor_block < num_blocks:
                        block_counts[neighbor_block] += 1
                # pick block with max link
                block_assignment[vertex] = np.argmax(block_counts)
        return Partition(num_blocks, out_neighbors, args, block_assignment)
    # End of from_sample()


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
        self.num_blocks = [0, 0, 0]  # type: List[int]
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
