"""Contains code for the block merge part of the baseline algorithm.
"""

import numpy as np

from partition_baseline_support import propose_new_partition
from partition_baseline_support import compute_new_rows_cols_interblock_edge_count_matrix
from partition_baseline_support import compute_new_block_degrees
from partition_baseline_support import compute_delta_entropy
from partition_baseline_support import carry_out_best_merges

from partition import Partition


def merge_blocks(partition: Partition, num_agg_proposals_per_block: int, use_sparse_matrix: bool,
    num_blocks_to_merge: int, out_neighbors: np.array) -> Partition:
    """The block merge portion of the algorithm.

        Parameters:
        ---------
        partition : Partition
                the current partitioning results
        num_agg_proposals_per_block : int
                the number of proposals to make for each block
        use_sparse_matrix : bool
                if True, then use the slower but smaller sparse matrix representation to store matrices
        num_blocks_to_merge : int
                the number of blocks to merge in this iteration
        out_neighbors : np.array
                the matrix representing neighboring blocks

        Returns:
        -------
        partition : Partition
                the updated partition
    """
    best_merge_for_each_block = np.ones(partition.num_blocks, dtype=int) * -1  # initialize to no merge
    delta_entropy_for_each_block = np.ones(partition.num_blocks) * np.Inf  # initialize criterion
    block_partition = range(partition.num_blocks)
    for current_block in range(partition.num_blocks):  # evaluate agglomerative updates for each block
        for _ in range(num_agg_proposals_per_block):
            # populate edges to neighboring blocks
            out_blocks = outgoing_edges(partition.interblock_edge_count, current_block, use_sparse_matrix)
            in_blocks = incoming_edges(partition.interblock_edge_count, current_block, use_sparse_matrix)

            # propose a new block to merge with
            proposal, num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges = propose_new_partition(
                current_block, out_blocks, in_blocks, block_partition, partition, True, use_sparse_matrix)

            # compute the two new rows and columns of the interblock edge count matrix
            new_interblock_edge_count_current_block_row, new_interblock_edge_count_new_block_row, new_interblock_edge_count_current_block_col, new_interblock_edge_count_new_block_col = \
                compute_new_rows_cols_interblock_edge_count_matrix(partition.interblock_edge_count, current_block, proposal,
                                                                out_blocks[:, 0], out_blocks[:, 1], in_blocks[:, 0],
                                                                in_blocks[:, 1],
                                                                partition.interblock_edge_count[current_block, current_block],
                                                                1, use_sparse_matrix)

            # compute new block degrees
            block_degrees_out_new, block_degrees_in_new, block_degrees_new = compute_new_block_degrees(current_block,
                                                                                                    proposal,
                                                                                                    partition,
                                                                                                    num_out_neighbor_edges,
                                                                                                    num_in_neighbor_edges,
                                                                                                    num_neighbor_edges)

            # compute change in entropy / posterior
            delta_entropy = compute_delta_entropy(current_block, proposal, partition,
                                                new_interblock_edge_count_current_block_row,
                                                new_interblock_edge_count_new_block_row,
                                                new_interblock_edge_count_current_block_col,
                                                new_interblock_edge_count_new_block_col, block_degrees_out_new, block_degrees_in_new,
                                                use_sparse_matrix)
            if delta_entropy < delta_entropy_for_each_block[current_block]:  # a better block candidate was found
                best_merge_for_each_block[current_block] = proposal
                delta_entropy_for_each_block[current_block] = delta_entropy

    # carry out the best merges
    partition.block_assignment, partition.num_blocks = carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block, partition.block_assignment,
                                                partition.num_blocks, num_blocks_to_merge)

    # re-initialize edge counts and block degrees
    partition.initialize_edge_counts(out_neighbors, use_sparse_matrix)
    
    return partition
# End of merge_blocks()


def outgoing_edges(adjacency_matrix: np.array, block: int, use_sparse_matrix: bool) -> np.array:
    """Finds the outgoing edges from a given block, with their weights.

        Parameters
        ----------
        adjacency_matrix : np.array [int]
                the adjacency matrix for all blocks in the current partition
        block : int
                the block for which to get the outgoing edges
        use_sparse_matrix : bool
                if True, then the adjacency_matrix is stored in a sparse format

        Returns
        -------
        outgoing_edges : np.array [int]
                matrix with two columns, representing the edge (as the other block's ID), and the weight of the edge
    """
    if use_sparse_matrix:
        out_blocks = adjacency_matrix[block, :].nonzero()[1]
        out_blocks = np.hstack((out_blocks.reshape([len(out_blocks), 1]),
                                adjacency_matrix[block, out_blocks].toarray().transpose()))
    else:
        out_blocks = adjacency_matrix[block, :].nonzero()
        out_blocks = np.hstack((np.array(out_blocks).transpose(), adjacency_matrix[block, out_blocks].transpose()))
    return out_blocks
# End of outgoing_edges()

def incoming_edges(adjacency_matrix: np.array, block: int, use_sparse_matrix: bool) -> np.array:
    """Finds the incoming edges to a given block, with their weights.

        Parameters
        ----------
        adjacency_matrix : np.array [int]
                the adjacency matrix for all blocks in the current partition
        block : int
                the block for which to get the incoming edges
        use_sparse_matrix : bool
                if True, then the adjacency_matrix is stored in a sparse format

        Returns
        -------
        incoming_edges : np.array [int]
                matrix with two columns, representing the edge (as the other block's ID), and the weight of the edge
    """
    if use_sparse_matrix:
        in_blocks = adjacency_matrix[:, block].nonzero()[0]
        in_blocks = np.hstack(
            (in_blocks.reshape([len(in_blocks), 1]), adjacency_matrix[in_blocks, block].toarray()))
    else:
        in_blocks = adjacency_matrix[:, block].nonzero()
        in_blocks = np.hstack((np.array(in_blocks).transpose(), adjacency_matrix[in_blocks, block].transpose()))
    return in_blocks
# End of incoming_edges()
