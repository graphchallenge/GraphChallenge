"""Contains code for the block merge part of the baseline algorithm.
"""

import numpy as np

from partition_baseline_support import propose_new_partition
from partition_baseline_support import compute_new_rows_cols_interblock_edge_count_matrix
from partition_baseline_support import compute_new_block_degrees
from partition_baseline_support import compute_delta_entropy
from partition_baseline_support import carry_out_best_merges
from partition_baseline_support import initialize_edge_counts

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
    for current_block in range(partition.num_blocks):  # evalaute agglomerative updates for each block
        for _ in range(num_agg_proposals_per_block):
            # populate edges to neighboring blocks
            if use_sparse_matrix:
                out_blocks = partition.interblock_edge_count[current_block, :].nonzero()[1]
                out_blocks = np.hstack((out_blocks.reshape([len(out_blocks), 1]),
                                        partition.interblock_edge_count[current_block, out_blocks].toarray().transpose()))
            else:
                out_blocks = partition.interblock_edge_count[current_block, :].nonzero()
                out_blocks = np.hstack(
                    (np.array(out_blocks).transpose(), partition.interblock_edge_count[current_block, out_blocks].transpose()))
            if use_sparse_matrix:
                in_blocks = partition.interblock_edge_count[:, current_block].nonzero()[0]
                in_blocks = np.hstack(
                    (in_blocks.reshape([len(in_blocks), 1]), partition.interblock_edge_count[in_blocks, current_block].toarray()))
            else:
                in_blocks = partition.interblock_edge_count[:, current_block].nonzero()
                in_blocks = np.hstack(
                    (np.array(in_blocks).transpose(), partition.interblock_edge_count[in_blocks, current_block].transpose()))

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
                                                                                                    partition.block_degrees_out,
                                                                                                    partition.block_degrees_in,
                                                                                                    partition.block_degrees,
                                                                                                    num_out_neighbor_edges,
                                                                                                    num_in_neighbor_edges,
                                                                                                    num_neighbor_edges)

            # compute change in entropy / posterior
            delta_entropy = compute_delta_entropy(current_block, proposal, partition.interblock_edge_count,
                                                new_interblock_edge_count_current_block_row,
                                                new_interblock_edge_count_new_block_row,
                                                new_interblock_edge_count_current_block_col,
                                                new_interblock_edge_count_new_block_col, partition.block_degrees_out,
                                                partition.block_degrees_in, block_degrees_out_new, block_degrees_in_new,
                                                use_sparse_matrix)
            if delta_entropy < delta_entropy_for_each_block[current_block]:  # a better block candidate was found
                best_merge_for_each_block[current_block] = proposal
                delta_entropy_for_each_block[current_block] = delta_entropy

    # carry out the best merges
    partition.block_assignment, partition.num_blocks = carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block, partition.block_assignment,
                                                partition.num_blocks, num_blocks_to_merge)

    # re-initialize edge counts and block degrees
    # TODO(): Change this to a call to partition.initialize_edge_counts()
    partition.interblock_edge_count, partition.block_degrees_out, partition.block_degrees_in, partition.block_degrees = initialize_edge_counts(out_neighbors,
                                                                                                    partition.num_blocks,
                                                                                                    partition.block_assignment,
                                                                                                    use_sparse_matrix)
    
    return partition
# End of merge_blocks()
