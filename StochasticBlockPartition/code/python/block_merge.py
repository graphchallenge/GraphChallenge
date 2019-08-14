"""Contains code for the block merge part of the baseline algorithm.
"""

from typing import Tuple

import numpy as np

from partition_baseline_support import propose_new_partition
# from partition_baseline_support import compute_new_rows_cols_interblock_edge_count_matrix
from partition_baseline_support import block_merge_edge_count_updates
from partition_baseline_support import compute_new_block_degrees
from partition_baseline_support import compute_delta_entropy
from partition_baseline_support import carry_out_best_merges

from partition import Partition
from evaluation import Evaluation
from block_merge_timings import BlockMergeTimings


def merge_blocks(partition: Partition, num_agg_proposals_per_block: int, use_sparse_matrix: bool,
    out_neighbors: np.array, evaluation: Evaluation) -> Partition:
    """The block merge portion of the algorithm.

        Parameters:
        ---------
        partition : Partition
                the current partitioning results
        num_agg_proposals_per_block : int
                the number of proposals to make for each block
        use_sparse_matrix : bool
                if True, then use the slower but smaller sparse matrix representation to store matrices
        out_neighbors : np.array
                the matrix representing neighboring blocks
        evaluation : Evaluation
                stores the evaluation metrics

        Returns:
        -------
        partition : Partition
                the updated partition
    """
    block_merge_timings = evaluation.add_block_merge_timings()

    block_merge_timings.t_initialization()
    best_merge_for_each_block = np.ones(partition.num_blocks, dtype=int) * -1  # initialize to no merge
    delta_entropy_for_each_block = np.ones(partition.num_blocks) * np.Inf  # initialize criterion
    block_partition = range(partition.num_blocks)
    block_merge_timings.t_initialization()

    for current_block in range(partition.num_blocks):  # evaluate agglomerative updates for each block
        for _ in range(num_agg_proposals_per_block):
            proposal, delta_entropy = propose_merge(current_block, partition, use_sparse_matrix, block_partition,
                                                    block_merge_timings)
            block_merge_timings.t_acceptance()
            if delta_entropy < delta_entropy_for_each_block[current_block]:  # a better block candidate was found
                best_merge_for_each_block[current_block] = proposal
                delta_entropy_for_each_block[current_block] = delta_entropy
            block_merge_timings.t_acceptance()

    # carry out the best merges
    block_merge_timings.t_merging()
    partition = carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block, partition)
    block_merge_timings.t_merging()

    # re-initialize edge counts and block degrees
    block_merge_timings.t_re_counting_edges()
    partition.initialize_edge_counts(out_neighbors, use_sparse_matrix)
    block_merge_timings.t_re_counting_edges()

    return partition
# End of merge_blocks()


def propose_merge(current_block: int, partition: Partition, use_sparse_matrix: bool, block_partition: np.array,
    block_merge_timings: BlockMergeTimings) -> Tuple[int, float]:
    """Propose a block merge, and calculate its delta entropy value.

        Parameters
        ----------
        current_block : int
                the block for which to propose merges
        partition : Partition
                the current partitioning results
        use_sparse_matrix : bool
                if True, the interblock edge count matrix is stored using a slower sparse representation
        block_partition : np.array [int]
                the current block assignment for every block
        block_merge_timings : BlockMergeTimings
                stores the timing details of the block merge step

        Returns
        -------
        proposal : int
                the proposed block to merge with
        delta_entropy : float
                the delta entropy of the proposed merge
    """
    # populate edges to neighboring blocks
    block_merge_timings.t_indexing()
    out_blocks = outgoing_edges(partition.interblock_edge_count, current_block, use_sparse_matrix)
    in_blocks = incoming_edges(partition.interblock_edge_count, current_block, use_sparse_matrix)
    block_merge_timings.t_indexing()

    # propose a new block to merge with
    block_merge_timings.t_proposal()
    proposal, num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges = propose_new_partition(
        current_block, out_blocks, in_blocks, block_partition, partition, True, use_sparse_matrix)
    block_merge_timings.t_proposal()

    # compute the two new rows and columns of the interblock edge count matrix
    block_merge_timings.t_edge_count_updates()
    edge_count_updates = block_merge_edge_count_updates(partition.interblock_edge_count, current_block, proposal,
                                                        out_blocks[:, 0], out_blocks[:, 1], in_blocks[:, 0],
                                                        in_blocks[:, 1],
                                                        partition.interblock_edge_count[current_block, current_block],
                                                        use_sparse_matrix)
    # edge_count_updates = compute_new_rows_cols_interblock_edge_count_matrix(partition.interblock_edge_count, current_block, proposal,
    #                                                     out_blocks[:, 0], out_blocks[:, 1], in_blocks[:, 0],
    #                                                     in_blocks[:, 1],
    #                                                     partition.interblock_edge_count[current_block, current_block],
    #                                                     1, use_sparse_matrix)
    block_merge_timings.t_edge_count_updates()

    # compute new block degrees
    block_merge_timings.t_block_degree_updates()
    # print("k_out: {} k_in: {} k: {}".format(num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges))
    block_degrees_out_new, block_degrees_in_new, block_degrees_new = compute_new_block_degrees(current_block,
                                                                                            proposal,
                                                                                            partition,
                                                                                            num_out_neighbor_edges,
                                                                                            num_in_neighbor_edges,
                                                                                            num_neighbor_edges)
    block_merge_timings.t_block_degree_updates()

    # compute change in entropy / posterior
    block_merge_timings.t_compute_delta_entropy()
    delta_entropy = compute_delta_entropy(current_block, proposal, partition, edge_count_updates, 
                                        block_degrees_out_new, block_degrees_in_new, use_sparse_matrix)
    block_merge_timings.t_compute_delta_entropy()
    return proposal, delta_entropy
# End of propose_merge()


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
        out_blocks = adjacency_matrix.getrow(block)  # .reshape((adjacency_matrix.ncols, 1))  # values
        out_blocks_nonzero = out_blocks.nonzero()[0]  # indices
        out_blocks = np.vstack((out_blocks_nonzero, out_blocks[out_blocks_nonzero])).T  # np.transpose([out_blocks_nonzero[0]]), np.transpose([out_blocks[out_blocks_nonzero]])))
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
        in_blocks = adjacency_matrix.getcol(block)
        in_blocks_nonzero = in_blocks.nonzero()[0]
        # print(in_blocks)
        # print(in_blocks_nonzero)
        # print(len(np.array(in_blocks_nonzero).transpose()))
        # print(len(in_blocks[in_blocks_nonzero].transpose()))
        in_blocks = np.vstack((in_blocks_nonzero, in_blocks[in_blocks_nonzero])).T
        # in_blocks = np.hstack((np.array(in_blocks_nonzero).transpose(), in_blocks[in_blocks_nonzero].transpose()))
                # in_blocks = adjacency_matrix[:, block].strip()
                # # print("in_blocks: ", in_blocks.rows, in_blocks.values)
                # in_blocks = np.hstack((in_blocks.rows, in_blocks.values))
        # print("in_blocks: ", in_blocks)
        # exit()
        # in_blocks = adjacency_matrix[:, block].nonzero()[0]
        # in_blocks = np.hstack(
            # (in_blocks.reshape([len(in_blocks), 1]), adjacency_matrix[in_blocks, block].toarray()))
    else:
        in_blocks = adjacency_matrix[:, block].nonzero()
        in_blocks = np.hstack((np.array(in_blocks).transpose(), adjacency_matrix[in_blocks, block].transpose()))
    return in_blocks
# End of incoming_edges()
