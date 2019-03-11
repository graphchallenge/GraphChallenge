"""Contains code for the node reassignment part of the baseline algorithm.
"""

from argparse import Namespace
from typing import List, Tuple

import numpy as np

from partition_baseline_support import compute_overall_entropy
from partition_baseline_support import propose_new_partition
from partition_baseline_support import compute_new_rows_cols_interblock_edge_count_matrix
from partition_baseline_support import compute_new_block_degrees
from partition_baseline_support import compute_Hastings_correction
from partition_baseline_support import compute_delta_entropy
from partition_baseline_support import update_partition

from partition import Partition
from partition import PartitionTriplet


def reassign_nodes(partition: Partition, num_nodes: int, num_edges: int, out_neighbors: List[np.ndarray],
    in_neighbors: List[np.ndarray], partition_triplet: PartitionTriplet, args: Namespace) -> Partition:
    """Reassigns nodes to different blocks based on Bayesian statistics.

        Parameters
        ---------
        partition : Partition
                the current partitioning results
        num_nodes : int
                the number of nodes in the graph
        num_edges : int
                the number of edges in the graph
        out_neighbors : List[np.ndarray]
                the list of outgoing edges per node
        in_neighbors : List[np.ndarray]
                the list of incoming edges per node
        partition_triplet : PartitionTriplet
                the triplet of partitions with the lowest overall entropy scores so far
        args : Namespace
                the command-line arguments

        Returns
        -------
        partition : Partition
                the updated partitioning results
    """
    # nodal partition updates parameters
    delta_entropy_threshold1 = 5e-4  # stop iterating when the change in entropy falls below this fraction of the overall entropy
                                    # lowering this threshold results in more nodal update iterations and likely better performance, but longer runtime
    delta_entropy_threshold2 = 1e-4  # threshold after the golden ratio bracket is established (typically lower to fine-tune to partition)
    delta_entropy_moving_avg_window = 3  # width of the moving average window for the delta entropy convergence criterion

    total_num_nodal_moves = 0
    itr_delta_entropy = np.zeros(args.iterations)

    # compute the global entropy for MCMC convergence criterion
    partition.overall_entropy = compute_overall_entropy(partition, num_nodes, num_edges, args.sparse)

    for itr in range(args.iterations):
        num_nodal_moves = 0
        itr_delta_entropy[itr] = 0

        for current_node in range(num_nodes):
            p_accept, delta_entropy = propose_new_assignment(current_node, partition, out_neighbors, in_neighbors, args)
            if p_accept >= 0.0:
                total_num_nodal_moves += 1
                num_nodal_moves += 1
                itr_delta_entropy[itr] += delta_entropy

        # End of iteration_over_nodes
        if args.verbose:
            print("Itr: {}, number of nodal moves: {}, delta S: {:0.5f}".format(
                itr, num_nodal_moves, itr_delta_entropy[itr] / float(partition.overall_entropy)))

        # exit MCMC if the recent change in entropy falls below a small fraction of the overall entropy
        if itr >= (delta_entropy_moving_avg_window - 1):  
            if not (np.all(np.isfinite(partition_triplet.overall_entropy))):  # golden ratio bracket not yet established
                if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                    delta_entropy_threshold1 * partition.overall_entropy)):
                    break
            else:  # golden ratio bracket is established. Fine-tuning partition.
                if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                    delta_entropy_threshold2 * partition.overall_entropy)):
                    break

    # compute the global entropy for determining the optimal number of blocks
    partition.overall_entropy = compute_overall_entropy(partition, num_nodes, num_edges, args.sparse)

    if args.verbose:
        print("Total number of nodal moves: {}, overall_entropy: {:0.2f}".format(
            total_num_nodal_moves, partition.overall_entropy))

    return partition
# End of reassign_nodes()


def propose_new_assignment(current_node: int, partition: Partition, out_neighbors: List[np.ndarray], 
    in_neighbors: List[np.ndarray], args: Namespace) -> Tuple[float, float]:
    """Proposes a block reassignment to for the given node.

        Parameters
        ----------
        current_node : int
                the node for which to propose a reassignment
        partition : Partition
                the current partitioning results
        out_neighbors : List[np.ndarray]
                the outgoing edges for every node
        in_neighbors : List[np.ndarray]
                the incoming edges for every node
        args : Namespace
                the command-line arguments provided

        Returns
        ------
        p_accept : float
                the probability of accepting the block proposal. If the proposal is the same as the current block,
                returns -1.0
        delta_entropy : float
                the change in entropy due to the node's block reassignment (if any). If the block isn't reassigned,
                returns -1.0
    """
    current_block = partition.block_assignment[current_node]

    # propose a new block for this node
    proposal, num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges = propose_new_partition(
        current_block, out_neighbors[current_node], in_neighbors[current_node], partition.block_assignment,
        partition, False, args.sparse)

    # determine whether to accept or reject the proposal
    if (proposal != current_block):
        # compute block counts of in and out neighbors
        blocks_out, inverse_idx_out = np.unique(partition.block_assignment[out_neighbors[current_node][:, 0]],
                                                return_inverse=True)
        count_out = np.bincount(inverse_idx_out, weights=out_neighbors[current_node][:, 1]).astype(int)
        blocks_in, inverse_idx_in = np.unique(partition.block_assignment[in_neighbors[current_node][:, 0]], return_inverse=True)
        count_in = np.bincount(inverse_idx_in, weights=in_neighbors[current_node][:, 1]).astype(int)

        # compute the two new rows and columns of the interblock edge count matrix
        self_edge_weight = np.sum(out_neighbors[current_node][np.where(
            out_neighbors[current_node][:, 0] == current_node), 1])  # check if this node has a self edge
        edge_count_updates = compute_new_rows_cols_interblock_edge_count_matrix(partition.interblock_edge_count, current_block, proposal,
                                                            blocks_out, count_out, blocks_in, count_in,
                                                            self_edge_weight, 0, args.sparse)

        # compute new block degrees
        block_degrees_out_new, block_degrees_in_new, block_degrees_new = compute_new_block_degrees(
            current_block, proposal, partition, num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges)

        # compute the Hastings correction
        if num_neighbor_edges>0:
            Hastings_correction = compute_Hastings_correction(blocks_out, count_out, blocks_in, count_in, proposal,
                                                            partition, edge_count_updates.block_row,
                                                            edge_count_updates.block_col, block_degrees_new,
                                                            args.sparse)
        else: # if the node is an island, proposal is random and symmetric
            Hastings_correction = 1

        # compute change in entropy / posterior
        delta_entropy = compute_delta_entropy(current_block, proposal, partition, edge_count_updates, 
                                            block_degrees_out_new, block_degrees_in_new, args.sparse)

        # compute probability of acceptance
        p_accept = np.min([np.exp(-args.beta * delta_entropy) * Hastings_correction, 1])
        if (np.random.uniform() <= p_accept):
            partition = update_partition(partition, current_node, current_block, proposal, edge_count_updates,
                                            block_degrees_out_new, block_degrees_in_new, block_degrees_new, 
                                            args.sparse)
        return p_accept, delta_entropy
    else:
        return -1.0, -1.0
# End of reassign_node()