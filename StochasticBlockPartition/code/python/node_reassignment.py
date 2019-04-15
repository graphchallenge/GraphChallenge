"""Contains code for the node reassignment part of the baseline algorithm.
"""

from argparse import Namespace
from typing import List, Tuple
import math

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
from graph import Graph
from evaluate import Evaluation
from mcmc_timings import MCMCTimings


def reassign_nodes(partition: Partition, graph: Graph, partition_triplet: PartitionTriplet, evaluation: Evaluation,
    args: Namespace) -> Partition:
    """Reassigns nodes to different blocks based on Bayesian statistics.

        Parameters
        ---------
        partition : Partition
                the current partitioning results
        graph : Graph
                the loaded Graph object
        partition_triplet : PartitionTriplet
                the triplet of partitions with the lowest overall entropy scores so far
        evaluation : Evaluation
                stores the evaluation metrics
        args : Namespace
                the command-line arguments

        Returns
        -------
        partition : Partition
                the updated partitioning results
    """
    # nodal partition updates parameters
    # delta_entropy_threshold1 = 5e-4  # stop iterating when the change in entropy falls below this fraction of the overall entropy
                                    # lowering this threshold results in more nodal update iterations and likely better performance, but longer runtime
    # delta_entropy_threshold2 = 1e-4  # threshold after the golden ratio bracket is established (typically lower to fine-tune to partition)
    delta_entropy_moving_avg_window = 3  # width of the moving average window for the delta entropy convergence criterion

    mcmc_timings = evaluation.add_mcmc_timings()

    mcmc_timings.t_initialization()
    itr_delta_entropy = np.zeros(args.iterations)
    delta_entropy_threshold1, delta_entropy_threshold2 = get_thresholds(evaluation.num_iterations, args)
    mcmc_timings.t_initialization()

    # compute the global entropy for MCMC convergence criterion
    mcmc_timings.t_compute_initial_entropy()
    partition.overall_entropy = compute_overall_entropy(partition, graph.num_nodes, graph.num_edges, args.sparse)
    mcmc_timings.t_compute_initial_entropy()

    for itr in range(args.iterations):
        evaluation.num_nodal_update_iterations += 1
        num_nodal_moves = 0
        itr_delta_entropy[itr] = 0

        for current_node in range(graph.num_nodes):
            delta_entropy, did_move = propose_new_assignment(current_node, partition, graph, args, mcmc_timings)
            if did_move:
                evaluation.num_nodal_updates += 1
                num_nodal_moves += 1
                itr_delta_entropy[itr] += delta_entropy

        # End of iteration_over_nodes
        if args.verbose:
            print("Itr: {}, number of nodal moves: {}, delta S: {:0.5f}".format(
                itr, num_nodal_moves, itr_delta_entropy[itr] / float(partition.overall_entropy)))

        # exit MCMC if the recent change in entropy falls below a small fraction of the overall entropy
        mcmc_timings.t_early_stopping()
        if itr >= (delta_entropy_moving_avg_window - 1):
            if not (np.all(np.isfinite(partition_triplet.overall_entropy))):  # golden ratio bracket not yet established
                if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                    delta_entropy_threshold1 * partition.overall_entropy)):
                    mcmc_timings.t_early_stopping()
                    break
            else:  # golden ratio bracket is established. Fine-tuning partition.
                if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                    delta_entropy_threshold2 * partition.overall_entropy)):
                    mcmc_timings.t_early_stopping()
                    break
        mcmc_timings.t_early_stopping()

    # compute the global entropy for determining the optimal number of blocks
    mcmc_timings.t_compute_final_entropy()
    partition.overall_entropy = compute_overall_entropy(partition, graph.num_nodes, graph.num_edges, args.sparse)
    mcmc_timings.t_compute_final_entropy()

    if args.verbose:
        print("Total number of nodal moves: {}, overall_entropy: {:0.2f}".format(
            evaluation.num_nodal_updates, partition.overall_entropy))

    return partition
# End of reassign_nodes()


def propose_new_assignment(current_node: int, partition: Partition, graph: Graph, 
    args: Namespace, mcmc_timings: MCMCTimings) -> Tuple[float, bool]:
    """Proposes a block reassignment to for the given node.

        Parameters
        ----------
        current_node : int
                the node for which to propose a reassignment
        partition : Partition
                the current partitioning results
        graph : Graph
                the Graph loaded from file
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
        did_move : bool
                True if the node was moved to a new block, False otherwise
    """
    mcmc_timings.t_indexing()
    current_block = partition.block_assignment[current_node]
    out_neighbors = graph.out_neighbors[current_node]
    in_neighbors = graph.in_neighbors[current_node]
    mcmc_timings.t_indexing()

    # propose a new block for this node
    mcmc_timings.t_proposal()
    proposal, num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges = propose_new_partition(
        current_block, out_neighbors, in_neighbors, partition.block_assignment, partition, False, args.sparse)
    mcmc_timings.t_proposal()
    did_move = False  # Has the graph node been moved to another block or not?

    # determine whether to accept or reject the proposal
    if (proposal != current_block):
        # compute block counts of in and out neighbors
        mcmc_timings.t_neighbor_counting()
        blocks_out, inverse_idx_out = np.unique(partition.block_assignment[out_neighbors[:, 0]], return_inverse=True)
        count_out = np.bincount(inverse_idx_out, weights=out_neighbors[:, 1]).astype(int)
        blocks_in, inverse_idx_in = np.unique(partition.block_assignment[in_neighbors[:, 0]], return_inverse=True)
        count_in = np.bincount(inverse_idx_in, weights=in_neighbors[:, 1]).astype(int)
        mcmc_timings.t_neighbor_counting()

        # compute the two new rows and columns of the interblock edge count matrix
        mcmc_timings.t_edge_count_updates()
        self_edge_weight = np.sum(out_neighbors[np.where(
            out_neighbors[:, 0] == current_node), 1])  # check if this node has a self edge
        edge_count_updates = compute_new_rows_cols_interblock_edge_count_matrix(partition.interblock_edge_count, current_block, proposal,
                                                            blocks_out, count_out, blocks_in, count_in,
                                                            self_edge_weight, 0, args.sparse)
        mcmc_timings.t_edge_count_updates()

        # compute new block degrees
        mcmc_timings.t_block_degree_updates()
        block_degrees_out_new, block_degrees_in_new, block_degrees_new = compute_new_block_degrees(
            current_block, proposal, partition, num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges)
        mcmc_timings.t_block_degree_updates()

        # compute the Hastings correction
        mcmc_timings.t_hastings_correction()
        if num_neighbor_edges > 0:
            Hastings_correction = compute_Hastings_correction(blocks_out, count_out, blocks_in, count_in, proposal,
                                                            partition, edge_count_updates.block_row,
                                                            edge_count_updates.block_col, block_degrees_new,
                                                            args.sparse)
        else: # if the node is an island, proposal is random and symmetric
            Hastings_correction = 1
        mcmc_timings.t_hastings_correction()

        # compute change in entropy / posterior
        mcmc_timings.t_compute_delta_entropy()
        delta_entropy = compute_delta_entropy(current_block, proposal, partition, edge_count_updates, 
                                            block_degrees_out_new, block_degrees_in_new, args.sparse)
        mcmc_timings.t_compute_delta_entropy()

        # compute probability of acceptance
        mcmc_timings.t_acceptance()
        p_accept = np.min([np.exp(-args.beta * delta_entropy) * Hastings_correction, 1])
        if (np.random.uniform() <= p_accept):
            partition = update_partition(partition, current_node, current_block, proposal, edge_count_updates,
                                            block_degrees_out_new, block_degrees_in_new, block_degrees_new, 
                                            args.sparse)
            did_move = True
        mcmc_timings.t_acceptance()
        return delta_entropy, did_move
    else:
        mcmc_timings.zeros()
        return -1.0, did_move
# End of reassign_node()


def get_thresholds(current_iteration: int, args: Namespace) -> Tuple[float, float]:
    """Returns the thresholds at which to stop the nodal update reassignment iterations. The type of calculation is
    dependent on the nodal update strategy and direction arguments.

    If direction is growth:
        the threshold will increase with every algorithm iteration, leading to less nodal updates being performed over
        time
    If decay:
        the threshold will decrease with every algorithm iteration, leading to more nodal updates being performed over
        time

    If nodal update strategy is original:
        return (original threshold, 0.0001). This is the only case where the two thresholds are different
    If step:
        the original threshold will increase or decrease by the value of (threshold * factor * current_iteration)
    If exponential:
        the original threshold will increase or decrease by a factor of (factor ^ current_iteration)
    if log:
        the original threshold will increase or decrease by a factor of ln(current_iteration + 3). This is because 
        ln(3) is the first natural logarithm of a whole number that's a whole number


        Parameters
        ---------
        current_iteration : int
                the current iteration
        args : Namespace
                the command-line arguments given

        Returns
        ------
        threshold1 : float
                the threshold to use until the golden ratio bracket is established
        threshold2 : float
                the threshold ot use after the golden ratio bracket is established
    """
    if args.nodal_update_strategy == "original":
        return args.threshold, 1e-4
    elif args.nodal_update_strategy == "step":
        if args.direction == "growth":
            new_threshold = args.threshold + (args.threshold * current_iteration * args.factor)
        else:  # direction == "decay"
            new_threshold = max(args.threshold - (args.threshold * current_iteration * args.factor), 1e-8)
        return new_threshold, new_threshold
    elif args.nodal_update_strategy == "exponential":
        if args.direction == "growth":
            factor = 1 + args.factor
        else:
            factor = 1 - args.factor
        new_threshold = args.threshold * math.pow(factor, current_iteration)
        return new_threshold, new_threshold
    elif args.nodal_update_strategy == "log":
        if args.direction == "growth":
            new_threshold = math.log(current_iteration + 3) * args.threshold
        else:  # direction == "decay"
            new_threshold = args.threshold / math.log(current_iteration + 3)
        return new_threshold, new_threshold
    else:
        raise NotImplementedError("The nodal update strategy {} is not implemented.".format(args.nodal_update_strategy))
# End of get_thresholds()
