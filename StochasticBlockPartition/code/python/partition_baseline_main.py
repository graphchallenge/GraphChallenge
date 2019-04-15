"""Runs the partitioning script.
"""

import timeit
import os, sys, argparse

from partition_baseline_support import plot_graph_with_partition
from partition_baseline_support import prepare_for_partition_on_next_num_blocks

from partition import Partition, PartitionTriplet
from block_merge import merge_blocks
from node_reassignment import reassign_nodes
from graph import Graph
from evaluate import evaluate_partition, Evaluation


def parse_arguments():
    """Parses command-line arguments.

        Returns
        -------
        args : argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parts", type=int, default=0, help="The number of streaming partitions to the dataset. If the dataset is static, 0. Default = 0")
    parser.add_argument("-o", "--overlap", type=str, default="low", help="(low|high). Default = low")
    parser.add_argument("-s", "--blockSizeVar", type=str, default="low", help="(low|high). Default = low")
    parser.add_argument("-t", "--type", type=str, default="static", help="(static|streamingEdge|streamingSnowball). Default = static")
    parser.add_argument("-n", "--numNodes", type=int, default=1000, help="The size of the dataset. Default = 1000")
    parser.add_argument("-d", "--directory", type=str, default="../../data", help="The location of the dataset directory. Default = ../../data")
    parser.add_argument("-v", "--verbose", action="store_true", help="If supplied, will print 'helpful' messages.")
    parser.add_argument("-b", "--blockProposals", type=int, default=10, help="The number of block merge proposals per block. Default = 10")
    parser.add_argument("-i", "--iterations", type=int, default=100, help="Maximum number of node reassignment iterations. Default = 100")
    parser.add_argument("-r", "--blockReductionRate", type=float, default=0.5, help="The block reduction rate. Default = 0.5")
    parser.add_argument("--beta", type=int, default=3, help="exploitation vs exploration: higher threshold = higher exploration. Default = 3")
    parser.add_argument("--sparse", action="store_true", help="If supplied, will use Scipy's sparse matrix representation for the matrices.")
    parser.add_argument("-c", "--csv", type=str, default="eval/benchmark", help="The filepath to the csv file in which to store the evaluation results.")
    # Nodal Update Strategy
    parser.add_argument("-u", "--nodal_update_strategy", type=str, default="original", help="(original|step|exponential|log). Default = original")
    parser.add_argument("--direction", type=str, default="growth", help="(growth|decay) Default = growth")
    parser.add_argument("-f", "--factor", type=float, default=0.0001, help="""The factor by which to grow or decay the nodal update threshold. 
                                                                            If the nodal update strategy is step:
                                                                                this value is added to or subtracted from the threshold with every iteration
                                                                            If the nodal update strategy is exponential:
                                                                                this (1 +/- this value) is multiplied by the threshold with every iteration
                                                                            If the nodal update strategy is log:
                                                                                this value is ignored
                                                                            Default = 0.0001""")
    parser.add_argument("-e", "--threshold", type=float, default=5e-4, help="The threshold at which to stop nodal block reassignment. Default = 5e-4")
    args = parser.parse_args()
    return args
# End of parse_arguments()


if __name__ == "__main__":
    args = parse_arguments()

    true_partition_available = True
    visualize_graph = False  # whether to plot the graph layout colored with intermediate partitions

    graph = Graph.load(args)

    if args.verbose:
        print('Number of nodes: {}'.format(graph.num_nodes))
        print('Number of edges: {}'.format(graph.num_edges))

    evaluation = Evaluation(args, graph)

    t_start = timeit.default_timer()

    partition = Partition(graph.num_nodes, graph.out_neighbors, args)

    # initialize items before iterations to find the partition with the optimal number of blocks
    partition_triplet = PartitionTriplet()
    graph_object = None

    # begin partitioning by finding the best partition with the optimal number of blocks
    while not partition_triplet.optimal_num_blocks_found:
        ##################
        # BLOCK MERGING
        ##################
        # begin agglomerative partition updates (i.e. block merging)
        t_block_merge_start = timeit.default_timer()

        if args.verbose:
            print("\nMerging down blocks from {} to {}".format(partition.num_blocks, partition.num_blocks - partition.num_blocks_to_merge))
        
        partition = merge_blocks(partition, args.blockProposals, args.sparse, graph.out_neighbors, evaluation)

        t_nodal_update_start = timeit.default_timer()

        ############################
        # NODAL BLOCK UPDATES
        ############################
        if args.verbose:
            print("Beginning nodal updates")

        partition = reassign_nodes(partition, graph, partition_triplet, evaluation, args)

        if visualize_graph:
            graph_object = plot_graph_with_partition(graph.out_neighbors, partition.block_assignment, graph_object)

        t_prepare_next_start = timeit.default_timer()

        # check whether the partition with optimal number of block has been found; if not, determine and prepare for the next number of blocks to try
        partition, partition_triplet = prepare_for_partition_on_next_num_blocks(
            partition, partition_triplet, args.blockReductionRate)

        t_prepare_next_end = timeit.default_timer()
        evaluation.update_timings(t_block_merge_start, t_nodal_update_start, t_prepare_next_start, t_prepare_next_end)
        evaluation.num_iterations += 1

        if args.verbose:
            print('Overall entropy: {}'.format(partition_triplet.overall_entropy))
            print('Number of blocks: {}'.format(partition_triplet.num_blocks))
            if partition_triplet.optimal_num_blocks_found:
                print('\nOptimal partition found with {} blocks'.format(partition.num_blocks))

    t_end = timeit.default_timer()
    evaluation.total_runtime(t_start, t_end)
    print('\nGraph partition took {} seconds'.format(t_end - t_start))

    # evaluate output partition against the true partition
    evaluate_partition(graph.true_block_assignment, partition.block_assignment, evaluation)
