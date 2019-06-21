"""Runs the partitioning script.
"""

import timeit
import os, sys, argparse
from typing import Tuple

import numpy as np

from partition_baseline_support import plot_graph_with_partition
from partition_baseline_support import prepare_for_partition_on_next_num_blocks

from partition import Partition, PartitionTriplet
from block_merge import merge_blocks
from node_reassignment import reassign_nodes, propagate_membership, fine_tune_membership
from graph import Graph
from evaluate import evaluate_partition, evaluate_subgraph_partition
from evaluation import Evaluation


def parse_arguments():
    """Parses command-line arguments.

        Returns
        -------
        args : argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parts", type=int, default=0, 
                        help="""The number of streaming partitions to the dataset. If the dataset is static, 0.
                             Default = 0""")
    parser.add_argument("-o", "--overlap", type=str, default="low", help="(low|high). Default = low")
    parser.add_argument("-s", "--blockSizeVar", type=str, default="low", help="(low|high). Default = low")
    parser.add_argument("-t", "--type", type=str, default="static", 
                        help="(static|streamingEdge|streamingSnowball). Default = static")
    parser.add_argument("-n", "--numNodes", type=int, default=1000, help="The size of the dataset. Default = 1000")
    parser.add_argument("-d", "--directory", type=str, default="../../data", 
                        help="The location of the dataset directory. Default = ../../data")
    parser.add_argument("-v", "--verbose", action="store_true", help="If supplied, will print 'helpful' messages.")
    parser.add_argument("-b", "--blockProposals", type=int, default=10, 
                        help="The number of block merge proposals per block. Default = 10")
    parser.add_argument("-i", "--iterations", type=int, default=100, 
                        help="Maximum number of node reassignment iterations. Default = 100")
    parser.add_argument("-r", "--blockReductionRate", type=float, default=0.5, 
                        help="The block reduction rate. Default = 0.5")
    parser.add_argument("--beta", type=int, default=3, 
                        help="exploitation vs exploration: higher threshold = higher exploration. Default = 3")
    parser.add_argument("--sparse", action="store_true", 
                        help="If supplied, will use Scipy's sparse matrix representation for the matrices.")
    parser.add_argument("-c", "--csv", type=str, default="eval/benchmark", 
                        help="The filepath to the csv file in which to store the evaluation results.")
    # Nodal Update Strategy
    parser.add_argument("-u", "--nodal_update_strategy", type=str, default="original", 
                        help="(original|step|exponential|log). Default = original")
    parser.add_argument("--direction", type=str, default="growth", help="(growth|decay) Default = growth")
    parser.add_argument("-f", "--factor", type=float, default=0.0001, 
                        help="""The factor by which to grow or decay the nodal update threshold. 
                            If the nodal update strategy is step:
                                this value is added to or subtracted from the threshold with every iteration
                            If the nodal update strategy is exponential:
                                this (1 +/- this value) is multiplied by the threshold with every iteration
                            If the nodal update strategy is log:
                                this value is ignored
                            Default = 0.0001""")
    parser.add_argument("-e", "--threshold", type=float, default=5e-4, 
                        help="The threshold at which to stop nodal block reassignment. Default = 5e-4")
    parser.add_argument("-z", "--sample_size", type=int, default=100, 
                        help="The percent of total nodes to use as sample. Default = 100 (no sampling)")
    parser.add_argument("-m", "--sample_type", type=str, default="none", 
                        choices=["uniform_random", "random_walk", "random_jump", "degree_weighted",
                                 "random_node_neighbor", "forest_fire", "none"],
                        help="""Sampling algorithm to use. Default = none""")
    parser.add_argument("--sample_iterations", type=int, default=1,
                        help="The number of sampling iterations to perform. Default = 1")
    args = parser.parse_args()
    return args
# End of parse_arguments()


def stochastic_block_partition(graph: Graph, args: argparse.Namespace) -> Tuple[Partition, Evaluation]:
    """The stochastic block partitioning algorithm
    """ 
    visualize_graph = False
    evaluation = Evaluation(args, graph)

    partition = Partition(graph.num_nodes, graph.out_neighbors, args)

    # initialize items before iterations to find the partition with the optimal number of blocks
    partition_triplet = PartitionTriplet()
    graph_object = None
    
    while not partition_triplet.optimal_num_blocks_found:
        ##################
        # BLOCK MERGING
        ##################
        # begin agglomerative partition updates (i.e. block merging)
        t_block_merge_start = timeit.default_timer()

        if args.verbose:
            print("\nMerging down blocks from {} to {}".format(partition.num_blocks,
                                                               partition.num_blocks - partition.num_blocks_to_merge))
        
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
    return partition, evaluation
# End of stochastic_block_partition()


if __name__ == "__main__":
    args = parse_arguments()

    true_partition_available = True
    visualize_graph = False  # whether to plot the graph layout colored with intermediate partitions

    t_start = timeit.default_timer()

    ## For num_iterations
    ##     samples.append(sample graph)
    ## partition sample
    ## For num_iterations
    ##     extend sample results
    ##     finetune sample results
    ## Evaluate innermost sample
    ## Evaluate results
    graphs = list()
    if args.sample_type != "none":
        full_graph = Graph.load(args)
        t_load = timeit.default_timer()
        for _ in args.sample_iterations:
            graph, mapping, block_assignment_mapping = full_graph.sample(args)
            graphs.append((graph, mapping, block_assignment_mapping))
        t_sample = timeit.default_timer()
        print("Performing stochastic block partitioning on sampled subgraph after {} sampling iterations".format(
            args.sample_iterations
        ))
        partition, evaluation = stochastic_block_partition(graphs[-1][0], args)
    else:
        graph = Graph.load(args)
        t_load = timeit.default_timer()
        t_sample = timeit.default_timer()
        print("Performing stochastic block partitioning")
        # begin partitioning by finding the best partition with the optimal number of blocks
        partition, evaluation = stochastic_block_partition(graph, args)

    if args.sample_type != "none":
        print('Combining sampled partition with full graph')
        for i in range(args.sample_iterations-2, 0, -1):
            t_start_merge_sample = timeit.default_timer()
            full_graph_partition = Partition(full_graph.num_nodes, full_graph.out_neighbors, args)
            full_graph_partition.block_assignment = np.full(full_graph_partition.block_assignment.shape, -1)
            for key, value in mapping.items():
                full_graph_partition.block_assignment[key] = partition.block_assignment[value]
            next_block = partition.num_blocks
            for vertex in range(full_graph.num_nodes):
                if full_graph_partition.block_assignment[vertex] == -1:
                    full_graph_partition.block_assignment[vertex] = next_block
                    next_block += 1
            full_graph_partition.num_blocks = next_block
            full_graph_partition.initialize_edge_counts(full_graph.out_neighbors, args.sparse)
            t_merge_sample = timeit.default_timer()

            full_graph_partition = propagate_membership(full_graph, full_graph_partition, partition, args)
            t_propagate_membership = timeit.default_timer()

            full_graph_partition = fine_tune_membership(full_graph_partition, full_graph, evaluation, args)
            t_fine_tune_membership = timeit.default_timer()

    t_end = timeit.default_timer()
    print('\nGraph partition took {} seconds'.format(t_end - t_start))

    evaluation.total_runtime(t_start, t_end)
    evaluation.loading = t_load - t_start
    evaluation.sampling = t_sample - t_load

    if args.sample_type != "none":
        evaluation.evaluate_subgraph_sampling(full_graph, graph, full_graph_partition, partition,
                                              block_assignment_mapping)
        evaluation.num_nodes = full_graph.num_nodes
        evaluation.num_edges = full_graph.num_edges
        evaluation.merge_sample = t_merge_sample - t_start_merge_sample
        evaluation.propagate_membership = t_propagate_membership - t_merge_sample
        evaluation.finetune_membership = t_fine_tune_membership - t_propagate_membership

        # evaluate output partition against the true partition
        evaluate_subgraph_partition(graph.true_block_assignment, partition.block_assignment, evaluation)
        evaluate_partition(full_graph.true_block_assignment, full_graph_partition.block_assignment, evaluation)
    else:
        evaluate_partition(graph.true_block_assignment, partition.block_assignment, evaluation)
