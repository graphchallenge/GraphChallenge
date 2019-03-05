"""Runs the partitioning script.
"""

from partition_baseline_support import *
use_timeit = True # for timing runs (optional)
if use_timeit:
    import timeit
import os, sys, argparse

from partition import Partition
from block_merge import merge_blocks
from node_reassignment import reassign_nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parts", type=int, required=False)
    parser.add_argument("input_filename", nargs="?", type=str, default="../../data/static/simulated_blockmodel_graph_500_nodes")
    args = parser.parse_args()

    input_filename = args.input_filename
    true_partition_available = True
    visualize_graph = False  # whether to plot the graph layout colored with intermediate partitions
    verbose = True  # whether to print updates of the partitioning

    if not os.path.isfile(input_filename + '.tsv') and not os.path.isfile(input_filename + '_1.tsv'):
        print("File doesn't exist: '{}'!".format(input_filename))
        sys.exit(1)

    if args.parts >= 1:
        print('\nLoading partition 1 of {} ({}) ...'.format(args.parts, input_filename + "_1.tsv"))
        out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=true_partition_available, strm_piece_num=1)
        for part in range(2, args.parts + 1):
            print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))
            out_neighbors, in_neighbors, N, E = load_graph(input_filename, load_true_partition=False, strm_piece_num=part, out_neighbors=out_neighbors, in_neighbors=in_neighbors)
    else:
        out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=true_partition_available)

    if verbose:
        print('Number of nodes: {}'.format(N))
        print('Number of edges: {}'.format(E))

    if use_timeit:
        t0 = timeit.default_timer()

    # partition update parameters
    use_sparse_matrix = False  # whether to represent the edge count matrix using sparse matrix
                            # Scipy's sparse matrix is slow but this may be necessary for large graphs

    partition = Partition(N, out_neighbors, use_sparse_matrix)

    # agglomerative partition update parameters
    num_agg_proposals_per_block = 10  # number of proposals per block
    num_block_reduction_rate = 0.5  # fraction of blocks to reduce until the golden ratio bracket is established

    # initialize items before iterations to find the partition with the optimal number of blocks
    partition_triplet, graph_object = initialize_partition_variables()
    num_blocks_to_merge = int(partition.num_blocks * num_block_reduction_rate)

    # begin partitioning by finding the best partition with the optimal number of blocks
    while not partition_triplet.optimal_num_blocks_found:
        ##################
        # BLOCK MERGING
        ##################
        # begin agglomerative partition updates (i.e. block merging)
        if verbose:
            print("\nMerging down blocks from {} to {}".format(partition.num_blocks, partition.num_blocks - num_blocks_to_merge))
        
        partition = merge_blocks(partition, num_agg_proposals_per_block, use_sparse_matrix, num_blocks_to_merge, out_neighbors)

        # perform nodal partition updates
        ############################
        # NODAL BLOCK UPDATES
        ############################

        if verbose:
            print("Beginning nodal updates")

        partition = reassign_nodes(partition, N, E, out_neighbors, in_neighbors, partition_triplet, use_sparse_matrix, verbose)

        if visualize_graph:
            graph_object = plot_graph_with_partition(out_neighbors, partition.block_assignment, graph_object)

        # check whether the partition with optimal number of block has been found; if not, determine and prepare for the next number of blocks to try
        partition, num_blocks_to_merge, partition_triplet = prepare_for_partition_on_next_num_blocks(
            partition, partition_triplet, num_block_reduction_rate)

        if verbose:
            print('Overall entropy: {}'.format(partition_triplet.overall_entropy))
            print('Number of blocks: {}'.format(partition_triplet.num_blocks))
            if partition_triplet.optimal_num_blocks_found:
                print('\nOptimal partition found with {} blocks'.format(partition.num_blocks))

    if use_timeit:
        t1 = timeit.default_timer()
        print('\nGraph partition took {} seconds'.format(t1 - t0))

    # evaluate output partition against the true partition
    evaluate_partition(true_partition, partition.block_assignment)
