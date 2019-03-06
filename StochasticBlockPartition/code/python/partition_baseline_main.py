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
    # parser.add_argument("input_filename", nargs="?", type=str, default="../../data/static/simulated_blockmodel_graph_500_nodes")
    args = parser.parse_args()
    return args
# End of parse_arguments()


def build_filepath(args: argparse.Namespace) -> str:
    """Builds the filename string.

        Parameters
        ---------
        args : argparse.Namespace
                the command-line arguments passed in
        
        Returns
        ------
        filepath : str
                the path to the dataset base directory
    """
    filepath_base = "{0}/{1}/{2}Overlap_{3}BlockSizeVar/{1}_{2}Overlap_{3}BlockSizeVar_{4}_nodes".format(
        args.directory, args.type, args.overlap, args.blockSizeVar, args.numNodes
    )

    if not os.path.isfile(filepath_base + '.tsv') and not os.path.isfile(filepath_base + '_1.tsv'):
        print("File doesn't exist: '{}'!".format(filepath_base))
        sys.exit(1)
        
    return filepath_base
# End of build_filepath()


if __name__ == "__main__":
    args = parse_arguments()

    input_filename = build_filepath(args)

    true_partition_available = True
    visualize_graph = False  # whether to plot the graph layout colored with intermediate partitions
    verbose = True  # whether to print updates of the partitioning

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

    partition = Partition(N, out_neighbors, args.sparse)

    # agglomerative partition update parameters
    # num_agg_proposals_per_block = 10  # number of proposals per block
    # num_block_reduction_rate = 0.5  # fraction of blocks to reduce until the golden ratio bracket is established

    # initialize items before iterations to find the partition with the optimal number of blocks
    partition_triplet, graph_object = initialize_partition_variables()
    num_blocks_to_merge = int(partition.num_blocks * args.blockReductionRate)

    # begin partitioning by finding the best partition with the optimal number of blocks
    while not partition_triplet.optimal_num_blocks_found:
        ##################
        # BLOCK MERGING
        ##################
        # begin agglomerative partition updates (i.e. block merging)
        if verbose:
            print("\nMerging down blocks from {} to {}".format(partition.num_blocks, partition.num_blocks - num_blocks_to_merge))
        
        partition = merge_blocks(partition, args.blockProposals, args.sparse, num_blocks_to_merge, out_neighbors)

        # perform nodal partition updates
        ############################
        # NODAL BLOCK UPDATES
        ############################

        if verbose:
            print("Beginning nodal updates")

        partition = reassign_nodes(partition, N, E, out_neighbors, in_neighbors, partition_triplet, args.sparse, args.verbose)

        if visualize_graph:
            graph_object = plot_graph_with_partition(out_neighbors, partition.block_assignment, graph_object)

        # check whether the partition with optimal number of block has been found; if not, determine and prepare for the next number of blocks to try
        partition, num_blocks_to_merge, partition_triplet = prepare_for_partition_on_next_num_blocks(
            partition, partition_triplet, args.blockReductionRate)

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
