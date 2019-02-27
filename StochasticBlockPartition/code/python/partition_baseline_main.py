"""Runs the partitioning script.
"""

from partition_baseline_support import *
use_timeit = True # for timing runs (optional)
if use_timeit:
    import timeit
import os, sys, argparse

from partition import Partition


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

    # initialize by putting each node in its own block (N blocks)
    # num_blocks = N
    # block_assignment = np.array(range(num_blocks))

    # partition update parameters
    beta = 3  # exploitation versus exploration (higher value favors exploitation)
    use_sparse_matrix = False  # whether to represent the edge count matrix using sparse matrix
                            # Scipy's sparse matrix is slow but this may be necessary for large graphs

    partition = Partition(N, out_neighbors, use_sparse_matrix)

    # agglomerative partition update parameters
    num_agg_proposals_per_block = 10  # number of proposals per block
    num_block_reduction_rate = 0.5  # fraction of blocks to reduce until the golden ratio bracket is established

    # nodal partition updates parameters
    max_num_nodal_itr = 100  # maximum number of iterations
    delta_entropy_threshold1 = 5e-4  # stop iterating when the change in entropy falls below this fraction of the overall entropy
                                    # lowering this threshold results in more nodal update iterations and likely better performance, but longer runtime
    delta_entropy_threshold2 = 1e-4  # threshold after the golden ratio bracket is established (typically lower to fine-tune to partition)
    delta_entropy_moving_avg_window = 3  # width of the moving average window for the delta entropy convergence criterion

    # initialize edge counts and block degrees
    # interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees = initialize_edge_counts(out_neighbors,
    #                                                                                                 partition.num_blocks,
    #                                                                                                 partition.block_assignment,
    #                                                                                                 use_sparse_matrix)

    # initialize items before iterations to find the partition with the optimal number of blocks
    partition_triplet, graph_object = initialize_partition_variables()
    # optimal_num_blocks_found, old_block_assignment, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks, graph_object = initialize_partition_variables()
    num_blocks_to_merge = int(partition.num_blocks * num_block_reduction_rate)

    # begin partitioning by finding the best partition with the optimal number of blocks
    ##################
    # BLOCK MERGING
    ##################
    while not partition_triplet.optimal_num_blocks_found:
        # begin agglomerative partition updates (i.e. block merging)
        if verbose:
            print("\nMerging down blocks from {} to {}".format(partition.num_blocks, partition.num_blocks - num_blocks_to_merge))
        best_merge_for_each_block = np.ones(partition.num_blocks, dtype=int) * -1  # initialize to no merge
        delta_entropy_for_each_block = np.ones(partition.num_blocks) * np.Inf  # initialize criterion
        block_partition = range(partition.num_blocks)
        for current_block in range(partition.num_blocks):  # evalaute agglomerative updates for each block
            for proposal_idx in range(num_agg_proposals_per_block):
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
                    current_block, out_blocks, in_blocks, block_partition, partition.interblock_edge_count, partition.block_degrees, partition.num_blocks,
                    1, use_sparse_matrix)

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
        partition.interblock_edge_count, partition.block_degrees_out, partition.block_degrees_in, partition.block_degrees = initialize_edge_counts(out_neighbors,
                                                                                                        partition.num_blocks,
                                                                                                        partition.block_assignment,
                                                                                                        use_sparse_matrix)

        # perform nodal partition updates
        ############################
        # NODAL BLOCK UPDATES
        ############################
        if verbose:
            print("Beginning nodal updates")
        total_num_nodal_moves = 0
        itr_delta_entropy = np.zeros(max_num_nodal_itr)

        # compute the global entropy for MCMC convergence criterion
        partition.overall_entropy = compute_overall_entropy(partition.interblock_edge_count, partition.block_degrees_out, partition.block_degrees_in, partition.num_blocks, N,
                                                E, use_sparse_matrix)

        for itr in range(max_num_nodal_itr):
            num_nodal_moves = 0
            itr_delta_entropy[itr] = 0

            for current_node in range(N):
                current_block = partition.block_assignment[current_node]
                # propose a new block for this node
                proposal, num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges = propose_new_partition(
                    current_block, out_neighbors[current_node], in_neighbors[current_node], partition.block_assignment,
                    partition.interblock_edge_count, partition.block_degrees, partition.num_blocks, 0, use_sparse_matrix)

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
                    new_interblock_edge_count_current_block_row, new_interblock_edge_count_new_block_row, new_interblock_edge_count_current_block_col, new_interblock_edge_count_new_block_col = \
                        compute_new_rows_cols_interblock_edge_count_matrix(partition.interblock_edge_count, current_block, proposal,
                                                                        blocks_out, count_out, blocks_in, count_in,
                                                                        self_edge_weight, 0, use_sparse_matrix)

                    # compute new block degrees
                    block_degrees_out_new, block_degrees_in_new, block_degrees_new = compute_new_block_degrees(
                        current_block, proposal, partition.block_degrees_out, partition.block_degrees_in, partition.block_degrees, num_out_neighbor_edges,
                        num_in_neighbor_edges, num_neighbor_edges)

                    # compute the Hastings correction
                    if num_neighbor_edges>0:
                        Hastings_correction = compute_Hastings_correction(blocks_out, count_out, blocks_in, count_in, proposal,
                                                                        partition.interblock_edge_count,
                                                                        new_interblock_edge_count_current_block_row,
                                                                        new_interblock_edge_count_current_block_col,
                                                                        partition.num_blocks, partition.block_degrees,
                                                                        block_degrees_new, use_sparse_matrix)
                    else: # if the node is an island, proposal is random and symmetric
                        Hastings_correction = 1

                    # compute change in entropy / posterior
                    delta_entropy = compute_delta_entropy(current_block, proposal, partition.interblock_edge_count,
                                                        new_interblock_edge_count_current_block_row,
                                                        new_interblock_edge_count_new_block_row,
                                                        new_interblock_edge_count_current_block_col,
                                                        new_interblock_edge_count_new_block_col, partition.block_degrees_out,
                                                        partition.block_degrees_in, block_degrees_out_new, block_degrees_in_new,
                                                        use_sparse_matrix)

                    # compute probability of acceptance
                    p_accept = np.min([np.exp(-beta * delta_entropy) * Hastings_correction, 1])

                    # if accept the proposal, update the block_assignment, inter_block_edge_count, and block degrees
                    if (np.random.uniform() <= p_accept):
                        total_num_nodal_moves += 1
                        num_nodal_moves += 1
                        itr_delta_entropy[itr] += delta_entropy
                        partition.block_assignment, partition.interblock_edge_count, partition.block_degrees_out, partition.block_degrees_in, partition.block_degrees = update_partition(
                            partition.block_assignment, current_node, current_block, proposal, partition.interblock_edge_count,
                            new_interblock_edge_count_current_block_row, new_interblock_edge_count_new_block_row,
                            new_interblock_edge_count_current_block_col, new_interblock_edge_count_new_block_col,
                            block_degrees_out_new, block_degrees_in_new, block_degrees_new, use_sparse_matrix)
            if verbose:
                print("Itr: {}, number of nodal moves: {}, delta S: {:0.5f}".format(itr, num_nodal_moves,
                                                                                    itr_delta_entropy[itr] / float(
                                                                                        partition.overall_entropy)))
            if itr >= (
                delta_entropy_moving_avg_window - 1):  # exit MCMC if the recent change in entropy falls below a small fraction of the overall entropy
                if not (np.all(np.isfinite(partition_triplet.overall_entropy))):  # golden ratio bracket not yet established
                    if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                        delta_entropy_threshold1 * partition.overall_entropy)):
                        break
                else:  # golden ratio bracket is established. Fine-tuning partition.
                    if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                        delta_entropy_threshold2 * partition.overall_entropy)):
                        break

        # compute the global entropy for determining the optimal number of blocks
        partition.overall_entropy = compute_overall_entropy(partition.interblock_edge_count, partition.block_degrees_out, partition.block_degrees_in, partition.num_blocks, N,
                                                E, use_sparse_matrix)

        if verbose:
            print(
            "Total number of nodal moves: {}, overall_entropy: {:0.2f}".format(total_num_nodal_moves, partition.overall_entropy))
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
