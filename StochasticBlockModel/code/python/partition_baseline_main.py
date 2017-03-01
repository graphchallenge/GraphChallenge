from partition_baseline_support import *
use_timeit = True # for timing runs (optional)
if use_timeit:
    import timeit

input_filename = '../../data/static/simulated_blockmodel_graph_500_nodes'
true_partition_available = True
visualize_graph = False  # whether to plot the graph layout colored with intermediate partitions
verbose = True  # whether to print updates of the partitioning

out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, true_partition_available)
if verbose:
    print('Number of nodes: {}'.format(N))
    print('Number of edges: {}'.format(E))

if use_timeit:
    t0 = timeit.default_timer()

# initialize by putting each node in its own block (N blocks)
num_blocks = N
partition = np.array(range(num_blocks))

# partition update parameters
beta = 3  # exploitation versus exploration (higher value favors exploitation)
use_sparse_matrix = False  # whether to represent the edge count matrix using sparse matrix
                           # Scipy's sparse matrix is slow but this may be necessary for large graphs

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
interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees = initialize_edge_counts(out_neighbors,
                                                                                                   num_blocks,
                                                                                                   partition,
                                                                                                   use_sparse_matrix)

# initialize items before iterations to find the partition with the optimal number of blocks
optimal_num_blocks_found, old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks, graph_object = initialize_partition_variables()
num_blocks_to_merge = int(num_blocks * num_block_reduction_rate)

# begin partitioning by finding the best partition with the optimal number of blocks
while not optimal_num_blocks_found:
    # begin agglomerative partition updates (i.e. block merging)
    if verbose:
        print("\nMerging down blocks from {} to {}".format(num_blocks, num_blocks - num_blocks_to_merge))
    best_merge_for_each_block = np.ones(num_blocks, dtype=int) * -1  # initialize to no merge
    delta_entropy_for_each_block = np.ones(num_blocks) * np.Inf  # initialize criterion
    block_partition = range(num_blocks)
    for current_block in range(num_blocks):  # evalaute agglomerative updates for each block
        for proposal_idx in range(num_agg_proposals_per_block):
            # populate edges to neighboring blocks
            if use_sparse_matrix:
                out_blocks = interblock_edge_count[current_block, :].nonzero()[1]
                out_blocks = np.hstack((out_blocks.reshape([len(out_blocks), 1]),
                                        interblock_edge_count[current_block, out_blocks].toarray().transpose()))
            else:
                out_blocks = interblock_edge_count[current_block, :].nonzero()
                out_blocks = np.hstack(
                    (np.array(out_blocks).transpose(), interblock_edge_count[current_block, out_blocks].transpose()))
            if use_sparse_matrix:
                in_blocks = interblock_edge_count[:, current_block].nonzero()[0]
                in_blocks = np.hstack(
                    (in_blocks.reshape([len(in_blocks), 1]), interblock_edge_count[in_blocks, current_block].toarray()))
            else:
                in_blocks = interblock_edge_count[:, current_block].nonzero()
                in_blocks = np.hstack(
                    (np.array(in_blocks).transpose(), interblock_edge_count[in_blocks, current_block].transpose()))

            # propose a new block to merge with
            proposal, num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges = propose_new_partition(
                current_block, out_blocks, in_blocks, block_partition, interblock_edge_count, block_degrees, num_blocks,
                1, use_sparse_matrix)

            # compute the two new rows and columns of the interblock edge count matrix
            new_interblock_edge_count_current_block_row, new_interblock_edge_count_new_block_row, new_interblock_edge_count_current_block_col, new_interblock_edge_count_new_block_col = \
                compute_new_rows_cols_interblock_edge_count_matrix(interblock_edge_count, current_block, proposal,
                                                                   out_blocks[:, 0], out_blocks[:, 1], in_blocks[:, 0],
                                                                   in_blocks[:, 1],
                                                                   interblock_edge_count[current_block, current_block],
                                                                   1, use_sparse_matrix)

            # compute new block degrees
            block_degrees_out_new, block_degrees_in_new, block_degrees_new = compute_new_block_degrees(current_block,
                                                                                                       proposal,
                                                                                                       block_degrees_out,
                                                                                                       block_degrees_in,
                                                                                                       block_degrees,
                                                                                                       num_out_neighbor_edges,
                                                                                                       num_in_neighbor_edges,
                                                                                                       num_neighbor_edges)

            # compute change in entropy / posterior
            delta_entropy = compute_delta_entropy(current_block, proposal, interblock_edge_count,
                                                  new_interblock_edge_count_current_block_row,
                                                  new_interblock_edge_count_new_block_row,
                                                  new_interblock_edge_count_current_block_col,
                                                  new_interblock_edge_count_new_block_col, block_degrees_out,
                                                  block_degrees_in, block_degrees_out_new, block_degrees_in_new,
                                                  use_sparse_matrix)
            if delta_entropy < delta_entropy_for_each_block[current_block]:  # a better block candidate was found
                best_merge_for_each_block[current_block] = proposal
                delta_entropy_for_each_block[current_block] = delta_entropy

    # carry out the best merges
    partition, num_blocks = carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block, partition,
                                                  num_blocks, num_blocks_to_merge)

    # re-initialize edge counts and block degrees
    interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees = initialize_edge_counts(out_neighbors,
                                                                                                       num_blocks,
                                                                                                       partition,
                                                                                                       use_sparse_matrix)

    # perform nodal partition updates
    if verbose:
        print("Beginning nodal updates")
    total_num_nodal_moves = 0
    itr_delta_entropy = np.zeros(max_num_nodal_itr)

    # compute the global entropy for MCMC convergence criterion
    overall_entropy = compute_overall_entropy(interblock_edge_count, block_degrees_out, block_degrees_in, num_blocks, N,
                                              E, use_sparse_matrix)

    for itr in range(max_num_nodal_itr):
        num_nodal_moves = 0;
        itr_delta_entropy[itr] = 0

        for current_node in range(N):
            current_block = partition[current_node]
            # propose a new block for this node
            proposal, num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges = propose_new_partition(
                current_block, out_neighbors[current_node], in_neighbors[current_node], partition,
                interblock_edge_count, block_degrees, num_blocks, 0, use_sparse_matrix)

            # determine whether to accept or reject the proposal
            if (proposal != current_block):
                # compute block counts of in and out neighbors
                blocks_out, inverse_idx_out = np.unique(partition[out_neighbors[current_node][:, 0]],
                                                        return_inverse=True)
                count_out = np.bincount(inverse_idx_out, weights=out_neighbors[current_node][:, 1]).astype(int)
                blocks_in, inverse_idx_in = np.unique(partition[in_neighbors[current_node][:, 0]], return_inverse=True)
                count_in = np.bincount(inverse_idx_in, weights=in_neighbors[current_node][:, 1]).astype(int)

                # compute the two new rows and columns of the interblock edge count matrix
                self_edge_weight = np.sum(out_neighbors[current_node][np.where(
                    out_neighbors[current_node][:, 0] == current_node), 1])  # check if this node has a self edge
                new_interblock_edge_count_current_block_row, new_interblock_edge_count_new_block_row, new_interblock_edge_count_current_block_col, new_interblock_edge_count_new_block_col = \
                    compute_new_rows_cols_interblock_edge_count_matrix(interblock_edge_count, current_block, proposal,
                                                                       blocks_out, count_out, blocks_in, count_in,
                                                                       self_edge_weight, 0, use_sparse_matrix)

                # compute new block degrees
                block_degrees_out_new, block_degrees_in_new, block_degrees_new = compute_new_block_degrees(
                    current_block, proposal, block_degrees_out, block_degrees_in, block_degrees, num_out_neighbor_edges,
                    num_in_neighbor_edges, num_neighbor_edges)

                # compute the Hastings correction
                Hastings_correction = compute_Hastings_correction(blocks_out, count_out, blocks_in, count_in, proposal,
                                                                  interblock_edge_count,
                                                                  new_interblock_edge_count_current_block_row,
                                                                  new_interblock_edge_count_current_block_col,
                                                                  num_blocks, block_degrees,
                                                                  block_degrees_new, use_sparse_matrix)

                # compute change in entropy / posterior
                delta_entropy = compute_delta_entropy(current_block, proposal, interblock_edge_count,
                                                      new_interblock_edge_count_current_block_row,
                                                      new_interblock_edge_count_new_block_row,
                                                      new_interblock_edge_count_current_block_col,
                                                      new_interblock_edge_count_new_block_col, block_degrees_out,
                                                      block_degrees_in, block_degrees_out_new, block_degrees_in_new,
                                                      use_sparse_matrix)

                # compute probability of acceptance
                p_accept = np.min([np.exp(-beta * delta_entropy) * Hastings_correction, 1])

                # if accept the proposal, update the partition, inter_block_edge_count, and block degrees
                if (np.random.uniform() <= p_accept):
                    total_num_nodal_moves += 1
                    num_nodal_moves += 1
                    itr_delta_entropy[itr] += delta_entropy
                    partition, interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees = update_partition(
                        partition, current_node, current_block, proposal, interblock_edge_count,
                        new_interblock_edge_count_current_block_row, new_interblock_edge_count_new_block_row,
                        new_interblock_edge_count_current_block_col, new_interblock_edge_count_new_block_col,
                        block_degrees_out_new, block_degrees_in_new, block_degrees_new, use_sparse_matrix)
        if verbose:
            print("Itr: {}, number of nodal moves: {}, delta S: {:0.5f}".format(itr, num_nodal_moves,
                                                                                itr_delta_entropy[itr] / float(
                                                                                    overall_entropy)))
        if itr >= (
            delta_entropy_moving_avg_window - 1):  # exit MCMC if the recent change in entropy falls below a small fraction of the overall entropy
            if not (np.all(np.isfinite(old_overall_entropy))):  # golden ratio bracket not yet established
                if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                    delta_entropy_threshold1 * overall_entropy)):
                    break
            else:  # golden ratio bracket is established. Fine-tuning partition.
                if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                    delta_entropy_threshold2 * overall_entropy)):
                    break

    # compute the global entropy for determining the optimal number of blocks
    overall_entropy = compute_overall_entropy(interblock_edge_count, block_degrees_out, block_degrees_in, num_blocks, N,
                                              E, use_sparse_matrix)

    if verbose:
        print(
        "Total number of nodal moves: {}, overall_entropy: {:0.2f}".format(total_num_nodal_moves, overall_entropy))
    if visualize_graph:
        graph_object = plot_graph_with_partition(out_neighbors, partition, graph_object)

    # check whether the partition with optimal number of block has been found; if not, determine and prepare for the next number of blocks to try
    partition, interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in, num_blocks, num_blocks_to_merge, old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks, optimal_num_blocks_found = \
        prepare_for_partition_on_next_num_blocks(overall_entropy, partition, interblock_edge_count, block_degrees,
                                                 block_degrees_out, block_degrees_in, num_blocks, old_partition,
                                                 old_interblock_edge_count, old_block_degrees, old_block_degrees_out,
                                                 old_block_degrees_in, old_overall_entropy, old_num_blocks,
                                                 num_block_reduction_rate)

    if verbose:
        print('Overall entropy: {}'.format(old_overall_entropy))
        print('Number of blocks: {}'.format(old_num_blocks))
        if optimal_num_blocks_found:
            print('\nOptimal partition found with {} blocks'.format(num_blocks))
if use_timeit:
    t1 = timeit.default_timer()
    print('\nGraph partition took {} seconds'.format(t1 - t0))

# evaluate output partition against the true partition
evaluate_partition(true_partition, partition)
