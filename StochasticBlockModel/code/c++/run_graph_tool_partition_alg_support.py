""" Supporting functions for reading the graph and evaluating the partition """

import pandas as pd
from munkres import Munkres # for correctness evaluation
import numpy as np
import scipy.misc as misc

def load_graph(input_filename, true_partition_available):
    """Load the graph from a TSV file with standard format, and the truth partition if available
        Parameters
        ----------
        input_filename : str
                input file name not including the .tsv extension
        true_partition_available : bool
                whether the truth partition is available
        Returns
        -------
        out_neighbors : list of ndarray; list length is N, the number of nodes
                each element of the list is a ndarray of out neighbors, where the first column is the node indices
                and the second column the corresponding edge weights
        in_neighbors : list of ndarray; list length is N, the number of nodes
                each element of the list is a ndarray of in neighbors, where the first column is the node indices
                and the second column the corresponding edge weights
        N : int
                number of nodes in the graph
        E : int
                number of edges in the graph
        true_b : ndarray (int) optional
                array of truth block assignment for each node
        Notes
        -----
        The standard tsv file has the form for each row: "from to [weight]" (tab delimited). Nodes are indexed from 0
        to N-1. If available, the true partition is stored in the file `filename_truePartition.tsv`."""

    # read the entire graph CSV into rows of edges
    edge_rows = pd.read_csv('{}.tsv'.format(input_filename), delimiter='\t').as_matrix()
    N = edge_rows[:, 0:1].max() + 1
    out_neighbors = [[] for i in range(N)]
    in_neighbors = [[] for i in range(N)]
    weights_included = edge_rows.shape[1] == 3

    # load edges to list of lists of out and in neighbors
    for i in range(edge_rows.shape[0]):
        if weights_included:
            edge_weight = edge_rows[i, 2]
        else:
            edge_weight = 1
        out_neighbors[edge_rows[i, 0]].append([edge_rows[i, 1], edge_weight])
        in_neighbors[edge_rows[i, 1]].append([edge_rows[i, 0], edge_weight])

    # convert each neighbor list to neighbor numpy arrays for faster access
    for i in range(N):
        out_neighbors[i] = np.array(out_neighbors[i], dtype=int)
    for i in range(N):
        in_neighbors[i] = np.array(in_neighbors[i], dtype=int)

    # find number of nodes and edges
    N = len(out_neighbors)
    E = sum(len(v) for v in out_neighbors)

    if true_partition_available:
        true_b = np.zeros(len(out_neighbors), dtype=int)
        # read the entire true partition CSV into rows of partitions
        true_b_rows = pd.read_csv('{}_truePartition.tsv'.format(input_filename), delimiter='\t').as_matrix()
        for i in range(true_b_rows.shape[0]):
            true_b[true_b_rows[i, 0]] = int(true_b_rows[i, 1])

    if true_partition_available:
        return out_neighbors, in_neighbors, N, E, true_b
    else:
        return out_neighbors, in_neighbors, N, E

def evaluate_partition(true_b, alg_b):
    """Evaluate the output partition against the truth partition and report the correctness metrics.
        Parameters
        ----------
        true_b : ndarray (int)
                    array of truth block assignment for each node
        alg_b : ndarray (int)
                    array of output block assignment for each node"""

    blocks_b1 = true_b
    B_b1 = len(set(blocks_b1))

    blocks_b2 = alg_b
    B_b2 = max(blocks_b2) + 1

    N = len(true_b)

    print('\nPartition Correctness Evaluation\n')
    print('Number of nodes: {}'.format(N))
    print('Number of partitions in partition 1: {}'.format(B_b1))
    print('Number of partitions in partition 2: {}'.format(B_b2))

    # populate the confusion matrix between the two partitions
    contingency_table = np.zeros((B_b1, B_b2))
    for i in range(0, N):
        contingency_table[blocks_b1[i], blocks_b2[i]] += 1

    # associate the labels between two partitions using linear assignment
    assignment = Munkres()  # use the Hungarian algorithm / Kuhn-Munkres algorithm
    if B_b1 > B_b2:  # transpose matrix for linear assignment (this implementation assumes #col >= #row)
        contingency_table = contingency_table.transpose()
    indexes = assignment.compute(-contingency_table)
    total = 0
    contingency_table_before_assignment = np.array(contingency_table)
    for row, column in indexes:
        contingency_table[:, row] = contingency_table_before_assignment[:, column]
        total += contingency_table[row, row]
    # fill in the un-associated columns
    unassociated_col = set(range(contingency_table.shape[1])) - set(np.array(indexes)[:, 1])
    counter = 0;
    for column in unassociated_col:
        contingency_table[:, contingency_table.shape[0] + counter] = contingency_table_before_assignment[:, column]
        counter += 1
    if B_b1 > B_b2:  # transpose back
        contingency_table = contingency_table.transpose()
    print('Contingency Table: \n{}'.format(contingency_table))
    joint_prob = contingency_table / sum(
        sum(contingency_table))  # joint probability of the two partitions is just the normalized contingency table
    accuracy = sum(joint_prob.diagonal())
    print('Accuracy (with optimal partition matching): {}'.format(accuracy))
    print('\n')

    # Compute pair-counting-based metrics
    def nchoose2(a):
        return misc.comb(a, 2)
    num_pairs = nchoose2(N)
    colsum = np.sum(contingency_table, axis=0)
    rowsum = np.sum(contingency_table, axis=1)
    # compute counts of agreements and disagreement (4 types) and the regular rand index
    sum_table_squared = sum(sum(contingency_table ** 2))
    sum_colsum_squared = sum(colsum ** 2)
    sum_rowsum_squared = sum(rowsum ** 2)
    count_in_each_b1 = np.sum(contingency_table, axis=1)
    count_in_each_b2 = np.sum(contingency_table, axis=0)
    num_same_in_b1 = sum(count_in_each_b1 * (count_in_each_b1 - 1)) / 2
    num_same_in_b2 = sum(count_in_each_b2 * (count_in_each_b2 - 1)) / 2
    num_agreement_same = 0.5 * sum(sum(contingency_table * (contingency_table - 1)));
    num_agreement_diff = 0.5 * (N ** 2 + sum_table_squared - sum_colsum_squared - sum_rowsum_squared);
    num_agreement = num_agreement_same + num_agreement_diff
    rand_index = num_agreement / num_pairs

    vectorized_nchoose2 = np.vectorize(nchoose2)
    sum_table_choose_2 = sum(sum(vectorized_nchoose2(contingency_table)))
    sum_colsum_choose_2 = sum(vectorized_nchoose2(colsum))
    sum_rowsum_choose_2 = sum(vectorized_nchoose2(rowsum))
    adjusted_rand_index = (sum_table_choose_2 - sum_rowsum_choose_2 * sum_colsum_choose_2 / num_pairs) / (
    0.5 * (sum_rowsum_choose_2 + sum_colsum_choose_2) - sum_rowsum_choose_2 * sum_colsum_choose_2 / num_pairs)
    print('Rand Index: {}'.format(rand_index))
    print('Adjusted Rand Index: {}'.format(adjusted_rand_index))
    print('Pairwise Recall: {}'.format(num_agreement_same / (num_same_in_b1)))
    print('Pairwise Precision: {}'.format(num_agreement_same / (num_same_in_b2)))
    print('\n')

    # compute the information theoretic metrics
    marginal_prob_b2 = np.sum(joint_prob, 0)
    marginal_prob_b1 = np.sum(joint_prob, 1)
    conditional_prob_b2_b1 = joint_prob / marginal_prob_b1[:, None]
    conditional_prob_b1_b2 = joint_prob / marginal_prob_b2[None, :]
    # compute entropy of the non-partition2 and the partition2 version
    idx = np.nonzero(marginal_prob_b2)
    H_b2 = -np.sum(marginal_prob_b2[idx] * np.log(marginal_prob_b2[idx]))
    idx = np.nonzero(marginal_prob_b1)
    H_b1 = -np.sum(marginal_prob_b1[idx] * np.log(marginal_prob_b1[idx]))
    # compute the conditional entropies
    idx = np.nonzero(joint_prob)
    H_b2_b1 = -np.sum(np.sum(joint_prob[idx] * np.log(conditional_prob_b2_b1[idx])))
    H_b1_b2 = -np.sum(np.sum(joint_prob[idx] * np.log(conditional_prob_b1_b2[idx])))
    # compute the mutual information (symmetric)
    marginal_prod = np.dot(marginal_prob_b1[:, None], np.transpose(marginal_prob_b2[:, None]))
    MI_b1_b2 = np.sum(np.sum(joint_prob[idx] * np.log(joint_prob[idx] / marginal_prod[idx])))
    print('Entropy of partition 1: {}'.format(H_b1))
    print('Entropy of partition 2: {}'.format(H_b2))
    print('Conditional entropy of partition 1 given partition 2: {}'.format(H_b1_b2))
    print('Conditional entropy of partition 2 given partition 1: {}'.format(H_b2_b1))
    print('Mututal informationion between partition 1 and partition 2: {}'.format(MI_b1_b2))
    print('Fraction of missed information: {}'.format(H_b1_b2 / H_b1))
    print('Fraction of erroneous information: {}'.format(H_b2_b1 / H_b2))