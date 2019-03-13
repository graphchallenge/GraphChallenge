"""Contains code for evaluating the resulting partition.
"""

from typing import Tuple
from argparse import Namespace

from munkres import Munkres # for correctness evaluation
import scipy.misc as misc
import numpy as np

from graph import Graph


class Evaluation(object):
    """Stores the evaluation results, and saves them to file.
    """

    def __init__(self, args: Namespace, graph: Graph) -> None:
        """Creates a new Evaluation object.

            Parameters
            ----------
            args : Namespace
                    the command-line arguments
            graph : Graph
                    the loaded graph to be partitioned
        """
        self.block_size_variation = args.blockSizeVar
        self.block_overlap = args.overlap
        self.streaming_type = args.type
        self.num_block_proposals = args.blockProposals
        self.beta = args.beta
        self.sparse = args.sparse
        self.csv_file = args.csv
        self.num_nodes = graph.num_nodes
        self.num_edges = graph.num_edges
        self.num_nodal_updates = 0
        self.num_iterations = 0
        self.num_blocks_algorithm = 0
        self.num_blocks_truth = 0
        self.accuracy = 0.0
        self.rand_index = 0.0
        self.adjusted_rand_index = 0.0
        self.pairwise_recall = 0.0
        self.pairwise_precision = 0.0
        self.entropy_algorithm = 0.0
        self.entropy_truth = 0.0
        self.entropy_algorithm_given_truth = 0.0
        self.entropy_truth_given_algorithm = 0.0
        self.mutual_info = 0.0
        self.missed_info = 0.0
        self.erroneous_info = 0.0
        self.total_partition_time = 0.0
        self.total_block_merge_time = 0.0
        self.total_nodal_update_time = 0.0
    # End of __init__()
# End of Evaluation()


def evaluate_partition(true_b: np.ndarray, alg_b: np.ndarray, eval: Evaluation):
    """Evaluate the output partition against the truth partition and report the correctness metrics.
       Compare the partitions using only the nodes that have known truth block assignment.

        Parameters
        ----------
        true_b : ndarray (int)
                array of truth block assignment for each node. If the truth block is not known for a node, -1 is used
                to indicate unknown blocks.
        alg_b : ndarray (int)
                array of output block assignment for each node. The length of this array corresponds to the number of
                nodes observed and processed so far.
        eval : Evaluation
                stores evaluation results
    """
    contingency_table = create_contingency_table(true_b, alg_b)
    joint_prob = evaluate_accuracy(contingency_table, eval)

    # Compute pair-counting-based metrics
    def nchoose2(a):
        return misc.comb(a, 2)

    N = eval.num_nodes
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
    num_agreement_same = 0.5 * sum(sum(contingency_table * (contingency_table - 1)))
    num_agreement_diff = 0.5 * (N ** 2 + sum_table_squared - sum_colsum_squared - sum_rowsum_squared)
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
    idx1 = np.nonzero(marginal_prob_b1)
    idx2 = np.nonzero(marginal_prob_b2)
    conditional_prob_b2_b1 = np.zeros(joint_prob.shape)
    conditional_prob_b1_b2 = np.zeros(joint_prob.shape)
    conditional_prob_b2_b1[idx1, :] = joint_prob[idx1, :] / marginal_prob_b1[idx1, None]
    conditional_prob_b1_b2[:, idx2] = joint_prob[:, idx2] / marginal_prob_b2[None, idx2]
    # compute entropy of the non-partition2 and the partition2 version
    H_b2 = -np.sum(marginal_prob_b2[idx2] * np.log(marginal_prob_b2[idx2]))
    H_b1 = -np.sum(marginal_prob_b1[idx1] * np.log(marginal_prob_b1[idx1]))

    # compute the conditional entropies
    idx = np.nonzero(joint_prob)
    H_b2_b1 = -np.sum(np.sum(joint_prob[idx] * np.log(conditional_prob_b2_b1[idx])))
    H_b1_b2 = -np.sum(np.sum(joint_prob[idx] * np.log(conditional_prob_b1_b2[idx])))
    # compute the mutual information (symmetric)
    marginal_prod = np.dot(marginal_prob_b1[:, None], np.transpose(marginal_prob_b2[:, None]))
    MI_b1_b2 = np.sum(np.sum(joint_prob[idx] * np.log(joint_prob[idx] / marginal_prod[idx])))

    if H_b1 > 0:
        fraction_missed_info = H_b1_b2 / H_b1
    else:
        fraction_missed_info = 0
    if H_b2 > 0:
        fraction_err_info = H_b2_b1 / H_b2
    else:
        fraction_err_info = 0
    print('Entropy of truth partition: {}'.format(abs(H_b1)))
    print('Entropy of alg. partition: {}'.format(abs(H_b2)))
    print('Conditional entropy of truth partition given alg. partition: {}'.format(abs(H_b1_b2)))
    print('Conditional entropy of alg. partition given truth partition: {}'.format(abs(H_b2_b1)))
    print('Mutual informationion between truth partition and alg. partition: {}'.format(abs(MI_b1_b2)))
    print('Fraction of missed information: {}'.format(abs(fraction_missed_info)))
    print('Fraction of erroneous information: {}'.format(abs(fraction_err_info)))
# End of evaluate_partition()


def create_contingency_table(true_b: np.ndarray, alg_b: np.ndarray) -> Tuple[np.ndarray, int]:
    """Creates the contingency table for the block assignment of the truth and algorithmically determined partitions..
    
        Parameters
        ---------
        true_b : ndarray (int)
                array of truth block assignment for each node. If the truth block is not known for a node, -1 is used
                to indicate unknown blocks.
        alg_b : ndarray (int)
                array of output block assignment for each node. The length of this array corresponds to the number of
                nodes observed and processed so far.
        
        Returns
        ------
        contingency_table : np.ndarray (int)
                the contingency table (confusion matrix) comparing the true block assignment to the algorithmically determined block assignment
        N : int
                the 
    """
    blocks_b1 = true_b
    blocks_b1_set = set(true_b)
    blocks_b1_set.discard(-1)  # -1 is the label for 'unknown'
    B_b1 = len(blocks_b1_set)

    blocks_b2 = alg_b
    B_b2 = max(blocks_b2) + 1

    print('\nPartition Correctness Evaluation\n')
    print('Number of nodes: {}'.format(len(alg_b)))
    print('Number of partitions in truth partition: {}'.format(B_b1))
    print('Number of partitions in alg. partition: {}'.format(B_b2))
    # populate the confusion matrix between the two partitions
    contingency_table = np.zeros((B_b1, B_b2))
    for i in range(len(alg_b)):  # evaluation based on nodes observed so far
        if true_b[i] != -1:  # do not include nodes without truth in the evaluation
            contingency_table[blocks_b1[i], blocks_b2[i]] += 1
    N = contingency_table.sum()

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
    counter = 0
    for column in unassociated_col:
        contingency_table[:, contingency_table.shape[0] + counter] = contingency_table_before_assignment[:, column]
        counter += 1
    if B_b1 > B_b2:  # transpose back
        contingency_table = contingency_table.transpose()
    print('Contingency Table: \n{}'.format(contingency_table))
    return contingency_table
# End of create_contingency_table()


def evaluate_accuracy(contingency_table: np.ndarray, eval: Evaluation) -> np.ndarray:
    """Evaluates the accuracy of partitioning.
    
        Parameters
        ---------
        contingency_table : np.ndarray (int)
                the contingency table (confusion matrix) comparing the true block assignment to the algorithmically determined block assignment
        eval : Evaluation
                stores evaluation results
        
        Returns
        -------
        joint_prob : np.ndarray (float)
                the normalized contingency table
    """
    joint_prob = contingency_table / sum(
        sum(contingency_table))  # joint probability of the two partitions is just the normalized contingency table
    accuracy = sum(joint_prob.diagonal())
    print('Accuracy (with optimal partition matching): {}'.format(accuracy))
    print()
    eval.accuracy = accuracy
    return joint_prob
# End of evaluate_accuracy()


def evaluate_pairwise_metrics(contingency_table: np.ndarray, eval: Evaluation):
    """Evaluates the accuracy of partitioning.
    
        Parameters
        ---------
        contingency_table : np.ndarray (int)
                the contingency table (confusion matrix) comparing the true block assignment to the algorithmically determined block assignment
        eval : Evaluation
                stores evaluation results
        
        Returns
        -------
        joint_prob : np.ndarray (float)
                the normalized contingency table
    """
# End of evaluate_pairwise_metrics()
