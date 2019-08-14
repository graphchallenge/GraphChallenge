"""Contains code for evaluating the resulting partition.
"""

from typing import Tuple, List, Callable

from munkres import Munkres # for correctness evaluation
import scipy.special as misc
import numpy as np

from evaluation import Evaluation


def evaluate_partition(true_b: np.ndarray, alg_b: np.ndarray, evaluation: Evaluation):
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
        evaluation : Evaluation
                stores evaluation results
        
        Returns
        ------
        evaluation : Evaluation
                the evaluation results, filled in with goodness of partitioning measures
    """
    contingency_table, N = create_contingency_table(true_b, alg_b, evaluation)
    joint_prob = evaluate_accuracy(contingency_table, evaluation)
    evaluate_pairwise_metrics(contingency_table, N, evaluation)
    evaluate_entropy_metrics(joint_prob, evaluation)
    evaluation.save()
# End of evaluate_partition()


def evaluate_subgraph_partition(true_b: np.ndarray, alg_b: np.ndarray, evaluation: Evaluation):
    """Evaluate the output partition against the truth partition and report the correctness metrics.
       Compare the partitions using only the nodes that have known truth block assignment.

        Parameters
        ----------
        true_b : ndarray (int)
                array of truth block assignment for each vertex in the subgrpah. If the truth block is not known for a 
                vertex, -1 is used to indicate unknown blocks.
        alg_b : ndarray (int)
                array of output block assignment for each vertex. The length of this array corresponds to the number of
                vertices observed and processed so far.
        evaluation : Evaluation
                stores evaluation results
        
        Returns
        ------
        evaluation : Evaluation
                the evaluation results, filled in with goodness of partitioning measures
    """
    contingency_table, N = create_contingency_table(true_b, alg_b, evaluation)
    joint_prob = evaluate_accuracy(contingency_table, evaluation, True)
    evaluate_pairwise_metrics(contingency_table, N, evaluation, True)
    evaluate_entropy_metrics(joint_prob, evaluation, True)
# End of evaluate_partition()


def create_contingency_table(true_b: np.ndarray, alg_b: np.ndarray, evaluation: Evaluation) -> Tuple[np.ndarray, int]:
    """Creates the contingency table for the block assignment of the truth and algorithmically determined partitions..
    
        Parameters
        ---------
        true_b : ndarray (int)
                array of truth block assignment for each node. If the truth block is not known for a node, -1 is used
                to indicate unknown blocks.
        alg_b : ndarray (int)
                array of output block assignment for each node. The length of this array corresponds to the number of
                nodes observed and processed so far.
        evaluation : Evaluation
                stores the evaluation results
        
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
    num_blocks_truth = len(blocks_b1_set)

    blocks_b2 = alg_b
    num_blocks_alg = max(blocks_b2) + 1

    evaluation.num_blocks_algorithm = num_blocks_alg
    evaluation.num_blocks_truth = num_blocks_truth

    print('\nPartition Correctness Evaluation\n')
    print('Number of nodes: {}'.format(len(alg_b)))
    print('Number of partitions in truth partition: {}'.format(num_blocks_truth))
    print('Number of partitions in alg. partition: {}'.format(num_blocks_alg))

    # populate the confusion matrix between the two partitions
    contingency_table = np.zeros((num_blocks_truth, num_blocks_alg))
    for i in range(len(alg_b)):  # evaluation based on nodes observed so far
        if true_b[i] != -1:  # do not include nodes without truth in the evaluation
            contingency_table[blocks_b1[i], blocks_b2[i]] += 1
    N = contingency_table.sum()

    if num_blocks_truth > num_blocks_alg:  # transpose matrix for linear assignment (this implementation assumes #col >= #row)
        contingency_table = contingency_table.transpose()
    contingency_table_before_assignment = np.array(contingency_table)

    # associate the labels between two partitions using linear assignment
    contingency_table, indexes = associate_labels(contingency_table, contingency_table_before_assignment)

    # fill in the un-associated columns
    contingency_table = fill_unassociated_columns(contingency_table, contingency_table_before_assignment, indexes)

    if num_blocks_truth > num_blocks_alg:  # transpose back
        contingency_table = contingency_table.transpose()
    print('Contingency Table: \n{}'.format(contingency_table))
    return contingency_table, N
# End of create_contingency_table()


def associate_labels(contingency_table: np.ndarray,
    contingency_table_before_assignment: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    """Uses linear assignment through Munkres to correctly pair up the block numbers in the truth and algorithmic
    partitions.

        Parameters
        ---------
        contingency_table : np.ndarray (int)
                the un-matched contingency table, will be modified in this function
        contingency_table_before_assignment : np.ndarray (int)
                the un-matched contingency table, will not be modified in this function
        
        Returns
        ------
        contingency_table : np.ndarray (int)
                the contingency table, after the rows and columns have been properly matched using Munkres
        indexes : List[Tuple[int,int]]
                the indexes for traversing the matrix, as determined by Munkres
    """
    # associate the labels between two partitions using linear assignment
    assignment = Munkres()  # use the Hungarian algorithm / Kuhn-Munkres algorithm
    indexes = assignment.compute(-contingency_table)
    total = 0
    for row, column in indexes:
        contingency_table[:, row] = contingency_table_before_assignment[:, column]
        total += contingency_table[row, row]
    return contingency_table, indexes
# End of associate_labels()


def fill_unassociated_columns(contingency_table: np.ndarray, contingency_table_before_assignment: np.ndarray,
    indexes: List[Tuple[int,int]]) -> np.ndarray:
    """Uses linear assignment through Munkres to correctly pair up the block numbers in the truth and algorithmic
    partitions.

        Parameters
        ---------
        contingency_table : np.ndarray (int)
                the current contingency table, will be modified in this function
        contingency_table_before_assignment : np.ndarray (int)
                the un-matched contingency table, will not be modified in this function
        indexes : List[Tuple[int,int]]
                the list of indexes for traversing the matrix, as determined by Munkres
        
        Returns
        ------
        contingency_table : np.ndarray (int)
                the contingency table, after the rows and columns have been properly matched using Munkres
    """
    # fill in the un-associated columns
    unassociated_col = set(range(contingency_table.shape[1])) - set(np.array(indexes)[:, 1])
    counter = 0
    for column in unassociated_col:
        contingency_table[:, contingency_table.shape[0] + counter] = contingency_table_before_assignment[:, column]
        counter += 1
    return contingency_table
# End of fill_unassociated_columns()


def evaluate_accuracy(contingency_table: np.ndarray, evaluation: Evaluation, is_subgraph: bool = False) -> np.ndarray:
    """Evaluates the accuracy of partitioning.
    
        Parameters
        ---------
        contingency_table : np.ndarray (int)
                the contingency table (confusion matrix) comparing the true block assignment to the algorithmically
                determined block assignment
        evaluation : Evaluation
                stores evaluation results
        is_subgraph : bool
                True if evaluation is for a subgraph. Default = False
        
        Returns
        -------
        joint_prob : np.ndarray (float)
                the normalized contingency table
    """
    joint_prob = contingency_table / sum(sum(contingency_table))  # joint probability of the two partitions is just the normalized contingency table
    accuracy = sum(joint_prob.diagonal())
    print('Accuracy (with optimal partition matching): {}'.format(accuracy))
    print()
    if is_subgraph:
        evaluation.subgraph_accuracy = accuracy
    else:
        evaluation.accuracy = accuracy
    return joint_prob
# End of evaluate_accuracy()


def evaluate_pairwise_metrics(contingency_table: np.ndarray, N: int, evaluation: Evaluation, is_subgraph: bool = False):
    """Evaluates the pairwise metrics for goodness of the partitioning. Metrics evaluated:
    rand index, adjusted rand index, pairwise recall, pairwise precision.
    
        Parameters
        ---------
        contingency_table : np.ndarray (int)
                the contingency table (confusion matrix) comparing the true block assignment to the algorithmically
                determined block assignment
        N : int
                the number of nodes in the confusion matrix
        evaluation : Evaluation
                stores evaluation results
        is_subgraph : bool
                True if evaluation is for a subgraph. Default = False
    """
    # Compute pair-counting-based metrics
    def nchoose2(a):
        return misc.comb(a, 2)

    num_pairs = nchoose2(N)
    colsum = np.sum(contingency_table, axis=0)
    rowsum = np.sum(contingency_table, axis=1)
    # compute counts of agreements and disagreement (4 types) and the regular rand index
    count_in_each_b1 = np.sum(contingency_table, axis=1)
    count_in_each_b2 = np.sum(contingency_table, axis=0)
    num_same_in_b1 = sum(count_in_each_b1 * (count_in_each_b1 - 1)) / 2
    num_same_in_b2 = sum(count_in_each_b2 * (count_in_each_b2 - 1)) / 2
    num_agreement_same = 0.5 * sum(sum(contingency_table * (contingency_table - 1)))
    num_agreement_diff = calc_num_agreement_diff(contingency_table, N, colsum, rowsum)
    num_agreement = num_agreement_same + num_agreement_diff
    rand_index = num_agreement / num_pairs

    adjusted_rand_index = calc_adjusted_rand_index(contingency_table, nchoose2, colsum, rowsum, num_pairs)

    if is_subgraph:
        evaluation.subgraph_rand_index = rand_index
        evaluation.subgraph_adjusted_rand_index = adjusted_rand_index
        evaluation.subgraph_pairwise_recall = num_agreement_same / num_same_in_b1
        evaluation.subgraph_pairwise_precision = num_agreement_same / num_same_in_b2
    else:
        evaluation.rand_index = rand_index
        evaluation.adjusted_rand_index = adjusted_rand_index
        evaluation.pairwise_recall = num_agreement_same / num_same_in_b1
        evaluation.pairwise_precision = num_agreement_same / num_same_in_b2

    print('Rand Index: {}'.format(rand_index))
    print('Adjusted Rand Index: {}'.format(adjusted_rand_index))
    print('Pairwise Recall: {}'.format(num_agreement_same / (num_same_in_b1)))
    print('Pairwise Precision: {}'.format(num_agreement_same / (num_same_in_b2)))
    print('\n')
# End of evaluate_pairwise_metrics()


def calc_num_agreement_diff(contingency_table: np.ndarray, N: int, colsum: np.ndarray, rowsum: np.ndarray) -> float:
    """Calculates the number of nodes that are different blocks in both the true and algorithmic block assignment.

        Parameters
        ---------
        contingency_table : np.ndarray (int)
                the contingency table (confusion matrix) comparing the true block assignment to the algorithmically
                determined block assignment
        N : int
                the number of nodes in the confusion matrix
        colsum : np.ndarray (int)
                the sum of values across the columns of the contingency table
        rowsum : np.ndarray (int)
                the sum of values across the rows of the contingency table

        Returns
        ------
        num_agreement_diff : float
                the number of nodes that are in different blocks in both the true and algorithmic block assignment
    """
    sum_table_squared = sum(sum(contingency_table ** 2))
    sum_colsum_squared = sum(colsum ** 2)
    sum_rowsum_squared = sum(rowsum ** 2)
    num_agreement_diff = 0.5 * (N ** 2 + sum_table_squared - sum_colsum_squared - sum_rowsum_squared)
    return num_agreement_diff
# End of calc_num_agreement_diff()


def calc_adjusted_rand_index(contingency_table: np.ndarray, nchoose2: Callable, colsum: np.ndarray, 
    rowsum: np.ndarray, num_pairs: int) -> float:
    """Calculates the adjusted rand index for the given contingency table.

        Parameters
        ---------
        contingency_table : np.ndarray (int)
                the contingency table (confusion matrix) comparing the true block assignment to the algorithmically
                determined block assignment
        nchoose2 : Callable
                the n choose 2 function
        colsum : np.ndarray (int)
                the sum of values across the columns of the contingency table
        rowsum : np.ndarray (int)
                the sum of values across the rows of the contingency table
        num_pairs : int
                the number of pairs (result of nchoose2(num_nodes_in_contingency_table))

        Returns
        ------
        adjusted_rand_index : float
                the adjusted rand index calculated here
    """
    vectorized_nchoose2 = np.vectorize(nchoose2)
    sum_table_choose_2 = sum(sum(vectorized_nchoose2(contingency_table)))
    sum_colsum_choose_2 = sum(vectorized_nchoose2(colsum))
    sum_rowsum_choose_2 = sum(vectorized_nchoose2(rowsum))
    adjusted_rand_index = (sum_table_choose_2 - sum_rowsum_choose_2 * sum_colsum_choose_2 / num_pairs) / (
        0.5 * (sum_rowsum_choose_2 + sum_colsum_choose_2) - sum_rowsum_choose_2 * sum_colsum_choose_2 / num_pairs)
    return adjusted_rand_index
# End of calc_adjusted_rand_index()


def evaluate_entropy_metrics(joint_prob: np.ndarray, evaluation: Evaluation, is_subgraph: bool = False):
    """Evaluates the entropy (information theoretics based) goodness of partition metrics.

        Parameters
        ---------
        joint_prob : np.ndarray
                the normalized contingency table
        evaluation : Evaluation
                stores the evaluation metrics
        is_subgraph : bool
                True if evaluation is for a subgraph. Default = False
    """
    # compute the information theoretic metrics
    marginal_prob_b2 = np.sum(joint_prob, 0)
    marginal_prob_b1 = np.sum(joint_prob, 1)
    idx_truth = np.nonzero(marginal_prob_b1)
    idx_alg = np.nonzero(marginal_prob_b2)
    evaluation = calc_entropy(marginal_prob_b1, marginal_prob_b2, idx_truth, idx_alg, evaluation, is_subgraph)
    evaluation = calc_conditional_entropy(joint_prob, marginal_prob_b1, marginal_prob_b2, idx_truth, idx_alg,
                                          evaluation, is_subgraph)

    if is_subgraph:
        if evaluation.subgraph_entropy_truth > 0:
            fraction_missed_info = evaluation.subgraph_entropy_truth_given_algorithm / evaluation.subgraph_entropy_truth
        else:
            fraction_missed_info = 0
        if evaluation.subgraph_entropy_algorithm > 0:
            fraction_err_info = (evaluation.subgraph_entropy_algorithm_given_truth /
                                 evaluation.subgraph_entropy_algorithm)
        else:
            fraction_err_info = 0

        evaluation.subgraph_missed_info = fraction_missed_info
        evaluation.subgraph_erroneous_info = fraction_err_info
    else:
        if evaluation.entropy_truth > 0:
            fraction_missed_info = evaluation.entropy_truth_given_algorithm / evaluation.entropy_truth
        else:
            fraction_missed_info = 0
        if evaluation.entropy_algorithm > 0:
            fraction_err_info = evaluation.entropy_algorithm_given_truth / evaluation.entropy_algorithm
        else:
            fraction_err_info = 0

        evaluation.missed_info = fraction_missed_info
        evaluation.erroneous_info = fraction_err_info

    print('Fraction of missed information: {}'.format(abs(fraction_missed_info)))
    print('Fraction of erroneous information: {}'.format(abs(fraction_err_info)))
# End of evaluate_entropy_metrics()


def calc_entropy(p_marginal_truth: np.ndarray, p_marginal_alg: np.ndarray, idx_truth: np.ndarray,
    idx_alg: np.ndarray, evaluation: Evaluation, is_subgraph: bool = False) -> Evaluation:
    """Calculates the entropy of the truth and algorithm partitions.

        Parameters
        ---------
        p_marginal_truth : np.ndarray (float)
                the marginal probabilities of the truth partition
        p_marginal_alg : np.ndarray (float)
                the marginal probabilities of the algorithm partition
        idx_truth : np.ndarray (int)
                the indexes of the non-zero marginal probabilities of the truth partition
        idx_alg : np.ndarray (int)
                the indexes of the non-zero marginal probabilities of the algorithm partition
        is_subgraph : bool
                True if evaluation is for a subgraph. Default = False

        Returns
        ------
        evaluation : Evaluation
                the evaluation object, updated with the entropy metrics
    """
    # compute entropy of the non-partition2 and the partition2 version
    entropy_truth = -np.sum(p_marginal_truth[idx_truth] * np.log(p_marginal_truth[idx_truth]))
    print('Entropy of truth partition: {}'.format(abs(entropy_truth)))
    entropy_alg = -np.sum(p_marginal_alg[idx_alg] * np.log(p_marginal_alg[idx_alg]))
    print('Entropy of alg. partition: {}'.format(abs(entropy_alg)))
    if is_subgraph:
        evaluation.subgraph_entropy_truth = entropy_truth
        evaluation.subgraph_entropy_algorithm = entropy_alg
    else:
        evaluation.entropy_truth = entropy_truth
        evaluation.entropy_algorithm = entropy_alg
    return evaluation
# End of calc_entropy()


def calc_conditional_entropy(joint_prob: np.ndarray, p_marginal_truth: np.ndarray, p_marginal_alg: np.ndarray,
    idx_truth: np.ndarray, idx_alg: np.ndarray, evaluation: Evaluation, is_subgraph: bool = False) -> Evaluation:
    """Calculates the conditional entropy metrics between the algorithmic and truth partitions. The following metrics
    are calculated: 
    
        entropy of the truth partition given the algorithm partition
        entropy of the algorithm partition given the truth partition
        the mutual information between the algorithm and truth partitions

        Parameters
        ---------

        Returns
        ------
        evaluation : Evaluation
                the evaluation object, updated with the entropy-based goodness of partition metrics
        is_subgraph : bool
                True if evaluation is for a subgraph. Default = False
        """
    conditional_prob_b2_b1 = np.zeros(joint_prob.shape)
    conditional_prob_b1_b2 = np.zeros(joint_prob.shape)
    conditional_prob_b2_b1[idx_truth, :] = joint_prob[idx_truth, :] / p_marginal_truth[idx_truth, None]
    conditional_prob_b1_b2[:, idx_alg] = joint_prob[:, idx_alg] / p_marginal_alg[None, idx_alg]

    # compute the conditional entropies
    idx = np.nonzero(joint_prob)
    H_b2_b1 = -np.sum(np.sum(joint_prob[idx] * np.log(conditional_prob_b2_b1[idx])))
    H_b1_b2 = -np.sum(np.sum(joint_prob[idx] * np.log(conditional_prob_b1_b2[idx])))
    # compute the mutual information (symmetric)
    marginal_prod = np.dot(p_marginal_truth[:, None], np.transpose(p_marginal_alg[:, None]))
    MI_b1_b2 = np.sum(np.sum(joint_prob[idx] * np.log(joint_prob[idx] / marginal_prod[idx])))
    
    if is_subgraph:
        evaluation.subgraph_entropy_truth_given_algorithm = H_b1_b2
        evaluation.subgraph_entropy_algorithm_given_truth = H_b2_b1
        evaluation.subgraph_mutual_info = MI_b1_b2
    else:
        evaluation.entropy_truth_given_algorithm = H_b1_b2
        evaluation.entropy_algorithm_given_truth = H_b2_b1
        evaluation.mutual_info = MI_b1_b2
    
    print('Conditional entropy of truth partition given alg. partition: {}'.format(abs(H_b1_b2)))
    print('Conditional entropy of alg. partition given truth partition: {}'.format(abs(H_b2_b1)))
    print('Mutual informationion between truth partition and alg. partition: {}'.format(abs(MI_b1_b2)))

    return evaluation
# End of calc_conditional_entropy()
