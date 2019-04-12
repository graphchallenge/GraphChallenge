"""Contains code for evaluating the resulting partition.
"""

import os
import csv
import timeit
from typing import Tuple, List, Callable
from argparse import Namespace

from munkres import Munkres # for correctness evaluation
import scipy.misc as misc
import numpy as np

from graph import Graph


class Evaluation(object):
    """Stores the evaluation results, and saves them to file.
    """

    FIELD_NAMES = [
        'block size variation',
        'block overlap',
        'streaming type',
        'num vertices',
        'num edges',
        'num block proposals',
        'beta',
        'sparse',
        'delta entropy threshold',
        'nodal update threshold strategy',
        'nodal update threshold factor',
        'nodal update threshold direction',
        'num blocks in algorithm partition',
        'num blocks in truth partition',
        'accuracy',
        'rand index',
        'adjusted rand index',
        'pairwise recall',
        'pairwise precision',
        'entropy of algorithm partition',
        'entropy of truth partition',
        'entropy of algorithm partition given truth partition',
        'entropy of truth partition given algorithm partition',
        'mutual information',
        'fraction of missed information',
        'fraction of erroneous information',
        'num nodal updates',
        'num nodal update iterations',
        'num iterations',
        'total partition time',
        'total block merge time',
        'total nodal update time'
    ]

    def __init__(self, args: Namespace, graph: Graph) -> None:
        """Creates a new Evaluation object.

            Parameters
            ----------
            args : Namespace
                    the command-line arguments
            graph : Graph
                    the loaded graph to be partitioned
        """
        # CSV file into which to write the results
        self.csv_file = args.csv
        # Dataset parameters
        self.block_size_variation = args.blockSizeVar
        self.block_overlap = args.overlap
        self.streaming_type = args.type
        self.num_nodes = graph.num_nodes
        self.num_edges = graph.num_edges
        # Algorithm parameters
        self.num_block_proposals = args.blockProposals
        self.beta = args.beta
        self.sparse = args.sparse
        self.delta_entropy_threshold = args.threshold
        self.nodal_update_threshold_strategy = args.nodal_update_strategy
        self.nodal_update_threshold_factor = args.factor
        self.nodal_update_threshold_direction = args.direction
        # Goodness of partition measures
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
        # Algorithm runtime measures
        self.num_nodal_updates = 0
        self.num_nodal_update_iterations = 0
        self.num_iterations = 0
        self.total_partition_time = 0.0
        self.total_block_merge_time = 0.0
        self.total_nodal_update_time = 0.0
        self.prepare_next_partition = 0.0
        self.mcmc_details = list()  # type: List[MCMCTimings]
        self.block_merge_details = list()  # type: List[BlockMergeTimings]
    # End of __init__()

    def update_timings(self, block_merge_start_t: float, node_update_start_t: float, node_update_end_t: float):
        """Updates the timings of a single iteration (block merge + nodal updates)

            Parameters
            ---------
            block_merge_start_t : float
                    the start time of the block merge step
            node_update_start_t : float
                    the start time of the nodal update step
            node_update_end_t : float
                    the end time of the nodal update step
        """
        block_merge_t = node_update_start_t - block_merge_start_t
        node_update_t = node_update_end_t - node_update_start_t
        self.total_block_merge_time += block_merge_t
        self.total_nodal_update_time += node_update_t
    # End of update_timings()

    def total_runtime(self, start_t: float, end_t: float):
        """Finalizes the runtime of the algorithm.

            Parameters
            ---------
            start_t : float
                    the start time of the partitioning
            end_t : float
                    the end time of the partitioning
        """
        runtime = end_t - start_t
        self.total_partition_time = runtime
    # End of total_runtime()

    def save(self):
        """Saves the evaluation to a CSV file. Creates a new CSV file one the path of csv_file doesn't exist. Appends
        results to the CSV file if it does.
        """
        write_header = False
        if not os.path.isfile(self.csv_file):
            directory = os.path.dirname(self.csv_file)
            if directory not in [".", ""]:
                os.makedirs(directory, exist_ok=True)
            write_header = True
        with open(self.csv_file, "a") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(Evaluation.FIELD_NAMES)
            writer.writerow([
                self.block_size_variation,
                self.block_overlap,
                self.streaming_type,
                self.num_nodes,
                self.num_edges,
                self.num_block_proposals,
                self.beta,
                self.sparse,
                self.delta_entropy_threshold,
                self.nodal_update_threshold_strategy,
                self.nodal_update_threshold_factor,
                self.nodal_update_threshold_direction,
                self.num_blocks_algorithm,
                self.num_blocks_truth,
                self.accuracy,
                self.rand_index,
                self.adjusted_rand_index,
                self.pairwise_recall,
                self.pairwise_precision,
                self.entropy_algorithm,
                self.entropy_truth,
                self.entropy_algorithm_given_truth,
                self.entropy_truth_given_algorithm,
                self.mutual_info,
                self.missed_info,
                self.erroneous_info,
                self.num_nodal_updates,
                self.num_nodal_update_iterations,
                self.num_iterations,
                self.total_partition_time,
                self.total_nodal_update_time,
                self.total_block_merge_time
            ])
    # End of save()
# End of Evaluation()


class MCMCTimings(object):
    """Stores timings for a single iteration of the MCMC update step.
    """
    def __init__(self, superstep: int) -> None:
        """Creates an MCMCTimings object.

            Parameters
            ----------
            superstep : int
                the superstep of the algorithm for which MCMC timings are being collected
        """
        self.superstep = superstep
        self._start_t = 0.0
        self._start_b = False
        self.initialization = 0.0
        self.compute_initial_entropy = 0.0
        self.iterations = 0
        self.indexing = list()  # type: List[float]
        self.proposal = list()  # type: List[float]
        self.neighbor_counting = list()  # type: List[float]
        self.edge_count_updates = list()  # type: List[float]
        self.block_degree_updates = list()  # type: List[float]
        self.hastings_correction = list()  # type: List[float]
        self.compute_delta_entropy = list()  # type: List[float]
        self.acceptance = list()  # type: List[float]
        self.early_stopping = list()  # type: List[float]
        self.compute_entropy = 0.0
    # End of __init__()

    def _start(self):
        """Stores the start time for this round of MCMC timings.

            Returns
            ------
            _start_b : bool
                True if time recorded was for the start of a step, False if it was recorded for the end of a step
        """
        if self._start_b:
            self._start_t = timeit.default_timer()
            self._start_b = True
        else:
            self._start_b = False
        return self._start_b
    # End of start()

    def t_initialization(self):
        """Stores the time taken for the initialization step.
        """
        if not self._start():
            self.initialization += timeit.default_timer() - self._start_t
    # End of t_initialization()

    def t_compute_initial_entropy(self):
        """Stores the time taken to compute initial entropy.
        """
        if not self._start():
            self.compute_initial_entropy += timeit.default_timer() - self._start_t
    # End of t_compute_initial_entropy()

    def t_indexing(self):
        """Stores the time taken to do indexing for the current iteration.
        """
        if not self._start():
            if len(self.indexing) == self.iterations:
                self.indexing.append(timeit.default_timer() - self._start_t)
            else:
                self.indexing[self.iterations] += timeit.default_timer() - self._start_t
    # End of t_indexing()

    def t_proposal(self):
        """Stores the time taken to propose a new block for the current iteration.
        """
        if not self._start():
            if len(self.proposal) == self.iterations:
                self.proposal.append(timeit.default_timer() - self._start_t)
            else:
                self.proposal[self.iterations] += timeit.default_timer() - self._start_t
    # End of t_proposal()

    def t_neighbor_counting(self):
        """Stores the time taken to do neighbor counting for the current iteration.
        """
        if not self._start():
            if len(self.neighbor_counting) == self.iterations:
                self.neighbor_counting.append(timeit.default_timer() - self._start_t)
            else:
                self.neighbor_counting[self.iterations] += timeit.default_timer() - self._start_t
    # End of t_neighbor_counting()

    def t_edge_count_updates(self):
        """Stores the time taken to calculate the edge count updates for the current iteration.
        """
        if not self._start():
            if len(self.edge_count_updates) == self.iterations:
                self.edge_count_updates.append(timeit.default_timer() - self._start_t)
            else:
                self.edge_count_updates[self.iterations] += timeit.default_timer() - self._start_t
    # End of t_edge_count_updates()

    def t_block_degree_updates(self):
        """Stores the time taken to calculate the block degree updates for the current iteration.
        """
        if not self._start():
            if len(self.block_degree_updates) == self.iterations:
                self.block_degree_updates.append(timeit.default_timer() - self._start_t)
            else:
                self.block_degree_updates[self.iterations] += timeit.default_timer() - self._start_t
    # End of t_block_degree_updates()

    def t_hastings_correction(self):
        """Stores the time taken to calculate the hastings correction for the current iteration.
        """
        if not self._start():
            if len(self.hastings_correction) == self.iterations:
                self.hastings_correction.append(timeit.default_timer() - self._start_t)
            else:
                self.hastings_correction[self.iterations] += timeit.default_timer() - self._start_t
    # End of t_hastings_correction()

    def t_compute_delta_entropy(self):
        """Stores the time taken to calculate the change in entropy for the current iteration.
        """
        if not self._start():
            if len(self.compute_delta_entropy) == self.iterations:
                self.compute_delta_entropy.append(timeit.default_timer() - self._start_t)
            else:
                self.compute_delta_entropy[self.iterations] += timeit.default_timer() - self._start_t
    # End of t_compute_delta_entropy()

    def t_acceptance(self):
        """Stores the time taken to accept or reject new block proposals for the current iteration.
        """
        if not self._start():
            if len(self.acceptance) == self.iterations:
                self.acceptance.append(timeit.default_timer() - self._start_t)
            else:
                self.acceptance[self.iterations] += timeit.default_timer() - self._start_t
    # End of t_acceptance()

    def t_early_stopping(self):
        """Stores the time taken to do early stopping for the current iteration.
        """
        if not self._start():
            if len(self.early_stopping) == self.iterations:
                self.early_stopping.append(timeit.default_timer() - self._start_t)
            else:
                self.early_stopping[self.iterations] += timeit.default_timer() - self._start_t
    # End of t_early_stopping()

    def t_compute_final_entropy(self):
        """Stores the time taken to compute the final entropy of the partition.
        """
        if not self._start():
            self.compute_entropy += timeit.default_timer() - self._start_t
    # End of t_compute_initial_entropy()
# End of MCMCTimings()


class BlockMergeTimings(object):
    """Stores timings of the block merge step.
    """
    def __init__(self, superstep: int) -> None:
        """creates a BlockMergeTimings object.

            Parameters
            ----------
            superstep : int
                the superstep of the algorithm for which block merge timings are being collected
        """
        self.superstep = superstep
        self._start_t = 0.0
        self._start_b = False
        self.initialization = 0.0
        self.indexing = 0.0
        self.proposal = 0.0
        self.edge_count_updates = 0.0
        self.block_degree_updates = 0.0
        self.compute_delta_entropy = 0.0
        self.acceptance = 0.0
        self.merging = 0.0
        self.re_counting_edges = 0.0
    # End of __init__()

    def _start(self):
        """Stores the start time for this round of block merge timings.

            Returns
            ------
            _start_b : bool
                True if time recorded was for the start of a step, False if it was recorded for the end of a step
        """
        if self._start_b:
            self._start_t = timeit.default_timer()
            self._start_b = True
        else:
            self._start_b = False
        return self._start_b
    # End of start()

    def t_initialization(self):
        """Stores the time taken for the initialization step.
        """
        if not self._start():
            self.initialization += timeit.default_timer() - self._start_t
    # End of t_initialization()

    def t_indexing(self):
        """Stores the time taken to do indexing for the current iteration.
        """
        if not self._start():
            self.indexing += timeit.default_timer() - self._start_t
    # End of t_indexing()

    def t_proposal(self):
        """Stores the time taken to propose a new block merge for the current iteration.
        """
        if not self._start():
            self.proposal += timeit.default_timer() - self._start_t
    # End of t_proposal()

    def t_edge_count_updates(self):
        """Stores the time taken to calculate the edge count updates for the current iteration.
        """
        if not self._start():
            self.edge_count_updates += timeit.default_timer() - self._start_t
    # End of t_edge_count_updates()

    def t_block_degree_updates(self):
        """Stores the time taken to calculate the block degree updates for the current iteration.
        """
        if not self._start():
            self.block_degree_updates += timeit.default_timer() - self._start_t
    # End of t_block_degree_updates()

    def t_compute_delta_entropy(self):
        """Stores the time taken to calculate the change in entropy for the current iteration.
        """
        if not self._start():
            self.compute_delta_entropy += timeit.default_timer() - self._start_t
    # End of t_compute_delta_entropy()

    def t_acceptance(self):
        """Stores the time taken to accept or reject new block merge proposals for the current iteration.
        """
        if not self._start():
            self.acceptance += timeit.default_timer() - self._start_t
    # End of t_acceptance()

    def t_merging(self):
        """Stores the time taken to perform a merging step.
        """
        if not self._start():
            self.merging += timeit.default_timer() - self._start_t
    # End of t_merging()

    def t_re_counting_edges(self):
        """Stores the time taken to perform an edge re-counting step.
        """
        if not self._start():
            self.re_counting_edges += timeit.default_timer() - self._start_t
    # End of t_re_counting_edges()
# End of BlockMergeTimings()


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


def evaluate_accuracy(contingency_table: np.ndarray, evaluation: Evaluation) -> np.ndarray:
    """Evaluates the accuracy of partitioning.
    
        Parameters
        ---------
        contingency_table : np.ndarray (int)
                the contingency table (confusion matrix) comparing the true block assignment to the algorithmically
                determined block assignment
        evaluation : Evaluation
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
    evaluation.accuracy = accuracy
    return joint_prob
# End of evaluate_accuracy()


def evaluate_pairwise_metrics(contingency_table: np.ndarray, N: int, evaluation: Evaluation):
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


def evaluate_entropy_metrics(joint_prob: np.ndarray, evaluation: Evaluation):
    """Evaluates the entropy (information theoretics based) goodness of partition metrics.

        Parameters
        ---------
        joint_prob : np.ndarray
                the normalized contingency table
        evaluation : Evaluation
                stores the evaluation metrics
    """
    # compute the information theoretic metrics
    marginal_prob_b2 = np.sum(joint_prob, 0)
    marginal_prob_b1 = np.sum(joint_prob, 1)
    idx_truth = np.nonzero(marginal_prob_b1)
    idx_alg = np.nonzero(marginal_prob_b2)
    evaluation = calc_entropy(marginal_prob_b1, marginal_prob_b2, idx_truth, idx_alg, evaluation)
    evaluation = calc_conditional_entropy(joint_prob, marginal_prob_b1, marginal_prob_b2, idx_truth, idx_alg,
                                          evaluation)

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
    idx_alg: np.ndarray, evaluation: Evaluation) -> Evaluation:
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
    evaluation.entropy_truth = entropy_truth
    evaluation.entropy_algorithm = entropy_alg
    return evaluation
# End of calc_entropy()


def calc_conditional_entropy(joint_prob: np.ndarray, p_marginal_truth: np.ndarray, p_marginal_alg: np.ndarray,
    idx_truth: np.ndarray, idx_alg: np.ndarray, evaluation: Evaluation) -> Evaluation:
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
    
    evaluation.entropy_truth_given_algorithm = H_b1_b2
    evaluation.entropy_algorithm_given_truth = H_b2_b1
    evaluation.mutual_info = MI_b1_b2
    
    print('Conditional entropy of truth partition given alg. partition: {}'.format(abs(H_b1_b2)))
    print('Conditional entropy of alg. partition given truth partition: {}'.format(abs(H_b2_b1)))
    print('Mutual informationion between truth partition and alg. partition: {}'.format(abs(MI_b1_b2)))

    return evaluation
# End of calc_conditional_entropy()
