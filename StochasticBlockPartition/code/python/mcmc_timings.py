"""Contains the MCMCTimings class for storing timing details for the MCMC (nodal updates) algorithm step.
"""
import csv
import timeit
from typing import List


class MCMCTimings(object):
    """Stores timings for a single iteration of the MCMC update step.
    """

    def __init__(self, superstep: int, step: str = "MCMC Updates") -> None:
        """Creates an MCMCTimings object.

            Parameters
            ----------
            superstep : int
                the superstep of the algorithm for which MCMC timings are being collected
            step : str
                the name of the step
        """
        self._step = step
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
        if not self._start_b:
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
            self.iterations += 1
    # End of t_early_stopping()

    def t_compute_final_entropy(self):
        """Stores the time taken to compute the final entropy of the partition.
        """
        if not self._start():
            self.compute_entropy += timeit.default_timer() - self._start_t
    # End of t_compute_initial_entropy()

    def save(self, writer: 'csv._writer'):
        """Save the MCMC update timing details.

            Parameters
            ----------
            writer : csv._writer
                the CSV writer pointing towards the CSV details file
        """
        self._writerow(writer, "Initialization", self.initialization)
        self._writerow(writer, "Compute Initial Entropy", self.compute_initial_entropy)
        for i in range(self.iterations):
            self._writerow(writer, "Indexing", self.indexing[i], i)
            self._writerow(writer, "Proposal", self.proposal[i], i)
            self._writerow(writer, "Neighbor Counting", self.neighbor_counting[i], i)
            self._writerow(writer, "Edge Count Updates", self.edge_count_updates[i], i)
            self._writerow(writer, "Block Degree Updates", self.block_degree_updates[i], i)
            self._writerow(writer, "Hastings Correction", self.hastings_correction[i], i)
            self._writerow(writer, "Compute Delta Entropy", self.compute_delta_entropy[i], i)
            self._writerow(writer, "Acceptance", self.acceptance[i], i)
            self._writerow(writer, "Early Stopping", self.early_stopping[i], i)
        self._writerow(writer, "Compute Final Entropy", self.compute_entropy)
    # End of save()

    def _writerow(self, writer: 'csv._writer', substep: str, time: float, iteration: int = -1):
        """Writes the timing information to the current csv writer.

            Parameters
            ----------
            writer : csv._writer
                the current CSV writer object
            substep : str
                the name of the substep for which the timing is recorded
            time : float
                the timing information to save
            iteration : int (default = -1)
                the current iteration
        """
        writer.writerow([self.superstep, self._step, iteration, substep, time])
    # End of _writerow()

    def zeros(self):
        """Adds zeros to all iteration-dependent variables after the proposal, for the case where the proposal is
        the same as the current block.
        """
        self.neighbor_counting.append(0.0)
        self.edge_count_updates.append(0.0)
        self.block_degree_updates.append(0.0)
        self.hastings_correction.append(0.0)
        self.compute_delta_entropy.append(0.0)
        self.acceptance.append(0.0)
    # End of zeros()
# End of MCMCTimings()