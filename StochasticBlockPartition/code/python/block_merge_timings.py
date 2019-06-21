"""Contains the BlockMergeTimings class, which stores the timing details for the block merge algorithm step.
"""
import csv
import timeit


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

    def save(self, writer: 'csv._writer'):
        """Save the MCMC update timing details.

            Parameters
            ---------
            writer : csv._writer
                the CSV writer pointing towards the CSV details file
        """
        self._writerow(writer, "Initialization", self.initialization)
        self._writerow(writer, "Indexing", self.indexing)
        self._writerow(writer, "Proposal", self.proposal)
        self._writerow(writer, "Edge Count Updates", self.edge_count_updates)
        self._writerow(writer, "Block Degree Updates", self.block_degree_updates)
        self._writerow(writer, "Compute Delta Entropy", self.compute_delta_entropy)
        self._writerow(writer, "Acceptance", self.acceptance)
        self._writerow(writer, "Merging", self.merging)
        self._writerow(writer, "Re-counting Edges", self.re_counting_edges)
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
        writer.writerow([self.superstep, "Block Merge", iteration, substep, time])
    # End of _writerow()
# End of BlockMergeTimings()
