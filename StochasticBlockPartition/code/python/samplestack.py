"""A Stack object for incremental sampling
"""
from typing import Tuple, Dict, List
import timeit

from graph import Graph
from partition import Partition
from evaluation import Evaluation
from node_reassignment import fine_tune_membership


class SampleStack(object):
    def __init__(self, args: 'argparse.Namespace') -> None:
        """Creates a SampleStack object.

        Parameters
        ---------
        args : argparse.Namespace
            the command-line arguments provided by the user
        """
        # Load graph
        # Create stack of samples
        # Use List as stack
        self.t_load_start = timeit.default_timer()
        graph = Graph.load(args)
        self.t_load_end = timeit.default_timer()
        self.stack = list()  # type: List[Tuple[Graph, Partition, Dict]]
        self.stack.append((graph, None, None))
        self._sample(args)
        self.t_sample_end = timeit.default_timer()
    # End of __init__()

    def _sample(self, args):
        # Iteratively perform sampling
        for _ in range(args.sample_iterations):
            graph = self.stack[-1][0]
            subgraph, vertex_mapping, block_mapping = graph.sample(args)
            self.stack.append((subgraph, vertex_mapping, block_mapping))
    # End of _sample()

    def _push(self):
        # Add a subsample to the stack
        raise NotImplementedError()

    def _pop(self) -> Tuple[Graph, Dict, Dict]:
        # Propagate a subsample's results up the stack
        return self.stack.pop(-1)
    # End of _pop()

    def unstack(self, partition: Partition, args: 'argparse.Namespace',
        evaluation: 'Evaluation') -> Tuple[Graph, Partition, Dict]:
        # Propagate results back through the stack
        subgraph_partition = partition
        _, vertex_mapping, block_mapping = self._pop()
        while len(self.stack) > 1:
            t1 = timeit.default_timer()
            graph, _, _ = self.tail()
            subgraph_partition = Partition.from_sample(subgraph_partition.num_blocks, graph.out_neighbors,
                                                       subgraph_partition.block_assignment, vertex_mapping, args)
            t2 = timeit.default_timer()
            subgraph_partition = fine_tune_membership(subgraph_partition, graph, evaluation, args)
            _, vertex_mapping, block_mapping = self._pop()
            t3 = timeit.default_timer()
            evaluation.propagate_membership += (t2 - t1)
            evaluation.finetune_membership += (t3 - t2)
        t1 = timeit.default_timer()
        full_graph, _, _ = self._pop()
        full_graph_partition = Partition.from_sample(subgraph_partition.num_blocks, full_graph.out_neighbors,
                                                     subgraph_partition.block_assignment, vertex_mapping, args)
        t2 = timeit.default_timer()
        full_graph_partition = fine_tune_membership(full_graph_partition, full_graph, evaluation, args)
        t3 = timeit.default_timer()
        evaluation.loading = self.t_load_end - self.t_load_start
        evaluation.sampling = self.t_sample_end - self.t_load_end
        evaluation.propagate_membership += (t2 - t1)
        evaluation.finetune_membership += (t3 - t2)
        return full_graph, full_graph_partition, block_mapping
    # End of unstack()

    def tail(self) -> Tuple[Graph, Partition, Dict]:
        # Get innermost sample
        return self.stack[-1]
    # End of tail()
# End of SampleStack()
