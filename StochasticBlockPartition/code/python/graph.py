"""Holds information about graphs loaded from .tsv files.
"""

class Graph():
    def __init__(self, out_neighbors, in_neighbors, num_nodes, num_edges, true_block_assignment=None):
        self.out_neighbors = out_neighbors
        self.in_neighbors = in_neighbors
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.true_block_assignment = true_block_assignment
    
    def update(self, graph: 'Graph'):
        self.out_neighbors = graph.out_neighbors
        self.in_neighbors = graph.in_neighbors
        self.num_nodes = graph.num_nodes
        self.num_edges = graph.num_edges
