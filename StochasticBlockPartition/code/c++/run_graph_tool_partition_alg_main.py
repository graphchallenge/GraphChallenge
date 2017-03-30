""" This Python script runs the graph-tool graph partition algorithm by Tiago Peixoto."""

from run_graph_tool_partition_alg_support import *
import graph_tool.all as gt
import timeit

input_filename = '../../data/static/simulated_blockmodel_graph_20000_nodes'
out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, True)
input_graph = gt.Graph()
input_graph.add_edge_list([(i,j) for i in range(len(out_neighbors)) for j in out_neighbors[i][:,0]])
t0 = timeit.default_timer()
# the parallel switch determines whether MCMC updates are run in parallel, epsilon is the convergence threshold for
# the nodal updates (smaller value is stricter), and the verbose option prints updates on each step of the algorithm.
# Please refer to the graph-tool documentation under graph-tool.inference for details on the input parameters
graph_tool_partition = gt.minimize_blockmodel_dl(input_graph, mcmc_args={'parallel':True},
                                                 mcmc_equilibrate_args={'verbose':False, 'epsilon':1e-4}, verbose=True)
t1 = timeit.default_timer()
print('\nGraph partition took {} seconds'.format(t1-t0))
evaluate_partition(true_partition, graph_tool_partition.get_blocks().a)