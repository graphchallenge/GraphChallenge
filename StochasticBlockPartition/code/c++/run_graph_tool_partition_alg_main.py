""" This Python script runs the graph-tool graph partition algorithm by Tiago Peixoto."""

from run_graph_tool_partition_alg_support import *
import graph_tool.all as gt
import timeit, os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parts", action="store", type=int)
parser.add_argument("input_filename", action="store", type=str, default="../../data/static/simulated_blockmodel_graph_500_nodes")
args = parser.parse_args()

print(args.parts)

input_filename = args.input_filename

if not os.path.isfile(input_filename + '.tsv') and not os.path.isfile(input_filename + '_1.tsv'):
	print("File doesn't exist: '{}'!".format(input_filename))
	sys.exit(1)

if args.parts != None:
	print('\nLoading partition 1 of {} ({}) ...'.format(args.parts, input_filename + "_1.tsv"))
	out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=True, strm_piece_num=1)
	for part in range(2, args.parts + 1):
		print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))
		out_neighbors, in_neighbors, N, E = load_graph(input_filename, load_true_partition=False, strm_piece_num=part, out_neighbors=out_neighbors, in_neighbors=in_neighbors)
else:
	out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=True)

input_graph = gt.Graph()
input_graph.add_edge_list([(i,j) for i in range(len(out_neighbors)) if len(out_neighbors[i]) > 0 for j in out_neighbors[i][:,0]])
t0 = timeit.default_timer()
# the parallel switch determines whether MCMC updates are run in parallel, epsilon is the convergence threshold for
# the nodal updates (smaller value is stricter), and the verbose option prints updates on each step of the algorithm.
# Please refer to the graph-tool documentation under graph-tool.inference for details on the input parameters
graph_tool_partition = gt.minimize_blockmodel_dl(input_graph, mcmc_args={'parallel':True},
                                                 mcmc_equilibrate_args={'verbose':False, 'epsilon':1e-5}, verbose=True)
t1 = timeit.default_timer()
print('\nGraph partition took {} seconds'.format(t1-t0))
evaluate_partition(true_partition, graph_tool_partition.get_blocks().a)
