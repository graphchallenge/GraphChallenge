
from triangle import triangle
import os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument("adj_mtx_file", nargs="?", action="store", type=str, default="../../../data/A_adj.mmio")
parser.add_argument("inc_mtx_file", nargs="?", action="store", type=str, default="../../../data/A_inc.mmio")
args = parser.parse_args()
 
inc_mtx_file = args.inc_mtx_file
adj_mtx_file = args.adj_mtx_file
 
if not os.path.isfile(inc_mtx_file):
	print("File doesn't exist: '{}'!".format(inc_mtx_file))
	sys.exit(1)
elif not os.path.isfile(adj_mtx_file):
	print("File doesn't exist: '{}'!".format(adj_mtx_file))
	sys.exit(1)

# this depends on the pandas package
triangle(adj_mtx_file, inc_mtx_file)


########################################################
# GraphChallenge Benchmark
# Developer : Dr. Siddharth Samsi (ssamsi@mit.edu)
#
# MIT
########################################################
# (c) <2017> Massachusetts Institute of Technology
########################################################
