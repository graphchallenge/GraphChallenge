import numpy as np
import scipy as sp
from ktruss import ktruss
import os, sys, argparse

#Use the pandas package if available
#import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("input_filename", nargs="?", action="store", type=str, default="../../../data/ktruss_example.tsv")
args = parser.parse_args()

inc_mtx_file = args.input_filename

if not os.path.isfile(inc_mtx_file):
	print("File doesn't exist: '{}'!".format(inc_mtx_file))
	sys.exit(1)

E=ktruss(inc_mtx_file,3)


###################################################
# Graph Challenge benchmark
# Developer: Dr. Vijay Gadepally (vijayg@mit.edu)
# MIT
###################################################
# (c) <2015> Vijay Gadepally
###################################################
