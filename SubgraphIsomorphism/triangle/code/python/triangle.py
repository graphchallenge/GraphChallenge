from time import clock
from glob import glob
import os,sys
import logging

def getlogger():
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)                                                                                                                                                                                                   
    return logger

# assume that the input file is in mmio format
# also assume that indexing is 1-based (MATLAB) not 0-based
def triangle(adj_mtx_file, inc_mtx_file):
    from pandas import read_csv
    from scipy.sparse import csr_matrix
    from scipy.sparse import coo_matrix

    logger = getlogger()
    dataset_name = os.path.split(os.path.split(adj_mtx_file)[0])[1]
    logger.info('processing ' + dataset_name)

    # figure out the shape of the adjacency matrix
    a = read_csv(adj_mtx_file, sep='\s+', header=-1, skiprows=2, nrows=1, dtype=float).as_matrix()
    M = a[0,0]
    N = a[0,1]
    
    # read adjacency matrix
    logger.info('reading adjacency matrix')
    t0 = clock()
    y = read_csv(adj_mtx_file, sep='\s+', header=-1, skiprows=3, dtype=float).as_matrix()
    t_read_adj = clock() - t0
    logger.info('read time: ' + str(t_read_adj) + ' seconds')

    # convert data to sparse matrix using the coo_matrix function
    t0 = clock()
    A = coo_matrix( ( y[:,2], (y[:,0]-1, y[:,1]-1) ) , shape=(M,N) )
    A = A + A.transpose()
    adjMtx = A.tocsr()
    t_adj_reshape = clock() - t0
    logger.info('COO to CSR time : ' + str(t_adj_reshape) + ' seconds')

    # figure out shape of incidence matrix
    a = read_csv(inc_mtx_file, sep='\s+', header=-1, skiprows=2, nrows=1, dtype=float).as_matrix()
    M = a[0,0]
    N = a[0,1]
    
    # read incidence matrix
    logger.info('reading incidence matrix')
    t0 = clock()
    y = read_csv(inc_mtx_file, sep='\s+', header=-1, skiprows=3, dtype=float).as_matrix()
    t_read_inc = clock() - t0
    logger.info('read time: ' + str(t_read_inc) + ' seconds')

    # reshape incidence matrix
    t0 = clock()
    B = coo_matrix( ( y[:,2], (y[:,0]-1, y[:,1]-1) ) , shape=(M,N) )
    incMtx = B.tocsr()
    t_inc_reshape = clock() - t0
    logger.info('COO to CSR time : ' + str(t_inc_reshape) + ' seconds')

    # count triangles
    logger.info('counting triangles')
    t0 = clock()
    C =  adjMtx * incMtx
    num_triangles = (C==2).nnz/3
    t_triangle_count = clock() - t0
    logger.info('triangle count time : ' + str(t_triangle_count) + ' seconds')
    logger.info('number of triangles in ' + dataset_name + ' : ' + str(num_triangles))

    return (num_triangles,t_read_adj,t_adj_reshape,t_read_inc,t_inc_reshape,t_triangle_count)


########################################################
# GraphChallenge Benchmark
# Developer : Dr. Siddharth Samsi (ssamsi@mit.edu)
#
# MIT
########################################################
# (c) <2017> Massachusetts Institute of Technology
########################################################
