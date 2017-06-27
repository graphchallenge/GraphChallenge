import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
import time
#Use the pandas package if available
#import pandas as pd
#import pdb ; pdb.set_trace()

import scipy.io as sio

###################################################
###################################################
def StrArrayRead(filename):
    
    f=open(filename,'r')
    edgelist = []
    with open(filename, 'r') as f:
        for line in f:
            edgelist.append(list(map(float, line.split('\t'))))
    f.close()
    return np.asarray(edgelist)


###################################################
###################################################
def set_zero_rows(sparray, rowNum):
    for row in rowNum:
        sparray.data[sparray.indptr[row]:sparray.indptr[row+1]]=0

###################################################
###################################################
def set_diag_val(sparray,val):
    r,c=sparray.shape
    for row in xrange(r):
        sparray[row,row]=val


###################################################
###################################################
def StrArrayWrite(nparray, filename):
    
    f=open(filename,"wt",buffering=20*(1024**2))
    f=open(filename,"w")
    data = [str(float(row[0])) + '\t' + str(float(row[1])) + '\n' for row in nparray]
    f.write(''.join(data))
    f.close()

#Use Pandas if you have it
#pd.DataFrame(nparray).to_csv(filename, sep='\t', header=False, index=False)


def ktruss (inc_mat_file,k):
    
    ii=StrArrayRead(inc_mat_file)
    
    startTime=time.clock()
    
    E=csr_matrix(( ii[:,2], (ii[:,0]-1, ii[:,1]-1)), shape=(max(ii[:,0]),max(ii[:,1])))
    
    readTime=time.clock()
    
    tmp=np.transpose(E)*E
    sizeX,sizeY=np.shape(tmp)
    
    print "Time to Read Data:  " + str(readTime-startTime) + "s"
    print "Computing k-truss"
    tmp.setdiag(np.zeros(sizeX),k=0)
    #set_diag_val(tmp,0)
    tmp.eliminate_zeros()
    R= E * tmp
    
    s=lil_matrix(((R==2).astype(float)).sum(axis=1))
    xc= (s >=k-2).astype(int)
    
    while xc.sum() != np.unique(sp.sparse.find(E)[0]).shape:
	x=sp.sparse.find(xc==0)[0]
	#x=np.where(xc==0)[0]
        set_zero_rows(E, x)
        E=(E>0).astype(int)
	#E.eliminate_zeros()
        tmp=np.transpose(E)*E
        (tmp).setdiag(np.zeros(np.shape(tmp)[0]),k=0)
        tmp.eliminate_zeros()
        R=E*tmp
	s=csr_matrix(((R==2).astype(float)).sum(axis=1))
        xc= (s >=k-2).astype(int)

    ktrussTime=time.clock()
    print "Time to Compute k=truss:  " + str(ktrussTime-startTime) + "s"
    return E


###################################################
# Graph Challenge benchmark
# Developer: Dr. Vijay Gadepally (vijayg@mit.edu)
# MIT
###################################################
# (c) <2015> Vijay Gadepally
###################################################
