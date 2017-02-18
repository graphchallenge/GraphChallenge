miniTri is a simple, triangle-based data analytics code.  miniTri is a miniapp
in the Mantevo project (http://www.mantevo.org) at Sandia National Laboratories
The primary authors of miniTri are Jon Berry and Michael Wolf (mmwolf@sandia.gov).

miniTri v. 1.0. Copyright (2016) Sandia Corporation.

For questions, contact Jon Berry (jberry@sandia.gov) or Michael Wolf (mmwolf@sandia.gov).

Please read the accompanying README and LICENSE files.

------------------------------------------------
linearAlgebra:
------------------------------------------------

This directory contains different implementations of a linear algebra based formulation of
miniTri.  These implementations are supposed to be prototypes of future Graph BLAS based
implementations.

Graph BLAS (graphblas.org) is an effort to standardize building blocks for graph algorithms in language of linear algebra.
The motivation behind Graph BLAS is that most graph computations can be expressed by overloading sparse
linear algebra kernels. The hope is that if vendors optimize small set of building blocks, this will enable users to build
high performing graph analysis applications


Graph BLAS is promising for data sciences but there are many challenges.
We believe that miniTri is an important stressor of the Graph BLAS standard 
requiring flexibility and asynchronous execution of its fine grain operations 
(underlying its high-level abstraction) in order to achieve high performance 
and scalability. An efficient Graph BLAS implementation of miniTri would go a long way to validating the Graph BLAS approach.


The implementations in this directory represent a miniTri linear algebra-based formulation where 
miniTri isrepresented in four compact linear algebra-based operations:

1. Triangle enumeration is computed using an overloaded sparse matrix, sparse matrix multiplication operation
2. Triangle vertex degrees are computed using an overloaded sparse matrix, dense vector multiplication operation
3. Triangle edge degrees are computed using an overloaded sparse matrix, dense vector multiplication operation
4. k values computed using 3 linear algebra based data structures.

A detailed description of this formulation can be found in the following paper:

> Wolf, M.M., J.W. Berry, and D.T. Stark. "A task-based linear algebra Building Blocks 
> approach for scalable graph analytics." High Performance Extreme Computing Conference (HPEC), 
> 2015 IEEE. IEEE, 2015. ([link](http://ieeexplore.ieee.org/document/7322450/?arnumber=7322450))

We have developed several different implementations of this linear algebra-based miniTri using 
different fundamental programming models.  These implementaions are organized as follows:

* __MPI__ -- distributed memory based implementation using MPI
* __openmp__ -- data parallel OpenMP implementation
* __serial__ -- serial reference implementation 




