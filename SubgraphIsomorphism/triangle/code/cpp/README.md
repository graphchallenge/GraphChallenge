miniTri is a simple, triangle-based data analytics code.  miniTri is a miniapp
in the Mantevo project (http://www.mantevo.org) at Sandia National Laboratories
The primary authors of miniTri are Jon Berry and Michael Wolf (mmwolf@sandia.gov).

miniTri v. 1.0. Copyright (2016) Sandia Corporation.

For questions, contact Jon Berry (jberry@sandia.gov) or Michael Wolf (mmwolf@sandia.gov).

Please read the accompanying README and LICENSE files.

------------------------------------------------
Description:
------------------------------------------------

miniTri is a proxy for a class of triangle based data analytics (Mantevo).
This simple code is a self-contained piece of C++ software that
uses triangle enumeration with a calculation of specific  vertex and edge properties.
Key uses related to miniTri include 
dense subgraph detection, characterizing graphs, improving community detection, and 
generating graphs.
Related applications exist in cyber security, intelligence, and functional biology.
miniTri attempts to be more application relevant than standard data analytics 
benchmarks such as Graph 500.

Authors: Jon Berry (jberry@sandia.gov), Michael Wolf (mmwolf@sandia.gov)
   and Dylan Stark

The objective of the miniTri miniapp is to calculate a specific number (k) for all triangles
in the graph. miniTri has the following basic steps (some of which can be combined):

1. Find all triangles in the graph;
2. For all triangles in the graph, calculate the vertex triangle degrees;
3. For all triangles in the graph, calculate the edge triangle degrees; and
4. For all each triangle, calculate integer k given triangle degree info.

From these k values,  an upper bound on the largest clique in the graph can be
calculated. 
