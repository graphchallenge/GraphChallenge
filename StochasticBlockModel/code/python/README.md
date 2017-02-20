# Python code for the baseline graph partition algorithm

## Source code

### Jupyter notebook file
- partition_baseline.ipynb : Contains the entire implementation, with documentation on the algorithmic details of each step of the algorithm. Run this to visualize and see the intermediate results through each step of the graph partitioning algorithm, to get a feel for how it works.

### Python files
- partition_baseline_support.py : Source code for all the supporting functions of the graph partitioning algorithm, with functional documentation detailiong all the inputs and outputs 

- partition_baseline_main.py : Source code for the main routine that invokes the supporting functions to perform graph partitioning


## Version
The source code is run and tested using Python 2.7

## Dependency
This baseline implementation uses the following Python modules:

### Required modules (standard or easy to install)
- pandas : for loading input graph TSV files

- sys : standard module for determining the smallest positive floating value of the computing platform

- numpy : used extensively for storage and computations

- scipy : for sparse matrix and some specific computations

- munkres : linear assignment module for computing the correctness metrics, by Brian Clapper (https://pypi.python.org/pypi/munkres/)

### Optional modules
- timeit : for timing each run (https://docs.python.org/2/library/timeit.html)

- graph_tool : for visualizing the intermediate partitioning results (https://graph-tool.skewed.de/)

