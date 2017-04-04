# Data sets and generator of static and streaming graphs

## The graph and the truth partition are stored in the standard TSV format
Graphs are stored as a row-wise edge list, where each row is: 

\<source node index\> \<target node index\> \<edge weight\>

Truth partitions are stored as a row-wise nodal partition list, where each row is: 

\<node index\> \<block index\>

All rows are tab-delimited, and all indices begin at 1. The nodes are indexed from 1 to N, where N is the number of nodes in the graph so far. The blocks are indexed from 1 to B, where B is the number of blocks in the graph so far.

## Graph generator code

### Jupyter notebook file
graph_generator.ipynb : Contains the generator code and visualization of the resulting graph.

### Python files
graph_generator.py : Source code for the generator 

### Version
The source code is run and tested using Python 2.7

### Dependency
This graph generator uses the following Python modules:

- graph_tool : for generating and visualizing graphs (https://graph-tool.skewed.de/). Tested with v.2.16.

- pandas : for reading and writing TSV files

- numpy : used extensively for storage, computations, and random number generation

- scipy.stats : for random number generation

- random : basic random number generation and sampling




