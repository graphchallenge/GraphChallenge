# Data sets of static and streaming graphs of varying sizes

## The graph and the truth partition are stored in the standard TSV format

### Graphs are stored as a row-wise edge list, where each row is: <source node index> <target node index> <edge weight>

### Truth partitions are stored as a row-wise nodal partition list, where each row is: <node index> <block index>

### All rows are tab-delimited, and all indices begin at 1. The nodes are indexed from 1 to N, where N is the number of nodes in the graph so far. The blocks are indexed from 1 to B, where B is the number of blocks in the graph so far.





