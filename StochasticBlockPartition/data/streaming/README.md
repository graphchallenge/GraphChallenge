# Data sets on streaming graphs generated in two ways:
## Emerging edges over time
Interactions and relationships take place and are observed over time
## Snowball sampling
Data is collected incrementally by exploring the graph from starting point(s)

# File format
Graphs and true partitions are stored in the standard TSV format specified in the /data directory. Streaming graphs are stored in pieces, each representing a part of the graph observed at a given stage. The pieces are numbered in the order of arrival. Nodes are indexed incrementally in the order of their arrival. The truth partition is for the entire graph, however, correctness evaluation at each stage of the graph is only on the nodes observed so far.


