# C++ code for the baseline graph partition algorithm

The baseline graph partition algorithm is implemented in the open source graph-tool repository by Tiago Peixoto: https://git.skewed.de/count0/graph-tool/tree/master

The implementation is in C++ with Python wrappers so the top level functions are accessible in Python. Specifically, the top level function for invoking the baseline algorithm is minimize_blockmodel_dl, found in "/src/graph_tool/inference/minimize.py". All the core C++ code is under "/src/graph/inference/"

The official graph-tool website is https://graph-tool.skewed.de/ and documentation on the baseline algorithm is at https://graph-tool.skewed.de/static/doc/inference.html

## Running this C++ implementation

To run this C++ implementation, simply install the graph-tool module, and run the example Python script included here: "run_graph_tool_partition_alg_main" with the appropriate arguments. For example: "python run_graph_tool_partition_alg_main.py ../../data/static/simulated_blockmodel_graph_5000_nodes" partitions the static graph with 5000 nodes and "python run_graph_tool_partition_alg_main.py ../../data/streaming/emergingEdges/500_nodes/simulated_blockmodel_graph_500_nodes_edgeSample -p 7" partitions stage 7 of the streaming graph with emerging edges.

The following Python modules are required for running this script:

- graph_tool : For running the partition algorithm (https://graph-tool.skewed.de/). Tested with v.2.16 (parallel MCMC updates seem to crash in later versions)

- pandas : For loading input graph TSV files

- numpy : For storing the graph

The modules below are optional for evaluating the resulting partition and timing the run:

- scipy : For the combinatoric computation in evaluation

- munkres : Linear assignment module for computing the correctness metrics, by Brian Clapper (https://pypi.python.org/pypi/munkres/)

- timeit : For timing each run (https://docs.python.org/2/library/timeit.html)

