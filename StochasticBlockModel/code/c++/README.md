# C++ code for the baseline graph partition algorithm

The baseline graph partition algorithm is implemented in the open source graph-tool repository by Tiago Peixoto: https://github.com/count0/graph-tool

The implementation is in C++ with Python wrappers so the top level functions are accessible in Python. Specifically, the top level function for invoking the baseline algorithm is minimize_blockmodel_dl, found in "/src/graph_tool/inference/minimize.py". All the core C++ code is under "/src/graph/inference/"

The official graph-tool website is https://graph-tool.skewed.de/ and documentation on the baseline algorithm is at https://graph-tool.skewed.de/static/doc/inference.html

## Running this C++ implementation

To run this C++ implementation, simply install the graph-tool module, and run the example Python script included here: "run_graph_tool_partition_alg_main"

The following Python modules are required for running this script:

- graph_tool : for running the partition algorithm (https://graph-tool.skewed.de/)

- pandas : for loading input graph TSV files

- numpy : for storing the graph

The modules below are optional for evaluating the resulting partition and timing the run:

- scipy : for the combinatoric computation in evaluation

- munkres : linear assignment module for computing the correctness metrics, by Brian Clapper (https://pypi.python.org/pypi/munkres/)

- timeit : for timing each run (https://docs.python.org/2/library/timeit.html)

