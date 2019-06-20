# Python code for the baseline graph partition algorithm

For clarity and simplicity, here is a pure Python implementation of the baseline algorithm. Runtime-wise, it is about an order of magnitude slower than the C++ implementation.

## Source code

### Jupyter notebook file

- partition_baseline.ipynb : Contains the entire implementation, with documentation on the algorithmic details of each step of the algorithm. Run this to visualize and see the intermediate results through each step of the graph partitioning algorithm, to get a feel for how it works.

### Python files

- partition_baseline_support.py : Source code for all the supporting functions of the graph partitioning algorithm, with functional documentation detailiong all the inputs and outputs 

- partition_baseline_main.py : Source code for the main routine that invokes the supporting functions to perform graph partitioning. Arguments may be passed to specify the data set and streaming stage to process. For example: "python partition_baseline_main.py ../../data/static/simulated_blockmodel_graph_5000_nodes" partitions the static graph with 5000 nodes and "python partition_baseline_main.py ../../data/streaming/emergingEdges/500_nodes/simulated_blockmodel_graph_500_nodes_edgeSample -p 7" partitions stage 7 of the streaming graph with emerging edges.

## Version

The source code is run and tested using Python 3.6.8, provided through Anaconda

## Dependency

This baseline implementation uses the following Python modules:

### Required modules (standard or easy to install)

- pandas : for loading input graph TSV files

- numpy : used extensively for storage and computations

- scipy : for sparse matrix and some specific computations

- munkres : linear assignment module for computing the correctness metrics, by Brian Clapper (https://pypi.python.org/pypi/munkres/)

### Optional modules

- graph_tool : for visualizing the intermediate partitioning results (https://graph-tool.skewed.de/). Tested with v.2.16.

To install these dependencies, use `pip` with the provided `requirements.txt` file, e.g.: `pip install -r requirements.txt`

## Usage

```
partition_baseline_main.py [-h] [-p PARTS] [-o OVERLAP]
                                  [-s BLOCKSIZEVAR] [-t TYPE] [-n NUMNODES]
                                  [-d DIRECTORY] [-v] [-b BLOCKPROPOSALS]
                                  [-i ITERATIONS] [-r BLOCKREDUCTIONRATE]
                                  [--beta BETA] [--sparse] [-c CSV]
                                  [-u NODAL_UPDATE_STRATEGY]
                                  [--direction DIRECTION] [-f FACTOR]
                                  [-e THRESHOLD] [-z SAMPLE_SIZE]
                                  [-m {uniform_random,random_walk,random_jump,degree_weighted,random_node_neighbor,forest_fire,none}]

optional arguments:
  -h, --help            show this help message and exit
  -p PARTS, --parts PARTS
                        The number of streaming partitions to the dataset. If
                        the dataset is static, 0. Default = 0
  -o OVERLAP, --overlap OVERLAP
                        (low|high). Default = low
  -s BLOCKSIZEVAR, --blockSizeVar BLOCKSIZEVAR
                        (low|high). Default = low
  -t TYPE, --type TYPE  (static|streamingEdge|streamingSnowball). Default =
                        static
  -n NUMNODES, --numNodes NUMNODES
                        The size of the dataset. Default = 1000
  -d DIRECTORY, --directory DIRECTORY
                        The location of the dataset directory. Default =
                        ../../data
  -v, --verbose         If supplied, will print 'helpful' messages.
  -b BLOCKPROPOSALS, --blockProposals BLOCKPROPOSALS
                        The number of block merge proposals per block. Default
                        = 10
  -i ITERATIONS, --iterations ITERATIONS
                        Maximum number of node reassignment iterations.
                        Default = 100
  -r BLOCKREDUCTIONRATE, --blockReductionRate BLOCKREDUCTIONRATE
                        The block reduction rate. Default = 0.5
  --beta BETA           exploitation vs exploration: higher threshold = higher
                        exploration. Default = 3
  --sparse              If supplied, will use Scipy's sparse matrix
                        representation for the matrices.
  -c CSV, --csv CSV     The filepath to the csv file in which to store the
                        evaluation results.
  -u NODAL_UPDATE_STRATEGY, --nodal_update_strategy NODAL_UPDATE_STRATEGY
                        (original|step|exponential|log). Default = original
  --direction DIRECTION
                        (growth|decay) Default = growth
  -f FACTOR, --factor FACTOR
                        The factor by which to grow or decay the nodal update
                        threshold. If the nodal update strategy is step: this
                        value is added to or subtracted from the threshold
                        with every iteration If the nodal update strategy is
                        exponential: this (1 +/- this value) is multiplied by
                        the threshold with every iteration If the nodal update
                        strategy is log: this value is ignored Default =
                        0.0001
  -e THRESHOLD, --threshold THRESHOLD
                        The threshold at which to stop nodal block
                        reassignment. Default = 5e-4
  -z SAMPLE_SIZE, --sample_size SAMPLE_SIZE
                        The percent of total nodes to use as sample. Default =
                        100 (no sampling)
  -m {uniform_random,random_walk,random_jump,degree_weighted,random_node_neighbor,forest_fire,none}, --sample_type {uniform_random,random_walk,random_jump,degree_weighted,random_node_neighbor,forest_fire,none}
                        Sampling algorithm to use. Default = none
```

The script expects the data to be stored using the following file and directory naming convention:

```{bash}
${DIRECTORY}/${TYPE}/${OVERLAP}Overlap_${BLOCKSIZEVAR}BlockSizeVar/${TYPE}_${OVERLAP}Overlap_${BLOCKSIZEVAR}BlockSizeVar_${NUMNODES}_nodes.tsv
```

if the graph is static, or

```{bash}
${DIRECTORY}/${TYPE}/${OVERLAP}Overlap_${BLOCKSIZEVAR}BlockSizeVar/${TYPE}_${OVERLAP}Overlap_${BLOCKSIZEVAR}BlockSizeVar_${NUMNODES}_nodes_{1..PARTS}.tsv
```

if it's a streaming graph.

The truth partition would then be in the following file:

```{bash}
${DIRECTORY}/${TYPE}/${OVERLAP}Overlap_${BLOCKSIZEVAR}BlockSizeVar/${TYPE}_${OVERLAP}Overlap_${BLOCKSIZEVAR}BlockSizeVar_${NUMNODES}_nodes_truePartition.tsv
```
