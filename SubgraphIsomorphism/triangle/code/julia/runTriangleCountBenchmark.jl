
include("triangle.jl");
include("trianglev2.jl");

adj_mtx_file = "../../../data/A_adj.tsv"
inc_mtx_file = "../../../data/A_inc.tsv"

T1 = @elapsed N1 = triangle(adj_mtx_file, inc_mtx_file)
T2 = @elapsed N2 = trianglev2(adj_mtx_file)

println("\nimplementation I  : time = ", T1, " triangle count = ", N1);
println("implementation II : time = ", T2, " triangle count = ", N2);


#######################################################
# Graph Challenge Benchmark
# Developer : Dr. Siddharth Samsi (ssamsi@mit.edu)
#
# MIT
#########################################################
# (c) <2017> Massachusetts Institute of Technology
########################################################


