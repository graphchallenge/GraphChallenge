
function triangle(adj_mtx_file)
    if ~isfile( adj_mtx_file ) 
        println("unable to open input file");
        return (-1);
    end

    # load input data    
    t_read_adj=@elapsed ii = readdlm( adj_mtx_file, '\t', Int64);
    println("adjacency matrix read time : ", t_read_adj);
    
    t_create_adj=@elapsed A = sparse( ii[:,1], ii[:,2], ii[:,3] );
    println("sparse adj. matrix creation time : ", t_create_adj);
    
    tic;
    nt = sum( A^2 .*  A )/6;
    t = toc;
    println("triangle count time : ", t);
    println("number of triangles : ", nt);
    
    return(nt)
end



#######################################################
# Graph Challenge Benchmark
# Developer : Dr. Siddharth Samsi (ssamsi@mit.edu)
#
# MIT
########################################################
# (c) <2017> Massachusetts Institute of Technology
########################################################
