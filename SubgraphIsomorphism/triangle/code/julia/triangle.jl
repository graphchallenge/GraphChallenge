
function triangle(adj_mtx_file, inc_mtx_file)
    if ~isfile( adj_mtx_file ) || ~isfile( inc_mtx_file )
        println("unable to open input files");
        return (-1);
    end

    # load input data    
    t_read_adj=@elapsed ii = readdlm( adj_mtx_file, '\t', Int64);
    println("adjacency matrix read time : ", t_read_adj);
    
    t_create_adj=@elapsed A = sparse( ii[:,1], ii[:,2], ii[:,3] );
    println("sparse adj. matrix creation time : ", t_create_adj);
    
    t_read_inc=@elapsed ii = readdlm( inc_mtx_file, '\t', Int64);
    println("incidence matrix read time : ", t_read_inc);

    t_create_inc=@elapsed B = sparse( ii[:,1], ii[:,2], ii[:,3] );
    println("sparse adj. matrix creation time : ", t_create_inc);
    
    t_mult=@elapsed C = A*B;
    println("matrix multiplication time : ", t_mult);
    
    t_find=@elapsed y = find( x->x==2, C);
    nt = length(y)/3;
    println("triangle count time : ", t_find+t_mult);
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
