
function ktruss(adj_mtx_file, k)
    if ~isfile( adj_mtx_file )
        println("unable to open input file");
        return (-1);
    end

    # load input data       
    t_read_inc=@elapsed ii = readdlm( adj_mtx_file, '\t', Int64);
    println("incidence matrix read time : ", t_read_inc);

    t_create_inc=@elapsed E = sparse( ii[:,1], ii[:,2], ii[:,3] );
    println("sparse adj. matrix creation time : ", t_create_inc);

    #
    tmp = E.'*E;
    R = E * ( tmp - spdiagm(diag(tmp)) );    
    id = sparse( R .== 2 );
    s = sum( id, 2);

    x = s .< (k-2);
    xc = s .>= (k-2); # full matrix unless we cast it to sparse
    while sum(xc) != sum( any(E,2) )
        E[find(x), :] = 0
        R = E * ( tmp - spdiagm(diag(tmp)) );
        id = sparse( R .== 2 );
        s = sum( id, 2);

        x  = s .< (k-2); # full matrix
        xc = s .>= (k-2); # full matrix unless we cast it to sparse
    end
    
    return E
end



#######################################################
# Graph Challenge Benchmark
# Architect : Dr. Jeremy Kepner (kepner@ll.mit.edu)
# Developer : Dr. Siddharth Samsi (ssamsi@mit.edu)
#
# MIT
########################################################
# (c) <2017> Massachusetts Institute of Technology
########################################################
