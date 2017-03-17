function calcx(E, m, n, k)
    tmp = E.'*E;
    R = E * ( tmp - spdiagm( diag(tmp) ) );
    r,c,v = findnz(R);
    id = v.==2;
    A = sparse( r[id], c[id], 1, m, n);
    s = sum(A, 2);
    x = s .< (k-2);    
    return(x, !x);
end

function ktruss(inc_mtx_file, k)
    if ~isfile( inc_mtx_file )
        println("unable to open input file");
        return (-1);
    end

    # load input data       
    t_read_inc=@elapsed ii = readdlm( inc_mtx_file, '\t', Int64);
    println("incidence matrix read time : ", t_read_inc);

    t_create_inc=@elapsed E = sparse( ii[:,1], ii[:,2], ii[:,3] );
    println("sparse adj. matrix creation time : ", t_create_inc);

    #
    tic();
    m,n = size(E);
    x, xc = calcx(E, m, n, k);
    while sum(xc) != sum( any(E,2) )
        E[find(x), :] = 0
        x, xc = calcx(E, m, n, k);
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
