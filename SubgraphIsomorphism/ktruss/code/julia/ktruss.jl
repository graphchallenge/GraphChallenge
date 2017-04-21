function calcx(Et, k)
    n, m = size(Et)
    tmp = Et*Et'
    Rt = ( tmp - Diagonal( diag(tmp) ) )*Et
    s = vec(sum(t -> t == 2, Rt, 1))
    x = find(t -> t < (k - 2), s)
    return x
end

function ktruss(inc_mtx_file, k; time = true)
    if !isfile( inc_mtx_file )
        println("unable to open input file")
        return (-1)
    end

    # load input data
    t_read_inc = @elapsed ii = readdlm( inc_mtx_file, '\t', Int64)
    time && println("incidence matrix read time : ", t_read_inc)

    t_create_inc = @elapsed Et = sparse( ii[:,2], ii[:,1], ii[:,3] )
    time && println("sparse adj. matrix creation time : ", t_create_inc)

    time && tic()
    m = size(Et, 2)
    x = calcx(Et, k)
    while (m - length(x)) != sum( any(Et,1) )
        Et[:, x] = 0
        x = calcx(Et, k)
    end
    time && toc()
    return Et'
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
