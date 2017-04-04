const Chol = if isdefined(Base, :SparseArrays)
    Base.SparseArrays.CHOLMOD
else
    Base.SparseMatrix.CHOLMOD
end

colptr0(A::Chol.Sparse, col) = unsafe_load(unsafe_load(A.p).p, col)
rowval(A::Chol.Sparse, j) = unsafe_load(unsafe_load(A.p).i, j) + 1
nzval(A::Chol.Sparse, j) = unsafe_load(unsafe_load(A.p).x, j)

function calcx(E, m, n, k)
    CSE = Chol.Sparse(E)
    tmp = CSE'*CSE

    # subtract spdiagm( diag(tmp) ) from tmp in-place by setting diagonals to 0
    tmp_ptr = unsafe_load(tmp.p)
    for col in 1:n
        for j in colptr0(tmp, col) + 1 : colptr0(tmp, col+1)
            if rowval(tmp, j) == col
                unsafe_store!(tmp_ptr.x, 0, j)
            end
        end
    end

    R = CSE * tmp
    s = zeros(Int, m)

    @inbounds for col in 1:n
        for j in colptr0(R, col) + 1 : colptr0(R, col+1)
            if nzval(R, j) == 2
                s[rowval(R, j)] += 1
            end
        end
    end

    rowflags = falses(m) # rows of E with at least one nonzero in them
    @inbounds for col in 1:n
        for j in colptr0(CSE, col) + 1 : colptr0(CSE, col+1)
            if nzval(CSE, j) != 0
                rowflags[rowval(CSE, j)] = true
            end
        end
    end

    return find(s .>= (k-2)), sum(rowflags)
end

function ktruss(inc_mtx_file, k)
    if !isfile( inc_mtx_file )
        println("unable to open input file")
        return (-1)
    end

    # load input data       
    t_read_inc=@elapsed ii = readdlm( inc_mtx_file, '\t', Int64)
    println("incidence matrix read time : ", t_read_inc)

    t_create_inc=@elapsed E = sparse( ii[:,1], ii[:,2], ii[:,3] )
    println("sparse adj. matrix creation time : ", t_create_inc)

    tic()
    m,n = size(E)
    xcinds, rowcount = calcx(E, m, n, k)
    while length(xcinds) != rowcount
        # keep rows of E where x is false, equivalent to E[find(x), :] = 0
        Ekeep = E[xcinds, :]
        E = SparseMatrixCSC(m, n, Ekeep.colptr, xcinds[Ekeep.rowval], Ekeep.nzval)
        xcinds, rowcount = calcx(E, m, n, k)
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
