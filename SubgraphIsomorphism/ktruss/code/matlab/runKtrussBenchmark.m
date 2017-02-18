
inc_mtx_file = '../../../data/ktruss_example.tsv';

E_expected =  [1  1  0  0  0; ...
               0  1  1  0  0; ...
               1  0  0  1  0; ...
               0  0  1  1  0; ...
               1  0  1  0  0; ...
               0  0  0  0  0];


E = ktruss(inc_mtx_file, 3);

if nnz( full(E) - E_expected )
    fprintf(2, 'Unable to verify results\n');
else
    disp(E);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graph Challenge benchmark
% Developer : Dr. Siddharth Samsi (ssamsi@mit.edu)
%
% MIT
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) <2017> Massachusetts Institute of Technology
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

