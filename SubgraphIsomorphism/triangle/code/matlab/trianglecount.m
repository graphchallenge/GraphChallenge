% TRIANGLECOUNT implements triangle counting graph challenge benchmark
%   Usage:
%       trianglecount(adj_mtx_file, inc_mtx_file)
%
%   Inputs:
%       adj_mtx_file - full path to TSV file containing input adjacency matrix
%       inc_mtx_file - full path to TSV file containing input incidence matrix
%

function [numTriangles, T] = trianglecount(adj_mtx_file, inc_mtx_file)

% read data
if exist(adj_mtx_file, 'file')
    t0 = clock;
    a = load(adj_mtx_file);
    t_read_adj = etime(clock, t0);
    fprintf('adjacency matrix read time : %f seconds \n', t_read_adj);
    t0 = clock;
    A = sparse(a(:,1), a(:,2), a(:,3));
    t_create_adj = etime(clock, t0);
    fprintf('adjacency matrix create time : %f seconds \n', t_create_adj);    
else
    error('Unable to read adjacency matrix');
end

if exist(inc_mtx_file, 'file')
    t0 = clock;
    a = load(inc_mtx_file);
    t_read_inc = etime(clock, t0);
    fprintf(2, 'incidence matrix read time : %f seconds \n', t_read_inc);
    t0 = clock;
    B = sparse(a(:,1), a(:,2), a(:,3));
    t_create_inc = etime(clock, t0);
    fprintf('incidence matrix create time : %f seconds \n', t_create_inc);    
else
    error('Unable to read incidence matrix');
end

% count triangles
fprintf('counting triangles\n');
t0 = clock;
C = A*B;
numTriangles = nnz( C==2 ) / 3;
t_triangle_count = etime(clock, t0);

fprintf('number of triangles : %d\ntime to count triangles : %f seconds\n', ...
    numTriangles, t_triangle_count );

T = struct('t_create_adj', t_create_adj, ...
	   't_create_inc', t_create_inc, ...
	   't_read_adj', t_read_adj, ...
	   't_read_inc', t_read_inc, ...
	   't_triangle_count', t_triangle_count);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graph Challenge benchmark
% Developer : Dr. Siddharth Samsi (ssamsi@mit.edu)
% MIT
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) <2017> Massachusetts Institute of Technology
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

