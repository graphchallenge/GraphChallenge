%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parallel Exact Kronecker Graph Generator
% Dr. Jeremy Kepner (MIT)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The goal of this script is to enable the fast, parallel, in-memory generation
% of immense power-law Kronecker graphs with exactly computable theoretical properties.
%
% This script reads in 2 sub-graphs A and B, divides the entries of A
% evenly among difference processors and then performs a local Kronecker
% product of the local sub graph of A with B.
% 
% The local parts of the generated graph can be saved in .tsv file formats.
%
% This script computes and saves the graphs 
% for three types of Kronecker graphs built up
% from star graphs or or B(1,m) bipartite graphs.
% The three classes of graphs are:
% Base: has no self-loops, many triangles, power-law degree distribution
% 1: center has a self-loop, many triangles, nearly power-law degree distribution
% 2: leaf has a self-loop, some triangles, nearly power-law degree distribution
%
%
% NOTE: Requires the pMatlab library to run the program in parallel
% https://www.ll.mit.edu//mission/cybersec/softwaretools/pmatlab/pmatlab.html
%
% To run locally in parallel on 4 processors type:
%     eval(pRUN('pExactKronGenGraph',4,{}));
%
% To run in serial without pMatlab, uncomment the following:
%     Np = 1;  Pid = 0; 
%
% Then type:
%     pExactKronGenGraph
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Flags for computing for saving in .tsv format.
SAVETSV = 1;

% Flag for setting which type graph to generate.
Btype = '';      % Base case (no triangles).
%Btype = '1';     % Type 1 (many triangles).
%Btype = '2';     % Type 2 (some triangles).

Bvar = ['B' Btype 'k'];
% Load sparse matrices.  Insert self-loop where necessary.
%fileA = '3-4-5-9-16';   fileB = '25-81-256';     fileABd = '3-4-5-9-16-25-81-256';
%fileA = '3-4-5-9-16';   fileB = '25-81';         fileABd = '3-4-5-9-16-25-81';
fileA = '3-4-5-9';      fileB = '16-25';         fileABd = '3-4-5-9-16-25';
%fileA = '3-4-5';        fileB = '9-16';          fileABd = '3-4-5-9-16';

system(['mkdir -p data/' num2str(Pid)]);

fid=fopen(['data/Theory-' fileA '-' Bvar '.tsv'],'r');
  ijvA = fscanf(fid,'%d\t%d\t%d');
fclose(fid);
A = sparse(ijvA(1:3:end),ijvA(2:3:end),ijvA(3:3:end));
fid=fopen(['data/Theory-' fileB '-' Bvar '.tsv'],'r');
  ijvB = fscanf(fid,'%d\t%d\t%d');
fclose(fid);
B = sparse(ijvB(1:3:end),ijvB(2:3:end),ijvB(3:3:end));

if (Btype == '1')
  A(1,1) = 1; 
  B(1,1) = 1;
end
if (Btype == '2')
  A(end,end) = 1;
  B(end,end) = 1;
end

disp(['A size: '  num2str(size(A)) ', A nonzeros: ' num2str(nnz(A))]);
disp(['B size: '  num2str(size(B)) ', B nonzeros: ' num2str(nnz(B))]);
disp(['kron(A,B) size: '  num2str(size(A).*size(B)) ', kron(A,B) nonzeros: ' num2str(nnz(A).*nnz(B))]);


% Get indices from A.
[iA jA vA]  = find(A);

% Distribute A indices amongst processors.
NiA = numel(iA);
%NiA = min(50.*Np,numel(iA));
%NiA = 2000;
iRangeAll = zeros(Np,3);
iRangeAll(:,1) = (0:(Np-1)).';
iRangeAll(:,2) = round(1:(NiA./Np):NiA).';
iRangeAll(:,3) = [iRangeAll(2:end,2)-1; NiA]
iRange = iRangeAll(Pid+1,2:3);

% Get local part of indices.
my_iA = iA(iRange(1):iRange(2));
my_jA = jA(iRange(1):iRange(2));
my_vA = vA(iRange(1):iRange(2));

% Create a local A.
myA = sparse(my_iA,my_jA-min(my_jA)+1,my_vA,max(iA),max(my_jA)-min(my_jA)+1);

% Create a local kron(A,B).
tic;
  myAkB = kron(myA,B);
kronTime = toc;
disp(['kronTime: ' num2str(kronTime)]);
disp(['Edges created/second: ' num2str(Np.*nnz(myAkB)./kronTime)]);

jOffset = (min(my_jA)-1).*size(B,2);

% Remove self-loop.
if (min(my_jA) == 1)
  myAkB(1,1) = 0;
end

if (max(my_jA) == size(A,2))
  myAkB(end,end) = 0;
end

nnzmyAkB = nnz(myAkB);

if SAVETSV
  % Write to tsv file.
  outFile = ['data/' num2str(Pid) '/' fileA '-' Bvar '-x-' fileB '-' Bvar '.' num2str(Pid) '.tsv'];
  disp(outFile);

  ijvAkB = zeros(3,nnz(myAkB));

  % Get indices of non-zero values.
  [ijvAkB(1,:) ijvAkB(2,:) ijvAkB(3,:)] = find(myAkB);

  clear('myAkB');

  % Offset my_jAkB
  ijvAkB(2,:) = ijvAkB(2,:) + jOffset;
    
  tic;
    fid=fopen(outFile,'w');
      fprintf(fid,'%d\t%d\t%d\n',ijvAkB);
    fclose(fid);
  tsvSaveTime = toc;
  disp(['tsvSaveTime: ' num2str(tsvSaveTime)]);
  disp(['Edges saved/second: ' num2str(Np.*nnzmyAkB./tsvSaveTime)]);

end

disp('END-OF-PROGRAM');
