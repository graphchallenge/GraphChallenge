%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parallel Exact Kronecker Graph Generator and Check
% Dr. Jeremy Kepner (MIT)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The goal of this script is to enable the fast, parallel, in-memory generation
% of immense power-law Kronecker graphs with exactly computable theoretical properties.
%
% This script reads in 2 sub-graphs A and B, divides the entries of A
% evenly among difference processors and then performs a local Kronecker
% product of the local sub graph of A with B.
% 
% The degree distributions can be computed and aggregated and plotted if desired.
% The local parts of the generated graph can also be saved in
% in .mat or .tsv file formats.
%
% This script computes and saves the graphs and their degree distributions
% for three types of Kronecker graphs built up
% from star graphs or or B(1,m) bipartite graphs.
% The three classes of graphs are:
% Base: has no self-loops, many triangles, power-law degree distribution
% 1: center has a self-loop, many triangles, nearly power-law degree distribution
% 2: leaf has a self-loop, some triangles, nearly power-law degree distribution
%
% NOTE: Requires D4M library to represent hypersparse distributions
%          https://github.com/Accla/d4m
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

% Flags for computing and plotting the actual degree distribution
% and for saving in .mat and/or .tsv format.
READMAT = 0;        READTSV = 1;
DEGACTUAL = 1;      MAKEPLOT = 0;
SAVEMAT   = 0;      SAVETSV  = 0;

% Flag for setting which type graph to generate.
%Btype = '';      % Base case (no triangles).
%Btype = '1';     % Type 1 (many triangles).
Btype = '2';     % Type 2 (some triangles).

Bvar = ['B' Btype 'k'];
% Load sparse matrices.  Insert self-loop where necessary.
fileA = '3-4-5-9-16';   fileB = '25-81-256';     fileABd = '3-4-5-9-16-25-81-256';
%fileA = '3-4-5-9-16';   fileB = '25-81';         fileABd = '3-4-5-9-16-25-81';
%fileA = '3-4-5-9';      fileB = '16-25';         fileABd = '3-4-5-9-16-25';
%fileA = '3-4-5';        fileB = '9-16';          fileABd = '3-4-5-9-16';

system(['mkdir -p data/' num2str(Pid)]);
if READMAT
  load(['data/Theory-' fileA '-' Bvar '.mat'],Bvar);
  eval(['A = ' Bvar ';']);
  clear(Bvar); 
  load(['data/Theory-' fileB '-' Bvar '.mat'],Bvar);
  eval(['B = ' Bvar ';']);
  clear(Bvar);
end
if READTSV
  fid=fopen(['data/Theory-' fileA '-' Bvar '.tsv'],'r');
    ijvA = fscanf(fid,'%d\t%d\t%d');
  fclose(fid);
  A = sparse(ijvA(1:3:end),ijvA(2:3:end),ijvA(3:3:end));
  fid=fopen(['data/Theory-' fileB '-' Bvar '.tsv'],'r');
    ijvB = fscanf(fid,'%d\t%d\t%d');
  fclose(fid);
  B = sparse(ijvB(1:3:end),ijvB(2:3:end),ijvB(3:3:end));
end

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

% Load degree distribution of combined.
dTheory = str2num(ReadCSV(['data/Theory-' fileABd '-d' Btype '.tsv']));

if (MAKEPLOT && (Pid == 0))
  figure(1);
  [drow temp dval] = find(dTheory);
  drow = str2num(drow);
  loglog(drow,dval,'bo');hold('on');
  loglog([1 max(drow)],[Val(dTheory('1,',1)) Val(dTheory(sprintf('%d,',max(drow)),1))],'-r');
  xlabel('vertex degree, d');
  ylabel('degree count, n(d)');
  %title(['Nvertex: ' num2str(NvertexTheory) ', Nedge: ' num2str(NedgeTheory) ', ratio: ' num2str(NedgeTheory/NvertexTheory) ', Ntri: ' num2str(NtriTheory) ]);
  hold('off');
end


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


if DEGACTUAL
  % Compute degree distribution.
  myAkBsum = sum(myAkB,1);

  % Get the boundary columns.
  jAr =  reshape(jA(iRangeAll(:,2:3)),[Np 2]);
  nB = size(B,2);

  % Loop over each processor.
  for  p = 1:Np
    % Compute the destination to send the lower boundary.
    pDest = min(find(jAr(p,1) == jAr(:,2)));
    if (p ~=  pDest)
      if (Pid == p-1)
        disp(['Sending sums to Pid: ' num2str(pDest-1)]);
        tic;
          SendMsg(pDest-1,'myAkBsum',myAkBsum(1,1:nB));
          myAkBsum(1,1:nB) = 0;
        toc
      end
    end
  end
  for  p = 1:Np
    % Compute the destination to recieve the lower boundary.
    % Replace with probe loop.
    pDest = min(find(jAr(p,1) == jAr(:,2)));
    if (p ~=  pDest)
      if (Pid == pDest-1)
        disp(['Receiving sums from Pid: ' num2str(p-1)]);
        tic;
          myAkBsum(1,(end-nB+1):end) = myAkBsum(1,(end-nB+1):end) + RecvMsg(p-1,'myAkBsum');
        toc
      end
    end
  end

  % Compute the degree distribution of the local sum.
  [temp tmp dActualVal] = find(myAkBsum);
  dActual = Assoc('','','');
  if nnz(dActualVal)
    dActual = Assoc(sprintf('%d,',dActualVal),'n(d) ',1,@sum);
  end

  % Aggregated all the distribution back to Pid = 0.
  disp(['Receiving distributions from all processors.']);
%  tic;
%   dActual = gagg(dActual);
%  toc 
% Replace with probe loop.
  if (Pid > 0)
    SendMsg(0,'dActual',dActual)
  else
    pRecvCount = 0;
    while (pRecvCount < (Np-1))
      [pSource, numeric_tag, string_tag] = ProbeMsg('*','dActual');
      if not(isempty(pSource))
        for p = pSource.'
          disp(['Receiving distributions from Pid: ' num2str(p)]);
          tic;
            dActual =  dActual + RecvMsg(p,'dActual');
          toc
          pRecvCount = pRecvCount + 1;
        end
      end
      pause(0.1)
    end
%    for p=1:1:(Np-1)
%      disp(['Receiving distributions from Pid: ' num2str(p)]);
%      tic;
%        dActual =  dActual + RecvMsg(p,'dActual');
%      toc
%    end
  end

  % Compare results.
  disp(['Predicted - Measured = ', num2str(nnz(abs(dTheory - dActual)))]);

  if (MAKEPLOT && (Pid == 0))
    figure(1);
    hold('on');
    [drow temp dval] = find(dActual);
    drow = str2num(drow);
    loglog(drow,dval,'k+');
    hold('off');
  end

  if SAVEMAT
    % Write to mat file.
    outFile = ['data/' num2str(Pid) '/' fileA '-' Bvar '-x-' fileB '-' Bvar '.' num2str(Pid) '.mat'];
    disp(outFile);
    tic;
      if exist('OCTAVE_VERSION','builtin')
        save(outFile,'myAkB','jOffset');
      else
        save(outFile,'-v7.3','-nocompression','myAkB','jOffset');
      end
    matSaveTime = toc;
    disp(['matSaveTime: ' num2str(matSaveTime)]);
    disp(['Edges saved/second: ' num2str(Np.*nnz(myAkB)./matSaveTime)]);
  end

  if SAVETSV
    % Write to tsv file.
    outFile = ['data/' num2str(Pid) '/' fileA '-' Bvar '-x-' fileB '-' Bvar '.' num2str(Pid) '.tsv'];
    disp(outFile);

    ijvAkB = zeros(3,nnz(myAkB));

    % Get indices of non-zero values.
    [ijvAkB(1,:) ijvAkB(2,:) ijvAkB(3,:)] = find(myAkB);

    % Offset my_jAkB
    ijvAkB(2,:) = ijvAkB(2,:) + jOffset;
    
    tic;
      fid=fopen(outFile,'w');
        fprintf(fid,'%d\t%d\t%d\n',ijvAkB);
      fclose(fid);
    tsvSaveTime = toc;
    disp(['tsvSaveTime: ' num2str(tsvSaveTime)]);
    disp(['Edges saved/second: ' num2str(Np.*nnz(myAkB)./tsvSaveTime)]);

  end

end

disp('END-OF-PROGRAM');
