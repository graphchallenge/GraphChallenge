%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exact Kronecker Graph Design
% Dr. Jeremy Kepner (MIT)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The goal of this script is to enable the exploration and design of immense
% power-law Kronecker graphs with exactly computable theoretical properties.
%
% This script computes the size, nnz, triangles, k=3 trusses, and
% plots the degree distributions
% for three types of Kronecker graphs built up from star graphs or
% or B(1,m) bipartite graphs.  The three classes of graphs are:
% Base: has no self-loops, many triangles, power-law degree distribution
% 1: center has a self-loop, many triangles, nearly power-law degree distribution
% 2: leaf has a self-loop, some triangles, nearly power-law degree distribution
%
% NOTE: For class 1, the k=3 truss prediction is not complete.
%
% NOTE: Requires D4M library to represent hypersparse distributions
%          https://github.com/Accla/d4m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set ACTUAL = 1 to compute and check theory with actual matrix.
% Only use with smaller graphs.
% Set ACTUAL = 0 when working with large graphs.
ACTUAL = 0;      

% Uncomment *one* set of m for B(1,m) bipartite graphs.
                                    % Type 1 and 2 edge counts:
%m = [3 4 5 9 16 25 81 256 625];    % 2300e12    edge/vert = 331
%m = [  4 5 9 16 25 81 256 625];    %  331e12    edge/vert = 189
%m = [    5 9 16 25 81 256 625];    %   36e12    edge/vert = 105
%m = [      9 16 25 81 256 625];    %  3.3e12    edge/vert = 57.5
%m = [        16 25 81 256 625];    %  176e9     edge/vert = 30.2
%m = [           25 81 256 625];    %    5e9     edge/vert = 15.5
%m = [              81 256 625];    %  104e6     edge/vert = 7.9
%m = [                 256 625];    %  641e3     edge/vert = 4.0
%m = [3 4 5 9 16 25 81 256    ];    %  1.8e12    edge/vert = 165
%m = [3 4 5 9 16 25 81        ];    %  3.6e9     edge/vert = 83
%m = [3 4 5 9 16 25           ];    %   22e6     edge/vert = 41
%m = [3 4 5 9 16              ];    %  434e3     edge/vert = 7.9
m = [3 4 5 9                 ];    %   13e3     edge/vert = 11
%m = [3 4 5                   ];    %    692     edge/vert = 5.8
%m = [3 4                     ];    %     62     edge/vert = 3.1

% Some really big graphs.
%m = [3 4 5 7 11 9 16 25 49 81 121 256 625 2401 14641];    % 2.7e30    edge/vert = 18776
%m = [3 4 5 7 11 9 16 25 49 81 121 256 625 2401];         % 9.2e25    edge/vert = 9388
%m = [3 4 5 9 16 25 81 256 625];    % 2300e12    edge/vert = 331
%m = [3 4 5 9 16 25 81 256    ];    %  1.8e12    edge/vert = 165
%m = [3 4 5 9 16              ];    %  434e3     edge/vert = 22
%m = [           25 81 256    ];    %    4e6     edge/vert = 7.7


disp(['Sub-graph leaf vertices: ' sprintf('%d ',m)]);

% Set the type of k-truss to compute (only works for k=3 at this time).
truss_k = 3;

% NOTE: k-truss relies on incidence matrices E that follow:
% EEk = kron(Ei,Ei) + kron(Ej,Ej);
% kron(A,A)  = (EEk.' * EEk) - diag(diag(EEk.' * EEk));

% Initialize arrays for storing sub-graphs.
B  = {};  Bs  = {};  Bt  = {};  Ei  = {};   Ej  = {};   Btr  = {};
B1 = {};  B1s = {};  B1t = {};  E1i = {};   E1j = {};   B1tr = {};
B2 = {};  B2s = {};  B2t = {};  E2i = {};   E2j = {};   B2tr = {};

% Create subgraphs.
for i=1:length(m)

  % Base type
  Bi = bipartite(1,m(i));                  % Bipartite adjacency matrix.
  B{m(i)} = Bi;
  Bn{m(i)} = size(Bi,1);                   % Size.
  Bs{m(i)} = full(sum(sum(Bi)));           % Sum.
  Bt{m(i)} = full(sum(sum(Bi*Bi.*Bi)));    % 6 x Triangles (zero for bipartite).
  Bd{m(i)} = OutDegree(Bi).';
  [ii jj tmp] = find((Bi));
  Ei{m(i)} = sparse(1:size(ii,1),ii,1,size(ii,1),size(Bi,1));
  Ej{m(i)} = sparse(1:size(jj,1),jj,1,size(jj,1),size(Bi,1));
  Btr{m(i)} = nnz(any(((Ei{m(i)} + Ej{m(i)})*Bi) == 2,2));


  % Type 1
  Bi(1,1) = 1;                             % Set to 1 to create triangles
  B1{m(i)} = Bi;
  B1s{m(i)} = full(sum(sum(Bi)));          % Sum.
  B1t{m(i)} = full(sum(sum(Bi*Bi.*Bi)));   % 6 x Triangles
  B1d{m(i)} = OutDegree(Bi).';
  [ii jj tmp] = find((Bi));
  E1i{m(i)} = sparse(1:size(ii,1),ii,1,size(ii,1),size(Bi,1));
  E1j{m(i)} = sparse(1:size(jj,1),jj,1,size(jj,1),size(Bi,1));
  B1tr{m(i)} = nnz(any(((E1i{m(i)} + E1j{m(i)})*Bi) == 2,2));

  % Type 2
  Bi(1,1) = 0; Bi(end,end) = 1;            % Set to 1 to create triangles
  B2{m(i)} = Bi;
  B2s{m(i)} = full(sum(sum(Bi)));          % Sum.
  B2t{m(i)} = full(sum(sum(Bi*Bi.*Bi)));   % 6 x Triangles
  B2d{m(i)} = OutDegree(Bi).';
  [ii jj tmp] = find((Bi));
  E2i{m(i)} = sparse(1:size(ii,1),ii,1,size(ii,1),size(Bi,1));
  E2j{m(i)} = sparse(1:size(jj,1),jj,1,size(jj,1),size(Bi,1));
  B2tr{m(i)} = nnz(any(((E2i{m(i)} + E2j{m(i)})*Bi) == 2,2));

end

% Initialize variables for computing macroscopic properties via property
% KronProd(A1,...,An)*KronProd(B1,...,Bn) = KronProd(A1*B1,...,An*Bn)
NvertexTheory = Bn{m(1)};
NedgeTheory = Bs{m(1)};
NtriTheory = Bt{m(1)};
[d tmp n] = find(Bd{m(1)});
dTheory = Assoc(sprintf('%d,',d),1,n,@sum);
NtrussTheory = Btr{m(1)};

N1vertexTheory = Bn{m(1)};
N1edgeTheory = B1s{m(1)};
N1triTheory = B1t{m(1)};
[d tmp n] = find(B1d{m(1)});
d1Theory = Assoc(sprintf('%d,',d),1,n,@sum);
N1trussTheory = B1tr{m(1)};


N2vertexTheory = Bn{m(1)};
N2edgeTheory = B2s{m(1)};
N2triTheory = B2t{m(1)};
[d tmp n] = find(B2d{m(1)});
d2Theory = Assoc(sprintf('%d,',d),1,n,@sum);
N2trussTheory = B2tr{m(1)};


% Recursively compute properties of graphs.
for i = 2:numel(m)

   % Iteratively compute macroscopic properties.
   NvertexTheory = NvertexTheory.*Bn{m(i)};
   NedgeTheory = NedgeTheory .* Bs{m(i)};
   NtriTheory = NtriTheory .* Bt{m(i)};
   [d  tmp n ] = find(dTheory);
   [di tmp ni] = find(Bd{m(i)});
   dTheory = Assoc(sprintf('%d,',kron(str2num(d),di)),1,kron(n,ni),@sum);
   NtrussTheory = NtrussTheory .* Btr{m(i)};

   N1vertexTheory = N1vertexTheory.*Bn{m(i)};
   N1edgeTheory = N1edgeTheory .* B1s{m(i)};
   N1triTheory = N1triTheory .* B1t{m(i)};
   [d  tmp n ] = find(d1Theory);
   [di tmp ni] = find(B1d{m(i)});
   d1Theory = Assoc(sprintf('%d,',kron(str2num(d),di)),1,kron(n,ni),@sum);
   N1trussTheory = N1trussTheory .* B1tr{m(i)};

   N2vertexTheory = N2vertexTheory.*Bn{m(i)};
   N2edgeTheory = N2edgeTheory .* B2s{m(i)};
   N2triTheory = N2triTheory .* B2t{m(i)};
   [d  tmp n ] = find(d2Theory);
   [di tmp ni] = find(B2d{m(i)});
   %d2Theory = sparse(kron(d,di),1,kron(n,ni));
   d2Theory = Assoc(sprintf('%d,',kron(str2num(d),di)),1,kron(n,ni),@sum);
   N2trussTheory = N2trussTheory .* B2tr{m(i)};

end

% Remove contribution of B1k(1,1) entry.
N1edgeTheory = N1edgeTheory - 1;
N2edgeTheory = N2edgeTheory - 1;

NtriTheory = NtriTheory/6;

% Remove contribution of B1k(1,1) entry.
N1triTheory = N1triTheory/6 - N1vertexTheory/2 + (1/3);
N2triTheory = N2triTheory/6  - 2^(length(m)-1) + (1/3);

NtrussTheory = NtrussTheory/2;
N1trussTheory = N1trussTheory/2;
N2trussTheory = N2trussTheory/2 - 1.5;


% Put into an associative array for display purposes.
%A = Assoc('Nvertex Nedge Ntri Nedge/Nvertex Ntruss N1vertex N1edge N1tri N1edge/N1vertex N1truss N2vertex N2edge N2tri N2edge/N2vertex N2truss ','Theory ',[NvertexTheory NedgeTheory NtriTheory NedgeTheory/NvertexTheory NtrussTheory N1vertexTheory N1edgeTheory N1triTheory N1edgeTheory/N1vertexTheory N1trussTheory N2vertexTheory N2edgeTheory N2triTheory N2edgeTheory/N2vertexTheory N2trussTheory].');
A = Assoc('Nvertex Nedge Ntri Nedge/Nvertex Ntruss N1vertex N1edge N1tri N1edge/N1vertex N2vertex N2edge N2tri N2edge/N2vertex N2truss ','Theory ',[NvertexTheory NedgeTheory NtriTheory NedgeTheory/NvertexTheory NtrussTheory N1vertexTheory N1edgeTheory N1triTheory N1edgeTheory/N1vertexTheory N2vertexTheory N2edgeTheory N2triTheory N2edgeTheory/N2vertexTheory N2trussTheory].');


% Remove contribution of self-loops to degree distribution.
d1max = max(str2num(Row(d1Theory)));
d1Theory = (d1Theory - d1Theory(sprintf('%d,',d1max),1)) + Assoc(sprintf('%d,',d1max-1),1,1);
d2fix = 2.^length(m);
d2Theory = (d2Theory - Assoc(sprintf('%d,', d2fix),1,1)) + Assoc(sprintf('%d,', d2fix-1),1,1);

figure(1);
[drow temp dval] = find(dTheory);
drow = str2num(drow);
loglog(drow,dval,'bo');hold('on');
loglog([1 max(drow)],[Val(dTheory('1,',1)) Val(dTheory(sprintf('%d,',max(drow)),1))],'-r');
xlabel('vertex degree, d'); ylabel('degree count, n(d)');
title(['Nvertex: ' num2str(NvertexTheory) ', Nedge: ' num2str(NedgeTheory) ', ratio: ' num2str(NedgeTheory/NvertexTheory) ', Ntri: ' num2str(NtriTheory) ]);

figure(2);
[d1row temp d1val] = find(d1Theory);
d1row = str2num(d1row);
loglog(d1row,d1val,'bo');hold('on');
loglog([1 max(d1row)],[Val(d1Theory('1,',1)) Val(d1Theory(sprintf('%d,',max(d1row)),1))],'-r');
xlabel('vertex degree, d'); ylabel('degree count, n(d)');
title(['Nvertex: ' num2str(N1vertexTheory) ', Nedge: ' num2str(N1edgeTheory) ', ratio: ' num2str(N1edgeTheory/N1vertexTheory) ', Ntri: ' num2str(N1triTheory) ]);


figure(3);
[d2row temp d2val] = find(d2Theory);
d2row = str2num(d2row);
loglog(d2row,d2val,'bo');hold('on');
loglog([1 max(d2row)],[Val(d2Theory('1,',1)) Val(d2Theory(sprintf('%d,',max(d2row)),1))],'-r');
xlabel('vertex degree, d'); ylabel('degree count, n(d)');
title(['Nvertex: ' num2str(N2vertexTheory) ', Nedge: ' num2str(N2edgeTheory) ', ratio: ' num2str(N2edgeTheory/N2vertexTheory) ', Ntri: ' num2str(N2triTheory) ]);


% Compare with actual.
if ACTUAL
  Bk = B{m(1)};
  B1k = B1{m(1)};
  B2k = B2{m(1)};
  Eki = Ei{m(1)};    Ekj = Ej{m(1)};
  E1ki = E1i{m(1)};  E1kj = E1j{m(1)};
  E2ki = E2i{m(1)};  E2kj = E2j{m(1)};

  EB2ki = E2i{m(1)}*B2{m(1)};       EB2kj = E2j{m(1)}*B2{m(1)};

  for i = 2:numel(m)
    Bk = kron(Bk,B{m(i)});
    B1k = kron(B1k,B1{m(i)});
    B2k = kron(B2k,B2{m(i)});
    Eki = kron(Eki,Ei{m(i)});       Ekj = kron(Ekj,Ej{m(i)});
    E1ki = kron(E1ki,E1i{m(i)});    E1kj = kron(E1kj,E1j{m(i)});
    E2ki = kron(E2ki,E2i{m(i)});    E2kj = kron(E2kj,E2j{m(i)});

    EB2ki = kron(EB2ki,E2i{m(i)}*B2{m(i)});       EB2kj = kron(EB2kj,E2j{m(i)}*B2{m(i)});

  end

  disp(['kron(B) - kron(E): '  num2str(nnz(Bk  - ((Eki.'*Ekj))))]);
  disp(['kron(B1) - kron(E1): '  num2str(nnz(B1k  - ((E1ki.'*E1kj))))]);
  disp(['kron(B2) - kron(E2): '  num2str(nnz(B2k  - ((E2ki.'*E2kj))))]);

  iB2kLoop = prod((m+1));

  B1k(1,1) = 0;
  B2k(iB2kLoop, iB2kLoop) = 0;

  E1ki = E1ki(2:end,:);      E1kj = E1kj(2:end,:);
  E2ki = E2ki(1:(end-1),:);  E2kj = E2kj(1:(end-1),:);

  disp(['kron(B) - kron(E): '  num2str(nnz(Bk  - ((Eki.'*Ekj))))]);
  disp(['kron(B1) - kron(E1): '  num2str(nnz(B1k  - ((E1ki.'*E1kj))))]);
  disp(['kron(B2) - kron(E2): '  num2str(nnz(B2k  - ((E2ki.'*E2kj))))]);

  % Compute number of vertices and edges.
  % Plot degree distribution.
  NvertexActual = size(Bk,1);
  N1vertexActual = size(B1k,1);
  N2vertexActual = size(B2k,1);
  NedgeActual = nnz(Bk);
  N1edgeActual = nnz(B1k);
  N2edgeActual = nnz(B2k);
  NtriActual = full(sum(sum(tril(Bk)*tril(Bk).*tril(Bk))));
  N1triActual = full(sum(sum(tril(B1k)*tril(B1k).*tril(B1k))));
  N2triActual = full(sum(sum(tril(B2k)*tril(B2k).*tril(B2k))));

  [i j tmp] = find(Bk);
  Ek = sparse(1:size(i,1),i,1,size(i,1),size(Bk,1)) + sparse(1:size(j,1),j,1,size(j,1),size(Bk,1));
  [Ektruss, t] = ktruss(Ek,truss_k);
  NtrussActual = nnz(any(Ektruss,2))/2;

  [i j tmp] = find(B1k);
  E1k = sparse(1:size(i,1),i,1,size(i,1),size(B1k,1)) + sparse(1:size(j,1),j,1,size(j,1),size(B1k,1));
  [E1ktruss, t] = ktruss(E1k,truss_k);
  N1trussActual = nnz(any(E1ktruss,2))/2;


  [i j tmp] = find(B2k);
  E2k = sparse(1:size(i,1),i,1,size(i,1),size(B2k,1)) + sparse(1:size(j,1),j,1,size(j,1),size(B2k,1));
  [E2ktruss, t] = ktruss(E2k,truss_k);
  N2trussActual = nnz(any(E2ktruss,2))/2;

  A = A + Assoc('Nvertex Nedge Ntri Nedge/Nvertex Ntruss N1vertex N1edge N1tri N1edge/N1vertex N1truss N2vertex N2edge N2tri N2edge/N2vertex N2truss ','Actual ',[NvertexActual NedgeActual NtriActual NedgeActual/NvertexActual NtrussActual N1vertexActual N1edgeActual N1triActual N1edgeActual/N1vertexActual N1trussActual  N2vertexActual N2edgeActual N2triActual N2edgeActual/N2vertexActual N2trussActual].');


  figure(1);
  loglog(OutDegree(Bk),'k+');

  figure(2);
  loglog(OutDegree(B1k),'k+');

  figure(3);
  loglog(OutDegree(B2k),'k+');


  [tmp d n] = find(OutDegree(Bk));
  dActual = Assoc(sprintf('%d,',d),1,n,@sum);

  [tmp d n] = find(OutDegree(B1k));
  d1Actual = Assoc(sprintf('%d,',d),1,n,@sum);

  [tmp d n] = find(OutDegree(B2k));
  d2Actual = Assoc(sprintf('%d,',d),1,n,@sum);

  disp(['B  diffs: ' num2str(Val(sum(Abs0(dActual  - dTheory),1)))]);
  disp(['B1 diffs: ' num2str(Val(sum(Abs0(d1Actual - d1Theory),1)))]);
  disp(['B2 diffs: ' num2str(Val(sum(Abs0(d2Actual - d2Theory),1)))]);


%  save('data/Bk.mat','Bk');
%  save('data/B1k.mat','B1k');

end

hold('off')

displayFull(A)






