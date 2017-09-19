% Computes the principal component analysis of matrix A.
function [PC,Eigs,Coeff] = myPCA(A)
Cov = A*A';
[PC,V] = eig(Cov);
Eigs = diag(V);
[tmp,idx] = sort(-1*Eigs);
Eigs = Eigs(idx);
PC = PC(:,idx);
Coeff = PC'*A;