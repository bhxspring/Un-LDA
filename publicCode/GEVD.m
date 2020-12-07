function model = GEVD(A,B)
[W,D] = eig(A,B);
v = diag(D);
[v,ind] = sort(v);
D = diag(v);
W = W(:,ind);
model.W = W;
model.D = D;
end
