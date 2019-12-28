function W = MNRML(X,Y,k,dim,numIt,q)

% Step 1: Compute KNN
nX = cell(length(X),1);
nY = cell(length(X),1);
for p = 1 : length(X)
    nX{p} = knnsearch(X{p}',X{p}','K',k+1);
    nY{p} = knnsearch(Y{p}',Y{p}','K',k+1);
    nX{p} = nX{p}(:,2:end);
    nY{p} = nY{p}(:,2:end);
end

err = Inf;
tol = 0.01;
r = 1;
beta = ones(length(X),1)./length(X);
% Step 2: Local Optimization
while r <= numIt && err > tol
    fprintf('Iteration %03d / %03d : ',r,numIt);
    
    % 2.1 Compute H1, H2, and H3
    H1 = cell(length(X),1);
    H2 = cell(length(X),1);
    H3 = cell(length(X),1);
    H = zeros(size(X{1},1));
    for p = 1 : length(X)
        [H1{p},H2{p},H3{p}] = computeH(X{p},Y{p},nX{p},nY{p},k);
        
        H = H + beta(p) .* (H1{p}+H2{p}-H3{p});
    end
    
    % 2.2 Solve eigenvalue problem
    [W,~] = eigs(H,dim);
    
    % 2.4 Update KNN and Beta
    den = 0;
    for p = 1 : length(X)
        nX{p} = knnsearch((W'*X{p})',(W'*X{p})','K',k+1);
        nY{p} = knnsearch((W'*Y{p})',(W'*Y{p})','K',k+1);
        nX{p} = nX{p}(:,2:end);
        nY{p} = nY{p}(:,2:end);
        
        beta(p) = (1/sum(diag(W'*(H1{p}+H2{p}-H3{p})*W)))^(1/(q-1));
        den = den + (1/sum(diag(W'*(H1{p}+H2{p}-H3{p})*W)))^(1/(q-1));       
    end
    beta = beta ./ den;
    
    % 2.5 Computer error tolerance
    if r > 2
        err = norm(W-prevW);
    end
    
    fprintf('Error : %f\n',err);
    
    prevW = W;
    r = r + 1;
end

return