function W = NRML(x,y,k,dim,numIt)

% Step 1: Compute KNN
nX = knnsearch(x',x','K',k+1);
nY = knnsearch(y',y','K',k+1);
nX = nX(:,2:end);
nY = nY(:,2:end);

err = Inf;
tol = 0.01;
r = 1;
% Step 2: Local Optimization
while r <= numIt && err > tol
    fprintf('Iteration %03d / %03d : ',r,numIt);
    % 2.1 Compute H1, H2, and H3
    [H1,H2,H3] = computeH(x,y,nX,nY,k);

    % 2.2 Solve Eigenvalue problem and 2.3 Obtain W^r
    H = H1+H2-H3;
    [W,~] = eigs(H,dim);

    % 2.4 Update KNN
    nX = knnsearch((W'*x)',(W'*x)','K',k+1);
    nY = knnsearch((W'*y)',(W'*y)','K',k+1);
%     nX = knnsearch(x',x','K',k+1,'distance','mahalanobis');
%     nY = knnsearch(y',y','K',k+1,'distance','mahalanobis');
    nX = nX(:,2:end);
    nY = nY(:,2:end);
    
    % 2.5 Computer error tolerance
    if r > 2
        err = norm(W-prevW);
    end
    
    fprintf('Error : %f\n',err);
    
    prevW = W;
    r = r + 1;
end

return 