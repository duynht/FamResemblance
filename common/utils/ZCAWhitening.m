function [output,U,S] = ZCAWhitening(patches)
% Input column vectors
    %PCA
    [U,S] = svd(patches,'econ');
    xRot = U' * patches;
    
    %Whiten and regularize
    epsilon = 0.00001;
    xPCAWhite = diag(1 ./ sqrt(diag(S) + epsilon)) * xRot;
    output = U * xPCAWhite;
end