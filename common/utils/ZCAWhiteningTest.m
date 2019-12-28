function [output] = ZCAWhiteningTest(patches,U,S)
% Input column vectors
    xRot = U' * patches;
    
    %Whiten and regularize
    epsilon = 0.00001;
    xPCAWhite = diag(1 ./ sqrt(diag(S) + epsilon)) * xRot;
    output = U * xPCAWhite;
end