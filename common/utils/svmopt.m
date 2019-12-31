
function [error] = svmopt(params, trainIds, trainImgs, testIds, testImgs)

trainParams = sprintf('-c %d -g %f -q', params(1), params(2));
model = svmtrain(trainIds, trainImgs, trainParams);

[predLabs, tempAcc, decVals] = svmpredict(testIds, testImgs, model, '-q');

% fprintf('-c %d -g %f : acc : %0.04f\n', params(1), params(2),acc);

% fminsearch MINIMIZES the returned value. So instead of returning the
% percentage correct, we want to return the percentage wrong
error = 100-tempAcc(1);

return