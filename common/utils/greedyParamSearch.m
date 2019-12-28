function [bestParams,bestAcc] = greedyParamSearch(trainIds, trainImgs, testIds, testImgs)


% cV = [2^-5,2^-3,2^-1,2^0,2^1,2^3,2^5,2^7,2^9,2^11,2^13,2^15];
cV = 1;
% cV = 1 : 0.001 : 50;
gV = [2^-15,2^-13,2^-11,2^-9,2^-7,2^-5,2^-3,2^-1,2^0,2^1,2^3];
% gV = [2^-15,2^-13,2^-11,2^-9,2^-7,2^-5,2^-3,2^-1,2^0,2^1,2^3,2^5,2^7,2^9,2^11,2^13,2^15];
% gV = 0 : 0.001 : 30;

pRes = zeros(length(cV),length(gV));

for c = 1 : length(cV)
    for g = 1 : length(gV)
        params = [cV(c) gV(g)];
%         pRes(c,g) = svmopt(params,[trnLabs(tind(1:t));ntrnLabs(tind(1:t))], [trnH(tind(1:t),:);ntrnH(tind(1:t),:)], [trnLabs(tind(t+1:end));ntrnLabs(tind(t+1:end))], [trnH(tind(t+1:end),:);ntrnH(tind(t+1:end),:)]);
        pRes(c,g) = svmopt(params,trainIds,trainImgs,testIds,testImgs);
    end
end

if length(cV) > 1
    [v,ic] = min(pRes,[],1);
    [v,iv] = min(v);
    bestParams = [cV(ic(iv)), gV(iv)];
else
    [v,iv] = min(pRes);
    bestParams = [cV(1) gV(iv)];
end


bestAcc = 100-v;