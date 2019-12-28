function [tpr,fpr] = computeROC(conf,gtLabs,t)

tpr = zeros(length(t),1);
fpr = zeros(length(t),1);

for i = 1 : length(t)
    ind = conf >= t(i);
    predLabs = ones(length(gtLabs),1);
    predLabs(~ind) = -1;
    
    tp = sum(predLabs(ind) == gtLabs(ind));
    fp = sum(predLabs(ind) ~= gtLabs(ind));
    pos = sum(gtLabs == 1);
    neg = sum(gtLabs == -1);
    
    fpr(i) = fp / neg;
    tpr(i) = tp / pos;
end

return