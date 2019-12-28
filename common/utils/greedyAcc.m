function predLabs = greedyAcc(conf,tstLabs,step)

th = min(conf) : step : max(conf);

acc = zeros(length(th),1);
for t = 1 : length(th)
    predLabs = ones(size(tstLabs));
    predLabs(conf < th(t)) = -1;
    
    acc(t) = sum(predLabs == tstLabs) / length(tstLabs);
end

[val,ind] = max(acc);
predLabs = ones(size(tstLabs));
predLabs(conf < th(ind)) = -1;

return