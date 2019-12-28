function [p,ranks] = computeCMC(scores,labels,ranks)

totRanks = size(labels,2);

% ranks = [1 5 10 100:50:totRanks totRanks];
% ranks = [1 5:5:totRanks totRanks]';

[ss,si] = sort(scores,2,'descend');
% si = labels(si);
for i = 1 : size(si,1)
    si(i,:) = labels(i,si(i,:));
end


p = ones(length(ranks),1);
for r = 1 : length(ranks)
    if ranks(r) > totRanks
        break;
    end
    
    c = sum(sum(si(:,1:ranks(r)),2) > 0);
    p(r) = c / size(labels,1);
end

return