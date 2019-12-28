function [scores,correspondence,labelchild,labelparent] = getScoresTable(data,nn,batchSze)

    scores = zeros(length(data.xlabs),length(data.ylabs));
    correspondence = zeros(length(data.xlabs),length(data.ylabs));
    labelparent = data.xlabs';
    labelchild = data.ylabs;
    
    %for each parent
    for i = 1 : length(data.xlabs)
        %for each offspring
        for j = 1 : length(data.ylabs)
            %get the multiplicative relationship
            tstDataX = data.x((i-1)*batchSze+1:i*batchSze,:); % Parent
            tstDataY = data.y((j-1)*batchSze+1:j*batchSze,:); % Offspring
                        
            data_H = ((data.wxf'*tstDataX').*(data.wyf'*tstDataY'))';
            
            %store scores
            scores(i,j) = mean(nnpredict_mine(nn, data_H));
            
            %set correspondances
            if(data.ylabs(j) == data.xlabs(i))
                correspondence(i,j) = 1;
            end
        end
    end
    
    scores = scores';
    correspondence = correspondence';
end