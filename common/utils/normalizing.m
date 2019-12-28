function normalized = normalizing(feat,flag)
normalized = zeros(size(feat,1),size(feat,2));
if(flag==1)
    for i=1:size(feat,1)
        if(norm(feat(i,:))~=0)
            normalized(i,:)=feat(i,:)./norm(feat(i,:));
        else
            normalized(i,:)=feat(i,:);
        end
    end
end

if(flag==2)
    for i=1:size(feat,1)
        if(max(feat(i,:))~=0)
            normalized(i,:)=(feat(i,:)-min(feat(i,:)))./max(feat(i,:));
        else
            normalized(i,:)=feat(i,:);
        end
    end
end

if flag == 3  
    mfeat = mean(feat,2);
    normalized = bsxfun(@minus,feat,mfeat);
end

if flag == 4
    mfeat = mean(feat,1);
    sfeat = std(feat,0,1);
    feat = bsxfun(@minus,feat,mfeat);
    ofeat = bsxfun(@rdivide,feat,sfeat);
    
    ind = isnan(ofeat);
    ofeat(ind) = feat(ind);
    
    normalized = ZCAWhitening(ofeat')';
end

end