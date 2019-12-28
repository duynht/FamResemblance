function sae = saetrain(sae, x, opts,nOfPatche)
features_AE=[];
all_feat=[];
nOfPatche_2{1,1} = nOfPatche;
nOfPatche_2{1,2} = nOfPatche/2;
nOfPatche_2{1,3} = nOfPatche/4;

for i = 1 : numel(sae.ae);
    disp(['Training AE ' num2str(i) '/' num2str(numel(sae.ae))]);
    sae.ae{i} = nntrain(sae.ae{i}, x, x, opts);
    t = nnff(sae.ae{i}, x, x);
    x = t.a{2};
    %remove bias term
    x = x(:,2:end);
    % keep the features of each layer and make a long feature vector
    features_AE_layer = reconstruct_box(x,nOfPatche_2{1,i}(1),nOfPatche_2{1,i}(2));
    features_AE{1,i}=features_AE_layer;
    all_feat = [all_feat features_AE_layer];
    %maxPooling
    cnt = 1;
    features_next_layer = [];
    if(mod(size(mappedX,1),4)==0)
        tmptmptmp = size(mappedX,1);
    else
        tmptmptmp = size(mappedX,1)-(mod(size(mappedX,1),4));
    end
    for iRow=1:4:tmptmptmp
        features_next_layer(cnt,:) = max(mappedX(iRow:iRow+3,:));
        cnt=cnt+1;
    end
    x = features_next_layer;
end
features_AE{1,4}=all_feat;
end

