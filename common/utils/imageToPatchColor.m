function patches = imageToPatchColor(image,patch_size,flag_minMax)

if mod(size(image,1),patch_size)~= 0 || mod(size(image,2),patch_size)~= 0
    error('Patch size does not match dimensions of image.');
end

% get the number of patches in each row
nOfPatche(1) = size(image,1)/patch_size;
nOfPatche(2) = size(image,2)/patch_size;
patches = zeros(nOfPatche(1)*nOfPatche(2),patch_size*patch_size*3);
cntRow = 1;
for ii=1:2:nOfPatche(1)-1
    for jj=1:2:nOfPatche(2)-1
        patch_image = image((ii-1)*patch_size+1:ii*patch_size,(jj-1)*patch_size+1:jj*patch_size,:);
        patch_image = double(patch_image);
        tmp = patch_image(:)';
        if(max(tmp)~=0)
            if flag_minMax == 1
                patches(cntRow,:) = tmp/norm(tmp);
            elseif flag_minMax == 0
                patches(cntRow,:) = (tmp-min(tmp))/max(tmp);
            elseif flag_minMax == 3
                tmp = tmp - mean(tmp(:));
%                 tmp = ContrastStretchNorm(tmp);
                patches(cntRow,:) = tmp;
            else
                patches(cntRow,:) = tmp;
            end
        else
            patches(cntRow,:) = tmp;
        end
        cntRow = cntRow+1;
        patch_image = image((ii-1)*patch_size+1:ii*patch_size,(jj)*patch_size+1:(jj+1)*patch_size,:);
        patch_image = double(patch_image);
        tmp = patch_image(:)';
        if(max(tmp)~=0)
            if flag_minMax == 1
                patches(cntRow,:) = tmp/norm(tmp);
            elseif flag_minMax == 0
                patches(cntRow,:) = (tmp-min(tmp))/max(tmp);
            elseif flag_minMax == 3
                tmp = tmp - mean(tmp(:));
%                 tmp = ContrastStretchNorm(tmp);
                patches(cntRow,:) = tmp;
            else
                patches(cntRow,:) = tmp;
            end
        else
            patches(cntRow,:) = tmp;
        end
        cntRow = cntRow+1;
        patch_image = image((ii)*patch_size+1:(ii+1)*patch_size,(jj-1)*patch_size+1:jj*patch_size,:);
        patch_image = double(patch_image);
        tmp = patch_image(:)';
        if(max(tmp)~=0)
            if flag_minMax == 1
                patches(cntRow,:) = tmp/norm(tmp);
            elseif flag_minMax == 0
                patches(cntRow,:) = (tmp-min(tmp))/max(tmp);
            elseif flag_minMax == 3
                tmp = tmp - mean(tmp(:));
%                 tmp = ContrastStretchNorm(tmp);
                patches(cntRow,:) = tmp;
            else
                patches(cntRow,:) = tmp;
            end
        else
            patches(cntRow,:) = tmp;
        end
        cntRow = cntRow+1;
        patch_image = image((ii)*patch_size+1:(ii+1)*patch_size,(jj)*patch_size+1:(jj+1)*patch_size,:);
        patch_image = double(patch_image);
        tmp = patch_image(:)';
        if(max(tmp)~=0)
            if flag_minMax == 1
                patches(cntRow,:) = tmp/norm(tmp);
            elseif flag_minMax == 0
                patches(cntRow,:) = (tmp-min(tmp))/max(tmp);
            elseif flag_minMax == 3
                tmp = tmp - mean(tmp(:));
%                 tmp = ContrastStretchNorm(tmp);
                patches(cntRow,:) = tmp;
            else
                patches(cntRow,:) = tmp;
            end
        else
            patches(cntRow,:) = tmp;
        end
        
        cntRow = cntRow+1;
    end
end
end