function [D, ni] = find_nn(X, k)
 


	if ~exist('k', 'var') || isempty(k)
		k = 12;
    end
    
    % Perform adaptive neighborhood selection if desired
    if ischar(k)
        [D, max_k] = find_nn_adaptive(X);
        ni = zeros(size(X, 1), max_k);
        for i=1:size(X, 1)
            tmp = find(D(i,:) ~= 0);
            tmp(tmp == i) = [];
            tmp = [tmp(2:end) zeros(1, max_k - length(tmp) + 1)];
            ni(i,:) = tmp;
        end
    
    % Perform normal neighborhood selection
    else
        
        % Compute distances in batches
        n = size(X, 1);
        sum_X = sum(X .^ 2, 2);
        batch_size = round(2e7 ./ n);
        D = zeros(n, k);
        ni = zeros(n, k);
        for i=1:batch_size:n
            batch_ind = i:min(i + batch_size - 1, n);
            DD = bsxfun(@plus, sum_X', bsxfun(@plus, sum_X(batch_ind), ...
                                                   -2 * (X(batch_ind,:) * X')));
            [DD, ind] = sort(DD, 2, 'ascend');
            D(batch_ind,:) = sqrt(DD(:,2:k + 1));
            ni(batch_ind,:) = ind(:,2:k + 1);
        end
        D(D == 0) = 1e-9;
        D = sparse(repmat(1:n, [1 k])', ni(:), D(:), n, n);
    end
    