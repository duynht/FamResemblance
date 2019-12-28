function [H1,H2,H3] = computeH(x,y,nX,nY,k)

H1 = zeros(size(x,1));
for i = 1 : size(x,2)
    dif = repmat(x(:,i),1,k) - y(:,nY(i,:));
    H1 = H1 + dif*dif';
end
H1 = H1 ./ (size(x,2)*k);

H2 = zeros(size(y,1));
for i = 1 : size(y,2)
    dif = x(:,nX(i,:)) - repmat(y(:,i),1,k);
    H2 = H2 + dif*dif';
end
H2 = H2 ./ (size(y,2)*k);

dif = x - y;
H3 = (dif * dif') ./ size(x,2);

return