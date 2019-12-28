% Column vectors
function z = distSqr(x,y)

[d,m] = size(y);
z = x'*y;
x2 = sum(x.^2)';
y2 = sum(y.^2);
for i = 1:m,
  z(:,i) = x2 + y2(i) - 2*z(:,i);
end

% Rounding errors occasionally cause negative entries in n2
% if any(any(n2<0))
%   n2(n2<0) = 0;
% end