function [x,y,xlabs,ylabs] = loadAllType2(inPath,type,xmean,xstd,ymean,ystd,U,S)

% U = 0;
% S = 0;
% xmean = 0;
% ymean = 0;
% xstd = 0;
% ystd = 0;

files = dir(fullfile(inPath,[type '_*.mat']));

x = [];
y = [];
xlabs = [];
ylabs = [];
for f = 1 : length(files)
    load(fullfile(inPath,files(f).name));
    
    x = [x; lfeat];
    y = [y; rfeat];
    xlabs = [xlabs; llabs];
    ylabs = [ylabs; rlabs];
end

x = bsxfun(@rdivide,bsxfun(@minus,x,xmean),xstd);
y = bsxfun(@rdivide,bsxfun(@minus,y,ymean),ystd);

x = ZCAWhiteningTest(x',U,S);
y = ZCAWhiteningTest(y',U,S);
x = x';
y = y';