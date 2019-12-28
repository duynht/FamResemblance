function [x,y,labs,xmean,xstd,ymean,ystd,U,S] = loadAllType(inPath,type)

files = dir(fullfile(inPath,[type '_*.mat']));

x = [];
y = [];
labs = [];
for f = 1 : length(files)
    load(fullfile(inPath,files(f).name));
    
    x = [x; lfeat];
    y = [y; rfeat];
    labs = [labs; tlabs];
end

xmean = mean(x,1);
xstd = std(x,0,1);
x = bsxfun(@rdivide,bsxfun(@minus,x,xmean),xstd);

ymean = mean(y,1);
ystd = std(y,0,1);
y = bsxfun(@rdivide,bsxfun(@minus,y,ymean),ystd);

[x,U,S] = ZCAWhitening(x');
[y,U,S] = ZCAWhitening(y');
x = x';
y = y';