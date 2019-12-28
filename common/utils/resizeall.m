function [output] = resizeall(input)
    numIm = length(input);
    trimrows = 14;
    trimcols = 10;
    imdim = 64;
    
    output = cell([numIm 1]);
    
    for i = 1 : numIm
        output = imresize(input(trimrows:end-(trimrows+1),trimcols:end-(trimcols+1),:),[imdim,imdim]);
    end
end