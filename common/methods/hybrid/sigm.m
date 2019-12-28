% simple wrapper for sigmoid function
function [y] = sigm(x)

    y = 1./(1 + exp (-x));