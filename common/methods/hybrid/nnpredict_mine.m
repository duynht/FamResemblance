function labels = nnpredict_mine(nn, x)
    nn.testing = 1;
    nn = nnff_mine(nn, x);
    nn.testing = 0;
    labels = nn.a{end}(:,1);
end
