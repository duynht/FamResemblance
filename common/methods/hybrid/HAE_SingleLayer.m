function [out,filter_out,z_bias,score_patches,filter_out_disc,nn] = HAE_SingleLayer(train_x,train_y,whf,z_bias,learningrate,facDim,mapDim,epochs,verbose,outPutActivation)


%% Get the batchsize
opts.batchsize = 100;
sizeSize = size(train_x,1) - mod(size(train_x,1),opts.batchsize);


%% initialize a forward NN
nn = nnsetup([facDim mapDim 2]);            %D by H by lables
nn.activation_function              = 'sigm';
nn.output             = outPutActivation;
nn.learningRate                     = learningrate; % Select a number between zero to one / Usually 0.5 to 0.8
nn.W{1} = [whf;z_bias]'; % initialize it using the weight from GAE/ it need both the 

%% get the final W
opts.numepochs =   epochs;
opts.verbose = verbose;
sizeSize = size(train_x,1) - mod(size(train_x,1),opts.batchsize);
nn = nntrain(nn, train_x(1:sizeSize,:), train_y(1:sizeSize,:), opts);

%% If it is the last layer save the discriminative weights too
filter_out = nn.W{1,1}(:,1:end-1);
z_bias = nn.W{1,1}(:,end-1,end);
filter_out_disc = nn.W{1,2}(:,1:end-1)';

%% Apply the final Z matrix and get the final features 
out = sigm([ones(size(train_x,1),1) train_x]*((nn.W{1,1})'))';

%% Get the scores for each pair
score_patches = nnpredict_mine(nn, train_x);

end

