%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Kinship Recognition Toolbox
%       Copyright (C) Jun 2014 Center for Research in Computer Vision
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This toolbox was created to foster research in kinship recognition. If you
% use any of the algorithms or datasets please cite the relevant literature:
%
%  1)  A. Dehghan, E.G. Ortiz, R. Villegas, and M. Shah. "Who Do I Look Like?
%  Determining Parent-Offspring Resemblance via Genetic Features." IEEE CVPR
%  2013.
%
%  2) R. Memisevic. "Learning to relate images." IEEE TPAMI, 2013.
%
%  3) J. Lu, X. Zhou, Y.-P. Tan, Y. Shang, and J. Zhou. "Neighborhood
%  Repulsed Metric Learning for Kinship Verification." IEEE TPAMI, 2013.
%
%  4) R. Fang, A. C. Gallagher, T. Chen, and A. Loui. "Kinship
%  Classification by Modeling Facial Feature Heredity." ICIP, 2013.
%
% This toolbox performs both kinship verification (KinFace) and
% identification (Fam101).
%
% testModels: Test the trained models for Fam101.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For more information, see http://enriquegortiz.com/fbfaces.
%
% Contact: Enrique G. Ortiz (eortiz@cs.ucf.edu)
%          Afshin Dehghan (adehghan@cs.ucf.edu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist(resDir,'dir'); mkdir(resDir); end

doDisc = 0;
numReps = 5;
facDim = 40;
mapDim = 10;
facDim = 160;
mapDim = 40;
learnRate = 0.001;    % Best Rate for Parts and RGB Patches
% learnRate = 0.01;     % Best Rate for Patches from Entire Image
numEpoc = 10000;
verbose = 1;
epochs_disc = 50;
iActivation = 1;
outPut_activation = {'softmax'};
ilearning = 0.4;

mc = zeros(length(files),1);
sc = zeros(length(files),1);

for f = 1 : length(files)
%for f = modelNum:modelNum
    inFile = files{f};
    outFile = [inFile(1:end-4) '_f_' num2str(facDim) '_m_' num2str(mapDim)];
    
    disp(inFile);
    
    tstFile = fullfile(dataPath,files2{f});
    trnFile = fullfile(resDir,[inFile(1:end-4) '_trn.mat']);
    ntrnFile = fullfile(resDir,[inFile(1:end-4) '_ntrn.mat']);
    mapFile = fullfile(resDir,[inFile(1:end-4) '_trn_map.mat']);
    tabFile = fullfile(resDir,[inFile(1:end-4) '_tab.mat']);
    
    load(mapFile);
    
    if ~exist('whf_disc','var')
        % Do Discriminative
        disp('Training Discriminative Part');
        load(trnFile);
        trnDataX = x;
        trnDataY = y;
        
        load(ntrnFile);
        ntrnDataX = x;
        ntrnDataY = y;
        
        HAELabels = [ones(size(trnDataX,1),1),zeros(size(ntrnDataX,1),1); ...
            zeros(size(ntrnDataX,1),1),ones(size(ntrnDataX,1),1)];
        x = [trnDataX;ntrnDataX];
        y = [trnDataY;ntrnDataY];

        [~,whf_disc,z_bias,~,filter_out_disc,nn] = HAE_SingleLayer(((wxf'*x') .* (wyf'*y'))', ...
            HAELabels, whf', z_bias', ilearning, facDim, mapDim, epochs_disc, verbose,outPut_activation{1,iActivation});
    end
    
    % Load Test Data
    load(tstFile);
    data.x = x;
    data.y = y;
    data.xlabs = xlabs(1:batchSze:size(x,1));
    data.ylabs = ylabs(1:batchSze:size(y,1));
    
    data.wxf = wxf;
    data.wyf = wyf;
    data.whf = whf_disc;
    data.z_bias = z_bias;
    
    % Apply the projection
    [scoretable,correspondence,labelchild,labelparent] = getScoresTable(data,nn,batchSze);
    
    ulabs = unique(labelparent);
    imcount = zeros(length(ulabs),1);
    for u = 1 : length(ulabs)
        curLab = ulabs(u);
        imcount(u) = sum(labelparent == curLab);
    end
    mc(f) = mean(imcount);
    sc(f) = sum(imcount);
    
    save(tabFile,'scoretable','correspondence','labelchild','labelparent');
    
    clear data whf wxf wyf whf_disc z_bias x y xlabs ylabs scoretable correspondence labelchild labelparent;
end

