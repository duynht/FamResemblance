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
% trainModels: train models on Fam101 dataset.
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
numEpoc = 10;
verbose = 1;
epochs_disc = 50;
iActivation = 1;
outPut_activation = {'softmax'};
ilearning = 0.4;

ilearning = 0.4;

for f = 1 : length(files)
    inFile = files{f};
    outFile = [inFile(1:end-4) '_f_' num2str(facDim) '_m_' num2str(mapDim)];
    if doDisc
        outFile = [outFile '_disc_bal_map.mat'];
    else
        outFile = [outFile '_bal_map.mat'];
    end

    % Load Data
    load(fullfile(dataPath,inFile));
    trnDataX = x;
    trnDataY = y;
    
    % Negative Data
    ind = 1 : batchSze : size(x,1);
    ntrnind = zeros(length(ind),1);
    for i = 1 : length(ind)
        v = randperm(length(ind));
        for j = 1 : length(v)
            if v(j) ~= ind(i) && labs(ind(j)) ~= labs(ind(i)); v = v(j); break; end
        end
        ntrnind(i) = v;
    end

    ntrnDataY = zeros(batchSze*length(ntrnind),size(y,2));
    for i = 1 : length(ntrnind)
        ntrnDataY((i-1)*batchSze+1:i*batchSze,:) = y(ntrnind(i):ntrnind(i)+batchSze-1,:);
    end
    ntrnDataX = x;
    
    y = ntrnDataY;
    ntrnFile = fullfile(resDir,[inFile(1:end-4) '_ntrn.mat']);
    save(ntrnFile,'x','y');
    
    % Perform GAE
    x = trnDataX;
    y = trnDataY;
    
    trnFile = fullfile(resDir,[inFile(1:end-4) '_trn.mat']);
    mapFile = fullfile(resDir,[inFile(1:end-4) '_trn_map.mat']);
    save(trnFile,'x','y');

    % Do Generative
    disp('Learning Feature Metric');
    cmd = sprintf('python ../common/methods/gae/gae_on_fam.py -i %s -o %s -f %d -m %d -l %f -e %d -n 1 -v %d', trnFile, mapFile, facDim, mapDim, learnRate, numEpoc, verbose);
    system(cmd);

    disp('Projecting Data');
    % Load Mapping
    load(mapFile);

    % Do Discriminative
    disp('Training Discriminative Part');
    HAELabels = [ones(size(trnDataX,1),1),zeros(size(trnDataX,1),1); ...
        zeros(size(ntrnDataX,1),1),ones(size(ntrnDataX,1),1)];
    x = [trnDataX;ntrnDataX];
    y = [trnDataY;ntrnDataY];

    [~,whf_disc,z_bias,~,filter_out_disc,nn] = HAE_SingleLayer(((wxf'*x') .* (wyf'*y'))', ...
        HAELabels, whf', z_bias', ilearning, facDim, mapDim, epochs_disc, verbose,outPut_activation{1,iActivation});
    
    save(mapFile,'wxf','wyf','whf','z_bias','whf_disc','filter_out_disc','nn');
end
