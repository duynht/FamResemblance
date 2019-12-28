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
% scriptKinGAE: Performs experiments for KinFace dataset using 5-fold
% cross-validation using Gated Autoencoders Method.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For more information, see http://enriquegortiz.com/fbfaces.
%
% Contact: Enrique G. Ortiz (eortiz@cs.ucf.edu)
%          Afshin Dehghan (adehghan@cs.ucf.edu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;
addpath(genpath('../common'));

dataPath = 'data';
resDir = 'results';

if ~exist(resDir,'dir'); mkdir(resDir); end



numReps = 2;        % Number of reptitions to perform experiment
doDisc = 0;         % False - Only perform Generative Layer using GAE
                    % True - Run Additional Discriminative Layer


% Don't need to modify these
doBalance = 1;      % Balance Training/Testing Data
doLinear = 1;       % Do linear or RBF SVM
splitNum = 4;       % Number of splits to analyze
multiview = 0;

% Generative Parameters
facDim = 160;           % Number of Filters
mapDim = 40;            % Number of Mapping Unites
learnRate = 0.01;       % GAE Learning Rate
numEpoc = 1000;         % Maximum Number of Iterations for GAE
verbose = 1;            % Vebose Output

% Discriminative Parameters
learnRateDisc = [0.4];              % Discriminative Learning Rate
outPut_activation = {'softmax'};    % Discrimative Activation Function
epochs_disc = 50;                   % Number Iterations for Disc. Layer


% Datasets to Process
files = {'kinfwi_fs_iasp_rgb_white_p8.mat','kinfwi_fd_iasp_rgb_white_p8.mat','kinfwi_ms_iasp_rgb_white_p8.mat','kinfwi_md_iasp_rgb_white_p8.mat', ...
         'kinfwii_fs_iasp_rgb_white_p8.mat','kinfwii_fd_iasp_rgb_white_p8.mat','kinfwii_ms_iasp_rgb_white_p8.mat','kinfwii_md_iasp_rgb_white_p8.mat'};

for ilearning = learnRateDisc
    for iActivation = 1 : length(outPut_activation)
        for f = 1 : length(files)
            inFile = files{f};
            outFile = [inFile(1:end-4) '_f_' num2str(facDim) '_m_' num2str(mapDim)];
            if doBalance
                if doDisc
                    outFile = [outFile '_disc_bal_map_ruben.mat'];
                else
                    outFile = [outFile '_bal_map_ruben.mat'];
                end
            else
                if doDisc
                    outFile = [outFile '_disc_map_ruben.mat'];
                else
                    outFile = [outFile '_map_ruben.mat'];
                end
            end

            fprintf('%s\n',inFile);
            
            for rep = 1 : numReps
                
                fprintf('Rep %03d / %03d\n', rep, numReps);
                
                load(fullfile(dataPath,inFile));
                
                ind = 1 : batchSze : size(x,1);
                split = regexp(inFile,'_','split');
                split = split{2};
                switch split
                    case 'fs'
                        ib = [1,32,65,97,125];
                    case 'fd'
                        ib = [1,28,55,82,109];
                    case 'ms'
                        ib = [1,24,47,70,93];
                    case 'md'
                        ib = [1,26,51,76,102];
                end
                
                
                acc = zeros(splitNum,1);
                pacc = acc;
                nacc = acc;
                ap = acc;
                tpr = [];
                fpr = [];
                for s = 1 : splitNum
                    fprintf('Split %02d / %02d\n', s, splitNum);
                    disp('Loading Data');
                    if s ~= splitNum
                        tstind = ind(ib(s):ib(s+1)-1);
                        trnind = ind;
                        trnind(ib(s):ib(s+1)-1) = [];
                    else
                        tstind = ind(ib(s):end);
                        trnind = ind;
                        trnind(ib(s):end) = [];
                    end
                    
                    if doBalance
                        ntrnind = randperm(length(trnind));
                        ntrnind = trnind(ntrnind);
                        while sum(trnind == ntrnind) > 0
                            ntrnind = randperm(length(trnind));
                            ntrnind = trnind(ntrnind);
                        end
                        
                        ntstind = randperm(length(tstind));
                        ntstind = tstind(ntstind);
                        while sum(tstind == ntstind) > 0
                            ntstind = randperm(length(tstind));
                            ntstind = tstind(ntstind);
                        end
                    end
                    
                    load(fullfile(dataPath,inFile));
                    
                    trnDataX = zeros(batchSze*length(trnind),size(x,2));
                    trnDataY = zeros(batchSze*length(trnind),size(y,2));
                    if doBalance
                        ntrnDataY = zeros(batchSze*length(trnind),size(y,2));
                    else
                        ntrnDataX = [];
                        ntrnDataY = [];
                    end
                    
                    for i = 1 : length(trnind)
                        trnDataX((i-1)*batchSze+1:i*batchSze,:) = x(trnind(i):trnind(i)+batchSze-1,:);
                        trnDataY((i-1)*batchSze+1:i*batchSze,:) = y(trnind(i):trnind(i)+batchSze-1,:);
                        
                        if doBalance
                            ntrnDataY((i-1)*batchSze+1:i*batchSze,:) = y(ntrnind(i):ntrnind(i)+batchSze-1,:);
                        else
                            tind = trnind;
                            tind(i) = [];
                            rind = randperm(length(tind));
                            t = tind(rind(1));
                            ntrnDataY = [ntrnDataY; y(t:t+batchSze-1,:)];
                            
                            t = tind(rind(2));
                            ntrnDataY = [ntrnDataY; y(t:t+batchSze-1,:)];
                            ntrnDataX = [ntrnDataX; repmat(x(trnind(i):trnind(i)+batchSze-1,:),2,1)];
                        end
                    end
                    
                    if doBalance
                        ntrnDataX = trnDataX;
                    end
                    
                    tstDataX = zeros(batchSze*length(tstind),size(x,2));
                    tstDataY = zeros(batchSze*length(tstind),size(y,2));
                    if doBalance
                        ntstDataY = zeros(batchSze*length(tstind),size(y,2));
                    else
                        ntstDataX = [];
                        ntstDataY = [];
                    end
                    
                    for i = 1 : length(tstind)
                        tstDataX((i-1)*batchSze+1:i*batchSze,:) = x(tstind(i):tstind(i)+batchSze-1,:);
                        tstDataY((i-1)*batchSze+1:i*batchSze,:) = y(tstind(i):tstind(i)+batchSze-1,:);
                        
                        if doBalance
                            ntstDataY((i-1)*batchSze+1:i*batchSze,:) = y(ntstind(i):ntstind(i)+batchSze-1,:);
                        else
                            ttstind = tstind;
                            ttstind(i) = [];
                            tmpneg = zeros(batchSze*length(ttstind),size(y,2));
                            for j = 1 : length(ttstind)
                                tmpneg((j-1)*batchSze+1:j*batchSze,:) = y(ttstind(j):ttstind(j)+batchSze-1,:);
                            end
                            ntstDataY = [ntstDataY; tmpneg];
                            ntstDataX = [ntstDataX; repmat(x(tstind(i):tstind(i)+batchSze-1,:),length(ttstind),1)];
                        end
                    end
                    
                    if doBalance
                        ntstDataX = tstDataX;
                    end
                    
                    x = trnDataX;
                    y = trnDataY;
                    
                    trnFile = fullfile(resDir,[inFile(1:end-4) '_trn_p8.mat']);
                    mapFile = fullfile(resDir,[inFile(1:end-4) '_trn_map_p8.mat']);
                    save(trnFile,'x','y');
                    
                    disp('Learning Feature Metric');
                    if multiview
                        cmd = sprintf('python ../common/methods/gae/multiview_fam.py -i %s -o %s -f %d -m %d -e %d -n 1', trnFile, mapFile, facDim, mapDim, numEpoc);
                    else
                        cmd = sprintf('python ../common/methods/gae/gae_on_fam.py -i %s -o %s -f %d -m %d -l %f -e %d -n 1 -v %d', trnFile, mapFile, facDim, mapDim, learnRate, numEpoc, verbose);
                    end
                    system(cmd);
                    
                    
                    disp('Projecting Data');
                    % Load Mapping
                    load(mapFile);
                    
                    delete(trnFile,mapFile);
                    
                    if doDisc
                        disp('Training Discriminative Part');
                        HAELabels = [ones(size(trnDataX,1),1),zeros(size(trnDataX,1),1); ...
                            zeros(size(ntrnDataX,1),1),ones(size(ntrnDataX,1),1)];
                        x = [trnDataX;ntrnDataX];
                        y = [trnDataY;ntrnDataY];

                        [~,whf_disc,z_bias,~,filter_out_disc,nn] = HAE_SingleLayer(((wxf'*x') .* (wyf'*y'))', ...
                            HAELabels, whf', z_bias', ilearning, facDim, mapDim, epochs_disc, verbose,outPut_activation{1,iActivation});
                    
                        % Apply the projection
                        ptstH = ((wxf'*tstDataX').*(wyf'*tstDataY'))';
                        ntstH = ((wxf'*ntstDataX').*(wyf'*ntstDataY'))';
                        tstH = [ptstH;ntstH];
                    
                        nn.output = outPut_activation{1,iActivation};
                        scores = nnpredict_mine(nn, tstH);

                        score_patches = zeros([size(scores,1)/batchSze 1]);
                        tstLabs = [ones(size(score_patches,1)/2,1); zeros(size(score_patches,1)/2,1)];

                        for i = 1 :batchSze: size(scores, 1)
                            score_patches(int64(i/batchSze) + 1,1) = mean(scores(i:i+batchSze-1,1));
                        end
                        
                        ap(s) = draw_nist_mdfa(tstLabs,score_patches);
                    else
                        % Map Train Data
                        trnH = sigm(bsxfun(@plus,whf*((wxf'*trnDataX').*(wyf'*trnDataY')),z_bias));
                        trnH = reshape(trnH,size(trnH,1)*batchSze,size(trnH,2)/batchSze)';
                        trnLabs = ones(size(trnH,1),1);

                        % Map Test Data
                        tstH = sigm(bsxfun(@plus,whf*((wxf'*tstDataX').*(wyf'*tstDataY')),z_bias));
                        tstH = reshape(tstH,size(tstH,1)*batchSze,size(tstH,2)/batchSze)';
                        tstLabs = ones(size(tstH,1),1);

                        % Map Negative Train Data
                        ntrnH = sigm(bsxfun(@plus,whf*((wxf'*ntrnDataX').*(wyf'*ntrnDataY')),z_bias));
                        ntrnH = reshape(ntrnH,size(ntrnH,1)*batchSze,size(ntrnH,2)/batchSze)';
                        ntrnLabs = -1*ones(size(ntrnH,1),1);

                        % Map Negative Test Data
                        ntstH = sigm(bsxfun(@plus,whf*((wxf'*ntstDataX').*(wyf'*ntstDataY')),z_bias));
                        ntstH = reshape(ntstH,size(ntstH,1)*batchSze,size(ntstH,2)/batchSze)';
                        tstH = [tstH; ntstH];
                        tstLabs = [tstLabs; -1*ones(size(ntstH,1),1)];

                        disp('Train/Test SVM');

                        % Train Linear SVM
                        if doLinear
                            svmModel = svmtrain(double([trnLabs;ntrnLabs]), double([trnH;ntrnH]), '-s 0 -t 0 -q');     % Linear SVM
                        else
                            % For RBF kernel we need to do parameter search.
                            disp('Performing Greedy Parameter Search');
                            t = 1 : floor(length(trnLabs) / 4) : length(trnLabs);
                            gind = 1 : length(trnLabs);
                            params = [];
                            tacc = [];
                            for g = 1 : 4
                                if g ~= 4
                                    gtstInd = gind(t(g):t(g+1)-1);
                                    gtrnInd = gind;
                                    gtrnInd(t(g):t(g+1)-1) = [];
                                else
                                    gtstInd = gind(t(g):end);
                                    gtrnInd = gind;
                                    gtrnInd(t(g):end) = [];
                                end

                                gtrnData = [trnH(gtrnInd,:); ntrnH(gtrnInd,:)];
                                gtrnLabs = [ones(length(gtrnInd),1); -1*ones(length(gtrnInd),1)];

                                gtstData = [trnH(gtstInd,:); ntrnH(gtstInd,:)];
                                gtstLabs = [ones(length(gtstInd),1); -1*ones(length(gtstInd),1)];

                                [bestParams,bestAcc] = greedyParamSearch(gtrnLabs,gtrnData,gtstLabs,gtstData);
                                tacc = [tacc; bestAcc];
                                params = [params; bestParams];
                            end
                            [val,mind] = max(tacc);
                            bestParams = params(mind,:);
                            svmModel = svmtrain([trnLabs;ntrnLabs], [trnH;ntrnH], sprintf('-c %d -g %f -q', bestParams(1), bestParams(2)));
                        end

                        % Test SVM
                        [predLabs, nothing, decVals] = svmpredict(double(tstLabs), double(tstH), svmModel,'-q');

                        predLabs = greedyAcc(decVals(:,1),tstLabs,0.01);

                        cAcc = sum(predLabs == tstLabs) / length(tstLabs);
                        acc(s) = cAcc;
                        tind = tstLabs == 1;
                        cAcc2 = sum(predLabs(tind) == tstLabs(tind)) / length(tstLabs(tind));
                        pacc(s) = cAcc2;
                        tind = tstLabs ~= 1;
                        cAcc3 = sum(predLabs(tind) == tstLabs(tind)) / length(tstLabs(tind));
                        nacc(s) = cAcc3;

                        tstLabs(tind) = 0;
                        ap(s) = draw_nist_mdfa(tstLabs,decVals);
                    end
                    if doDisc
                        fprintf('MAP: %3.3f\n',ap(s)*100);
                    else
                        fprintf('Acc: %3.3f Pos: %3.3f Neg: %3.3f MAP: %3.3f\n',acc(s)*100,pacc(s)*100,nacc(s)*100,ap(s)*100);
                    end
                end
                
                mAcc = mean(acc);
                mpAcc = mean(pacc);
                mnAcc = mean(nacc);
                map = mean(ap);
                

                if doDisc
                    resFile = [inFile(1:end-4) '_res_disc.csv'];
                else
                    resFile = [inFile(1:end-4) '_res.csv'];
                end
                fp = fopen(fullfile(resDir,resFile),'a+');
                if doDisc
                    fprintf(fp,'%d,%d,%d,%3.3f\n',ilearning,iActivation,rep,map*100);
                else
                    fprintf(fp,'%3.3f,%3.3f,%3.3f,%3.3f\n',mAcc*100,mpAcc*100,mnAcc*100,map*100);
                end
                fclose(fp);
            end
        end
    end
end