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
% trainTestSplits: Create training and testing data splits for Fam101.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For more information, see http://enriquegortiz.com/fbfaces.
%
% Contact: Enrique G. Ortiz (eortiz@cs.ucf.edu)
%          Afshin Dehghan (adehghan@cs.ucf.edu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxImgs = 20;
fams = 1 : 101;
rind = randperm(101);
trnFams = fams(rind(1:51));
tstFams = fams(rind(52:101));
batchSze = 16;

if ~exist(outPath,'dir'); mkdir(outPath); end
if ~exist(outPath2,'dir'); mkdir(outPath2); end
if ~exist(resDir,'dir'); mkdir(resDir); end

if exist('trntstind.mat','file')
    load('trntstind.mat');
else
    save('trntstind.mat','trnFams','tstFams');
end

% Setup Training Data
disp('Processing Training Data');
for t = 1 : length(trnFams)
    curFam = trnFams(t);
    
    % Parent Offspring Pairings
    for p = 0 : 1
        pFiles = dir(fullfile(dataPath,sprintf('%04d_%04d_*.mat',curFam-1,p)));
        
        nump = length(pFiles);
        if nump == 0; continue; end
        
        for o = 2 : 3
            lfeat = [];
            rfeat = [];
            
            oFiles = dir(fullfile(dataPath,sprintf('%04d_%04d_*.mat',curFam-1,o)));

            numo = length(oFiles);
            if numo == 0; continue; end
            
            if numo > maxImgs; numo = maxImgs; end
            if nump > maxImgs; nump = maxImgs; end
            
            for pf = 1 : nump
                load(fullfile(dataPath,pFiles(pf).name));
                pfeat = feat;
                for of = 1 : numo
                    lfeat = [lfeat; pfeat];
                    load(fullfile(dataPath,oFiles(of).name));
                    rfeat = [rfeat; feat]; 
                end
            end
            
            file = '';
            if p == 0;
                file = [file 'f'];
            else
                file = [file 'm'];
            end

            if o == 2
                file = [file 'd'];
            else
                file = [file 's'];
            end
            
            tlabs = ones(size(rfeat,1),1).*curFam-1;

            file = sprintf('%s_%04d_train_p16.mat',file,t-1);
            save(fullfile(outPath,file),'lfeat','rfeat','tlabs');
        end
    end
end

% Load All Data
[x,y,labs,xmean,xstd,ymean,ystd,U,S] = loadAllType(outPath,'fs');
save(fullfile(resDir,'fs_train_p16.mat'),'x','y','labs','xmean','xstd','ymean','ystd','batchSze','U','S');

[x,y,labs,xmean,xstd,ymean,ystd,U,S] = loadAllType(outPath,'fd');
save(fullfile(resDir,'fd_train_p16.mat'),'x','y','labs','xmean','xstd','ymean','ystd','batchSze','U','S');

[x,y,labs,xmean,xstd,ymean,ystd,U,S] = loadAllType(outPath,'ms');
save(fullfile(resDir,'ms_train_p16.mat'),'x','y','labs','xmean','xstd','ymean','ystd','batchSze','U','S');

[x,y,labs,xmean,xstd,ymean,ystd,U,S] = loadAllType(outPath,'md');
save(fullfile(resDir,'md_train_p16.mat'),'x','y','labs','xmean','xstd','ymean','ystd','batchSze','U','S');


%%
disp('Processing Testing Data');
for t = 1 : length(tstFams)
    curFam = tstFams(t);
    
    % Parent Offspring Pairings
    for p = 0 : 1
        lfeat = [];
        pFiles = dir(fullfile(dataPath,sprintf('%04d_%04d_*.mat',curFam-1,p)));
        nump = length(pFiles);
        if nump == 0; continue; end;
        
        % Load All Parent           
        for pf = 1 : nump
            load(fullfile(dataPath,pFiles(pf).name));
            lfeat = [lfeat; feat];
        end
        llabs = ones(size(lfeat,1),1).*curFam-1;
        
        for o = 2 : 3
            rfeat = [];
            
            oFiles = dir(fullfile(dataPath,sprintf('%04d_%04d_*.mat',curFam-1,o)));

            numo = length(oFiles);
            if numo == 0; continue; end;

            % Load All Offspring
            for of = 1 : numo
                load(fullfile(dataPath,oFiles(of).name));
                rfeat = [rfeat; feat]; 
            end
            
            file = '';
            if p == 0;
                file = [file 'f'];
            else
                file = [file 'm'];
            end

            if o == 2
                file = [file 'd'];
            else
                file = [file 's'];
            end
            
            rlabs = ones(size(rfeat,1),1).*curFam-1;
            
            file = sprintf('%s_%04d_test_p16.mat',file,t-1);
            save(fullfile(outPath2,file),'lfeat','rfeat','llabs','rlabs');
        end
    end
end

% Load All Data
load(fullfile(resDir,'fs_train_p16.mat'));
[x,y,xlabs,ylabs] = loadAllType2(outPath2,'fs',xmean,xstd,ymean,ystd,U,S);
save(fullfile(resDir,'fs_test_p16.mat'),'x','y','xlabs','ylabs','batchSze');

load(fullfile(resDir,'fd_train_p16.mat'));
[x,y,xlabs,ylabs] = loadAllType2(outPath2,'fd',xmean,xstd,ymean,ystd,U,S);
save(fullfile(resDir,'fd_test_p16.mat'),'x','y','xlabs','ylabs','batchSze');

load(fullfile(resDir,'ms_train_p16.mat'));
[x,y,xlabs,ylabs] = loadAllType2(outPath2,'ms',xmean,xstd,ymean,ystd,U,S);
save(fullfile(resDir,'ms_test_p16.mat'),'x','y','xlabs','ylabs','batchSze');

load(fullfile(resDir,'md_train_p16.mat'));
[x,y,xlabs,ylabs] = loadAllType2(outPath2,'md',xmean,xstd,ymean,ystd,U,S);
save(fullfile(resDir,'md_test_p16.mat'),'x','y','xlabs','ylabs','batchSze');