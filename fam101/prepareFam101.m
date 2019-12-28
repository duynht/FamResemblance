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
% prepareFam101: This file parepres the data for experiments.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For more information, see http://enriquegortiz.com/fbfaces.
%
% Contact: Enrique G. Ortiz (eortiz@cs.ucf.edu)
%          Afshin Dehghan (adehghan@cs.ucf.edu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[index role name] = textread(fullfile(dataPath,'FAMILY101.txt'),'%d %s %s');

if ~exist(outPath,'dir'); mkdir(outPath); end

famInd = find(index == 0);
oneInd = find(index == 1);

patchSz = 16;
pNorm = 3;

for f = 1 : length(famInd)
    disp(f);
    if f == length(famInd)
        ind = oneInd(oneInd > famInd(f-1));
    else
        ind = oneInd(oneInd < famInd(f+1) & oneInd >= famInd(f));
    end
    fam = name{famInd(f)};
    
    for p = 1 : length(ind)
        person = name{ind(p)};
        imgs = dir(fullfile(dataPath,fam,person,'*.jpg'));
        
        switch role{ind(p)}
            case 'HUSB'
                r = 0;
                icount = 0;
            case 'WIFE'
                r = 1;
                icount = 0;
            case 'DAUG'
                r = 2;
            case 'SONN'
                r = 3;
        end
        
        for i = 1 : length(imgs)
            
            im = imread(fullfile(dataPath,fam,person,imgs(i).name));
            
            [sr,sc,sch] = size(im);
            if sch == 1
                nim(:,:,1) = im;
                nim(:,:,2) = im;
                nim(:,:,3) = im;
                im = nim; clear nim;
            end            
            
            % Extract Features
            im = resizeall(im);
            hsvim = rgb2hsv(im);
            hsvim(:,:,3) = histeq(hsvim(:,:,3));
            im = hsv2rgb(hsvim);
            feat = imageToPatchColor(im,patchSz,pNorm);
            
            outfile = fullfile(outPath,sprintf('%04d_%04d_%04d.mat',f-1,r,i+icount-1));
            save(outfile,'feat');
            if r == 2 || r == 3
                icount = i + icount;
            end
        end
    end
end