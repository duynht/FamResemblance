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
% processResults: Process Fam101 results.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For more information, see http://enriquegortiz.com/fbfaces.
%
% Contact: Enrique G. Ortiz (eortiz@cs.ucf.edu)
%          Afshin Dehghan (adehghan@cs.ucf.edu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fName = 'fam101_disc.eps';
oFile = 'fam101_disc.mat';

balData = -1;

val = 0.2;
fSize = 15;


colors = [255,0,0; 11,204,51; 0,0,255; 221,113,219; 255,162,0;209,207,32;0,201,203;85,85,85;255,162,0;0,255,0;0,0,255;221,113,219;209,207,32;0,201,203;85,85,85;255,162,0;255,0,0;0,255,0;0,0,255;221,113,219;209,207,32;0,201,203;85,85,85;255,162,0;255,0,0;11,204,51;0,0,255;221,113,219;0,201,203;255,162,0;255,0,0;209,207,32;0,201,203;85,85,85;255,162,0;];
% 0,201,203;
% Compute Chance
% ranks = 1 : 5 : 100;

xMax = 870;
ranks = [1 2 3 4 5:5:xMax]';
x = 8;
c = ones(length(ranks),1);
for r = 1 : length(ranks)
    if ranks(r) >= xMax; break; end
%     c(r) = 1 - (nchoosek(xMax-1,ranks(r)) / nchoosek(xMax,ranks(r)));
%     c(r) = 1 - (xMax - ranks(r))/xMax;
	c(r) = 1 - newc(xMax-ranks(r),x) / newc(xMax,x);
end

leg1 = {'Chance'};
figure(1);
semilogx(ranks,c,'linewidth',3,'color',[1,0,0]); set(gca,'FontSize',fSize);
hold on;

psAll = zeros(length(ranks),length(splits));

msize = 0;
for s = 1 : length(splits)
    cSplit = splits{s};
    
    leg1 = [leg1 upper(cSplit)];
    
%     cFile = ['*_' cSplit '_*_' num2str(val) '_full.mat'];
%     cFile = ['*' cSplit '*.mat'];
    cFile = [cSplit '_*_tab.mat'];
    
    files = dir(fullfile(tPath,cFile));
    
    ps = [];
    vrs = [];
    fars = [];
    for f = 1 : length(files)
    
        load(fullfile(tPath,files(f).name));
        
        % Balance Data
        newind = [];
        if balData > 0
            uLabs = unique(labelparent);
            for u = 1 : length(uLabs)
                curLab = uLabs(u);
                ind = find(labelparent == curLab);
                if length(ind) > balData
                    newind = [newind; ind(1:balData)];
                else
                    newind = [newind; ind];
                end
            end
            correspondence = correspondence(:,newind);
            scoretable = scoretable(:,newind);
            length(newind)
        end        
                
        [p,ranks] = computeCMC(scoretable,correspondence,ranks);
        ps = [ps p];
    end
    msize = max(msize,size(scoretable,1));
    ps = mean(ps,2);
    vrs = mean(vrs,2);
    fars = mean(fars,2);
    
    psAll(:,s) = ps;
    
    figure(1);
    hold on;
    semilogx(ranks,ps,'linewidth',3,'color',colors(s+1,:)./255); set(gca,'FontSize',fSize);
end

figure(1); legend(leg1,'Location','SouthEast','FontSize',fSize); grid; xlim([1 1000])
xlabel('Rank','FontSize',fSize,'FontWeight','bold'); ylabel('Identification Rate','FontSize',fSize,'FontWeight','bold');

xMaxStr = num2str(xMax);

% set(gca,'XTick',[1 10 100 1000]);

pbaspect([2 1 1]);

print(gcf,fullfile(tPath,fName),'-depsc');

psAll = mean(psAll,2);
save(fullfile(tPath,oFile),'ranks','psAll');