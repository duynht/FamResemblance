% load('C:\project\Aladdin\release_event\bases\isa1layer_16_10_300_1.mat')
a = nn.W{1};
a = a(:,1:end-1)';
%a=Atica;
for i =1:500
    f = a(:,i);
    f = reshape(f,[10 10]);
    %bg = zeros(16,16*10+9*2);
    %for j = 1:10
    %   bg(:,1+(j-1)*16+(j-1)*2:1+(j-1)*16+(j-1)*2+15)=f(:,:,j);
    %end
    %figure (1);
    imshow(f,[]);%title(num2str(i));
    pause;
    %saveas(gcf,['C:\Users\vision2011\Dropbox\ASLAN\new_filter\' num2str(i) '.jpg']);
    
    %AxesH = gca;   % Not the GCF
    %F = getframe(AxesH);
    %imwrite(F.cdata, ['C:\Users\vision2011\Dropbox\ASLAN\new_filter\' num2str(i) '.jpg']);
end