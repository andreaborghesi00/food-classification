%% read images

%template
boxImage = imread('../datasets/immaginiObjectDetection/stapleRemover.jpg')

%desk
sceneImage = imread('../datasets/immaginiObjectDetection/clutteredDesk.jpg')

%figure(1), clf, imshow(boxImage)
%figure(2), clf, imshow(sceneImage)

figure(1), clf, imagesc(boxImage)
figure(2), clf, imagesc(sceneImage)

boxImage = imCrop(boxImage);

%% compute the scale factor
% so that we can perform sliding window with a fixed scale we compute it
% (manually) as the ratio of the same box dimension in the two images

fs = 2.82; % computation done by manually from the professor w1/h1 x = w2/h2, x is this scaling factor
boxImage = imresize(boxImage, 1/fs);

%% sliding window
boxImage = im2double(boxImage); % convert to double 
sceneImage = im2double(sceneImage); 
Sb = size(boxImage); 
Ss = size(sceneImage);
step = 2; % how many pixels we move the window each time
Map = []; % this will store the values of the sum of squared differences in each location

tic
for rr = 1:step:(Ss(1)-Sb(1)) % begin:step:end
    tmp = [];
    for cc = 1:step:Ss(2)-Sb(2)
        
        D = sceneImage(rr:rr+Sb(1)-1, cc:cc+Sb(2)-1, :) - boxImage; % we stop before the end of the image, as much as boxImage
        % compute the sum of squared differences
        D = D.^2;
        D = sum(D, 'all');
        tmp = [tmp, D];
    end
    Map = [Map; tmp]; % append the row to the map
    figure(3), clf, imagesc(Map), colorbar, drawnow
end
toc
