clear all
close all
clc

%template
boxImage = imread('../datasets/immaginiObjectDetection/stapleRemover.jpg')

%desk
sceneImage = imread('../datasets/immaginiObjectDetection/clutteredDesk.jpg')

figure(1), clf, imshow(boxImage)
figure(2), clf, imshow(sceneImage)

%figure(1), clf, imagesc(boxImage)
%figure(2), clf, imagesc(sceneImage)

%% keypoint detection
boxPoints = detectSURFFeatures(boxImage)
scenePoints = detectSURFFeatures(sceneImage)

figure(1), clf
imshow(boxImage), hold on
plot(selectStrongest(boxPoints, 100)), hold off

figure(2), clf
imshow(sceneImage), hold on
plot(selectStrongest(scenePoints, 100)), hold off

% keypoint description
[boxFeatures, boxPoints] = extractFeatures(boxImage, boxPoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);

% feature matching
boxPairs = matchFeatures(boxFeatures, sceneFeatures);
matchedBoxPoints = boxPoints(boxPairs(:, 1), :);
matchedScenePoints = scenePoints(boxPairs(:, 2), :);
showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, matchedScenePoints, 'montage')


%% geometric consistency check
[tform, inlierBoxPoints, inlierScenePoints] = estimateGeometricTransform(matchedBoxPoints, matchedScenePoints, 'affine');
showMatchedFeatures(boxImage, sceneImage, inlierBoxPoints, inlierScenePoints, 'montage')

% Bounding box
boxPolygon = [1, 1;                           % top-left
              size(boxImage, 2), 1;            % top-right
              size(boxImage, 2), size(boxImage, 1); % bottom-right
              1, size(boxImage, 1);            % bottom-left
              1, 1];                   % top-left again to close the polygon

newBoxPolygon = transformPointsForward(tform, boxPolygon);

figure, clf
imshow(sceneImage), hold on
line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y')
hold off


% more precise bounding box
figure, clf
imshow(boxImage)
[x, y] = ginput(4);

%%
x = [x; x(1)];
y = [y; y(1)];
newBoxPolygon = transformPointsForward(tform, [x, y]);
figure, clf
imshow(sceneImage), hold on
line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y')
hold off


% we have a more difficult pattern for the assignment, we need to change some parameters in the function
% for example we can change the parameters in detectSURFFeatures in order to force the detector to detect more point (too contrast, not strong enough, etc.)
% we can also change the parameters in matchFeatures in order to force the detector to match more points (too far, too close, etc.)
% we can also change the parameters in estimateGeometricTransform in order to force the detector to find more inliers (too far, too close, etc.) (the y)

% written by copilot:
% we can also change the parameters in transformPointsForward in order to force the detector to find more inliers (too far, too close, etc.) (the x)
% we can also change the parameters in ginput in order to force the detector to find more inliers (too far, too close, etc.) (the x)


% try to draw a polygon that is similar to the shape of the elephant
% every round will have a different result

