close all
clc

% read images
% template
boxImage = imread('./elephant.jpg');
% desk
sceneImage = imread('./clutteredDesk.jpg');

% keypoint detection
boxPoints = detectSURFFeatures(boxImage, NumOctaves=12, MetricThreshold=100, NumScaleLevels=10); % 1614
scenePoints = detectSURFFeatures(sceneImage, NumOctaves=12, MetricThreshold=100, NumScaleLevels=10); % 5819
 
% boxPoints = detectSURFFeatures(boxImage); % 272
% scenePoints = detectSURFFeatures(sceneImage); % 1129

% hey there, we also experimented a bit with SIFT but did not documented
% it, we aimed to achieve a better fit by forcing to look at the rear of
% the elephant, where we had problem fitting. Turns out that the results
% are pretty similar and that our problem has not been solved ¯\_(ツ)_/¯

% boxPoints = detectSIFTFeatures(boxImage, "ContrastThreshold", 0);
% scenePoints = detectSIFTFeatures(sceneImage, "ContrastThreshold", 0);

% keypoint description
[boxFeatures, boxPoints]=extractFeatures(boxImage, boxPoints);
[sceneFeatures, scenePoints]=extractFeatures(sceneImage, scenePoints);

% feature matching
boxPairs = matchFeatures(boxFeatures, sceneFeatures, Method="Exhaustive", MatchThreshold=30, MaxRatio=1, Unique=true, Metric="SSD");
% boxPairs = matchFeatures(boxFeatures, sceneFeatures);

matchedBoxPoints = boxPoints(boxPairs(:,1),:);
matchedScenePoints = scenePoints(boxPairs(:,2),:);
showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, matchedScenePoints, 'montage');

% geometric consistency check
[tform, inlierBoxPoints, inlierScenePoints]= estimateGeometricTransform(matchedBoxPoints, matchedScenePoints, 'similarity','MaxNumTrials', 10000, 'Confidence', 99, 'MaxDistance', 30);
% [tform, inlierBoxPoints, inlierScenePoints]= estimateGeometricTransform(matchedBoxPoints, matchedScenePoints, 'similarity');


showMatchedFeatures(boxImage, sceneImage, inlierBoxPoints, inlierScenePoints, 'montage');

% more precise bounding box
% figure, clf
% imshow(boxImage)
% [xb,yb]=ginput(4);
% csvwrite('elephant-box.csv', [xb,yb]);

%ground truth
% figure, clf
% imshow(sceneImage)
% [xt,yt] = ginput(128);
% csvwrite('ground-truth.csv', [xt,yt]);

%read ground truth
mt = readtable("ground-truth.csv");
xt =  table2array(mt(:,1));
yt =  table2array(mt(:,2));
xt=[xt; xt(1,:)];
yt=[yt; yt(1,:)];
polyTruth = polyshape(xt, yt);

mb = readtable("elephant-box.csv");
m = readtable("elephant-outline.csv");

x = table2array(m(:,1));
y = table2array(m(:,2));

xb = table2array(mb(:,1));
yb = table2array(mb(:,2));

xb=[xb; xb(1,:)];
yb=[yb; yb(1,:)];

x=[x; x(1,:)];
y=[y; y(1,:)];

elephantPoly=transformPointsForward(tform,[x y]);
boxPoly = transformPointsForward(tform, [xb, yb]);

figure, clf
imshow(sceneImage), hold on
line(elephantPoly(:,1),elephantPoly(:,2),'Color','y')
line(boxPoly(:,1), boxPoly(:,2), Color='g')
hold off