%% negative class
neg = dir('../datasets/CaltechFaces/my_train_non_face_scenes/*.jpg');


% negative class augmentation
mkdir('../datasets/CaltechFaces/my2_train_non_face_scenes/')
outdir = '../datasets/CaltechFaces/my2_train_non_face_scenes';
for ii=1:size(neg,1)
    im = imread([neg(ii).folder filesep neg(ii).name]);
    imwrite(im,[outdir filesep neg(ii).name]);
    
    [pp,ff,ee]=fileparts(neg(ii).name);
    
    im_flip = fliplr(im);
    imwrite(im_flip,[outdir filesep ff '_flip' ee]);
    % <----------- consider also up-down versions? TBD
    
    for nrot = 1:2 % <-----how many? TBD
        imr = imrotate(im, 40*rand(1)-20,'crop'); % <--- rotation range? TBD
        imwrite(imr,[outdir filesep ff '_r' num2str(nrot) ee]);
    end
end


% negativeFolder = './CaltechFaces/my_train_non_face_scenes';
negativeFolder = '../datasets/CaltechFaces/my2_train_non_face_scenes';
negativeImages = imageDatastore(negativeFolder);

%% positive class
faces = dir('../datasets/CaltechFaces/my_train_faces/*.jpg');
sz = [size(faces,1) 2];
varTypes = {'cell','cell'};
varNames = {'imageFilename','face'};
facesIMDB = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);

for ii=1:size(faces,1)
    facesIMDB.imageFilename(ii) = {[faces(ii).folder filesep faces(ii).name]};
    facesIMDB.face(ii) = {[1 1 32 32]};
end
positiveInstances = facesIMDB;

%% VJ detector training
trainCascadeObjectDetector('myFaceDetector.xml',positiveInstances,negativeImages,...
    'FalseAlarmRate',0.1,'NumCascadeStages',2);
% <---- TBD: change values and have a look at the available parameters

%% visualize results
detector = vision.CascadeObjectDetector('myFaceDetector.xml');
% detector = vision.CascadeObjectDetector();

imgs = dir('../datasets/CaltechFaces/test_scenes/test_jpg/*.jpg');

for ii=1:size(imgs,1)
    img = imread([imgs(ii).folder filesep imgs(ii).name]);
    bbox = step(detector,img);
    
    detectedImg = insertObjectAnnotation(img,'rectangle',bbox,'face');
    detectedImg = imresize(detectedImg,800/max(size(detectedImg)));
    
    figure(1),clf
    imshow(detectedImg)
    waitforbuttonpress
end

%% visualize results and GTs
load('../datasets/CaltechFaces/test_scenes/GT.mat');

detector = vision.CascadeObjectDetector('myFaceDetector.xml');
% detector = vision.CascadeObjectDetector();

imgs = dir('../datasets/CaltechFaces/test_scenes/test_jpg/*.jpg');

for ii=1:size(imgs,1)
    img = imread([imgs(ii).folder filesep imgs(ii).name]);
    bbox = step(detector,img);
    
    detectedImg = insertObjectAnnotation(img,'rectangle',bbox,'face');
    detectedImg_1 = imresize(detectedImg,800/max(size(detectedImg)));
    
    bbox = GT.face{ii};
    
    detectedImg = insertObjectAnnotation(img,'rectangle',bbox,'face');
    detectedImg_2 = imresize(detectedImg,800/max(size(detectedImg)));
    
    
    figure(1),clf
    imshow([detectedImg_1 detectedImg_2])
    waitforbuttonpress
end

%% create data structure that contains our results

load('../datasets/CaltechFaces/test_scenes/GT.mat');

detector = vision.CascadeObjectDetector('myFaceDetector.xml');
% detector = vision.CascadeObjectDetector();

imgs = dir('../datasets/CaltechFaces/test_scenes/test_jpg/*.jpg');

numImages = size(imgs,1);
results = table('Size',[numImages 2],...
    'VariableTypes',{'cell','cell'},...
    'VariableNames',{'face','Scores'});

for ii=1:size(imgs,1)
    img = imread([imgs(ii).folder filesep imgs(ii).name]);
    bbox = step(detector,img);
    results.face{ii}=bbox;
    results.Scores{ii}=0.5+zeros(size(bbox,1),1);
end

% compute average precision
[ap, recall, precision] = evaluateDetectionPrecision(results,GT,0.2);
figure(2),clf
plot(recall,precision)
grid on
title(sprintf('Average Precision = %.1f',ap))