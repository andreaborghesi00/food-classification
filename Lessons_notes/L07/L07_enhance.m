clear
clc
% sequentially training the model, then sequentially selecting the best


neg = dir('../datasets/CaltechFaces/my_train_non_face_scenes/*.jpg'); % load the data

% we want to augment the negative class by generating random variations
% negative class augmentation:

mkdir('../datasets/CaltechFaces/my2_train_non_face_scenes/')
outdir = '../datasets/CaltechFaces/my2_train_non_face_scenes';

for ii = 1:size(neg, 1)
    im = imread([neg(ii).folder, '/', neg(ii).name]); % read the image
    imwrite(im, [outdir, '/', neg(ii).name]); % save the original image
    [pp, ff, ee] = fileparts(neg(ii).name); % divide the name into parts
    im_flip = fliplr(im); % flip the image
    imwrite(im_flip, [outdir, '/', ff, '_flip', ee]); % save the flipped image
    
    % we can also flip the image up-down:
    im_updw = flipud(im); % flip the image up-down
    imwrite(im_updw, [outdir, '/', ff, '_updw', ee]); % save the up-down flipped image

    % in general, we want multiple variations of the image
    for nrot = 1:10 % how many rotation is to be seen
        im_rot = imrotate(im, 40*rand(1)-20, 'crop'); % rotate the image by 20 degrees in one of the directions randomly
        imwrite(im_rot, [outdir, '/', ff, '_rot', num2str(nrot), ee]); % save the rotated image
    end
end

negativeFolder = '../datasets/CaltechFaces/my2_train_non_face_scenes';
negativeImages = imageDatastore(negativeFolder); % load the negative images, it's a matlab object dataloader

%----------------------------------------------------------------------------------------------------------------

faces = dir('../datasets/CaltechFaces/my_train_faces/*.jpg'); % load the datafaces = dir('./CaltechFaces/my_train_faces/*.jpg');

%% enhance positive images
mkdir('../datasets/CaltechFaces/my2_train_face_scenes/')
outdir = '../datasets/CaltechFaces/my2_train_face_scenes';

for ii = 1:size(faces, 1)
    im = imread([faces(ii).folder, '/', faces(ii).name]); % read the image
    imwrite(im, [outdir, '/', faces(ii).name]); % save the original image
    [pp, ff, ee] = fileparts(faces(ii).name); % divide the name into parts
    im_flip = fliplr(im); % flip the image
    imwrite(im_flip, [outdir, '/', ff, '_flip', ee]); % save the flipped image
    
    % we can also flip the image up-down:
    im_updw = flipud(im); % flip the image up-down
    imwrite(im_updw, [outdir, '/', ff, '_updw', ee]); % save the up-down flipped image

    % in general, we want multiple variations of the image
    for nrot = 1:10 % how many rotation is to be seen
        im_rot = imrotate(im, 40*rand(1)-20, 'crop'); % rotate the image by 20 degrees in one of the directions randomly
        imwrite(im_rot, [outdir, '/', ff, '_rot', num2str(nrot), ee]); % save the rotated image
    end
end

%----------------------------------------------------------------------------------------------------------------
%% load the positive images

faces_enhanced = dir('../datasets/CaltechFaces/my2_train_face_scenes/*.jpg')

positiveInstances = struct('imageFilename', {}, 'face', {});
sz = [size(faces_enhanced,1) 2];
varTypes = {'cell','cell'};
varNames = {'imageFilename','face'};
facesIMDB = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);

for ii = 1:size(faces_enhanced, 1)
    facesIMDB.imageFilename(ii) = {[faces_enhanced(ii).folder, '/', faces_enhanced(ii).name]};
    facesIMDB.face(ii) = {[1 1 32 32]}; % position of the face in the image, which is the whole image in this case
end
positiveInstances = facesIMDB;

positiveInstances


%% VJ detector training

trainCascadeObjectDetector('caltechFaceDetector.xml', positiveInstances, negativeFolder, FalseAlarmRate = 0.5, NumCascadeStages = 5, FeatureType='HOG'); % try  with 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, change the number of stages and have a look at the available options
% it creates integral images and trains the detector, then it deletes them

%----------------------------------------------------------------------------------------------------------------
% visualize the results

detector = vision.CascadeObjectDetector('caltechFaceDetector.xml'); % load the detector
imgs = dir('../datasets/CaltechFaces/test_scenes/test_jpg/*.jpg'); % load the data

for ii = 1:size(imgs, 1)
    im = imread([imgs(ii).folder, '/', imgs(ii).name]); % read the image
    bbox = step(detector, im); % detect the faces

    detectedIm = insertObjectAnnotation(im, 'rectangle', bbox, 'face'); % insert the annotations
    detectedIm = imresize(detectedIm, 800/max(size(detectedIm))); % resize the image
    
    figure(1),clf % create a figure
    imshow(detectedIm) % show the image
    %pause(0.1); % pause for 0.1 seconds
    waitforbuttonpress % wait for a button press
end

%----------------------------------------------------------------------------------------------------------------
%% measure the performance

load('../datasets/CaltechFaces/test_scenes/GT.mat'); % load the ground truth
detector = vision.CascadeObjectDetector('caltechFaceDetector.xml'); % load the detector
imgs = dir('../datasets/CaltechFaces/test_scenes/test_jpg/*.jpg'); % load the data

numImages = size(imgs, 1);
results = table('Size', [numImages, 2], 'VariableTypes', {'cell', 'cell'}, 'VariableNames', {'face', 'Scores'});

for ii = 1:size(imgs, 1)
    img = imread([imgs(ii).folder, '/', imgs(ii).name]); % read the image
    bbox = step(detector, img); % detect the faces
    results.face{ii} = bbox; % save the results
    results.Scores{ii} = 0.5 + zeros(size(bbox, 1), 1); % save the scores
end

% compute average precision
[ap, recall, precision] = evaluateDetectionPrecision(results, GT, 0.2);
figure(2), clf
plot(recall, precision)
grid on
title(sprintf('Average Precision = %.1f', ap))

% Assigment: we want to measure the performance and obtain the highest one