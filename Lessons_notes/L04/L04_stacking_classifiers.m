clc
clear

%% load dataset
load Datasets/dataset.mat

%% plot data
u = find(labels_tr == 1);
figure(1), hold on
plot(data_tr(u,1), data_tr(u,2), 'r.')
u = find(labels_tr==2)
plot(data_tr(u,1), data_tr(u,2), 'b.')
hold off


% train classifiers using this data, and then using their features to train
% another set of classifiers

%% split training set, stratified sampling

rng('default'); % for reproducibility

idx_f1 = []; % fold 1
idx_f2 = []; % fold 2

for nclass = 1:2
    u = find(labels_tr == nclass);
    idx = randperm(numel(u));
    idx_f1 = [idx_f1; u(idx(1:round(numel(idx)/2)))];
    idx_f2 = [idx_f2; u(idx(1 + round(numel(idx)/2):end))];
end
labels_f1 = labels_tr(idx_f1);
labels_f2 = labels_tr(idx_f2);
data_f1 = data_tr(idx_f1,:);
data_f2 = data_tr(idx_f2,:);

%% train level-1 classifiers on fold1

mdl = {}; % different type of container, in each position we can put whatever we want, with differet types

% SVM with gaussian kernel

rng('default');
mdl{1} = fitcsvm(data_f1, labels_f1, 'KernelFunction', 'gaussian', 'KernelScale', 5);

%SVM with polynomial kernel

rng('default');
mdl{2} = fitcsvm(data_f1, labels_f1, 'KernelFunction', 'polynomial', 'KernelScale', 10);

% Decision tree

rng('default');
mdl{3} = fitctree(data_f1, labels_f1, 'SplitCriterion', 'gdi', 'MaxNumSplits', 20);

% Naive Bayes

rng('default');
mdl{4} = fitcnb(data_f1, labels_f1);

% Ensemble of decision trees

rng('default');
mdl{5} = fitcensemble(data_f1, labels_f1);


%% make the predictions on fold2 (to be used to train the second stacked classifier)

N = numel(mdl);
Predictions = zeros(size(data_f2, 1), N);
Scores = zeros(size(data_f2, 1), N);

for ii = 1:N
    [predictions, scores] = predict(mdl{ii}, data_f2);
    Predictions(:,ii) = predictions;
    Scores(:,ii) = scores(:,1); % confidence interval, I take just the first point of the interval, it's centered in 0
end

%% train the stacked classifier on fold2

rng('default');
% stackedModel = fitcensemble(Scores, labels_f2, 'Method','AdaBoostM1'); % try with parameter 'NumLearningCycles'
stackedModel = fitcensemble(Scores, labels_f2, 'Method', 'Bag');

%% the meta-classifier when trained on Predictions (i.e. the predicted class) instead of the classification Scores
stackedModel_predict = fitcensemble(Predictions, labels_f2, 'Method','Bag');




mdl{N+1} = stackedModel;
mdl{N+2} = stackedModel_predict;

Predictions = zeros(size(data_te, 1), N);
Scores = zeros(size(data_te, 1), N);

for ii = 1:N
    [predictions, scores] = predict(mdl{ii}, data_te);
    Predictions(:, ii) = predictions;
    Scores(:, ii) = scores(:,1);
    ACC(ii) = numel(find(predictions == labels_te))/numel(labels_te);
end

predictions = predict(mdl{N+1}, Scores);
ACC(N+1) = numel(find(predictions == labels_te))/numel(labels_te);

predictions = predict(mdl{N+2}, Predictions); % not using the scores (confidence intervals) but the predictions of the classifiers as scores
ACC(N+2) = numel(find(predictions == labels_te))/numel(labels_te);



%% The meta-classifier when the training split is not perfomed and the same data is used to train the level-1 classifiers and the meta-classifier, same data, no split.


rng('default');
mdl{1} = fitcsvm(data_tr, labels_tr, 'KernelFunction', 'gaussian', 'KernelScale', 5);
rng('default');
mdl{2} = fitcsvm(data_tr, labels_tr, 'KernelFunction', 'polynomial', 'KernelScale', 10);
rng('default');
mdl{3} = fitctree(data_tr, labels_tr, 'SplitCriterion', 'gdi', 'MaxNumSplits', 20);
rng('default');
mdl{4} = fitcnb(data_tr, labels_tr);
rng('default');
mdl{5} = fitcensemble(data_tr, labels_tr);

Scores = zeros(size(data_tr, 1), N);

for ii = 1:N
    [~, scores] = predict(mdl{ii}, data_tr);
    Scores(:,ii) = scores(:,1); % confidence interval, I take just the first point of the interval, it's centered in 0
end

stackedModel_no_split = fitcensemble(Scores, labels_tr, 'Method', 'Bag');
mdl{N+3} = stackedModel_no_split;

Predictions = zeros(size(data_te, 1), N);
Scores = zeros(size(data_te, 1), N);

for ii = 1:N
    [predictions, scores] = predict(mdl{ii}, data_te);
    Predictions(:, ii) = predictions;
    Scores(:, ii) = scores(:,1);
end

ACC(N+3) = numel(find(predictions == labels_te))/numel(labels_te);



%ACC
fprintf(['Accuracies: \n' ...
    'SVM Gaussian:                                  %f\n' ...
    'SVM Polynomial:                                %f\n' ...
    'Classification Tree:                           %f\n'...
    'Naive Bayes:                                   %f\n'...
    'Ensemble Decision Tree:                        %f\n'...
    'Meta-Classifier:                               %f\n'...
    'Meta-Classifier trained with predictions:      %f\n'...
    'Meta-Classifier trained on same training data: %f\n\n'
    ],ACC(1), ACC(2), ACC(3), ACC(4), ACC(5), ACC(6), ACC(7), ACC(8));

