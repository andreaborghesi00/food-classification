clc
clear 
 
%% Starting parameters and initialization
nalgs = 4;
ndatasets = 4;
t = 5;
qalpha = 2.569;
accuracy_5x2=zeros(nalgs, ndatasets);
accuracy_times = zeros(nalgs, t);
kfolds = 2; % implementation works for 2-fold only
 
 
%% Tuning hyperparameters
sum_param_svm_linear = 0;
sum_param_svm_gaussian = 0;
sum_param_knn = 0;
sum_param_tree = 0;
 
for ndataset = 1:ndatasets
    switch ndataset
        case 1, load Datasets/dataset1.mat;
        case 2, load Datasets/dataset2.mat;
        case 3, load Datasets/dataset3.mat;
        case 4, load Datasets/dataset4.mat;
        otherwise
    end
        % Stratified sampling
        [idx_tr, idx_te] = twoFoldSampling(labels);
 
        labels_tr = labels(idx_tr); % here we have the solutions
        labels_te = labels(idx_te); % also here
        data_tr = data(idx_tr,:); % access to data, go to row idx_tr and take all the values of the columns
        data_te = data(idx_te,:);
 
        % Define algorithm parameters, the parameters to be tuned will be
        % overwritten during the tuning procedure
        params_svm_linear = {'KernelFunction', 'linear', 'Kernelscale', 1};
        params_svm_gaussian = {'KernelFunction', 'gaussian', 'Kernelscale', 0.3};
        params_knn = {'Distance', 'Euclidean', 'NumNeighbors', 10};
        params_tree = {'SplitCriterion', 'gdi', 'MaxNumSplits', 15};
 
        % tuning parameters
        range_svm_linear = [0.1, 2]; % range for linear SVM kernel scale
        range_svm_gaussian = [0.1, 2]; % range for Gaussian SVM kernel scale
        range_knn = [1, 50]; % range for number of neighbors in kNN
        range_tree = [5, 20]; % range for maximum number of splits in decision tree
 
        % training classifiers
        sum_param_svm_linear = sum_param_svm_linear + Tuning(data_tr, labels_tr, data_te, labels_te, @fitcsvm, params_svm_linear, 'Kernelscale', range_svm_linear, 0.1);
        sum_param_svm_gaussian = sum_param_svm_gaussian + Tuning(data_tr, labels_tr, data_te, labels_te, @fitcsvm, params_svm_gaussian, 'Kernelscale', range_svm_gaussian, 0.1);
        sum_param_knn = sum_param_knn + Tuning(data_tr, labels_tr, data_te, labels_te, @fitcknn, params_knn, 'NumNeighbors', range_knn, 1);
        sum_param_tree = sum_param_knn + Tuning(data_tr, labels_tr, data_te, labels_te, @fitctree, params_tree, 'MaxNumSplits', range_tree, 1);
 
        sum_param_svm_linear = sum_param_svm_linear + Tuning(data_te, labels_te, data_tr, labels_tr, @fitcsvm, params_svm_linear, 'Kernelscale', range_svm_linear, 0.1);
        sum_param_svm_gaussian = sum_param_svm_gaussian + Tuning(data_te, labels_te, data_tr, labels_tr, @fitcsvm, params_svm_gaussian, 'Kernelscale', range_svm_gaussian, 0.1);
        sum_param_knn = sum_param_knn + Tuning(data_te, labels_te, data_tr, labels_tr, @fitcknn, params_knn, 'NumNeighbors', range_knn, 1);
        sum_param_tree = sum_param_tree + Tuning(data_te, labels_te, data_tr, labels_tr, @fitctree, params_tree, 'MaxNumSplits', range_tree, 1);
 
end
 
tuned_param_svm_linear = sum_param_svm_linear / (2*ndatasets)
tuned_param_svm_gaussian = sum_param_svm_gaussian / (2*ndatasets)
tuned_param_knn = round(sum_param_knn / (2*ndatasets))
tuned_param_tree = round(sum_param_tree / (2*ndatasets))
 
 
%% Dataset loading
for ndataset = 1:ndatasets
    switch ndataset
        case 1, load Datasets/dataset1.mat;
        case 2, load Datasets/dataset2.mat;
        case 3, load Datasets/dataset3.mat;
        case 4, load Datasets/dataset4.mat;
        otherwise
    end
 
    accuracy_times = [];
 
    for ntimes = 1:t
        %% Stratified sampling
        [idx_tr, idx_te] = twoFoldSampling(labels);
 
        labels_tr = labels(idx_tr); % here we have the solutions
        labels_te = labels(idx_te); % also here
        data_tr = data(idx_tr,:); % access to data, go to row idx_tr and take all the values of the columns
        data_te = data(idx_te,:);
 
        data_fold = zeros(size(data_tr, 1), size(data_tr, 2), 2);
        labels_fold = zeros(size(labels_tr, 1), size(labels_tr, 2), 2);
        data_fold(:, :, 1) = data_tr;
        data_fold(:, :, 2) = data_te;
        labels_fold(:, 1) = labels_tr;
        labels_fold(:, 2) = labels_te;
 
 
        %% Training classifiers
        % current_classifier = fitcsvm(data_tr, labels_tr, data_te, labels_te, 'KernelFunction', 'linear', 'Kernelscale', 1);
        for nalg = 1:4
            total_accuracy = 0;
            for fold = 1:kfolds
                switch nalg
                    case 1, current_classifier = fitcsvm(data_fold(:, :, fold), labels_fold(:, fold), 'KernelFunction', 'linear', 'Kernelscale', tuned_param_svm_linear);
                    case 2, current_classifier = fitcsvm(data_fold(:, :, fold), labels_fold(:, fold), 'KernelFunction', 'gaussian', 'Kernelscale', tuned_param_svm_gaussian); % gaussian2
                    case 3, current_classifier = fitcknn(data_fold(:, :, fold), labels_fold(:, fold), 'Distance', 'Euclidean', 'NumNeighbors', tuned_param_knn); % using euclidean distance
                    case 4, current_classifier = fitctree(data_fold(:, :, fold), labels_fold(:, fold), 'SplitCriterion','gdi', 'MaxNumSplits', tuned_param_tree); % decision tree, setting the number of splits and the split criterion
                    otherwise
                end
 
                next_fold = mod(fold, kfolds)+1;
                prediction = predict(current_classifier, data_fold(:, :, next_fold));
 
                % find all the times the prediction is equal to the label, and count it,
                % then divide by the total number in order to have a percentage:
                total_accuracy = total_accuracy + numel(find(prediction == labels_fold(:, next_fold)))/numel(labels_fold(:, next_fold));                
            end
            accuracy_times(nalg, ntimes) = total_accuracy/kfolds; 
        end
    end
 
    %fprintf("\n\nDataset %d\n", ndataset);
    accuracy_times;
    for nalg=1:4
        accuracy_5x2(nalg, ndataset) = mean(accuracy_times(nalg, :));
    end 
end
accuracy_5x2'
 
 
%% Find ranks and mean ranks
%find ranks, descending order
ranks = findRanks(accuracy_5x2, nalgs, ndatasets);
ranks'
% Compute average ranks
average_rank = zeros(1, nalgs);
for nalg=1:nalgs
    average_rank(nalg) = mean(ranks(nalg,:));
end
 
average_rank
p = friedman(accuracy_5x2')

% critical value
cdvalue = criticalDifference(qalpha, nalgs, ndatasets);
 
cdvalue
 
%% Plot
figure;
hold on;
 
%plot CD lines
for nalg=1:nalgs
    % plot(X, Y)
    plot([average_rank(nalg)-(cdvalue/2), average_rank(nalg)+(cdvalue/2)], [nalg, nalg], 'LineWidth',2);
end
 
%plot average rank dots
scatter(average_rank, 1:nalgs, 'filled');
 
xlabel('Average Rank');
ylabel('Algorithms');
title('Critical Difference Diagram');
yticks(1:nalgs);
yticklabels({'SVM Linear', 'SVM Gauss', 'K-NN', 'Tree predictor'});
grid on;
legend('CD', 'Average Rank');
hold off;
 
 
%% Functions
function cdvalue = criticalDifference(qalpha, nalgs, ndatasets)
    cdvalue = qalpha*sqrt(nalgs*(nalgs+1)/(6*ndatasets));
end
 
 
function param = Tuning(data_tr, labels_tr, data_te, labels_te, algorithm, params, tuning_param, range, step)
    best = 0;
    score = 0;
    for i = range(1):step:range(2)
        % Modify the tuning parameter, we pass momentarily to a struct to
        % search through it like a hashmap with the tuning_param name
        parameters = struct(params{:});
        parameters.(tuning_param) = i;
 
        % Convert parameters struct back to a cell array of name-value
        % pairs; the return values of these functions are both cell arrays,
        % this allows us to concatenate the return values in an alternating
        % manner; the ' simply transposes the cell array
        modified_params = [fieldnames(parameters), struct2cell(parameters)]';
 
        classifier = algorithm(data_tr, labels_tr, modified_params{:});
        prediction = predict(classifier, data_te);
        actual_score = numel(find(prediction == labels_te))/numel(labels_te);
 
        if actual_score > score % > or >= doesn't matter, > is just slightly less computationally expensive
            score = actual_score;
            best = i;
        end
    end
    param = best;
end
 
 
function ranks = findRanks(values, nalgs, ndatasets)
    ranks = zeros(nalgs, ndatasets);
    values = 1-values;
    for ndataset=1:ndatasets
        ranks(:, ndataset) = tiedrank(values(:, ndataset));       
    end
end
 
 
function [idx_tr, idx_te] = twoFoldSampling(labels)
        idx_tr = [];
        idx_te = [];
        for nclass=1:2
            u = find(labels == nclass);
            idx = randperm(numel(u)); % shuffle samples
            idx_tr = [idx_tr; u(idx(1:round(numel(idx)/2)))]; % training set indices
            idx_te = [idx_te; u(idx(1 + round(numel(idx)/2):end))]; % test set indices
        end
end