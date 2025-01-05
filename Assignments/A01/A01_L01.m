clc

%% Starting parameters
nalgs = 4;
ndatasets = 4;
qalpha = 2.569;
accuracy_5x2=zeros(nalgs, ndatasets);

%% Tuning parameters

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
    %% Define algorithm parameters
    params_svm_linear = {'KernelFunction', 'linear', 'Kernelscale', 1};
    params_svm_gaussian = {'KernelFunction', 'gaussian', 'Kernelscale', 0.3};
    params_knn = {'Distance', 'Euclidean', 'NumNeighbors', 10};
    params_tree = {'SplitCriterion', 'gdi', 'MaxNumSplits', 15};
    
    %% Define tuning parameters
    range_svm_linear = [0.1, 2]; % example range for linear SVM kernel scale
    range_svm_gaussian = [0.1, 2]; % example range for Gaussian SVM kernel scale
    range_knn = [1, 20]; % example range for number of neighbors in kNN
    range_tree = [5, 20]; % example range for maximum number of splits in decision tree
    
    %% Training classifiers
    sum_param_svm_linear = sum_param_svm_linear + Tuning(data_tr, labels_tr, data_te, labels_te, @fitcsvm, params_svm_linear, 'Kernelscale', range_svm_linear, 0.1);
    sum_param_svm_gaussian = sum_param_svm_gaussian + Tuning(data_tr, labels_tr, data_te, labels_te, @fitcsvm, params_svm_gaussian, 'Kernelscale', range_svm_gaussian, 0.1);
    sum_param_knn = sum_param_knn + Tuning(data_tr, labels_tr, data_te, labels_te, @fitcknn, params_knn, 'NumNeighbors', range_knn, 1);
    sum_param_tree = sum_param_knn + Tuning(data_tr, labels_tr, data_te, lax    bels_te, @fitctree, params_tree, 'MaxNumSplits', range_tree, 1);
end

tuned_param_svm_linear = sum_param_svm_linear / ndatasets
tuned_param_svm_gaussian = sum_param_svm_gaussian / ndatasets
tuned_param_knn = sum_param_knn / ndatasets
tuned_param_tree = sum_param_tree / ndatasets


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
    
    for ntimes = 1:5
        
        %% Stratified sampling
        [idx_tr, idx_te] = twoFoldSampling(labels);

        labels_tr = labels(idx_tr); % here we have the solutions
        labels_te = labels(idx_te); % also here
        data_tr = data(idx_tr,:); % access to data, go to row idx_tr and take all the values of the columns
        data_te = data(idx_te,:);


        %% Training classifiers
        % current_classifier = fitcsvm(data_tr, labels_tr, data_te, labels_te, 'KernelFunction', 'linear', 'Kernelscale', 1);
        
        for nalg = 1:4
            switch nalg
                case 1, current_classifier = fitcsvm(data_tr, labels_tr, 'KernelFunction', 'linear', 'Kernelscale', tuned_param_svm_linear);
                case 2, current_classifier = fitcsvm(data_tr, labels_tr, 'KernelFunction', 'gaussian', 'Kernelscale', tuned_param_svm_gaussian); % radial basis function
                case 3, current_classifier = fitcknn(data_tr, labels_tr, 'Distance', 'Euclidean', 'NumNeighbors', tuned_param_knn); % using euclidean distance
                case 4, current_classifier = fitctree (data_tr, labels_tr, 'SplitCriterion','gdi', 'MaxNumSplits', tuned_param_tree); % decision tree, setting the number pf splits and the split criterion
                otherwise
            end
            

            prediction = predict(current_classifier, data_te);
            
            % find all the times the prediction is equal to the label, and count it,
            % then divide by the total number in order to have a percentage:
            accuracy1 = numel(find(prediction == labels_te))/numel(labels_te);
            
    
            % reversing role of training and test:
            % train on test split, test on train split
            switch nalg
                case 1, current_classifier = fitcsvm(data_tr, labels_tr, 'KernelFunction', 'linear', 'Kernelscale', tuned_param_svm_linear);
                case 2, current_classifier = fitcsvm(data_tr, labels_tr, 'KernelFunction', 'gaussian', 'Kernelscale', tuned_param_svm_gaussian); % radial basis function
                case 3, current_classifier = fitcknn(data_tr, labels_tr, 'Distance', 'Euclidean', 'NumNeighbors', tuned_param_knn); % using euclidean distance
                case 4, current_classifier = fitctree (data_tr, labels_tr, 'SplitCriterion','gdi', 'MaxNumSplits', tuned_param_tree); % decision tree, setting the number pf splits and the split criterion
                otherwise
            end
            prediction = predict(current_classifier, data_te);
            
            accuracy2 = numel(find(prediction == labels_tr))/numel(labels_tr);
            
            % fpritf(''%d)
            mean_accuracy = (accuracy1 + accuracy2)/2;
            accuracy_times(nalg, ntimes) = mean_accuracy; 
        end
    end
    %fprintf("\n\nDataset %d\n", ndataset);
    accuracy_times;
    for nalg=1:4
        accuracy_5x2(nalg, ndataset) = mean(accuracy_times(nalg, :));
    end 
end
accuracy_5x2

%% Find ranks and mean ranks

%find ranks, descending order
ranks = findRanks(accuracy_5x2, nalgs, ndatasets)

% Compute average ranks
average_rank = [];
for nalg=1:4
    average_rank(nalg) = mean(ranks(nalg,:));
end

average_rank

% Compute critical value
cdvalue = criticalDifference(qalpha, nalgs, ndatasets);

%% Plot
figure;
hold on;

%plot CD lines
for nalg=1:nalgs
    plot([average_rank(nalg)-cdvalue, average_rank(nalg)+cdvalue], [nalg, nalg], 'LineWidth',2);
end

%plot average rank dots
scatter(average_rank, 1:nalg, 'filled');

% plot personalization
xlabel('Average Rank');
ylabel('Algorithms');
title('Critical Difference Diagram');
yticks(1:nalgs);
yticklabels({'Algorithm 1', 'Algorithm 2', 'Algorithm 3', 'Algorithm 4'});
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
        % Constructing parameters struct
        parameters = struct(params{:});
        % Modify the tuning parameter
        parameters.(tuning_param) = i;
        % Convert parameters struct back to a cell array of name-value pairs
        modified_params = [fieldnames(parameters), struct2cell(parameters)]';
        
        % Call the algorithm with the constructed parameters
        classifier = algorithm(data_tr, labels_tr, modified_params{:});
        prediction = predict(classifier, data_te);
        actual_score = numel(find(prediction == labels_te))/numel(labels_tr);

        if actual_score >= score
            score = actual_score;
            best = i;
        end
    end
    param = best;
end



function ranks = findRanks(values, nalgs, ndatasets)
    ranks = zeros(nalgs, ndatasets);
    for ndataset=1:ndatasets
        [sortedArray, sortedIndices] = sort(values(:, ndataset), 'descend');
        ranks(sortedIndices, ndataset) = 1:ndatasets;

        for nalg = 1:nalgs-1
            if sortedArray(nalg) == sortedArray(nalg+1)
                ranks(sortedIndices(nalg+1)) = ranks(nalg);
            end
        end
    end
end




function [idx_tr, idx_te] = twoFoldSampling(labels)
        idx_tr = [];
        idx_te = [];
        for nclass=1:2
            u = find(labels == nclass);
            idx = randperm(numel(u)); % creating the samples
            idx_tr = [idx_tr; u(idx(1:round(numel(idx)/2)))]; % training set, ; means append
            idx_te = [idx_te; u(idx(1 + round(numel(idx)/2):end))]; % test set
        end
end


