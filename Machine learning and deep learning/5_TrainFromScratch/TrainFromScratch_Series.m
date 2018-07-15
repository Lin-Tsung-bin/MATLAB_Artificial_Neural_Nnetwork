%% Create Simple Deep Learning Network for Classification
%%
% This example shows how to create and train a simple convolutional neural
% network for deep learning classification. Convolutional neural networks
% are essential tools for deep learning, and are especially suited for
% image recognition. Learn how to set up network layers, image data, and
% training options, train the network, and test the classification
% accuracy.

%% Load and Explore the Image Data
clc; clear; close all;
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Check the number of images in each category.
CountLabel = digitData.countEachLabel

%% Specify Training and Test Sets
trainingNumFiles = 750;
rng(1) % For reproducibility
[trainDigitData,testDigitData] = splitEachLabel(digitData, ...
    trainingNumFiles, 'randomize');

%% Need a starting point? Check the documentation!
% search "deep learning"
web(fullfile(docroot, 'nnet/deep-learning-training-from-scratch.html'))

%% Define the Network Layers
% Define the convolutional neural network architecture.

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(5, 20)
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% Specify the Training Options
options = trainingOptions(...
    'sgdm',...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128,...
    'InitialLearnRate', 0.01,...
    'ExecutionEnvironment', 'cpu',...
    'Plots', 'training-progress');

%% Train the Network Using Training Data
convnet = trainNetwork(trainDigitData, layers, options);

% It uses a GPU by default if there is one available (requires
% Parallel Computing Toolbox (TM) and a CUDA-enabled GPU with compute
% capability 3.0 and higher).

%======================================================%
%% Decrease initial learning rate 
options = trainingOptions(...
    'sgdm',...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128,...
    'InitialLearnRate', 0.0001,...
    'ExecutionEnvironment', 'cpu',...
    'Plots', 'training-progress');

%% Train the Network Using Training Data
convnet = trainNetwork(trainDigitData, layers, options);

%% Classify the Images in the Test Data and Compute Accuracy
predictedLabels  = classify(convnet, testDigitData);
valLabels  = testDigitData.Labels;

%% Calculate the accuracy.
accuracy = mean(predictedLabels == valLabels)

%% Calculate the confusion matrix
figure
[cmat,classNames] = confusionmat(valLabels,predictedLabels);
h = heatmap(classNames,classNames,cmat);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix');

%======================================================%
%% Add network validation during training
% Specify the Training Options
options = trainingOptions(...
    'sgdm',...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128,...
    'InitialLearnRate', 0.0001,...
    'ExecutionEnvironment', 'cpu',...
    'Plots', 'training-progress',...
    'ValidationData', testDigitData,...
    'ValidationFrequency', 30);

%% Train the Network Using Training Data
convnet = trainNetwork(trainDigitData, layers, options);
