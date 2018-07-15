%% Transfer Learning Using AlexNet
% Fine-tune a pretrained convolutional neural network to learn the features
% on a new collection of images.
%
% Transfer learning is commonly used in deep learning applications. You can
% take a pretrained network and use it as a starting point to learn a new
% task. Fine-tuning a network with transfer learning is much faster and
% easier than training from scratch. You can quickly transfer learning to a
% new task using a smaller number of training images.
clc;clear;close all;
%%
% Load the sample images as |ImageDatastore| objects.
[merchImagesTrain,merchImagesTest] = merchData;

%%
% Load a pretrained AlexNet network.
net = alexnet;

%%
% The last three layers of the pretrained network |net| are configured for
% 1000 classes. These three layers must be fine-tuned for the new
% classification problem. Extract all the layers except the last three from
% the pretrained network, |net|.
layersTransfer = net.Layers(1:end-3);

%%
% Determine the number of classes from the training data.
numClasses = numel(categories(merchImagesTrain.Labels))

% Create the layer array by combining the transferred layers with the new
% layers.
layers = [...
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%%
% Create the training options.
options = trainingOptions('sgdm',...
    'MiniBatchSize',10,...
    'MaxEpochs',4,...
    'InitialLearnRate',1e-4,...
    'ExecutionEnvironment','cpu',...
    'Plots', 'training-progress');

%%
% Fine-tune the network using |trainNetwork| on the new layer array.
netTransfer = trainNetwork(merchImagesTrain,layers,options);

%%
% Classify the test images using |classify|.
predictedLabels = classify(netTransfer,merchImagesTest);

%%
% Calculate the classification accuracy.
testLabels = merchImagesTest.Labels;

accuracy = mean(predictedLabels==testLabels)
