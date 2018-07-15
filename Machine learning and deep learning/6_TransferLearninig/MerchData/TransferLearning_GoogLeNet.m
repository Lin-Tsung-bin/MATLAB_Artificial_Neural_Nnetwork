%% Transfer Learning Using GoogLeNet
% This example shows how to use transfer learning to retrain GoogLeNet, a pretrained convolutional
% neural network, to classify a new set of images.
clc;clear;close all;
%%
% Load the sample images as |ImageDatastore| objects.
[merchImagesTrain, merchImagesTest] = merchData;
merchImagesTrain.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
merchImagesTest.ReadFcn = @(loc)imresize(imread(loc),[224,224]);

%%
% Load the pretrained GoogLeNet network.
net = googlenet;

%%
% Extract the layer graph from the trained network and plot the layer
% graph.
lgraph = layerGraph(net);
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
plot(lgraph)

%%
% Replace the last three layers of the network.
lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});

numClasses = numel(categories(merchImagesTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

%%
% Connect the last of the transferred layers remaining in the network
% to the new layers.
lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%%
% Specify the training options, including learning rate, mini-batch size,
% and validation data.
options = trainingOptions('sgdm',...
    'MiniBatchSize',10,...
    'MaxEpochs',3,...
    'InitialLearnRate',1e-4,...
    'ExecutionEnvironment','cpu',...
    'Plots','training-progress');

%%
% Train the network using the training data.
net = trainNetwork(merchImagesTrain,lgraph,options);

%%
% Classify the validation images using the fine-tuned network, and
% calculate the classification accuracy.
predictedLabels = classify(net,merchImagesTest);
accuracy = mean(predictedLabels == merchImagesTest.Labels)
