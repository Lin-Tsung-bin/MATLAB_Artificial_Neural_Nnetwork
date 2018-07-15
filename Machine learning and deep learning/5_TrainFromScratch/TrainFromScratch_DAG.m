%% Create Simple Deep Learning Network for Classification
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
CountLabel = digitData.countEachLabel;

%% Specify Training and Test Sets
trainingNumFiles = 750;
rng(1) % For reproducibility
[trainDigitData,testDigitData] = splitEachLabel(digitData, ...
    trainingNumFiles, 'randomize');

%% Define the Network Layers
% Create the main branch of the network as a layer array. The addition
% layer sums multiple inputs element-wise. Specify the number of inputs
% that the addition layer should sum. All layers must have names and all
% names must be unique.
layers = [
    imageInputLayer([28 28 1],'Name','input')
    
    convolution2dLayer(5,16,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    convolution2dLayer(3,32,'Padding','same','Stride',2,'Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    convolution2dLayer(3,32,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    
    additionLayer(2,'Name','add')
    
    averagePooling2dLayer(2,'Stride',2,'Name','avpool')
    fullyConnectedLayer(10,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')];

%%
% Create a layer graph from the layer array. |layerGraph| connects all the
% layers in |layers| sequentially. Plot the layer graph.
lgraph = layerGraph(layers);

figure
plot(lgraph)

%%
% Create the 1-by-1 convolutional layer and add it to the layer graph.
% Specify the number of convolutional filters and the stride so that the
% activation size matches the activation size of the |'relu_3'| layer. This
% enables the addition layer to add the outputs of the |'skipConv'| and
% |'relu_3'| layers. To check that the layer has been added, plot the layer
% graph.
skipConv = convolution2dLayer(1,32,'Stride',2,'Name','skipConv');
lgraph = addLayers(lgraph,skipConv);
figure
plot(lgraph)

%%
% Create the shortcut connection from the |'relu_1'| to the |'add'| layer.
% Because you specified the number of inputs to the addition layer to be
% two when you created the layer, the layer has two inputs with the names
% |'in1'| and |'in2'|. The |'relu_3'| layer is already connected to the
% |'in1'| input. Connect the |'relu_1'| layer to the |'skipConv'| layer and
% the |'skipConv'| layer to the |'in2'| input of the |'add'| layer. The
% addition layer now sums the outputs of the |'relu_3'| and |'skipConv'|
% layers. To check that the layers are correctly connected, plot the layer
% graph.
lgraph = connectLayers(lgraph,'relu_1','skipConv');
lgraph = connectLayers(lgraph,'skipConv','add/in2');
figure
plot(lgraph);

%%
% Specify training options and train the network. |trainNetwork| validates
% the network using the validation data every |ValidationFrequency|
% iterations.
options = trainingOptions('sgdm',...
    'MaxEpochs',6,...
    'ExecutionEnvironment', 'cpu',...
    'Plots', 'training-progress');

%% Train the Network Using Training Data
convnet = trainNetwork(trainDigitData, lgraph, options);

% It uses a GPU by default if there is one available (requires
% Parallel Computing Toolbox (TM) and a CUDA-enabled GPU with compute
% capability 3.0 and higher).

%% Classify the Images in the Test Data and Compute Accuracy
predictedLabels  = classify(convnet, testDigitData);
valLabels  = testDigitData.Labels;

%% Calculate the accuracy.
accuracy = mean(predictedLabels == valLabels)

