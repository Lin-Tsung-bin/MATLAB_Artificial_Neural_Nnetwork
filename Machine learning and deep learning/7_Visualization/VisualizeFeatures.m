%% Visualize Features of a Convolutional Neural Network
% Visualize what the learned features look like by using
% "deepDreamImage" to generate images that
% strongly activate a particular channel of the network layers.
clc;clear;close all;
%% Load Pretrained Network
% Load a pretrained AlexNet network.
net = alexnet;

%% Visualize Convolutional Layers
% There are five 2-D convolutional layers in the AlexNet network.
% Notice that the 2-D convolutional layers are layers
% 2, 6, 10, 12, and 14.
net.Layers

%% *Features on Convolutional Layer 1*
% Set |layer| to be the first convolutional layer. This layer is the second
% layer in the network and is named |'conv1'|.
layer = 2;
name = net.Layers(layer).Name

% |deepDreamImage|
channels = 1:56;
I = deepDreamImage(net,layer,channels, ...
    'PyramidLevels',1);

figure
montage(I)
title(['Layer ',name,' Features'])

%% *Features on Convolutional Layers 5*
layer = 14;
channels = 1:30;
I = deepDreamImage(net,layer,channels,...
    'PyramidLevels',1);

figure
montage(I)
name = net.Layers(layer).Name;
title(['Layer ',name,' Features'])

%% Visualize Fully Connected Layers
% There are three fully connected layers in the AlexNet model.
% To produce images that resemble each class the most closely, select the
% final fully connected layer, and set |channels| to be the indices of the
% classes.
layer = 23;
channels = [9 188 231 563];

% You can view the names of the selected classes by
% selecting the entries in |channels|.
net.Layers(end).ClassNames(channels)


% Generate detailed images that strongly activate these classes.
I = deepDreamImage(net,layer,channels, ...
    'NumIterations',50);

figure
montage(I)
name = net.Layers(layer).Name;
title(['Layer ',name,' Features'])

% ==============================================%
%% Visualize Activations of a Convolutional Neural Network
% Shows how to feed an image to a convolutional neural network
% and display the activations of different layers of the network.
clear;
net = alexnet;

% Read and show an image. Save its size for future use.
im = imread(fullfile(matlabroot,'examples','nnet','face.jpg'));
imshow(im)
imgSize = size(im);
imgSize = imgSize(1:2);

%% Show Activations of First Convolutional Layer
% Investigate features by observing which areas in the convolutional layers
% activate on an image.
act1 = activations(net,im,'conv1','OutputAs','channels');

%Reshape the array to 4-D. 
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);

%Normalize the output using |mat2gray|. All activations are scaled so that
% the minimum activation is 0 and the maximum is 1.
montage(mat2gray(act1))

%% Visualize the activations of the relu5 layer.
% Investigate channels 3 and 5 further.
act5relu = activations(net,im,'relu5','OutputAs','channels');
sz = size(act5relu);
act5relu = reshape(act5relu,[sz(1) sz(2) 1 sz(3)]);

montage(imresize(mat2gray(act5relu(:,:,:,[3 5])),imgSize))
