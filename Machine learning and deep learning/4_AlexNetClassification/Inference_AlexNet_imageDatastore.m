clc;clear;close all;
%% Import model
net = alexnet;

%% Show what AlexNet does with random images without being retrained
samples = imageDatastore('SampleImages',...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Count files in ImageDatastore labels
countEachLabel(samples)

%% Split ImageDatastore labels by proportions
samplespart = splitEachLabel(samples, 2, 'randomized');
countEachLabel(samplespart)

%% Read image from imageDatastore
img = readimage(samplespart, 2);
whos img
 
%% Change ReadFunction in imageDatastore 
% Resize image before reading it
samplespart.ReadFcn = @preprocessImg;

img = readimage(samplespart, 1);
whos img

% Make prediction for one image
classLabel = classify(net, img);

% Show image
imshow(img); 
title(char(classLabel));

%% Make prediction for all images in datastore
datastoreLabel = classify(net, samplespart);

