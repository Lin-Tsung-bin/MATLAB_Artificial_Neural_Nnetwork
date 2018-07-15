%% Human Activity Learning Using Mobile Phone Data
% Human activity sensor data contains observations derived from
% sensor measurements taken from smartphones worn by people while doing
% different activities (walking, lying, sitting etc). The goal of this 
% example is to provide a strategy to build a classifier that can 
% automatically identify the activity type given the sensor measurements. 
%
% Copyright (c) 2017, MathWorks, Inc.
%%
clc;clear;close all;
DataPreprocess;

%% Use the new features to train a model and assess its performance
classificationLearner

%% Load Raw Sensor Data for testing
load('SensorData.mat', 'rawSensorDataTest', 'testActivity');

%% Test classifier performance on new data
% Extract features from raw sensor data
T_mean = varfun(@Wmean, rawSensorDataTest);
T_stdv = varfun(@Wstd , rawSensorDataTest);
T_pca  = varfun(@Wpca1, rawSensorDataTest);

humanActivityDataTest = [T_mean, T_stdv, T_pca];
% humanActivityDataTest.activity = testActivity;
humanActivityDataTest.Properties.VariableNames = humanActivityDataTrain.Properties.VariableNames(1:end-1);

%%
% Step 3: Use trained model to predict activity on new sensor data
PredictActivity = trainedModel.predictFcn(humanActivityDataTest);

%% Plot the confusion matrix 
confMat0 = confusionmat(testActivity, PredictActivity);
confMat = bsxfun(@rdivide,confMat0,sum(confMat0,2));

figure,
h = heatmap(categories(testActivity),categories(testActivity),confMat);
h.FontSize = 12;
h.CellLabelFormat = '%.2f';
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
h.Title = ['Accurancy on validation set:',sprintf('%.2f',sum(diag(confMat0))/sum(sum(confMat0)))];

%%
plotActivityResults(trainedModel,rawSensorDataTest,humanActivityDataTest,testActivity,0.05)
