%% Human Activity Learning Using Mobile Phone Data
% Human activity sensor data contains observations derived from
% sensor measurements taken from smartphones worn by people while doing
% different activities (walking, lying, sitting etc). The goal of this 
% example is to provide a strategy to build a classifier that can 
% automatically identify the activity type given the sensor measurements. 
%
% Copyright (c) 2017, MathWorks, Inc.

%% Load Raw Sensor Data for training
close all; clear; clc

load('SensorData.mat', 'rawSensorDataTrain', 'trainActivity');

%% Display data summary
plotRawSensorData(rawSensorDataTrain,trainActivity,1000)

%% Pre-process Training Data: *Feature Extraction*
% The sensor data contain windows of 2.56sec (128 readings/window) 
% Lets start with a simple average feature for every 128 points

T_mean = varfun(@(x) mean(x,2), rawSensorDataTrain);
% T_mean = varfun(@Wmean, rawSensorDataTrain);
T_stdv = varfun(@Wstd, rawSensorDataTrain);
T_pca  = varfun(@Wpca1, rawSensorDataTrain);

humanActivityDataTrain = [T_mean, T_stdv, T_pca];
humanActivityDataTrain.activity = trainActivity;










