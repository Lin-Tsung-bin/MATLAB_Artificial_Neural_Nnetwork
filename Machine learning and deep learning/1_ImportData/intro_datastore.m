%% Using Datastore in MATLAB
% In this example will be looking at a large amount of vibration
% measurement data.

%% What is a DataStore?
% A |datastore| is an object useful for reading collections of data that are
% too large to fit in memory. It is a repository for multiple files and
% folders with the same structure and formatting.

%% Creating the DataStore
% By default, the datastore will use the first line as your variable
% names, since our file does not contain variable names - we specify some manually.
close all; clear; clc

datapath = fullfile(pwd, 'Subset');
ds = datastore(datapath,'Type','Tabulartext','Delimiter','\t','ReadVariableNames',false);         
ds.VariableNames = {'Ch1','Ch2','Ch3','Ch4','Ch5','Ch6','Ch7','Ch8'};

%% Preview the Data
a = preview(ds)

%% Select the variable of interest
% If we are only interested in one variable, we can discard the rest
ds.SelectedVariableNames = 'Ch4';

%% Read in First Chunk
% the default read size is 20000 rows
testdata = read(ds);
whos testdata

%% Read one file at a time
% We can also read one file at a time
ds.ReadSize = 'file';

%% Reset the datastore  
% If we want to start reading from the beginning, we need to reset the
% datastore
reset(ds);  % Start at first record again

%% Visualizing and analyzing data
% With the help of datastore, we can continuously read and analyze data.
warning off %suppress warnings

% while there's data in datastore
while hasdata(ds)

    % Read in Chunk
    dataChunk = read(ds);
    
    figure,
    plot(0:height(dataChunk)-1, dataChunk.Ch4);
    axis tight
    title('Time series data')
   end
