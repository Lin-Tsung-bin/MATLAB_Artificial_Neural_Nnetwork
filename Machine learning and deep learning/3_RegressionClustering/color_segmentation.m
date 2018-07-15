%% Color-Based Segmentation Using K-Means Clustering
close all; clear; clc

% Read Image
RGB_I = imread('yellowlily.jpg');

imshow(RGB_I)
title('RGB true-color image');

%% Convert Image from RGB Color Space to L*a*b* Color Space
% The L*a*b* space consists of :
% luminosity layer 'L*', 
% chromaticity-layer 'a*'indicating where color falls along the red-green axis
% chromaticity-layer b*' indicating where color falls along the blue-yellow axis.

LAB_I = rgb2lab(RGB_I);

figure,
subplot(1,3,1)
imshow(LAB_I(:,:,1),[]), title('L*(intensity)');

subplot(1,3,2)
imshow(LAB_I(:,:,2),[]), title('a*(red-green)');

subplot(1,3,3)
imshow(LAB_I(:,:,3),[]), title('b*(blue-yellow)');

%% Classify the Colors in 'a*b*' Space Using K-Means Clustering
% Clustering is a way to separate groups of objects.  K-means clustering treats
% each object as having a location in space. It finds partitions such that
% objects within each cluster are as close to each other as possible, and as far
% from objects in other clusters as possible.

ab = double(LAB_I(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 3;

[cluster_idx, cluster_center] = kmeans(ab, nColors,...
    'distance','sqeuclidean', 'Replicates', 10);

%% clusters visualization                                  
figure,
scatter(ab(:,1),ab(:,2),[],cluster_idx,'.')
hold on
plot(cluster_center(1,1),cluster_center(1,2),'Color','red','Marker','*','MarkerSize',12)
plot(cluster_center(2,1),cluster_center(2,2),'Color','red','Marker','*','MarkerSize',12)
plot(cluster_center(3,1),cluster_center(3,2),'Color','red','Marker','*','MarkerSize',12)
xlabel('a*');
ylabel('b*');
%% Label Every Pixel in the Image Using the Results from KMEANS
% Label every pixel in the image with its |cluster_index|.

pixel_labels = reshape(cluster_idx,nrows,ncols);

figure,
imshow(pixel_labels,[])
title('image labeled by cluster index');

%% Create Images that Segment the flower Image by Color.

% initiate space for saving three different segmented images
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = RGB_I;
    color(rgb_label ~= k) = 255;
    segmented_images{k} = color;
end

figure,
subplot(1,3,1)
imshow(segmented_images{1}), title('objects in cluster 1');

subplot(1,3,2)
imshow(segmented_images{2}), title('objects in cluster 2');

subplot(1,3,3)
imshow(segmented_images{3}), title('objects in cluster 3');

