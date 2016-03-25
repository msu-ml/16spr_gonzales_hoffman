%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simple_saliency_map_test.m
%
% This program performs a simple test where it passes an input image
% through a CNN to get the output of the first fully connected layer. It
% then back-propagates the data back through the network and creates the
% saliency map for that data input.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parameters
net_file_path = fullfile('..','nets','imagenet-vgg-f.mat');
net_download_path = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat';
test_img_path = fullfile('..','data','peppers.jpg');

%% Setup MatConvNet and Load CNN

% Must run vl_setupnn.m whenever you open matlab in order to use MatConvNet
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'MatConvNet', 'matlab', 'vl_setupnn.m')) ;

% download a pre-trained CNN from the web (needed once)
if ~exist(net_file_path,'file')
    urlwrite(net_download_path, net_file_path) ;
end

% load the pre-trained CNN
net = load(net_file_path) ;

%% Load the image and pass it through the CNN
img = imread(test_img_path) ;
[feat_vect, layer] = get_first_fully_connected_output(net, img);

back_prop = back_propagate_data(net, img, feat_vect, layer);

%% TODO: Convert back propogation result into saliency map using process talked about in the paper.
