%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simple_matconvnet_test.m
%
% This Program performs a simple test of MatConvNet to show how to use its
% API and to confirm that it has been correctly installed. It was modified
% from a program taken from http://www.vlfeat.org/matconvnet/quick/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

% load and preprocess an image
im = imread(test_img_path) ;
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;

% run the CNN
res = vl_simplenn(net, im_) ;

% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;