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
test_img_path = fullfile('..','data','eagle.jpg');

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

%% Load the image, pass it through the CNN, and get saliency map results
img = imread(test_img_path) ;
[feat_vect, layer] = get_first_fully_connected_output(net, img);
dzdx = back_propagate_data(net, img, feat_vect, layer);
sm = compute_class_saliency_map_dzdx(dzdx);

%% Obtain saliency map from last fully connected and strip away other classes
[feat_vect_last, layer_last] = get_nth_layer_output(net, img, -2); % -1 is softmax
[max_val,idx] = max(feat_vect_last);
feat_vect_last = [zeros(idx-1,1,'single') ; max_val ; zeros(numel(feat_vect_last)-idx,1,'single')];
dzdx_last = back_propagate_data(net, img, feat_vect_last, layer_last);
sm_last = compute_class_saliency_map_dzdx(dzdx_last);


%% Display results

% Resize image to be consistent and create pretty saliency map
img_sq = imresize(img, net.meta.normalization.imageSize(1:2)) ;
psm = get_pretty_saliency_map(sm);
psm_last = get_pretty_saliency_map(sm_last);

% Plot results
figure;
subplot(1,2,1);
imshow(img_sq,[]);
subplot(1,2,2);
imshow(psm,[]);
title('Saliency from first fc layer');

% Plot results
figure;
subplot(1,2,1);
imshow(img_sq,[]);
subplot(1,2,2);
imshow(psm_last,[]);
title('Saliency from class-specific last layer');


