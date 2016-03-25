function [ vect, layer ] = get_first_fully_connected_output( net, image )
% get_first_fully_connected_output Returns output of 1st fc layer
%   This function passes the input image through the provided CNN,
%   returning the output of the first fully connected layer of the network.
%   It also returns the index of the first fully connected layer.
%   It assumes that the image has already been loaded to memory but has not 
%   had the mean subtracted from it. This function will resize the image to 
%   the appropriate size for the CNN and subtract the mean out. The output
%   is a column vector.

% Must run vl_setupnn.m whenever you open matlab in order to use MatConvNet
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'MatConvNet', 'matlab', 'vl_setupnn.m')) ;

% load and preprocess an image
im_ = single(image) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;

% run the CNN
res = vl_simplenn(net, im_) ;

% Find the first fully connected layer
for layer=1:length(res)
    if (sum((size(res(layer).x) == 1)) >= 2)
        break;
    end
end

% Return the output of the first fully connected layer
vect = res(layer).x;
vect = reshape(vect, numel(vect),1);

% Subtract 1 from layer because res(1) is the input here, not a layer of
% the network
layer = layer - 1;
end

