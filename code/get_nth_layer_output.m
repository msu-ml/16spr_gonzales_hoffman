function [ vect, layer ] = get_nth_layer_output( net, image, n )
% get_nth_layer_output Returns output of nth layer
%   This function passes the input image through the provided CNN,
%   returning the output of the nth layer of the network.
%   The value of n may be negative, meaning the reverse order. For example,
%   n=-1 means the last layer and n=-2 means the 2nd to last layer.
%   The function also returns the positive index of the layer.
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

% Determine the layer if n is negative
if n < 0
    layer = size(net.layers,2) + n + 1;
else
    layer = n;
end


% Return the output of the nth layer
vect = res(layer+1).x;
vect = reshape(vect, numel(vect),1);
end

