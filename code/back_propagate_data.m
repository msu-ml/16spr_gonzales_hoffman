function [ dzdx ] = back_propagate_data( net, image, dzdy, start_layer )
% back_propagate_data Back propagates dzdy through net, starting at start_layer
%   This function back-propagates the data from dzdy, starting at the
%   specified layer and flowing backwards through the network back to the
%   input layer. It returns the change in the output in respect to the
%   change in the input, dzdx, which is of the same dimensionality as the
%   original image. The parameter image should be the image originally used
%   to create dzdy. It assumes that the image has already been loaded to  
%   memory but has not had the mean subtracted from it. This function  
%   will resize the image to the appropriate size for the CNN and subtract 
%   the mean out. A few empirical results show that providing instead an
%   array of zeros the same size as the original image should produce a
%   similar dzdx.

%% Find the data inputs (x) to each layer, captured in res().x
% Must run vl_setupnn.m whenever you open matlab in order to use MatConvNet
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'MatConvNet', 'matlab', 'vl_setupnn.m')) ;

% load and preprocess an image
im_ = single(image) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;

% run the CNN
res = vl_simplenn(net, im_) ;

% Confirm that dzdy is in the correct shape for the network
dzdy = reshape(dzdy, size(res(start_layer+1).x));

%% Back-Propagate the data through the various layers
for layer = start_layer:-1:1
    
    % Determine the type of the current layer and back-propagate
    layer_info = net.layers{1,layer};
    type = layer_info.type;
    if strcmp(type, 'conv') % Convolutional layer
        
        x = res(layer).x;
        w = layer_info.weights{1,1};
        b = layer_info.weights{1,2};
        stride = layer_info.stride;
        pad = layer_info.pad;
        [dzdx, dzdw] = vl_nnconv(x, w, b, dzdy, 'Stride', stride, 'Pad', pad);
        
    elseif strcmp(type, 'pool') % Pooling layer
        
        x = res(layer).x;
        pool = layer_info.pool;
        method = layer_info.method;
        stride = layer_info.stride;
        pad = layer_info.pad;
        dzdx = vl_nnpool(x, pool, dzdy, 'Stride', stride, 'Pad', pad, 'Method', method);
    
    elseif strcmp(type, 'relu') % RELU layer
        
        x = res(layer).x;
        leak = layer_info.leak;
        dzdx = vl_nnrelu(x, dzdy, 'Leak', leak);
        
    elseif strcmp(type, 'lrn') % Local Response Normalization layer
        
        x = res(layer).x;
        param = layer_info.param;
        dzdx = vl_nnnormalize(x, param, dzdy);
        
    elseif strcmp(type, 'softmax') % Softmax layer
        
        x = res(layer).x;
        dzdx = vl_nnsoftmax(x, dzdy);
        
    elseif strcmp(type, 'dropout') % Dropout layer
        
        x = res(layer).x;
        rate = layer_info.rate;
        dzdx = vl_nndropout(x, dxdy, 'Rate', rate);
    end
    
    % Update so that dzdy for next layer is the dzdx of this layer
    dzdy = dzdx;

end

