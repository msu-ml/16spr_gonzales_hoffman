function [feature_length, layer] = get_first_fully_connected_feature_length( net )
%get_first_fully_connected_feature_length Gives the number of features
%returned by the first fully connected layer
%   This function determines the number of features that will be returned
%   by the first fully connected layer of the provided network. It also
%   returns the index of the first fully connected layer.

% Get a dummy image (just needs to be right size)
dummy_img = zeros(size(net.meta.normalization.averageImage),'single');

% run the CNN
res = vl_simplenn(net, dummy_img) ;

% Find the first fully connected layer
for layer=1:length(res)
    if (sum((size(res(layer).x) == 1)) >= 2)
        break;
    end
end

% Return the number of features returned by the first fully connected layer
feature_length = numel(res(layer).x);

% Subtract 1 from layer because res(1) is the input here, not a layer of
% the network
layer = layer - 1;

end

