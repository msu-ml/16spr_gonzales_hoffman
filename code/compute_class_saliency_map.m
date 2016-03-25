function [ sm ] = compute_class_saliency_map( net, image, dzdy, start_layer )
% compute_class_saliency_map Computes the class saliency map.
%   This function computes the class saliency map as defined in 
%   "Deep Inside Convolutional Networks: Visualising Image Classification 
%   Models and Saliency Maps" by Simonyan et al. It does this by first
%   obtaining the result from the function back_propagate_data() and then
%   takes the maximum absolute pixel value for the three color channels at
%   each pixel location. This results in the saliency map.

dzdx = back_propagate_data( net, image, dzdy, start_layer );
sm = compute_class_saliency_map_dzdx(dzdx);


end

