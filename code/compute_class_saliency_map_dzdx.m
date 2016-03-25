function [ sm ] = compute_class_saliency_map_dzdx( dzdx )
% compute_class_saliency_map_dzdx Computes the class saliency map.
%   This function computes the class saliency map as defined in 
%   "Deep Inside Convolutional Networks: Visualising Image Classification 
%   Models and Saliency Maps" by Simonyan et al. It does this by taking
%   the result from the function back_propagate_data() as a parameter and then
%   takeing the maximum absolute pixel value for the three color channels at
%   each pixel location of that parameter. This results in the saliency map.

sm = abs(dzdx);
sm = max(sm, [], 3);

end

