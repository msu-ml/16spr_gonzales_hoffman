function [ psm ] = get_pretty_saliency_map( sm )
% get_pretty_saliency_map Creates a pretty, colored saliency map.
%   Takes the grayscale saliency map and converts it into a pretty, colored
%   saliency map.

% Set parameters for the coldest and hottest hue
coldest_hue = 240.0;

% Normalize image
psm = sm / max(max(sm));

% Set the Hue, Saturation, and Value parameters
s = psm;
v = psm;
percentage = 1 - psm;
h = (percentage * coldest_hue) / 360.0;

% Compute the RGB equivalent of the pretty saliency map
psm = zeros(size(sm,1), size(sm,2), 3, 'single');
psm(:,:,1) = h;
psm(:,:,2) = s;
psm(:,:,3) = v;
psm = hsv2rgb(psm);


end

