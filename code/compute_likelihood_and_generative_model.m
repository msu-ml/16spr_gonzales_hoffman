function [ likelihoods, gen_model ] = compute_likelihood_and_generative_model( ...
    target_spec_sal_maps, bbs, bb_samples, num_target_filters, curr_frame)
%compute_likelihood_and_generative_model Computes the likelihood of each
%sample and the generative model.
%   Computes the likelihood of each sample bounding box. To compute this,
%   the generative model (aka, the target appearance model H_t in the
%   paper) is also generated.
%
%   Input:
%       - target_spec_sal_maps: An [H x W x N] array containing the
%       target-specific saliency maps that have been generated so far where
%       H,W = the height and width respectively of the original input
%       frames and N = the number of frames. Note that only the last
%       num_target_filters number of maps generated before the current
%       frame, and the saliency map for the current frame, will be used in 
%       this function.
%
%       - bbs: An [N x 4] array where N = the number of frames.
%       bbs(i,:) = [X, Y, Wbb, Hbb] is the bounding box where the target object
%       (the object that is being tracked) was found in the ith frame.
%       X,Y indicates the top-left corner of the bounding box and
%       Wbb,Hbb indicate the width and height. Note that all bounding boxes
%       should have the same width and height.
%
%       - bb_samples: An [Nbb, 4] array where Nbb = the number of samples.
%       The format of this variable is the same as that of bbs. However,
%       this holds one bounding box for each of the samples which we want
%       to generate a likelihood for.
%
%       - num_target_filters: This is the max number of target filters (i.e.
%       target-specific saliency maps) to use when generating the model.
%
%       - curr_frame: The number of the frame for which the likelihoods and
%       generative model are currently being computed.
%
%   Output:
%       - likelihoods: An [Nbb x 1] array containing one likelihood 
%       probability for each of the sample bounding boxes.
%
%       - gen_model: An [Hbb x Wbb] array containing the generative model
%       (aka the target appearance model) that was computed in order to
%       find the likelihoods.
%

% Get needed parameters
W = size(target_spec_sal_maps,2);
H = size(target_spec_sal_maps,1);
Wbb = bbs(1,3);
Hbb = bbs(1,4);
Nbb = size(bb_samples,1);

% Update num_target_filters if there haven't been enough frames to have
% that many filters
if (curr_frame - num_target_filters) <= 0
    
    num_target_filters = curr_frame - 1;
end

% Compute the Generative Model
gen_model = zeros(Hbb,Wbb,'uint8');
sm = zeros(H, W, 1, 1);
for k = (curr_frame - num_target_filters):(curr_frame - 1)
    
    sm(:,:,1,1) = target_spec_sal_maps(:,:,k);
    gen_model = gen_model + ...
        squeeze(crop_img_to_bbs(sm, bbs(k,:)));
end
gen_model = (1 / num_target_filters) * gen_model;


% Compute the Likelihoods for each sample
likelihoods = zeros(Nbb,1);
sm(:,:,1,1) = target_spec_sal_maps(:,:,curr_frame);
for i = 1:Nbb
    
    % Because the filter and sample are the same size, convolving the
    % filter with the sample is the same as performing the dot-product on
    % their vectorized form
    sm_sample = squeeze(crop_img_to_bbs(sm, bb_samples(i,:)));
    likelihoods(i) = dot(single(gen_model(:)),single(sm_sample(:)));
end

end

