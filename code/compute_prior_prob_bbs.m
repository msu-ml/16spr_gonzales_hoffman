function [ priors ] = compute_prior_prob_bbs(bb_samples, bb_pos_samples, bb_samples_lf, bb_target_lf, posteriors_lf )
%compute_prior_prob_bb Computes the prior probability of each BB containing
%the object that is being tracked
%   This method computes the prior probability of each of the sample
%   bounding boxes as containing the object that is being tracked.
%
%   Input:
%       - bb_samples: An [N x 4] array where N = the number of samples.
%       bb_samples(i,:) = [X, Y, W, H] is the bounding box of the ith
%       sample. X,Y indicates the top-left corner of the bounding box and
%       W,H indicate the width and height. Note that all bounding boxes
%       should have the same width and height.
%
%       - bb_pos_samples: A [P x 4] array of same format as bb_samples.
%       However, this array only contains the bounding boxes of the P
%       positive samples.
%
%       - bb_samples_lf: An [N x 4] array of the same format as bb_samples.
%       However, this array contains the bounding boxes from samples
%       generated from the previous, i.e. last, frame.
%
%       - bb_target_lf: A [1 x 4] array of the same format as bb_samples.
%       This array contains the bounding box that was determined to contain
%       the target object (the object being tracked) in the previous, i.e.
%       last, frame.
%
%       - posteriors_lf: An [N x 1] array containing the posterior
%       probability for each of the sample bounding boxes from the
%       previous, i.e. last, frame.
%
%   Output:
%       - priors: An [N x 1] array containing one prior probability for
%       each of the sample bounding boxes.

% Get needed parameters
N = size(bb_samples,1);

% Compute the displacement vector and noise covariance matrix of positive
% samples
mu = mean(bb_pos_samples(:,1:2),1);
Sigma = cov(bb_pos_samples(:,1:2));
d = mu - bb_target_lf(1:2);

% Loop through all samples, computing a prior for each sample
priors = zeros(N,1);
for i = 1:N
   
    % Get the X,Y coordinates of current sample BB
    coord = bb_samples(i,1:2);
    
    % Loop through all the samples from the previous/last frame
    for j = 1:N
       
        coord_lf = bb_samples_lf(j,1:2);
        priors(i) = priors(i) + ...
            mvnpdf(coord - coord_lf, d, Sigma) * posteriors_lf(j);
    end
end

end

