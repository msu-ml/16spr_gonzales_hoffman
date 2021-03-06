function [ tssm ] = compute_target_specific_saliency_map( frameSize, csm, bb_pos_samples )
%compute_target_specific_saliency_map Generates the Target-Specific
%Saliency Map
%   This function computes the target-specific saliency map by aggregating
%   the class saliency maps, taking the largest absolute value of all the
%   saliency maps at every pixel.
%
%   Input:
%       - frameSize: An array containing the size of the original video
%       frame from which the class saliency maps were generated. The first
%       two dimensions of frameSize should be [H x W], where H = the height
%       of the frame and W = the width of the frame.
%
%       - csm: The class saliency maps. This should be an array of size
%       [Hcsm x Wcsm x N], where Hcsm = the height of every class saliency
%       map, Wcsm = the width of every class saliency map, and N = the
%       number of class saliency maps.
%
%       - bb_pos_samples: The bounding boxes of describing where each class
%       saliency map was extracted from the original video frame. It should
%       be an array of size [N x 4]. The ith bounding box should be 
%       bb_pos_samples(i,:) = [X x Y x Wcsm x Hcxm], where X,Y are the
%       coordinates of the top-left corner of the bounding box.
%
%   Output:
%       - tssm: The target-specific saliency map of size [H x W].

% Get needed info from parameters
H = frameSize(1);
W = frameSize(2);
Hcsm = size(csm,1);
Wcsm = size(csm,2);
N = size(csm,3);

% Initialize target-specific saliency map
tssm = zeros(H,W);

% Loop throught all class saliency maps, considering their values for the
% Target-specific saliency map
for i = 1:N
  
    % Get the bounding box of current csm
    x = bb_pos_samples(i,1);
    y = bb_pos_samples(i,2);
    Wcsm_i = Wcsm;
    Hcsm_i = Hcsm;
    
    % Adjust bb if it is to the left of the image frame
    if x < 1
        diff = 1 - x;
        x = 1;
        Wcsm_i = Wcsm - diff;
    end
    
    % Adjust bb if it is above the imagge frame
    if y < 1
        diff = 1 - y;
        y = 1;
        Hcsm_i = Hcsm - diff;
    end
    
    % Adjust bb if it is to the right of the image frame
    if (x + Wcsm - 1) > W
        Wcsm_i = W - x + 1;
    end
    
    % Adjust bb if it is to the right of the image frame
    if (y + Hcsm - 1) > H
        Hcsm_i = H - y + 1;
    end
    
    % Add the class saliency map info to the target-saliency map
    for r = 1:Hcsm_i
        for c = 1:Wcsm_i
            
            tssm(y+r-1, x+c-1) = max(tssm(y+r-1, x+c-1), abs(csm(r,c,i)));
        end
    end
end

end

