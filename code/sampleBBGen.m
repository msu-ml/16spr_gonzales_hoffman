function [ bbSamples ] = sampleBBGen( bb, N )
%samplePatchesGen This function generates N normally distributed bounding boxes 
%   Given the mean bounding box bb, this method generates N bounding boxes
%   from a normal distribution with mean of bb and standard deviation of
%   sqrt(bb(3)*bb(4))/2. Thus, the return value is an [N x 4] matrix.
%   Note that the first sample will always be the provided bounding box bb,
%   such that bbSamples(1,:) = bb.

% With initial bounding box information we can build samples in a
% normal distribution around the center.
x = bb(1);
y = bb(2);
w = bb(3);
h = bb(4);
samples = zeros(N,2);
bbSamples = zeros(N,4);
bbSamples(1,:) = bb;
for i = 2:N
    samples(i,:) = round(normrnd([x y], (sqrt(w*h)/2))); %Sample coords    
    bbSamples(i,:) = [samples(i,1), samples(i,2), w , h];
end
     
end

