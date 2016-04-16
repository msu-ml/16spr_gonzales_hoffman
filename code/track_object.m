%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% track_object.m
%
% This program is meant to perform the object tracking algorithm originally
% proposed by Hong, et al. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ------------------------------------------------------------------------
%                                                         Setup Parameters
% -------------------------------------------------------------------------
% Location of saved CNN/Location to save CNN
net_file_path = fullfile('..','nets','imagenet-vgg-f.mat'); 

% Location to download CNN if doesn't exist
net_download_path = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat'; 

% The folder containing the frames of the video in which to perform tracking
test_video = 'MountainBike'; % 'Basketball' 'Matrix' 'MountainBike' 'Girl'
test_video_dir = fullfile('..','data',test_video,'img'); 

% The file containing the ground truth bounding box labels (enter 'NA' to
% use User defined bounding boxes)
test_video_gt = fullfile('..','data',test_video,'groundtruth_rect.txt');
% test_video_gt = 'NA';

% The number of initial video frames for which ground truth bounding boxes
% should be provided.
num_initial_frames = 1;

% The max number of frames in a video that this algorithm should track 
% the object, after the initial frames. Set to intmax to consider all the 
% frames in the video
max_num_frames = 3; % Set to 3 just for quick tests
% max_num_frames = intmax;

%% ------------------------------------------------------------------------
%                                            Setup MatConvNet and Load CNN
% -------------------------------------------------------------------------

% Must run vl_setupnn.m whenever you open matlab in order to use MatConvNet
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'MatConvNet', 'matlab', 'vl_setupnn.m')) ;

% download a pre-trained CNN from the web (needed once)
if ~exist(net_file_path,'file')
    urlwrite(net_download_path, net_file_path) ;
end

% load the pre-trained CNN
net = load(net_file_path) ;

%% ------------------------------------------------------------------------
%                                                       Setup and load SVM
% -------------------------------------------------------------------------

% -------------------
%      Todo          |
% -------------------

%% ------------------------------------------------------------------------
%                                              Setup and load video frames
% -------------------------------------------------------------------------

% Load the video frames
frames = imgCompile(test_video_dir);

% Determine initial bounding box(es)
initial_bbs = zeros(num_initial_frames,4);
if strcmp(test_video_gt,'NA')
    
    % User should define initial bounding box(es)
    for i = 1:num_initial_frames
        initial_bbs(i,:) = bbGen(frames(:,:,:,i));
    end
else
    
    % Find initial bounding box(es) from ground truth file
    bb = load(test_video_gt);
    initial_bbs = bb(1:num_initial_frames,:);
    clear bb;
end

% Show initial bounding box(es) to confirm correct placement
for i = 1:num_initial_frames
    figure;
    imshow(frames(:,:,:,i),[]);
    rectangle('Position', initial_bbs(i,:) , 'EdgeColor', 'red');
    title(['Initial Bounding Box Placement: Frame ' num2str(i)]);
end

%% ------------------------------------------------------------------------
%                                        Train SVM on initial video frames
% -------------------------------------------------------------------------

% -------------------
%      Todo          |
% -------------------

%% ------------------------------------------------------------------------
%                                                 Begin Main Tracking loop
% -------------------------------------------------------------------------

% Create variable to store bounding box locations, inputting initial
% bounding box(es)
num_frames = min(num_initial_frames + max_num_frames, size(frames,4));
bbs = zeros(num_frames,4);
bbs(1:num_initial_frames,:) = initial_bbs;

% Create main tracking loop, looking for object in frames
for i = (num_initial_frames+1):num_frames
    
    % Get current frame
    frame = frames(:,:,:,i);

    % Generate samples from last frame's BB
    num_samples = 120;
    bbSamples = sampleBBGen(bbs(i-1,:), num_samples);
    samples = crop_img_to_bbs(frame, bbSamples);
    
    % Pass samples through CNN to get sample features
    [feature_length, layer] = get_first_fully_connected_feature_length(net);
    sample_features = zeros(num_samples, feature_length, 'single');
    for j = 1:num_samples
    
        feat_vect = get_first_fully_connected_output(net, samples(:,:,:,j));
        sample_features(j,:) = feat_vect;
    end
    clear feat_vect;
    
    % Pass sample features through SVM, retaining only positive samples
    % -------------------
    %      Todo          |
    % -------------------
    
    % Retrieve Target-specific features from SVM weights and positive
    % samples
    % -------------------
    %      Todo          |
    % -------------------
    target_specific_features = sample_features; % Obviously, we need to update this. This is just filler code for now so we can have features to test saliency maps with
    
    % Backpropagate target-specific features to get class-saliency maps
    class_saliency_maps = zeros([size(samples,1:2) 1 num_samples]);
    for j = 1:num_samples
       
        class_saliency_maps(:,:,:,j) = compute_class_saliency_map(net, ...
            samples(:,:,:,j), target_specific_features(j,:), layer);
    end
    
    % Generate overall target-specific saliency map from class-saliency
    % maps
    % -------------------
    %      Todo          |
    % -------------------
    
    % Apply/update Generative model
    % -------------------
    %      Todo          |
    % -------------------
    
    % Retrieve BB in current frame from target posterior
    % -------------------
    %      Todo          |
    % -------------------
    
    % Update SVM
    % -------------------
    %      Todo          |
    % -------------------
    
end
    
%% ------------------------------------------------------------------------
%                                                          Provide Results
% -------------------------------------------------------------------------

% Here, we may either return the bounding boxes to a different program,
% that will then display them. Or we may just want to show the video here
% in this program, painting the bounding boxes on each frame.

% -------------------
%      Todo          |
% -------------------