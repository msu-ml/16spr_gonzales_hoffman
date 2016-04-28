%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% track_object.m
%
% This program is meant to perform the object tracking algorithm originally
% proposed by Hong, et al. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic 

%% ------------------------------------------------------------------------
%                                                         Setup Parameters
% -------------------------------------------------------------------------
clear all;
clc; 

% Directory of saved CNN/Location to save CNN
net_file_dir = fullfile('..','nets');

% Filename of saved CNN/Location to save CNN
net_file_path = fullfile(net_file_dir,'imagenet-vgg-f.mat'); 

% Location to download CNN if doesn't exist
net_download_path = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat'; 

% Boolean: True if priors should be used, false otherwise
use_priors = false;

% The folder containing the frames of the video in which to perform tracking
test_video = 'Skiing'; % 'Basketball' 'Matrix' 'Girl' 'Deer'
test_video_dir = fullfile('..','data',test_video,'img'); 

% The file containing the ground truth bounding box labels (enter 'NA' to
% use User defined bounding boxes)
test_video_gt = fullfile('..','data',test_video,'groundtruth_rect.txt');
test_video_gt = 'NA';

if use_priors
    
    % The file to save the final bounding boxes to
    bb_out_filename = fullfile('..','results',[test_video '-bbs.mat']);

    % The file to save the target specific saliency maps to
    saliency_out_filename = fullfile('..','results',[test_video '-ts-sal-maps.mat']);

    % The directory to save outputs to (other than bounding boxes and saliency maps)
    result_out_dir = fullfile('..','results',test_video);
else
    
    % The file to save the final bounding boxes to
    bb_out_filename = fullfile('..','results',[test_video '-NoPrior-bbs.mat']);

    % The file to save the target specific saliency maps to
    saliency_out_filename = fullfile('..','results',[test_video '-NoPrior-ts-sal-maps.mat']);

    % The directory to save outputs to (other than bounding boxes and saliency maps)
    result_out_dir = fullfile('..','results',[test_video '-NoPrior']);  
end

% The number of initial video frames for which ground truth bounding boxes
% should be provided.
num_initial_frames = 10;

% The max number of frames in a video that this algorithm should track 
% the object, after the initial frames. Set to intmax to consider all the 
% frames in the video
max_num_frames = 3; % Set to 3 just for quick tests
max_num_frames = 50; % Set to 120 for  ~2hr tests
% max_num_frames = 210; % Set to 210 for longer tests
max_num_frames = intmax;


% The number of sample patches to generate in each frame when looking for the
% object
num_samples = 121; % 120 + 1 for the bb from last frame

% The number of target filters to be used in computing the generative model
num_target_filters = 30;

%% ------------------------------------------------------------------------
%                                            Setup MatConvNet and Load CNN
% -------------------------------------------------------------------------

% Must run vl_setupnn.m whenever you open matlab in order to use MatConvNet
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'MatConvNet', 'matlab', 'vl_setupnn.m')) ;

% download a pre-trained CNN from the web (needed once)
if ~exist(net_file_path,'file')
    
    if ~exist(net_file_dir,'dir')
        mkdir(net_file_dir);
    end
    urlwrite(net_download_path, net_file_path) ;
end

% load the pre-trained CNN
net = load(net_file_path) ;

%% ------------------------------------------------------------------------
%                                                       Setup and load SVM
% -------------------------------------------------------------------------

% Necessary SVM variables
global KTYPE
global KSCALE
global online
global visualize
global doloo
global terse
terse = 1;
doloo = 0;
KTYPE = 1;          %Determines the type of kernel used. 
KSCALE = 0.25;
online = 1;
visualize = 0;
C = 1;              % Soft Marging constraint in SVM train

% How many negative samples we allow per positive sample
samp_per_frame = 25;
% How large we will allow our training set to become
max_training_sz = 300;      %Corresponds to 12 frame training
% How many frames will we skip before updating the SVM again?
svm_step = 3;

%% ------------------------------------------------------------------------
%                                              Setup and load video frames
% -------------------------------------------------------------------------

% Load the video frames
frames = imgCompile(test_video_dir);

% Determine initial bounding box(es)
initial_bbs = zeros(num_initial_frames,4);
if strcmp(test_video_gt,'NA')
    
    % User should define initial bounding box(es)
    for t = 1:num_initial_frames
        initial_bbs(t,:) = bbGen(frames(:,:,:,t));
    end
else
    
    % Find initial bounding box(es) from ground truth file
    bb = load(test_video_gt);
    initial_bbs = bb(1:num_initial_frames,:);
    clear bb;
end

% % % % Show initial bounding box(es) to confirm correct placement
% % % for i = 1:num_initial_frames
% % %     figure;
% % %     imshow(frames(:,:,:,i),[]);
% % %     rectangle('Position', initial_bbs(i,:) , 'EdgeColor', 'red');
% % %     title(['Initial Bounding Box Placement: Frame ' num2str(i)]);
% % % end

%% ------------------------------------------------------------------------
%                                        Train SVM on initial video frames
% -------------------------------------------------------------------------
[fc_feature_length, first_fc_layer] = get_first_fully_connected_feature_length(net);
%Threshold for separating neutral/positive samples
overlap_threshold = 0.3;
svm_training_imgs = uint8.empty;
svm_training_truth = [];
%Ground truth variable can have at most n rows, assuming every sample 
%beyond the initial bb is negative
groundTruth = zeros(num_samples,1);
for t = 1:num_initial_frames
    initial_bbs(t,3:4) = initial_bbs(1,3:4);
    % Get current frame
    frame = frames(:,:,:,t);
    count = 0;
    % Generate samples from last frame's BB
    bb_samples = sampleBBGen(initial_bbs(t,:), num_samples);
    samples = crop_img_to_bbs(frame, bb_samples);
    for j = 1: length(bb_samples)        
        if count == samp_per_frame
            break;
        end
        currSamp = samples(:,:,:,j);
        overlapRatio = overlap(bb_samples(1,:), bb_samples(j,:));
        if overlapRatio == 1 
            count = count + 1;
            svm_training_imgs = cat(4, svm_training_imgs, currSamp);
            svm_training_truth = [svm_training_truth;1];
        elseif overlapRatio <= overlap_threshold 
            count = count + 1;
            svm_training_imgs = cat(4, svm_training_imgs, currSamp);
            svm_training_truth = [svm_training_truth;-1];
        end
    end
end

% Pass samples through CNN to get sample features
num_training = length(svm_training_truth);
svm_training_feat = zeros(num_training, fc_feature_length, 'single');
for j = 1:num_training    
   feat_vect = get_first_fully_connected_output(net, svm_training_imgs(:,:,:,j));
   svm_training_feat(j,:) = feat_vect;
end
addpath('../onlinesvm');
[coeff,bias,~,inds,inde,~] = svcm_train(svm_training_feat,svm_training_truth,C);
clear feat_vect;


%% ------------------------------------------------------------------------
%                   Initialize remaining variables needed in tracking loop
% -------------------------------------------------------------------------
weights = 0;
% Iterate through all the support vectors
for i = 1:length(inds)
    samp = coeff(inds(i)) * svm_training_feat(inds(i),:) * svm_training_truth(inds(i));
    weights = weights + samp;
end
weights(weights<=0) = 0;

% Create variable to store bounding box locations, inputting initial
% bounding box(es)
num_frames = min(num_initial_frames + max_num_frames, size(frames,4));
bbs = zeros(num_frames,4); % This will hold the location of the object at each frame
bbs(1:num_initial_frames,:) = initial_bbs;

% Create variables for performing Sequential Bayesian Filtering
bb_samples_last_frame = sampleBBGen(bbs(num_initial_frames,:), num_samples);
posteriors_last_frame = zeros(num_samples,1);
posteriors_last_frame(1) = 1; % Probability of object at its BB is 1, 0 elsewhere for ground truth
target_spec_sal_maps = zeros(size(frames,1), size(frames,2),num_frames);

% Initialize Target-Specific Saliency Maps for the initial frames to be 
% used by the Sequential Bayesian Filtering process. We will
% only use the sample with ground truth bounding box as it is most relevant
% to creating the generative model.
for t = 1:num_initial_frames
    
    % Get current frame and the sample from ground truth bounding box
    frame = frames(:,:,:,t);
    sample = crop_img_to_bbs(frame, bbs(t,:));
    
    % Pass sample through CNN to get sample features
    sample_features = get_first_fully_connected_output(net, sample);
    
    % Retrieve Target-specific features from SVM weights
        % Since the weights vector is sparse, many columns are removed from
        % our sample features
    target_specific_features = sample_features.*weights';

    % Backpropagate target-specific features to get class saliency map
    class_saliency_maps = zeros([size(sample,1) size(sample,2) 1]);
    sm = compute_class_saliency_map(net, ...
        sample, target_specific_features, first_fc_layer);
    class_saliency_maps(:,:,1) = imresize(sm, [size(sample,1), size(sample,2)]);
    
    % Generate overall target-specific saliency map from class-saliency
    % maps
    target_spec_sal_maps(:,:,t) = compute_target_specific_saliency_map(...
        size(frame), class_saliency_maps, bbs(t,:));
end
clear target_specific_features;
%% ------------------------------------------------------------------------
%                                                 Begin Main Tracking loop
% -------------------------------------------------------------------------
init_time = toc;
fprintf('Time to initialize algor: %f sec\n', init_time);
tic
updateSVM = 0;
num_true_positive = num_initial_frames*2;
posIdx = 1;
% Create main tracking loop, looking for object in frames
for t = (num_initial_frames+1):num_frames
    
    % Update SVM counter to tell it when to train
    updateSVM = updateSVM + 1;
    
    % Get current frame
    frame = frames(:,:,:,t);
    
    % Generate samples from last frame's BB
    bb_samples = sampleBBGen(bbs(t-1,:), num_samples);
    samples = crop_img_to_bbs(frame, bb_samples);
    
    % Pass samples through CNN to get sample features
    sample_features = zeros(num_samples, fc_feature_length, 'single');
    for j = 1:num_samples
    
        feat_vect = get_first_fully_connected_output(net, samples(:,:,:,j));
        sample_features(j,:) = feat_vect;
    end
    clear feat_vect;
    
    % Pass sample features through SVM, retaining only positive samples 
    ytest = ones(num_samples,1);
    [ypred,~] = svcm_test(sample_features, ytest, svm_training_feat, svm_training_truth, coeff, bias);
    
    pos_samples = samples(:,:,:,ypred>0); 
    bb_pos_samples = bb_samples(ypred>0,:); 
    num_pos_samples = sum(ypred>0);
    pos_ind = find(ypred>0);
    % Retrieve Target-specific features from SVM weights and positive
    % samples
    for i = 1:num_pos_samples
        target_specific_features(i,:) = sample_features(pos_ind(i),:).*weights;
    end
    if num_pos_samples == 0
        fprintf('No positive samples were found for frame %d\n',t);
        % Decrementing updateSVM here will ensure the SVM training set does
        % not attempt to train on the frame with no positives
        updateSVM = updateSVM - 1; 
        bbs(t,:) = bbs(t-1,:); 
        continue;
    end
    
    % Backpropagate target-specific features to get class saliency maps
    class_saliency_maps = zeros([size(samples,1) size(samples,2) num_pos_samples]);
    for j = 1:num_pos_samples
       
        sm = compute_class_saliency_map(net, ...
            pos_samples(:,:,:,j), target_specific_features(j,:), first_fc_layer);
        class_saliency_maps(:,:,j) = imresize(sm, [size(samples,1), size(samples,2)]);
    end
    
    % Generate overall target-specific saliency map from class-saliency
    % maps
    target_spec_sal_maps(:,:,t) = compute_target_specific_saliency_map(...
        size(frame), class_saliency_maps, bb_pos_samples );
    
    % Compute the prior probabilities of each sample BB for Sequential
    % Bayesian Filtering
    if use_priors
        priors = compute_prior_prob_bbs(bb_samples, bb_pos_samples, ...
            bb_samples_last_frame, bbs(t-1,:), posteriors_last_frame );
    end
    
    % Compute the likelihood probabilities of each sample BB by computing a
    % Generative model.
    [likelihoods, gen_model] = compute_likelihood_and_generative_model( ...
        target_spec_sal_maps, bbs, bb_samples, num_target_filters, t);
    
    % Retrieve BB in current frame from target posterior
    if use_priors
        posteriors = likelihoods .* priors;
    else
        posteriors = likelihoods;
    end
    [~,idx] = max(posteriors);
    bbs(t,:) = bb_samples(idx,:);
    
    % Update Sequential Bayesian Filtering variables (posteriors and sampleBBs)
    posteriors_last_frame = posteriors;
    bb_samples_last_frame = bb_samples;   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Update SVM every third frame
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~(mod(updateSVM,svm_step))
        %Print statement for debugging purposes
        fprintf('Update SVM at frame %d, update #%d\n',t,floor(t/svm_step));
        count = 0;        
        for j = 1:num_samples
            if count == samp_per_frame
                break;
            end
            currSamp = samples(:,:,:,j);
            overlapRatio = overlap(bbs(t,:), bb_samples(j,:));
            if overlapRatio == 1
                count = count + 1;
                svm_training_imgs = cat(4, svm_training_imgs, currSamp);
                svm_training_truth = [svm_training_truth;1];
            elseif overlapRatio <= overlap_threshold
                count = count + 1;
                svm_training_imgs = cat(4, svm_training_imgs, currSamp);
                svm_training_truth = [svm_training_truth;-1];
            end
        end
        % In training we'd like to keep the initial ground truth Positive
        % samples, and replace old negative samples to prevent our training
        % set from growing too large
        if length(svm_training_truth) > max_training_sz
            for j = 1:samp_per_frame
                if (svm_training_truth(j)>0) && (posIdx < num_true_positive)
                    %Keep track of the old positive samples we keep, we
                    %only want to keep a certain amount
                    posIdx = j;
                end
            end
            svm_training_truth = [svm_training_truth(1:posIdx);svm_training_truth((posIdx+samp_per_frame):end)];
            new_training_imgs = svm_training_imgs(:,:,:,1:posIdx);
            new_training_imgs = cat(4,new_training_imgs,svm_training_imgs(:,:,:,(posIdx+samp_per_frame):end));
            svm_training_imgs = new_training_imgs;
        end
        
        %Retrain the SVM
        num_training = length(svm_training_truth);
        svm_training_feat = zeros(num_training, fc_feature_length, 'single');
        for j = 1:num_training
            feat_vect = get_first_fully_connected_output(net, svm_training_imgs(:,:,:,j));
            svm_training_feat(j,:) = feat_vect;
        end
        [coeff,bias,~,inds,inde,~] = svcm_train(svm_training_feat,svm_training_truth,C);
        clear feat_vect;
        
        % Update the weights
        weights = 0;
        % Iterate through all the support vectors
        for i = 1:length(inds)
            samp = coeff(inds(i)) * svm_training_feat(inds(i),:) * svm_training_truth(inds(i));
            weights = weights + samp;
        end
        weights(weights<=0) = 0;
    end 
end
    
%% ------------------------------------------------------------------------
%                                                          Provide Results
% -------------------------------------------------------------------------

% Here, we may either return the bounding boxes to a different program,
% that will then display them. Or we may just bbwant to show the video here
% in this program, painting the bounding boxes on each frame.

if ~exist(result_out_dir,'dir')
    mkdir(result_out_dir);
end
save(bb_out_filename,'bbs');
save(saliency_out_filename, 'target_spec_sal_maps');
makeVid(result_out_dir, frames(:,:,:,1:num_frames), bbs);
save_annotated_vid_frames(result_out_dir, frames(:,:,:,1:num_frames), bbs);

track_time = toc;
fprintf('Time to track: %f sec\n', track_time);
fprintf('Total time: %f sec\n', init_time + track_time);