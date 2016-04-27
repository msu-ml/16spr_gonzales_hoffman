test_vid = 'Deer';
res_dir = fullfile('..','results_complete');
out_dir = fullfile(res_dir,[test_vid '-WithPrior']);
bbs_file = fullfile(res_dir,[test_vid '-WithPrior-bbs.mat']);
data_dir = fullfile('..','data',test_vid,'img');

frames = imgCompile(data_dir);
load(bbs_file);
save_annotated_vid_frames( out_dir, frames(:,:,:,1:size(bbs,1)), bbs);