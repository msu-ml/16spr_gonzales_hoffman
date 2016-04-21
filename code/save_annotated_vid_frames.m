function save_annotated_vid_frames( result_out_dir, frames, bbs)
%save_annotated_vid_frames Draws BBs in video frames and saves them to dir
%   This function draws the desired bounding boxes around the video frames
%   and then saves them all to the desired output directory
%
% INPUT
%   result_out_dir  - The output directory to save the video to
%   frames          - Each unaltered frame in our video
%   bbs             - [X,Y,W,H] info for each frame's predicted target bb
%                   (bounding box)

for t = 1:size(frames,4)
    
    img = insertShape(frames(:,:,:,t),'Rectangle',bbs(t,:),'Color','red');
    imwrite(img, fullfile(result_out_dir,sprintf('%04d.jpg',t)));
end

end

