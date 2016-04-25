%%% makeVid
% INPUT
%   result_out_dir  - The output directory to save the video to
%   frames          - Each unaltered frame in our video
%   bbs             - [X,Y,W,H] info for each frame's predicted target bb

function makeVid(result_out_dir, frames, bbs)
    
    % Parameters
    color = uint8([255 0 0]); % = red
    framerate = 20;

    [width,height,~,t] = size(frames);       %4-D object. Width x Height x # of Channels x # of Frames
    % We will assume we're only dealing with RGB images
    bbInsert = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',color);
    trackingVid = VideoWriter(fullfile(result_out_dir,'tracked.avi'));
    trackingVid.FrameRate = framerate;
    open(trackingVid);
    for i = 1:t
        frame = frames(:,:,:,i);
        newFrame = step(bbInsert,frame,int32(bbs(i,:)));
        writeVideo(trackingVid,newFrame);
    end
    close(trackingVid);
end