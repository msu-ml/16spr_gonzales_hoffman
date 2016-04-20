%%% makeVid
% INPUT
%   frames  - Each unaltered frame in our video
%   bbs     - [X,Y,W,H] info for each frames predicted target bb

function makeVid(frames, bbs)
    [width,height,~,t] = size(frames);       %4-D object. Width x Height x # of Channels x # of Frames
    % We will assume we're only dealing with RGB images
    bbInsert = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',uint8([255 0 0]));
    red = uint8([255 0 0]);
    framerate = 60;
    trackingVid = VideoWriter('tracked.avi');
    trackingVid.FrameRate = 60;
    open(trackingVid);
    for i = 1:t
        frame = frames(:,:,:,i);
        newFrame = step(bbInsert,frame,int32(bbs(i,:)));
        writeVideo(trackingVid,newFrame);
    end
    close(trackingVid);
end