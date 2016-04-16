function [bb, img, bbSamples] = bbGen(img)
%bbGen Lets the user define a bounding box around the object to track
%   This function allows the user to specify a bounding box around the
%   object they wish to track in the image provided. This function returns
%   the img cropped to that bounding box and the bounding box itself from
%   the original frame. The bounding box is a vector of form
%   [X, Y, W, H] where the top left corner of the bounding box is at X,Y
%   and the bounding box is of width W and height H. This function also
%   generates random samples around the bb specified by the user and
%   returns those.

    % We need some user input here to define an initial bounding box around
    % the target of interest. We'll need height, width, and center pixel
    % information.
    figure 
    imshow(img);
    done = 0;
    while ~done
        h = input('Height of bounding box: ');
        w = input('Width of bounding box: ');
        % Here we assume we're only interested in tracking a single target
        % in any given video sequence. N = 1 reflects that. 
        n = 1;
        [r,c] = ginput(n);
        x = round(r)-round(w/2);
        y = round(c)-round(h/2);
        bb = rectangle('Position', [x y w h], 'EdgeColor', 'red');
        done = input('Done? ' );
        if ~done
            delete(bb)
        end
    end
    bb = [x y w h];
    img = imcrop(img, [x y w h]);           %Return the cropped sample
    % With initial bounding box information we can build samples in a
    % normal distribution around the center.
    samples = [0 0];
    bbSamples = [0 0 0 0];
    for i = 1:120
        samples(i,:) = round(normrnd([x y], (sqrt(w*h)/2))); %Sample coords    
        bbSamples(i,:) = [samples(i,1), samples(i,2), w , h];
        if i <= 20                                            %Draw a few of the sample boxes
            rectangle('Position', [samples(i,1) samples(i,2) w h], 'EdgeColor','yellow');
        end
    end
    %close all;
end
