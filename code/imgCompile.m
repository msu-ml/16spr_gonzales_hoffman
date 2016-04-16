function [ vidFrames ] = imgCompile( directory )
%imgCompile Loads all image frames in a directory, saving to a 4-D struct
%   This function loads all the images in the given directory and returns
%   them in a W x H x C x F struct, where W = width of the frames, 
%   H = height of the frames, C = the number of channels in the images (1
%   or 3), and F = the number of frames. Note, this only works for '*.jpg'
%   images.

    files = dir(fullfile(directory,'*.jpg'));
    files = {files.name};
    vidFrames = [];
    for index = 1: length(files)
        frame = imread(fullfile(directory, files{index}));
        vidFrames = cat(4, vidFrames, frame);
    end

end

