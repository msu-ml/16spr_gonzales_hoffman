function Main()
    clear;
    clc;
    
    path = 'C:\Users\gonza647\Downloads\Data\Dog1\img\0001.jpg';

    % setup MatConvNet
    run  ../matconvnet/matlab/vl_setupnn

    % load the pre-trained CNN
    net = load('imagenet-vgg-f.mat') ;
    
    im = bbGen(path);

    % load and preprocess an image
    %im = imread(path) ;
    im_ = single(im) ; % note: 0-255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;
    
    if (numel(size(im_)) == 2 || size(im_, 3) == 1) 
        im_ = [im_ ; im_ ; im_];
    end
    size(im_)

    % run the CNN
    res = vl_simplenn(net, im_) ;

    % show the classification result
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores) ;
    figure(1) ; clf ; imagesc(im) ;
    title(sprintf('%s (%d), score %.3f',...
    net.meta.classes.description{best}, best, bestScore)) ;


end
function img = bbGen(path)
    % We need some user input here to define an initial bounding box around
    % the target of interest. We'll need height, width, and center pixel
    % information.
    img = imread(path);
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
    img = imcrop(img, [x y w h]);           %Return the cropped sample
    % With initial bounding box information we can build samples in a
    % normal distribution around the center.
    samples = [0 0];
    for i = 1:120
        samples(i,:) = round(normrnd([x y], (sqrt(w*h)/2))); %Sample coords     
        if i <= 15                                            %Draw a few of the sample boxes
            rectangle('Position', [samples(i,1) samples(i,2) w h], 'EdgeColor','yellow');
        end
    end
    close all;
end

function vidFrames = imgCompile(directory)
    loc = cd(directory);
    files = dir('*.jpg');
    vidFrames = [];
    for index = 1: length(files)
        frame = imread([loc '/' files{index}]);
        vidFrames = cat(4, vidFrames, frame);
    end
end