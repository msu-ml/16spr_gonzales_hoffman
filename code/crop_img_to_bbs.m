function [ cropped_imgs ] = crop_img_to_bbs( img, bbs )
%crop_imgs_to_bbs Crops img to the BBs provided
%    This function returns a H x W x C x N array of cropped images where W
%    and H are respectively the Width and Height to crop img to, C = the
%    number of channels in img, and N = the number of bounding boxes
%    provided. Note that the width and height of all bounding boxes is
%    assumed to be the same, that is, the width and height of the first
%    bounding box. The bbs parameters should be an [N x 4] array where each
%    bounding box is ordered [X, Y, W, H] with X,Y being the coordinate of
%    the top left corner of the box.

% Get basic parameters
w = bbs(1,3);
h = bbs(1,4);
if size(size(img)) == 2
    c = 1;
else
    c = size(img,3);
end
n = size(bbs,1);

% Crop the img
cropped_imgs = zeros(h,w,c,n, 'uint8');
for i = 1:n
    
    % Get bounding box info
    x = bbs(i,1);
    y = bbs(i,2);
    wi = w;
    hi = h;
    x_offset = 0;
    y_offset = 0;
    
    % Check for BB past the left border of the img
    if x <= 0
        
        x_offset = 1 - x;
        wi = w - x_offset;
        x = 1;
    end
    
    % Check for BB past the top border of the img
    if y <= 0
        
        y_offset = 1 - y;
        hi = h - y_offset;
        y = 1;
    end
    
    % Check for BB past the right border of the img
    if (x+wi-1) > size(img,2)
       
        wi = size(img,2) - x + 1;
    end
    
    % Check for BB past the right border of the img
    if (y+hi-1) > size(img,1)
       
        hi = size(img,1) - y + 1;
    end
   
    cropped_imgs((1+y_offset):(y_offset+hi),(1+x_offset):(x_offset+wi),1:c,i) = ...
        img(y:(y+hi-1),x:(x+wi-1),1:c);
%     cropped_imgs(:,:,:,i) = imcrop(img,bbs(i,:));
end

end

