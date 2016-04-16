function [ cropped_imgs ] = crop_img_to_bbs( img, bbs )
%crop_imgs_to_bbs Crops img to the BBs provided
%    This function returns a W x H x C x N array of cropped images where W
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
cropped_imgs = zeros(w,h,c,n);
for i = 1:n
   
    cropped_imgs(:,:,:,i) = imcrop(img,bbs(i,:));
end

end

