function display_saliency_map_results( frame, tssm )
%display_saliency_map_results Displays Saliency Map results
%   This function displays the results of the target-specific saliency map
%   (tssm) that was generated based on the image frame.

figure;
imshow(frame,[]);
title('Original Image Frame');

figure;
imshow(tssm,[]);
title('Target-Specific Saliency Map - Original Colors');

figure;
psm = get_pretty_saliency_map(tssm);
imshow(psm,[]);
title('Target-Specific Saliency Map');

figure;
image(frame);
image('CData',psm,'AlphaData',0.75);
title('TSSM overlayed onto image');
axis off;

end

