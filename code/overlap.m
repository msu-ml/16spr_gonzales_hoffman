function result = overlap(original, sample)
    xOrig = original(1);
    yOrig = original(2);
    w = original(3);
    h = original(4);
    
    xSamp = sample(1);
    ySamp = sample(2);
    Area = w*h;
    diffX = abs(xOrig-xSamp);
    diffY = abs(yOrig-ySamp);
    
    %The bounding boxes have no overlap
    if diffX >= w || diffY >= h
        result = 0;
        return;   
    end
    sharedArea = (w - diffX) * (h - diffY);
    %In case you'd like to see how bb overlap
    %display(original,sample);
    result = sharedArea/Area;
end

function display(orig,samp)
    figure;
    rectangle('Position', [orig(1) orig(2) orig(3) orig(4)], 'EdgeColor', 'red');
    rectangle('Position', [samp(1) samp(2) samp(3) samp(4)], 'EdgeColor', 'blue');
end