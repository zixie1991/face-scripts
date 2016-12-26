function [cropImg]=f5pt_align(image, facial5points)
imgSize = [112, 96];
coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
Tfm =  cp2tform(facial5points', coord5points', 'similarity');
cropImg = imtransform(image, Tfm, 'XData', [1 imgSize(2)],...
'YData', [1 imgSize(1)], 'Size', imgSize);
end
