clear;
face_dir='';
faces='';
ec_mc_y=60;
ec_y=30;
img_size=128;
save_dir='';

% center of eyes (ec), center of l&r mouth(mc), rotate and resize
% ec_mc_y: y_mc-y_ec, diff of height of ec & mc, to scale the image.
% ec_y: top of ec, to crop the face.

clck = clock();
%log_fn = sprintf('fa2_%4d%02d%02d%02d%02d%02d.log', [clck(1:5) floor(clck(6))]);
%log_fid = fopen(log_fn, 'w');

crop_size = img_size;


f = fopen(faces, 'r');

while ~feof(f)
    line = fgetl(f);
    %[filename, n, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5] = strread(line, '%s %d %f %f %f %f %f %f %f %f %f %f');
    [filename, n, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5] = strread(line, '%s %d %d %d %d %d %d %d %d %d %d %d');
    f5pt = [x1 y1; x2 y2; x3 y3; x4 y4; x5 y5];
	img=imread(strcat(face_dir, filename{1}));

    if size(img, 3) == 1
        % gray img to rgb
        img = gray2rgb(img);
    end

    [img2, eyec, img_cropped, resize_scale] = ec_mc_align(img, f5pt, crop_size, ec_mc_y, ec_y);

    if isempty(img2)
        filename
        continue
    end

    img_final = imresize(img_cropped, [img_size img_size], 'Method', 'bicubic');
    %if size(img_final,3)>1
        %img_final = rgb2gray(img_final);
    %end
    save_fn = [save_dir filename{1}];
    save_dn = save_dir;
    fn=strsplit(filename{1}, '/');
    save_dn = strcat(save_dir, fn(1));
    fn_size = size(fn);
    for n=fn(2:fn_size(2)-1)
        save_dn = [save_dn '/' n];
    end
    if exist(save_dn{1}, 'dir')  == 0
        mkdir(save_dn{1});
    end
    imwrite(img_final, save_fn);
end

fclose(f);


%fclose(log_fid);
