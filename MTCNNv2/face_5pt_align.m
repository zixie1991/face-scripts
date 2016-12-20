clear;
face_dir='/mnt/ssd-data-1/mydata/lfw_raw/';
faces='/mnt/ssd-data-1/mydata/lfw_raw/labeled_faces_5pt.txt';
save_dir='/mnt/ssd-data-1/mydata/lfw_5pt_96/';
img_size = [120, 96];

f = fopen(faces, 'r');

while ~feof(f)
    line = fgetl(f);
    %[filename, n, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5] = strread(line, '%s %d %f %f %f %f %f %f %f %f %f %f');
    [filename, n, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5] = strread(line, '%s %d %d %d %d %d %d %d %d %d %d %d');
    f5pt = [x1 x2 x3 x4 x5; y1 y2 y3 y4 y5];
	img=imread(strcat(face_dir, filename{1}));

    if size(img, 3) == 1
        % gray img to rgb
        img = gray2rgb(img);
    end

    img_cropped=f5pt_align(img, f5pt, img_size);

    if isempty(img_cropped)
        filename
        continue
    end

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
    imwrite(img_cropped, save_fn);
end

fclose(f);
