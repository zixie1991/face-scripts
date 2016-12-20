clear;
%list of images
root='/mnt/ssd-data-1/CASIA-WebFace-washed/';
imglist=importdata('/mnt/ssd-data-1/training_set/CASIA-WebFace-washed-lightened/imglist.txt');
labeled_faces='./labeled_faces.txt';

%minimum size of face
minsize=20;

%path of toolbox
caffe_path='/home/zixie1991/mygit/caffe/matlab';
pdollar_toolbox_path='/home/zixie1991/mygit/toolbox'
caffe_model_path='./model'
addpath(genpath(caffe_path));
addpath(genpath(pdollar_toolbox_path));

%use cpu
%caffe.set_mode_cpu();
gpu_id=0;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);

%three steps's threshold
threshold=[0.6 0.7 0.7]

%scale factor
factor=0.709;

%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir =  strcat(caffe_model_path,'/det4.prototxt');
model_dir =  strcat(caffe_model_path,'/det4.caffemodel');
LNet=caffe.Net(prototxt_dir,model_dir,'test');
faces=cell(0);	
points_max=zeros(10,length(imglist));

file = fopen(labeled_faces, 'w');

%points_max = [];
for i=1:length(imglist)
	img=imread(strcat(root, imglist{i}));
    if size(img, 3) == 1
        % gray img to rgb
        img = gray2rgb(img);
    end

	%we recommend you to set minsize as x * short side
	%minl=min([size(img,1) size(img,2)]);
	%minsize=fix(minl*0.1)
%    tic
    [boudingboxes points]=detect_face(img,minsize,PNet,RNet,ONet,LNet,threshold,false,factor);
    if isempty(points)
        continue;
    end
    img_size = size(img); % height, width, channels
    dist = ((boudingboxes(:,1) + boudingboxes(:,3)) - img_size(2)) .^ 2 + (boudingboxes(:,2) + boudingboxes(:,4) - img_size(1)) .^ 2;
    [r c]=min(dist);
	numbox=size(boudingboxes,1);
    %points(:,c)
    %fprintf(file, '%s %d', imglist{i}, 1);
    %fprintf(file, ' %f %f %f %f %f %f %f %f %f %f', (points(1,c)), (points(6,c)), (points(2, c)), (points(7, c)), (points(3, c)), (points(8, c)), (points(4, c)), (points(9, c)), (points(5, c)), (points(10,c)));
    %fprintf(file, ' %d %d %d %d %d %d %d %d %d %d', int32(points(1,c)), int32(points(6,c)), int32(points(2, c)), int32(points(7, c)), int32(points(3, c)), int32(points(8, c)), int32(points(4, c)), int32(points(9, c)), int32(points(5, c)), int32(points(10,c)));
    %fprintf(file, ' %d %d %d %d', int32(boudingboxes(c,1)), int32(boudingboxes(c,2)), int32(boudingboxes(c,3)), int32(boudingboxes(c,4)));
    fprintf(file, '%s %d', imglist{i}, numbox);
    for j=1:numbox
        fprintf(file, ' %d %d %d %d', int32(boudingboxes(j,1)), int32(boudingboxes(j,2)), int32(boudingboxes(j,3)), int32(boudingboxes(j,4)));
    end
    fprintf(file, '\n');
%    W=WH(:,1);
%    H=WH(:,2);
%    area=W.*H
    %points_max = [points_max, points(:,c)];
    %    boudingboxes
%    toc
    %faces{i,1}={boudingboxes};
	%faces{i,2}={points'};
	%show detection result
	%numbox=size(boudingboxes,1);
	%imshow(img)
	%hold on; 
	%for j=1:numbox
	%	plot(points(1:5,j),points(6:10,j),'g.','MarkerSize',10);
	%	r=rectangle('Position',[boudingboxes(j,1:2) boudingboxes(j,3:4)-boudingboxes(j,1:2)],'Edgecolor','g','LineWidth',3);
    %end
    %savename=strcat('dec_',num2str(i))
    %name=strcat(savename,'.jpg')
    %print ('-f1' ,'-r600', '-djpeg',[ num2str(i) '_dre.jpg'])  
    %saveas(gcf,[ num2str(i) '_dr_dre.jpg'])
    %hold off; 
	%pause(5)
end
fclose(file);
%save result box landmark
