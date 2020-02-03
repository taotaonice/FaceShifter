clear;clc;close all

pts = [
     93.6009  119.6385
  162.0618  119.2599
  128.0490  158.5740
   99.9244  198.6530
  156.6181  198.3394
  ];
convex = convhull(pts);
bw = poly2mask(pts(convex, 1), pts(convex, 2), 256, 256);
g =  double(bw);

% m = imgaussfilt(g, 21, 'filtersize', 51);

se = strel('disk', 70);
d = imdilate(g, se);
d = imgaussfilt(d, 41, 'filtersize', 111);
subplot 121
imshow(d)
subplot 122
imshow(g)
imwrite(d, './prior.png');