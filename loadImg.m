function [I,imgFiles] = loadImg(directory, h, w, fmt, fps)
%%
% Loads a collection of images from a directory and returns timestamp and
% images data in the matrix I. 
% 
% Note, this function assumes 24-bit RGB images.
% 
% Legend: 
%   directory = the location of the .png file
%   I = a matrix containing the timestamp + image data 
%   fmt = filename extension for images (e.g., 'png' or 'jpg')

assert(nargin > 4, 'Usage: [I] = loadImg(directory, h, w, fmt, fps)' );

bpp = 3; % assume 24-bit images, with 3 bytes per pixel

% Get all the images from a single directory
D = strcat(directory, '*.', fmt);
imgFiles = dir(D);
numImages = numel(imgFiles);

% Pre-allocate the data matrix
imgLen = h*w*bpp;
I = zeros(numImages, imgLen+1);

% Copy each image into the matrix
for i = 1:numImages,
    % Get the frame number
    I(i,1) = (i-1) / fps;
    
    % Get the image
    filename = fullfile(directory, imgFiles(i).name);
    im = imread(filename);
    im = im2double(im);
    
    % Resize the image
    im = imresize(im, [h w]);
    I(i,2:end) = reshape(im, 1, imgLen);
end

