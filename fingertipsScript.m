%
% MATLAB script for the paper:
%   Sheldon Andrews, Marc Jarvis, and Paul G. Kry. (2013) 
%   "Data-driven Fingertip Appearance for Interactive Hand Simulation"
%
%  Please see README.txt for a brief description on the usage of this script.
%
%  Author: Sheldon Andrews
%  Email: sheldon.andrews@mail.mcgill.ca
%  Webpage: http://www.cs.mcgill.ca/~sandre17/fingertip/
%
w = 256;
h = 256;
bpp = 3;

imgFilenames = { ...
    'data\Sheldon\Sheldon_Index_neutral.jpg' ...
    'data\Sheldon\Sheldon_Index_mag2.jpg' ...
    'data\Sheldon\Sheldon_Index_mag3.jpg' ...
    'data\Sheldon\Sheldon_Index_mag4.jpg' ...
    'data\Sheldon\Sheldon_Index_mag5.jpg' ...
    'data\Sheldon\Sheldon_Index_mag6.jpg' ...
    'data\Sheldon\Sheldon_Index_mag7.jpg' ...
    'data\Sheldon\Sheldon_Index_mag8.jpg' ...
    'data\Sheldon\Sheldon_Index_mag10.jpg' ... 
    };
pressureFilenames = { ...
    'data\Sheldon\Data001_Neutral.csv' ...
    'data\Sheldon\Data003_Mag2.csv' ...
    'data\Sheldon\Data004_Mag3.csv' ...
    'data\Sheldon\Data005_Mag4.csv' ...
    'data\Sheldon\Data006_Mag5.csv' ...
    'data\Sheldon\Data007_Mag6.csv' ...
    'data\Sheldon\Data008_Mag7.csv' ...
    'data\Sheldon\Data009_Mag8.csv' ...
    'data\Sheldon\Data011_Mag10.csv' ...
    };
selectIdx = 1:length(imgFilenames);
numFiles = length(selectIdx);
sampleI = cell(numFiles);
for i = 1:length(selectIdx),
    filename = char(imgFilenames(selectIdx(i)));
    fprintf(1, 'Loading %s \n', filename);
    sampleI{i} = im2double( imresize(imread(filename), [h w]) );
end

% Load the pressure data.
fingerId = 3;
maxPSI = 12.0;
sampleP = zeros(numFiles, 16);
X = zeros(numFiles, 3);
for i = 1:length(selectIdx),
    filename = char( pressureFilenames(selectIdx(i)) );
    fprintf(1, 'Loading %s \n', filename);
    tmpP = loadPressure(filename, fingerId, 1, 1, maxPSI);
    sampleP(i,:) = tmpP(1,2:end);
    
    % The input is center of pressure + magnitude,
    %  but currently only magnitude is used in the shader.
    [ X(i,1) X(i,2) X(i,3) ] = computeCOP(reshape(sampleP(i,:), [4 4]));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Building the model.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Construct variables for the log space model.
%   2D images are flattened into 1D arrays and then 
%   assembled into log color matrices.
logI = cell(numFiles);
logR = zeros(numFiles, h*w); % red log color matrix
logG = zeros(numFiles, h*w); % green log color matrix
logB = zeros(numFiles, h*w); % blue log color matrix
for i = 1:numFiles,
    I = sampleI{i};
    logR(i,:) = -log( reshape(I(:,:,1), [1 h*w]) );
    logG(i,:) = -log( reshape(I(:,:,2), [1 h*w]) );
    logB(i,:) = -log( reshape(I(:,:,3), [1 h*w]) );
    logI{i} = [ logR(i,:); logG(i,:); logB(i,:) ];
end
% Here, the neutral fingertip image is used as the mean image.
muLogR = -log( reshape(sampleI{1}(:,:,1), [1 h*w]) ); 
muLogG = -log( reshape(sampleI{1}(:,:,2), [1 h*w]) ); 
muLogB = -log( reshape(sampleI{1}(:,:,3), [1 h*w]) ); 
muLogI = [ muLogR; muLogG; muLogB ];

A = zeros(3, (numFiles-1)*h*w);
len = h*w;
rng = 1:len;
for i = 1:numFiles,
    A(1,rng) = logR(i,:) - muLogR;
    A(2,rng) = logG(i,:) - muLogG;
    A(3,rng) = logB(i,:) - muLogB;
    rng = rng + len;
end

[logPC,logEigs,logCoeff] = myPCA(A);

% Project each image onto the first PC vector, which is hemoglobin.
hemoData = zeros(numFiles, h*w);
hemoVec = logPC(:,1);
for i = 1:numFiles,
   hemoData(i,:) = hemoVec' * (logI{i} - muLogI);
end

numEigs = 5;
hemoMu =  zeros(h*w,1); % mu is zero, since we've already removed the mean.
[U,Eigs] = pc_evectors(hemoData', numEigs, hemoMu);

% Project hemoglobin data onto the Eigen vectors.
Y = zeros(numFiles,numEigs);
for i = 1:numFiles,
    x = hemoData(i,:)';
    Y(i,:) = U'*x;
end

% Train the RBF model.
polyOrder = 2;
basisfunction = 'polyharmonicspline'; % use 'gaussian' or 'polyharmonicspline'
Xt = [ X(:,3); X(end,3)*1.2 ];
Yt = [ Y; Y(end,:) ];                       % training output
Xc = Xt;                                    % training centers
numRBF = size(Xt, 1);                       % number of centers
W = zeros(numRBF, numEigs);
for i = 1:numEigs
    [W(:,i)] = train_rbf(Xt, Yt(:,i), Xc, polyOrder, basisfunction);
end

% Training image reconstruct.
logRecon = cell(numFiles);
reconRGB = cell(numFiles);
reconI = cell(numFiles);
x = Xt;
for i = 1:size(x,1)-1,
    fprintf(1, 'Reconstructing test data %d \n', i);

    [y] = sim_rbf(Xc, x(i,:), W, polyOrder, basisfunction);    
    C = zeros(h*w,1);
    for j = 1:numEigs,
        C = C + U(:,j)*y(j);
    end
    
    logRecon{i} = hemoVec * C' + muLogI;
    reconRGB{i} = exp(-logRecon{i});
    reconI{i} = zeros(h,w,3);
    reconI{i}(:,:,1) = reshape(reconRGB{i}(1,:), [h w 1]);
    reconI{i}(:,:,2) = reshape(reconRGB{i}(2,:), [h w 1]);
    reconI{i}(:,:,3) = reshape(reconRGB{i}(3,:), [h w 1]);

    filename = sprintf('orig%d.png',i);
    imwrite( sampleI{i}, filename, 'png');
    
    filename = sprintf('recon%d.png',i);
    imwrite( reconI{i}, filename, 'png');        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing and cross validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
testImgFilenames = { ...
    'data\Sheldon\Sheldon_Index_mag1.jpg' ...
    'data\Sheldon\Sheldon_Index_mag9.jpg' ...    
};
testPressureFilenames = { ...
    'data\Sheldon\Data002_Mag1.csv' ...
    'data\Sheldon\Data010_Mag9.csv' ...
};  
% Load the test data.
numTestFiles = length(testImgFilenames);
testI{i} = cell(numTestFiles);
testP = zeros(numTestFiles, 16);
Xtest = zeros(numTestFiles, 1);
for i = 1:numTestFiles,
    % Texture data.
    filename = char(testImgFilenames(i));
    fprintf(1, 'Loading test image %s \n', filename);
    testI{i} = im2double( imresize(imread(filename), [h w]) );
    % Pressure data.
    filename = char(testPressureFilenames(i));
    tmpP = loadPressure(filename, fingerId, 1, 1, maxPSI);
    testP(i,:) = tmpP(1,2:end);    
    [ posx, posy, Xtest(i,1) ] = computeCOP(reshape(testP(i,:), [4 4]));
end

% Test image reconstruction.
logRecon = cell(numTestFiles);
reconRGB = cell(numTestFiles);
reconI = cell(numTestFiles);
errI = cell(numTestFiles);
x = Xtest;
errTot = 0.0;
figure;
for i = 1:size(x,1),
    fprintf(1, 'Reconstructing test data %d \n', i);

    [y] = sim_rbf(Xc, x(i,:), W, polyOrder, basisfunction);    
    C = zeros(h*w,1);
    for j = 1:numEigs,
        C = C + U(:,j)*y(j);
    end
    
    % Reconstruction in log color space.
    logRecon{i} = hemoVec * C' + muLogI;
    reconRGB{i} = exp(-logRecon{i});
    reconI{i} = zeros(h,w,3);
    reconI{i}(:,:,1) = reshape(reconRGB{i}(1,:), [h w 1]);
    reconI{i}(:,:,2) = reshape(reconRGB{i}(2,:), [h w 1]);
    reconI{i}(:,:,3) = reshape(reconRGB{i}(3,:), [h w 1]);
    errI{i} = 4.0*abs(reconI{i}-testI{i});
    
    subplot(3,numTestFiles,i);
    imshow( testI{i} );
    subplot(3,numTestFiles,i+numTestFiles);
    imshow( reconI{i} );
    subplot(3,numTestFiles,i+2*numTestFiles);
    imshow( errI{i} );
    
    tmpImg = zeros(h, 3*w, 3);
    tmpImg(:,1:w, :) = testI{i};
    tmpImg(:,w+1:2*w, :) = reconI{i};
    tmpImg(:,2*w+1:end, :) = errI{i};
    
    % Output the side-by-side comparison images.
    filename = sprintf('compare%d.png',i);
    imwrite( tmpImg, filename, 'png');
    
    % Compute the error of the reconstruction.
    mseSq = (testI{i} - reconI{i}).^2;
    mse = sum(mseSq(:)) / (h*w);
    psnr = 10 * log10( 256^2 / mse);
    errTot = errTot + psnr;
end

mag = 0:0.1:10.0;
Ytest = zeros(length(mag),numEigs);
for j = 1:length(mag),
    x = [ mag(j) ];
    Ytest(j,:) = sim_rbf(Xc, x, W, polyOrder, basisfunction);  
end    

figure;
hold on;
plot(mag, Ytest);
labels = cellstr( num2str([1:size(Xt,1)-1]') );  % labels correspond to their order
plot(Xt(1:end-1), Yt(1:end-1,:), 'o');
xlabel('Force (N)');
ylabel('PCA basis coordinate value');
legend('c_0', 'c_1');
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write the model data to a file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = 'model.dat';
fid = fopen(filename, 'w');

% Line 1: Write out the number of RBF points, number of eigen vectors,
%  and the width and height of the eigen textures.
fprintf(fid, '%d %d %d %d \n', numRBF, numEigs, w, h);

% Line 2: Write out the hemoglobin vector in RGB space.
fprintf(fid, '%g %g %g \n', hemoVec(1), hemoVec(2), hemoVec(3));

% Line 3+: Write out the RBF centers.
for i = 1:numRBF,
    fprintf(fid, '%g \n', Xc(i,1));
end

% ... followed by the corresponding weights of each center.
for i = 1:numRBF,
    for j = 1:numEigs,
        fprintf(fid, '%g ', W(i,j));
    end
    fprintf(fid, ' \n');
end

% Write out the mean log image, which is the neutral pressure image.
img = zeros(h,w,3);
img(:,:,1) = reshape(muLogI(1,:), [h w 1]);
img(:,:,2) = reshape(muLogI(2,:), [h w 1]);
img(:,:,3) = reshape(muLogI(3,:), [h w 1]);
filename = 'muLogI.tif';
writeTIFF(im2single(img), filename); 
fprintf(fid, '%s \n', filename);

% Write out the hemo logarithm basis.
for i = 1:numEigs,
    filename = sprintf('%s%d.tif', 'eig', i);
    img(:,:,1) = reshape(U(:,i), [h w 1]);
    img(:,:,2) = reshape(U(:,i), [h w 1]);
    img(:,:,3) = reshape(U(:,i), [h w 1]);
    writeTIFF(im2single(img), filename);
    fprintf(fid, '%s \n', filename);
end

% Done.
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Other validation Section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
validationImgFilenames = { ...
    'data\Marc\Marc_Ring_Neutral.png' ... 
    'data\Marc\Marc_Ring_Mag3.png' ...
    'data\Marc\Marc_Ring_Mag4.png' ...
    'data\Marc\Marc_Ring_Mag5.png' ...
    'data\Marc\Marc_Ring_Mag7.png' ...
    'data\Marc\Marc_Ring_Mag8.png' ...
    'data\Marc\Marc_Ring_Mag11.png' ...
};
validationPressureFilenames = { ...
    'data\Marc\Data001_neutral.csv' ... 
    'data\Marc\Data004_mag3.csv' ...
    'data\Marc\Data005_mag4.csv' ...
    'data\Marc\Data006_mag5.csv' ...
    'data\Marc\Data008_mag7.csv' ...
    'data\Marc\Data009_mag8.csv' ...
    'data\Marc\Data012_mag11.csv' ...
};
numValidationFiles = length(validationImgFilenames);
validationI = cell(numValidationFiles);
P2 = zeros(numValidationFiles, 16);
X2 = zeros(numValidationFiles,1);
for i = 1:numValidationFiles,
    % Image data.
    filename = char(validationImgFilenames(selectIdx(i)));
    fprintf(1, 'Loading validation image %s \n', filename);
    validationI{i} = im2double( imresize(imread(filename), [h w]) );
    % Pressure data.
    filename = char(validationPressureFilenames(i));
    fprintf(1, 'Loading validation pressure %s \n', filename);
    tmpP = loadPressure(filename, 4, 1, 1, maxPSI);
    P2(i,:) = tmpP(1,2:end);    
    [ posx, posy, X2(i,1) ] = computeCOP(reshape(P2(i,:), [4 4]));
    
end
logI2 = cell(numValidationFiles);
logR2 = zeros(numValidationFiles, h*w);
logG2 = zeros(numValidationFiles, h*w);
logB2 = zeros(numValidationFiles, h*w);
for i = 1:numValidationFiles,
    I = validationI{i};
    logR2(i,:) = -log( reshape(I(:,:,1), [1 h*w]) );
    logG2(i,:) = -log( reshape(I(:,:,2), [1 h*w]) );
    logB2(i,:) = -log( reshape(I(:,:,3), [1 h*w]) );
    logI2{i} = [ logR2(i,:); logG2(i,:); logB2(i,:) ];
end
muLogR2 = -log( reshape(validationI{1}(:,:,1), [1 h*w]) ); 
muLogG2 = -log( reshape(validationI{1}(:,:,2), [1 h*w]) ); 
muLogB2 = -log( reshape(validationI{1}(:,:,3), [1 h*w]) ); 
muLogI2 = [ muLogR2; muLogG2; muLogB2 ];

A2 = zeros(3, (numValidationFiles-1)*h*w);
len = h*w;
rng = 1:len;
for i = 1:numValidationFiles,
    A2(1,rng) = logR2(i,:) - muLogR2;
    A2(2,rng) = logG2(i,:) - muLogG2;
    A2(3,rng) = logB2(i,:) - muLogB2;
    rng = rng + len;
end

[PC] = myPCA(A2);

% Project each image onto the first PC vector, which is hemoglobin basis
hemoData2 = zeros(numValidationFiles, h*w);
hemoVec2 = PC(:,1);
for i = 1:numValidationFiles,
   hemoData2(i,:) = hemoVec2' * (logI2{i} - muLogI2);
end

hemoMu2 =  zeros(h*w,1); % mu is zero, since we've already removed the mean.
[U2,Eigs2] = pc_evectors(hemoData2', numEigs, hemoMu2);

% Generate a big image showing the hemoglobin content.
%
bigHemoImage = zeros([h 5*w]);
bigHemoImage(:,1:w) = reshape(hemoData(2,:), [h w 1]);
bigHemoImage(:,w+1:2*w) = reshape(hemoData(3,:), [h w 1]);
bigHemoImage(:,2*w+1:3*w) = reshape(hemoData(5,:), [h w 1]);
bigHemoImage(:,3*w+1:4*w) = reshape(hemoData(7,:), [h w 1]);
bigHemoImage(:,4*w+1:5*w) = reshape(hemoData(9,:), [h w 1]);
figure;
imagesc(bigHemoImage);
colormap('gray');
colorbar;
set(gca,'xtick',[]);
set(gca,'xticklabel',[]);
set(gca,'ytick',[]);
set(gca,'yticklabel',[]);

% Generate a big image showing the basis textures.
%
bigEigImage = zeros([h 5*w]);
bigEigImage(:,1:w) = reshape(U(:,1), [h w 1]);
bigEigImage(:,w+1:2*w) = reshape(U(:,2), [h w 1]);
bigEigImage(:,2*w+1:3*w) = reshape(U(:,3), [h w 1]);
bigEigImage(:,3*w+1:4*w) = reshape(U(:,4), [h w 1]);
bigEigImage(:,4*w+1:5*w) = reshape(U(:,5), [h w 1]);
h = figure;
imagesc(bigEigImage);
colormap('gray');
colorbar;
set(gca,'xtick',[]);
set(gca,'xticklabel',[]);
set(gca,'ytick',[]);
set(gca,'yticklabel',[]);
