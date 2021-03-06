%
% Loads TekScan pressure data from a CSV file.
%
% Inputs:
%   filename = the CSV file to load
%   fingerId = 1 = thumb ... 5 = pinky
%   frameCount = range starting from 0
%   headerlineCount = amount of lines to skip at the start of the .csv
%   P = pressureData (a 1x2 cell) where: 
%               pData{1} = the frame #
%               pData{1}{frameNum} = access specific frame number
%               pData{2} = the frame's pressure data  
%               pData{2}{frameNum} = access specific frame's data
%
% Related info: 
%   www.mathworks.com/help/matlab/examples/reading-arbitrary-format-text-files-with-textscan.html
%
function [P] = loadPressure(filename, fingerId, frameCount, fps, maxPSI)

assert(nargin > 3 && fingerId > 0, 'Usage: [P] = loadPressure(filename, fingerId, frameCount, fps)' );

sc = (maxPSI / 255.0);

% Open the .csv file
fid = fopen(filename, 'r');  

% Skip the header lines. 
% We edit the CSV files to remove the header generated by TekScan. 
% Otherwise, set headerlineCount = 26 to skip the TekScan header.
headerlineCount = 0;
textscan(fid, '%*s', headerlineCount, 'delimiter', '\n');

% Set formatting for fingers (not thumb)
if( fingerId > 1 ),
    format = strcat('%*', int2str(12 + (fingerId - 2)*10), 'c %f %*c %f %*c %f %*c %f %*s');
else
    format = '%f %*c %f %*c %f %*c %f %*s';
end

startFrame = 0;
endFrame = startFrame + (frameCount - 1);
frameSize = 16; % 4x4 pressure data
P = zeros(frameCount, frameSize + 1); % each frame is timestamp + pressure

if( fingerId == 1 ), % case for thumb
    
    for i = startFrame:endFrame,
        
        % Read the frame number
        frameInfo = textscan(fid, '%*s %s', 1, 'whitespace', ', '); %optional: get the current frame from the csv file
        textscan(fid, '%*s', 1, 'whitespace', ', ', 'delimiter', '\n', 'collectOutput', 1);
        P(i+1,1) = i / fps; % Optional: use frameInfo{1}
        
        % Read the pressure data
        pressureStr = textscan(fid, format, 4, ...
            'headerlines', 15, 'delimiter', '\n', 'CollectOutput', 1); % pressureStr stores the original 4x4 configuration
        P(i+1,2:end) = sc * reshape(pressureStr{1}, 1, frameSize);
        
        % Skip the empty data
        textscan(fid, '%*s', 11, 'delimiter', '\n', 'CollectOutput', 0); 
    end    
    
else  % case for other fingers
    
    for i = startFrame:endFrame,
        % Read the frame number
        frameInfo = textscan(fid, '%*s %s', 1, 'Whitespace', ', '); 
        textscan(fid, '%*s', 1, 'Whitespace', ', ', 'delimiter', '\n', 'CollectOutput', 1);
        P(i+1,1) = i / fps; % Optional: use frameInfo{1}
        
        %Get the pressure data
        pressureStr = textscan(fid, format, 4, 'delimiter', '\n', 'CollectOutput', 1); %tmp stores the original 4x4 configuration
        pressureData = pressureStr{1};
        P(i+1,2:end) = sc * reshape(pressureData, 1, frameSize);

        % Skip the empty data
        textscan(fid, '%*s', 26, 'delimiter', '\n', 'CollectOutput', 0);
    end    
    
end

% Done. Close the file
fclose(fid);

