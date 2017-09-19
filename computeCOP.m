function [x,y,mag] = computeCOP(P)

[rows,cols] = size(P);

PSI = sum( sum(P) );

if( PSI < 1e-9 ), % zero check
    PSI = 0.0;
    x = cols/2;
    y = rows/2;
else
    % Compute the center-of-pressure by linear interpolation of pressure.
    x = 0.0;
    y = 0.0;
    for i = 1:rows,
        for j = 1:cols,
            % Weighting by (P(i,j)/mag) ensures convexity.
            sc = (P(i,j) / PSI);
            x = x + j * sc;
            y = y + i * sc;
        end
    end   
end

% TekScan 4256E specs - each grid is 0.59in x 0.59in = 0.3481 / 16 = .0218
sqInPerSensel = 0.0218;
poundsToNewtons = 4.448;
mag = poundsToNewtons * (sqInPerSensel * PSI);
