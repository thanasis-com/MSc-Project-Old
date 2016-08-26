function derivate = secondDerivatives( im, x, y, n )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if n == 1
    derivate = xDerivative(im, x, y);
end

if n== 2
    derivate = yDerivative(im, x, y);
end

end


function Rx = xDerivative(im, x, y)

Rx = (im(y,x+1) - im(y,x-1))/2;

end

function Ry = yDerivative(im, x, y)

Ry = (im(y+1,x) - im(y-1,x))/2;

end
