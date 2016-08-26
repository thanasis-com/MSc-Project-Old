function gaussMatrix = gaussfilt(n, sd, F)
%Implement Gaussian filter 
% Parameters:
%    n: size of Gaussian filter
%    sd: standard deviation
%    F: image
%
% Returns:
%    gaussMatrix: image after using Gaussian filter 

gaussvec = zeros(1, n);
for i =1:n
    x_u = i-(n+1)/2;
    gaussvec(i) = exp(-0.5*(x_u/sd)^2)/(sqrt(2*pi)*sd);
end
gaussvec = gaussvec/sum(gaussvec);


if nargin == 3
    gaussMatrix = conv2(double(F), gaussvec, 'same');
    gaussMatrix = conv2(gaussMatrix, gaussvec', 'same');
else
    gaussMatrix = gaussvec'*gaussvec;
end

end
