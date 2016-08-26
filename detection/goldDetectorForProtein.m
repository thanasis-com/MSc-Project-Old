function [ thRes ] = goldDetectorForProtein( I, fobiddenLines )
%Detect gold markers in the nanodisc images. 
%This function implemented the gold marker detection method in the paper:
%R. Wang, H. Pokhariya, S. J. McKenna, and J. Lucocq. Recognition of immunogold
%markers in electron micrographs. Journal of Structural Biology, 176:151–158, 2011.
%
% Parameters:
%    I: input image
%    fobiddenLines: forbidden line region
%    
% Returns:
%    thRes: detected gold markers, 4 columns are corresponding to:
%           x-coordinate, y-coordinate, DoG response and its corresponding Gaussian scale
%
 

I = 255 - double(I);
[rows, cols] = size(I);

%% Gold marker detection parameters
rmin = 7;  % minimum radius for gold markers
rmax = 20;  % maximum radius for gold markers

InitalTh = 150; % threshold of pixel values, used to ignore possible wrong detections 
DOGTh = 25; % threshold for the response of difference of Gaussians (DoG)
hessianRate = 1.88;  % eigenvalue ratio for Hessian matrix

forbidWidS = fobiddenLines.forbidWidS;
forbidWidE = fobiddenLines.forbidWidE;
forbidHeiS = fobiddenLines.forbidHeiS;
forbidHeiE = fobiddenLines.forbidHeiE;

%% Gaussian scales
rmid = (rmin + rmax)/2;
sig = 0.6 * ( (rmid-(rmax-rmin+1)/2):0.5:(rmid+(rmax-rmin+1)/2) );
sigL = length(sig);

%% Calculate the response of difference of Gaussians 
DOGResponse = zeros(rows+2,cols+2,sigL);
for i=1:sigL
    n = round(8*sig(i));
    if mod(n,2) == 0
        n = n+1;
    end
    im1 = gaussfilt(n, sig(i), I);
    n = round(8*sig(i)*sqrt(2));
    if mod(n,2) == 0
        n = n+1;
    end    
    im2 = gaussfilt(n, sqrt(2)*sig(i), I);
    im3 = zeros(rows+2, cols+2);
    im3 = im3+min(min(im1-im2));
    im3(2:1+rows, 2:1+cols) = im1-im2;  
    DOGResponse(:,:,i) = im3; 

end

%% find the key points which have local maxima in the responses of DoG
res = keypoints(sigL, rows, cols, DOGResponse);
resL = length(res);
res = reshape(res, 4, resL/4);
res = res';
res(:,4) = sig(res(:,4)); % 4 columns in res are: x-coordinate, y-coordinate, 
                          % DoG response and its corresponding scale 


%% Ignore the detections which are too light
i = 1;
while (i<=size(res,1))
    if I(res(i,2), res(i,1))<InitalTh
        res(i,:) = [];
    else
        i = i+1;
    end
end


%% Ignore the detections whose DoG responses are too weak
res = sortrows(res,-3);
r = res(:,3);
lr = length(r);
for i = 1:lr;
    if r(i)<=DOGTh
        break;
    end
end
thRes = res(1:i-1,:);


%% Elimilate markers based on the distance between centers
sign = 0;
i = 2;
while(i <= size(thRes,1))
    for j = 1:i-1
        d = sqrt((thRes(i,1)-thRes(j,1))^2+(thRes(i,2)-thRes(j,2))^2);
        % if the distance is smaller than either of the two detections,
        % ignore the detection which has the smaller response
        if d<thRes(j,4)/0.6 || d<thRes(i,4)/0.6
            thRes(i,:) = [];
            sign = 1;
            break;
        end
    end    
    if sign == 0
        i = i+1;
    else
        sign = 0;
    end
end

%% Elimilate markers based on Hessian Matrix
i = 1;
while i<=size(thRes,1)
    sign = 0;
    j = 1;
    while j<=size(thRes,1)
        if j~=i
            d = sqrt((thRes(i,1)-thRes(j,1))^2+(thRes(i,2)-thRes(j,2))^2);
            if d < 2*(thRes(j,4)/0.6+thRes(i,4)/0.6)
                sign = 1;
                break;
            end
        end
        j = j+1;
    end 
    if sign == 0
        s = find(sig == thRes(i,4));
        im = DOGResponse(:,:,s);
        x = thRes(i,1)+1;
        y = thRes(i,2)+1;
        xx = (secondDerivatives( im, x+1, y, 1 ) - ...
            secondDerivatives( im, x-1, y, 1 ))/2;
        yy = (secondDerivatives( im, x, y+1, 2 ) - ...
            secondDerivatives( im, x, y-1, 2 ))/2;
        xy = (secondDerivatives( im, x, y+1, 1 ) - ...
            secondDerivatives( im, x, y-1, 1 ))/2; 
        H = [xx xy; xy yy];
        p = trace(H)^2/det(H);
        if p<0 || p>((hessianRate+1)^2/hessianRate)
            thRes(i,:) = [];
        else
            i = i+1;
        end
    else
        i = i+1;
    end
end


%% Forbidden line restriction
alpha=0:pi/20:2*pi; % [0,2*pi]
i = 1;
while i<=size(thRes,1)
    x = thRes(i,4)/0.6*cos(alpha);
    y = thRes(i,4)/0.6*sin(alpha);
    x = x+thRes(i,1);
    y = y+thRes(i,2);        
    if ~isempty(find(x>forbidWidE, 1)) || ~isempty(find(y<forbidHeiS, 1)) ...
            || isempty(find(x>forbidWidS, 1)) || isempty(find(y<forbidHeiE, 1))     
        thRes(i,:) = [];
    else
        i = i+1;
    end
end
    

end

