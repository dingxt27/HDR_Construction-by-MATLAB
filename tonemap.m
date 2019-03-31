function M = tonemap(Enorm3)

L1 = rgb2gray(Enorm3);
L2 = log(L1);
sumL = sum(L2(:));
n = size(Enorm3,1)*size(Enorm3,2);
Lavg = exp((1/n)*sumL);
a = 0.18;
T = (a/Lavg)*L1;
Tmax2 = max(T(:))*max(T(:));
b = 1+ T/Tmax2;
% Ltone = T.*b/(1+T);
Ltone = zeros(480,640);
for p=1:480
    for h=1:640
        Ltone(p,h) = (T(p,h)*b(p,h))/(1+T(p,h));
    end
end

M = zeros(480,640);
for k1 = 1:480
    for k2 = 1:640
        M(k1,k2) = Ltone(k1,k2)/L1(k1,k2);
    end
end


end


