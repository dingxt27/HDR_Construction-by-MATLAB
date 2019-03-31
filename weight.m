 function w = weight(z)

Zmin = 1;
Zmax = 256;

if z<=0.5*(Zmin+Zmax)
    w = z - Zmin;
else
    w = Zmax - z;
end

end
