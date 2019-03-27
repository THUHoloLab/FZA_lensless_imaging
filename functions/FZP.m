%% FZA mask generate
function mask = FZP(L,dx,b)
% L = 10;         % aperture diameter

N = L/dx;

[x,y] = meshgrid(linspace(-L/2,L/2-dx,N));
r2 = x.^2+y.^2;

mask = 0.5*(1 + cos(b*r2));
mask = imbinarize(mask,0.5);
% mask(r2>L^2/4) = 0;

end