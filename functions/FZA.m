%% FZA mask generate
function mask = FZA(S,N,r1)
% S = 10;         % aperture diameter

[x,y] = meshgrid(linspace(-S/2,S/2-S/N,N));
r_2 = x.^2+y.^2;

mask = 0.5*(1 + cos(pi*r_2/r1^2));
% mask = imbinarize(mask,0.5);

end