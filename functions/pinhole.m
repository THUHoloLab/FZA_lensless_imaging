%% pinhole imaging function
function I = pinhole(O,di,x,y,z,Lx,dp,Nx)

if size(O,3) == 3
    O = rgb2gray(O);
end
[m,n] = size(O);
% Lx = 400;         % object width, unit:mm
Ly = m*Lx/n;      % object height, unit:mm

% do = 500;       % object distance
% di = 2;         % image distance
% dp = 0.01;      % pixel size
% Nx = 1024;      % pixels number

M = di/z;      % magnification
Lxi = M*Lx;       % image width
Lyi = M*Ly;
xi = M*x;
yi = M*y;
ds = Lxi/n;

Ny = Nx;
W = Nx*dp;      % sensor size
H = Ny*dp;
[X,Y] = meshgrid(linspace(xi-Lxi/2,xi+Lxi/2-ds,n),linspace(yi+Lyi/2-ds,yi-Lyi/2,m));
[Xq,Yq] = meshgrid(linspace(-W/2,W/2-dp,Nx),linspace(H/2-dp,-H/2,Ny));

I = interp2(X,Y,O,Xq,Yq,'linear',0);

end