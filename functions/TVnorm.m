function y = TVnorm(x)

[Nx,Ny,Nz] = size(x);
x = reshape(x,Nx,Ny*Nz);

y = sum(sum(sqrt(diffh(x).^2+diffv(x).^2)));
