clearvars; clc; close all
% FZA_lensless_imaging
addpath('./functions');

%% Pingole imaging
img = im2double(imread('.\image\THU.png'));
% img = im2double(imread('cameraman.tif'));

di = 3;
z1 = 20;    x1 = 0;    y1 = 0;

Lx1 = 20;

dp = 0.01;
Nx = 512;
Ny = 512;

Im = pinhole(img,di,x1,y1,z1,Lx1,dp,Nx);
figure,imagesc(Im);title('Original image')
colormap gray;
axis image off

%% Imaging processing
S = 2*dp*Nx;         % aperture diameter
% b = pi/(Nx*dp^2);  % optimal beta

b = 30;
M = di/z1;
bi = b/(1+M)^2;

fu_max = 0.5 / dp;
fv_max = 0.5 / dp;
du = 2*fu_max / (Nx);
dv = 2*fv_max / (Ny);

[u,v] = meshgrid(-fu_max:du:fu_max-du,-fv_max:dv:fv_max-dv);
H = 1i*exp(-1i*(pi^2/bi)*(u.^2 + v.^2));  % fresnel transfer function 

mask = FZP(S,dp,bi);

I = conv2(Im,mask,'same');
% I1 = MyForwardOperatorPropagation(Im,H);

figure,imagesc(mask);title('FZA pattern')
colormap gray;
axis image off
figure,imagesc(I);title('Observed imaging')
colormap gray;
axis image off

%% back propagation

Or = MyAdjointOperatorPropagation(I,H);

figure,imagesc(real(Or));title('Reconstructed image (BP)')
colormap gray;
% colorbar
axis equal off tight

%% Propagation operator (4)
A = @(obj) MyForwardOperatorPropagation(obj,H);  % forward propagation operator
AT = @(I) MyAdjointOperatorPropagation(I,H);  % backward propagation operator

%% TwIST algorithm (5)

% denoising function;
tv_iters = 2;
Psi = @(x,th) tvdenoise(x,2/th,tv_iters);
% TV regularizer;
Phi = @(x) TVnorm(x);
% Phi = @(x) sum(sum(sqrt(diffh(x).^2+diffv(x).^2)));

tau = 100; 
tolA = 1e-6;
iterations = 50;
[f_reconstruct,dummy,obj_twist,...
    times_twist,dummy,mse_twist]= ...
    TwIST(I,A,tau,...
    'AT', AT, ...
    'Psi',Psi,...
    'Phi',Phi,...
    'Initialization',2,...
    'Monotone',1,...
    'StopCriterion',1,...
    'MaxIterA',iterations,...
    'MinIterA',iterations,...
    'ToleranceA',tolA,...
    'Verbose', 1);

figure;imagesc(real(f_reconstruct));title('Reconstructed image (CS)')
colormap gray;
axis image off
