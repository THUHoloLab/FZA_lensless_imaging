clearvars; clc; close all 

addpath('./functions');

Im = im2double(imread('./image/cameraman.png'));

Xc = 1760; % Xc = 1760;
Yc = 1700; % Yc = 1700;

di = 3;
z1 = 300; 

dp = 0.00375;      % pixel pitch
Nx = 2400;
Ny = 2400;

I = Im(Yc-Ny/2:Yc+Ny/2-1,Xc-Nx/2:Xc+Nx/2-1);    % Image cropping

figure(1),imagesc(I(831:1630,781:1580));title('Sensor measurements')
colormap gray;
axis equal off tight

%% Imaging processing

b = 30;
bi = 29.75;

fu_max = 0.5 / dp;
fv_max = 0.5 / dp;
du = 2*fu_max / Nx;
dv = 2*fv_max / Ny;

[u,v] = meshgrid(-fu_max:du:fu_max-du,-fv_max:dv:fv_max-dv);


H = 1i*exp(-1i*(pi^2/bi)*(u.^2 + v.^2));  % fresnel transfer function 

%% back propagation

Or = MyAdjointOperatorPropagation(I,H);

figure(2),imagesc(real(Or(831:1630,781:1580)));title('Reconstructed image (BP)')
colormap gray;
axis equal off tight
drawnow


%% Propagation operator (4)
A = @(obj) MyForwardOperatorPropagation(obj,H);  % forward propagation operator
AT = @(I) MyAdjointOperatorPropagation(I,H);  % backward propagation operator

%% TwIST algorithm (5)

% denoising function;
tv_iters = 5;
Psi = @(x,th) tvdenoise(x,2/th,tv_iters);
% TV regularizer;
Phi = @(x) TVnorm(x);

tau = 0.003; 
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

f_reconstruct = mat2gray(real(f_reconstruct));

figure(3),imagesc(f_reconstruct(831:1630,781:1580));title('Reconstructed image (CS)')
colormap gray;
axis image off
