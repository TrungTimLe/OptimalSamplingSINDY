close all, clear all
%% Find Xi_true
%% Generate Data
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3; %number of state
x0=[-8; 8; 27];  % Initial condition

% Integrate
dt = 0.04;
tspan=[dt:dt:20];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);

%% Compute derivative using Total Variation Regularized Numerical Differentiation (TVDiff)
for i=1:length(x)
    ds_true(i,:) = lorenz(0,x(i,:),Beta);
end
% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);
% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,ds_true,lambda,n); 
poolDataLIST({'x','y','z'},Xi_true,n,polyorder);
%% Create the environment using the custom function handles
% Define environment constants
% Lambda_SINDy for R_SINDy
envConstants.Lambda_SINDy = 0.001;
% Lambda_TVDiff for R_TVDiff
envConstants.Lambda_TVDiff = 0.1;
% Lambda_TVDiff for mutual information
envConstants.Lambda_MI = 1000;
% Penalty for training time
envConstants.Lambda_train = 1;
% The lower limit for Ts
envConstants.Ts_low = 0.005;
% The upper limit for Ts
envConstants.Ts_high = 0.64;
% The Lorenz parameters
envConstants.Beta = [10; 28; 8/3];
% The Lorenz process noise level gamma
envConstants.Gamma = 0;
% The true Xi vector 
envConstants.Xi_true = Xi_true;
%% Validate
[~, LoggedSignals] = LorenzResetFunction(envConstants.Gamma);
Gt = 0;
IsDone = 0;
count = 0;
while ~IsDone
    [NextObs,Reward,IsDone,LoggedSignals] =...
        LorenzStepFunction(0,LoggedSignals,envConstants);
    count = count + 1;
    Gt = Gt + Reward;
end