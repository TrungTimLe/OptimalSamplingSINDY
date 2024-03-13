function [InitialObservation, LoggedSignals] = LorenzResetFunction(eta,Xi_true,lambda)
% Randomize the initial state.
% x0 = [-8*2*rand() 8*2*rand() 27*2*rand()];  % Random initial condition of Lorenz system;
n = 3;
x0 = [-8 8 27];  % Fixed inital condition
% Return initial environment state variables 
Ts_init = 0.04;
% Create the inital data
Beta = [10; 28; 8/3];
noisy_x = x0 + eta*rand(1,3);
% Compute derivative
noisy_dx(:) = lorenz(0,x0,Beta) + eta*rand(3,1);
polyorder = 3;
% calculate F-norm
Theta = poolData(noisy_x,n,polyorder);
Xi_hat = sparsifyDynamics(Theta,noisy_dx,lambda,n);
Fnorm_error = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
LoggedSignals.State = [noisy_x(end,:) Ts_init Fnorm_error 0]';
LoggedSignals.CurrentNoisySample = [noisy_x noisy_dx];
LoggedSignals.CurrentSample = x0;
LoggedSignals.Time = 0;
InitialObservation = LoggedSignals.State;
end
