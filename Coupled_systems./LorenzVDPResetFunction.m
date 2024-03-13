function [InitialObservation, LoggedSignals] = LorenzVDPResetFunction(envConstants)
n = 5;
x0 = [2 0 -8 8 27];
% Initialize parameters
par_vec = [envConstants.mu,envConstants.sigma,envConstants.rho,...
    envConstants.beta,envConstants.tau_fast,envConstants.tau_slow,...
    envConstants.c1,envConstants.c2];
% Return initial environment state variables 
Ts_init = 0.04;
% Create the inital data
noisy_x = x0 + envConstants.eta*rand(1,n);
% Compute derivative
noisy_dx = coupled_vdp_lorenz(0,x0,par_vec) + envConstants.eta*rand(n,1);
noisy_dx = noisy_dx';
polyorder = 3;
% calculate F-norm
Theta = poolData(noisy_x,n,polyorder);
Xi_hat = sparsifyDynamics(Theta,noisy_dx,envConstants.lambda,n);
Fnorm_error = norm(abs(envConstants.Xi_true)-abs(Xi_hat),'fro')^2;
LoggedSignals.State = [noisy_x(end,:) Ts_init Fnorm_error 0]';
LoggedSignals.CurrentNoisySample = [noisy_x noisy_dx];
LoggedSignals.CurrentSample = x0;
LoggedSignals.Time = 0;
InitialObservation = LoggedSignals.State;
end
