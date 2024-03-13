function [NextObs,Reward,IsDone,LoggedSignals] =...
    CoupledVDPStepFunction(Action,LoggedSignals,envConstants)
% Custom step function to construct cart-pole environment for the function
% handle case.
%
% This function applies the given action to the environment and evaluates
% the system dynamics for one simulation step.
% Check if the given action is valid.
% if Action == 1 && 
% error('Action must be %g for going left and %g for going right.',...
% -EnvConstants.MaxForce,EnvConstants.MaxForce);
% end
n = 4;
if Action == 1 && LoggedSignals.State(n+1) < envConstants.Ts_high
    LoggedSignals.State(n+1) = LoggedSignals.State(n+1)*2;
end

if Action == -1 && LoggedSignals.State(n+1) > envConstants.Ts_low
    LoggedSignals.State(n+1) = LoggedSignals.State(n+1)/2;
end
% Unpack the state vector from the logged signals.
LoggedSignals.Time = LoggedSignals.Time + LoggedSignals.State(n+1);
Ts = LoggedSignals.State(n+1);
state = LoggedSignals.CurrentSample(end,:);
tspan= [Ts:Ts:3*Ts];
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
par_vec = [envConstants.mu1,envConstants.mu2,envConstants.tau_fast,...
    envConstants.tau_slow,envConstants.c1,envConstants.c2];
[~,x_next] = ode45(@(t,x) coupled_vdp(t,x,par_vec),tspan,state,options);
for i = 1:n
    LoggedSignals.State(i) = x_next(2,i) + envConstants.eta*rand(); %% Add noise to the state
end
noisy_dx_next =  coupled_vdp(0,x_next(2,:),par_vec)' + envConstants.eta*rand(1,n);
% Transform state to observation.
% Compute reward R_SINDy
% Compute Sparse regression: sequential least squares
LoggedSignals.CurrentNoisySample = [LoggedSignals.CurrentNoisySample; ...
    [LoggedSignals.State(1:n)' noisy_dx_next]];
LoggedSignals.CurrentSample = [LoggedSignals.CurrentSample; x_next(2,:)];
x = LoggedSignals.CurrentNoisySample;
polyorder = 3;
Theta = poolData(LoggedSignals.CurrentNoisySample(:,1:n),n,polyorder);
Xi_hat = sparsifyDynamics(Theta,x(:,(n+1):(n+4)),envConstants.lambda,n);
Fnorm_error = norm(abs(envConstants.Xi_true)-abs(Xi_hat),'fro')^2;
LoggedSignals.State(n+2) = Fnorm_error;
% Check terminal condition.
IsDone = Fnorm_error <= envConstants.tol;
% Compute mutual information
acf_x = autocorr(x(:,1));
acf_y = autocorr(x(:,2));
acf_z = autocorr(x(:,3));
acf_t = autocorr(x(:,4));
MI = abs(mean([acf_x(2) acf_y(2) acf_z(2) acf_t(2)]));
LoggedSignals.State(n+3) = MI;
NextObs = LoggedSignals.State;
% Output reward.
if ~IsDone
    Reward = -envConstants.Lambda_SINDy*log(1+Fnorm_error)-envConstants.Lambda_MI*MI-...
        envConstants.Lambda_train*LoggedSignals.Time;
else
    Reward = 0;
end
