function [NextObs,Reward,IsDone,LoggedSignals] =...
    LorenzStepFunction(Action,LoggedSignals,EnvConstants)
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
n = 3;
if Action == 1 && LoggedSignals.State(4) < EnvConstants.Ts_high
    LoggedSignals.State(4) = LoggedSignals.State(4)*2;
end

if Action == -1 && LoggedSignals.State(4) > EnvConstants.Ts_low
    LoggedSignals.State(4) = LoggedSignals.State(4)/2;
end
LoggedSignals.Time = LoggedSignals.Time + LoggedSignals.State(4);
% Unpack the state vector from the logged signals.
Ts = LoggedSignals.State(4);
Lorenz_state = LoggedSignals.CurrentSample(end,:);
tspan= [Ts:Ts:3*Ts];
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[~,x_next] = ode45(@(t,x) lorenz(t,x,EnvConstants.Beta),tspan,Lorenz_state,options);
for i = 1:n
    LoggedSignals.State(i) = x_next(2,i) + EnvConstants.eta*rand(); %% Add noise to the state
end
noisy_dx_next =  lorenz(0,x_next(2,:),EnvConstants.Beta)' + EnvConstants.eta*rand(1,3);
% Transform state to observation.
% Compute reward R_SINDy
% Compute Sparse regression: sequential least squares
LoggedSignals.CurrentNoisySample = [LoggedSignals.CurrentNoisySample; ...
    [LoggedSignals.State(1:3)' noisy_dx_next]];
LoggedSignals.CurrentSample = [LoggedSignals.CurrentSample; x_next(2,:)];
x = LoggedSignals.CurrentNoisySample;
polyorder = 3;
Theta = poolData(LoggedSignals.CurrentNoisySample(:,1:3),n,polyorder);
Xi_hat = sparsifyDynamics(Theta,x(:,4:6),EnvConstants.lambda,n);
Fnorm_error = norm(abs(EnvConstants.Xi_true)-abs(Xi_hat),'fro')^2;
LoggedSignals.State(5) = Fnorm_error;
% Check terminal condition.
IsDone = Fnorm_error <= EnvConstants.tol;
% Compute mutual information
acf_x = autocorr(x(:,1));
acf_y = autocorr(x(:,2));
acf_z = autocorr(x(:,3));
MI = abs(mean([abs(acf_x(2)) abs(acf_y(2)) abs(acf_z(2))]));
LoggedSignals.State(6) = MI;
% LoggedSignals.State(7) = cond(Theta);
NextObs = LoggedSignals.State;
% Output reward.
if ~IsDone
    Reward = -EnvConstants.Lambda_SINDy*log(1+Fnorm_error)-EnvConstants.Lambda_MI*MI-...
     EnvConstants.Lambda_train*LoggedSignals.Time-EnvConstants.Lambda_MI*MI;
else
    Reward = 0;
end
