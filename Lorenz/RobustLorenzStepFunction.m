function [NextObs,Reward,IsDone,LoggedSignals] =...
    RobustLorenzStepFunction(Action,LoggedSignals,EnvConstants)
n = 3;
if Action == 1 && LoggedSignals.Ts < EnvConstants.Ts_high
    LoggedSignals.Ts = LoggedSignals.Ts*2;
end

if Action == -1 && LoggedSignals.Ts > EnvConstants.Ts_low
    LoggedSignals.Ts = LoggedSignals.Ts/2;
end
LoggedSignals.Time = LoggedSignals.Time + LoggedSignals.Ts;
% Unpack the state vector from the logged signals.
Ts = LoggedSignals.Ts;
Lorenz_state = LoggedSignals.CurrentSample(end,:);
tspan= [Ts:Ts:3*Ts];
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[~,x_next] = ode45(@(t,x) lorenz(t,x,EnvConstants.Beta),tspan,Lorenz_state,options);
noisy_x(1,1:3) = x_next(2,1:3) + EnvConstants.eta*rand(1,3); %% Add noise to the state
noisy_dx_next =  lorenz(0,x_next(2,:),EnvConstants.Beta)' + EnvConstants.eta*rand(1,3);
% Transform state to observation.
% Compute reward R_SINDy
% Compute Sparse regression: sequential least squares
LoggedSignals.CurrentNoisySample = [LoggedSignals.CurrentNoisySample; ...
    [noisy_x noisy_dx_next]];
LoggedSignals.CurrentSample = [LoggedSignals.CurrentSample; x_next(2,:)];
x = LoggedSignals.CurrentNoisySample;
polyorder = 3;
Theta = poolData(LoggedSignals.CurrentNoisySample(:,1:3),n,polyorder);
Xi_hat = sparsifyDynamics(Theta,x(:,4:6),EnvConstants.lambda,n);
LoggedSignals.Xi_hat = Xi_hat;
cond_theta = cond(Theta);
% Check terminal condition.
IsDone = cond_theta <= EnvConstants.cond_tol && size(x,1) >= EnvConstants.n_steps;
NextObs = LoggedSignals.State;
% Output reward.
if ~IsDone
%     Reward = -EnvConstants.Lambda_SINDy*log(1+Fnorm_error)-EnvConstants.Lambda_MI*MI-...
%      EnvConstants.Lambda_train*LoggedSignals.Time-EnvConstants.Lambda_cond*cond_theta;
    Reward = -EnvConstants.Lambda_cond*log(cond_theta)-EnvConstants.Lambda_train*LoggedSignals.Time;
else
    Reward = 0;
end
