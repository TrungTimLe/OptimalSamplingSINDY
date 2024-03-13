function [NextObs,Reward,IsDone,LoggedSignals] =...
    WendlingStepFunction(Action,LoggedSignals,EnvConstants)
n = 8;
if Action == 1 && LoggedSignals.State(1) < EnvConstants.Ts_high
    LoggedSignals.State(1) = LoggedSignals.State(1)*2;
end

if Action == -1 && LoggedSignals.State(1) > EnvConstants.Ts_low
    LoggedSignals.State(1) = LoggedSignals.State(1)/2;
end
dt = LoggedSignals.dt;
LoggedSignals.Time = LoggedSignals.Time + LoggedSignals.State(1)*dt;
% Unpack the state vector from the logged signals.
tnow = LoggedSignals.Time;
% Sample new data
cx = LoggedSignals.cx;
dx = LoggedSignals.dx;

LoggedSignals.Xcur = [LoggedSignals.Xcur; cx(ceil(tnow/dt)+5,:)];
LoggedSignals.dXcur = [LoggedSignals.dXcur; dx(ceil(tnow/dt)+5,:)];
LoggedSignals.sampled_data_idx(end+1) = ceil(tnow/dt)+5;
% Transform state to observation.
polyorder = 1;
Theta = poolData(LoggedSignals.Xcur,n,polyorder);
Theta = [Theta LoggedSignals.S_term(:,LoggedSignals.sampled_data_idx)'];
Xi_hat = sparsifyDynamics(Theta,LoggedSignals.dXcur,EnvConstants.lambda,12);
Fnorm_error = norm(abs(EnvConstants.Xi_true)-abs(Xi_hat),'fro')^2;
% Ts, Condition number, MI, Trace(I_mat)
LoggedSignals.State(2) = log(cond(Theta));
LoggedSignals.State(3) = real(log(real(trace(inv(Theta'*Theta)))));
% Check terminal condition.
IsDone = Fnorm_error <= EnvConstants.tol;
% IsDone = length(LoggedSignals.Xcur(:,1)) >= 200;
% Compute mutual information
x = LoggedSignals.Xcur;
acf_vec = zeros(size(x,2),1);
for i = 1:size(x,2)
    tmp = autocorr(x(:,i));
    acf_vec = abs(tmp(2));
end
MI = mean(acf_vec);
LoggedSignals.State(4) = MI;
LoggedSignals.State(5:end) = [LoggedSignals.Xcur(end,:) LoggedSignals.dXcur(end,:)];
% NextObs = LoggedSignals.State(2:4)';
% NextObs = [LoggedSignals.State(1:3) LoggedSignals.State(5:14)]';
NextObs = LoggedSignals.State';
% Output reward.
if ~IsDone
    % Reward = -EnvConstants.Lambda_condnum*LoggedSignals.State(2);
    % Reward = -EnvConstants.Lambda_trace*LoggedSignals.State(3);
    % Reward = -EnvConstants.Lambda_train*LoggedSignals.Time;
    Reward = -EnvConstants.Lambda_trace*LoggedSignals.State(3)-EnvConstants.Lambda_condnum*LoggedSignals.State(2)-EnvConstants.Lambda_train*LoggedSignals.Time;
else
    Reward = 0;
end