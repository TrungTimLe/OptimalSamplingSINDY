function [NextObs,Reward,IsDone,LoggedSignals] =...
    StepFunction_CVDP(Action,LoggedSignals,EnvConstants)
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
if Action == 1 && LoggedSignals.State(1) < EnvConstants.Ts_high
    LoggedSignals.State(1) = LoggedSignals.State(1)*2;
end

if Action == -1 && LoggedSignals.State(1) > EnvConstants.Ts_low
    LoggedSignals.State(1) = LoggedSignals.State(1)/2;
end
LoggedSignals.Time = LoggedSignals.Time + LoggedSignals.State(1);
% Unpack the state vector from the logged signals.
Ts = LoggedSignals.State(1);
tnow = LoggedSignals.Time;
% Sample new data
cx = LoggedSignals.cx;
dx = LoggedSignals.dx;
idx = ceil(tnow/LoggedSignals.dt);
if idx > 38000
    idx = randi([1000, 5000],1,1);
end
LoggedSignals.Xcur = [LoggedSignals.Xcur; cx(idx,:)];
LoggedSignals.dXcur = [LoggedSignals.dXcur; dx(idx,:)];
% Transform state to observation.
polyorder = 3;
Theta = poolData(LoggedSignals.Xcur,n,polyorder);
Xi_hat = sparsifyDynamics(Theta,LoggedSignals.dXcur,EnvConstants.lambda,n);
Fnorm_error = norm(abs(EnvConstants.Xi_true)-abs(Xi_hat),'fro')^2;
% Ts, Condition number, MI, Trace(I_mat)
LoggedSignals.State(2) = real(log(cond(Theta)));
LoggedSignals.State(3) = real(log(real(trace(inv(Theta'*Theta)))));
% Check terminal condition.
IsDone = Fnorm_error <= EnvConstants.tol;
% IsDone = length(LoggedSignals.Xcur(:,1)) >= 500;
% Compute mutual information
x = LoggedSignals.Xcur;
acf_x = autocorr(x(:,1));
acf_y = autocorr(x(:,2));
acf_z = autocorr(x(:,3));
acf_u = autocorr(x(:,4));
MI = abs(mean([abs(acf_x(2)) abs(acf_y(2)) abs(acf_z(2)) abs(acf_u(2))]));
LoggedSignals.State(4) = MI;
LoggedSignals.State(5:12) = [LoggedSignals.Xcur(end,:) LoggedSignals.dXcur(end,:)];
% NextObs = LoggedSignals.State(2:4)';
% NextObs = [LoggedSignals.State(1:3) LoggedSignals.State(5:14)]';
NextObs = LoggedSignals.State';
% Output reward.
if ~IsDone
    % Reward = -EnvConstants.Lambda_condnum*LoggedSignals.State(2);
        % Reward = -EnvConstants.Lambda_trace*LoggedSignals.State(3);
    % Reward = -EnvConstants.Lambda_train*LoggedSignals.Time;
        Reward = -EnvConstants.Lambda_trace*LoggedSignals.State(3)-EnvConstants.Lambda_condnum*LoggedSignals.State(2)-EnvConstants.Lambda_train*LoggedSignals.Time;
elseif idx > 38000
    Reward = -40;
else
    Reward = 0;
end
