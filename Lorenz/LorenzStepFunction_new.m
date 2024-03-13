function [NextObs,Reward,IsDone,LoggedSignals] =...
    LorenzStepFunction_new(Action,LoggedSignals,EnvConstants)
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
LoggedSignals.Xcur = [LoggedSignals.Xcur; cx(ceil(tnow/LoggedSignals.dt),:)];
LoggedSignals.dXcur = [LoggedSignals.dXcur; dx(ceil(tnow/LoggedSignals.dt),:)];
% Transform state to observation.
polyorder = 2;
Theta = poolData(LoggedSignals.Xcur,n,polyorder);
Xi_hat = sparsifyDynamics(Theta,LoggedSignals.dXcur,EnvConstants.lambda,n);
Fnorm_error = norm(abs(EnvConstants.Xi_true)-abs(Xi_hat),'fro')^2;
% Ts, Condition number, MI, Trace(I_mat)
LoggedSignals.State(2) = log(cond(Theta));
LoggedSignals.State(3) = real(log(real(trace(inv(Theta'*Theta)))));
% Check terminal condition.
IsDone = Fnorm_error <= EnvConstants.tol;
% IsDone = length(LoggedSignals.Xcur(:,1)) >= 50;
% Compute mutual information
x = LoggedSignals.Xcur;
acf_x = autocorr(x(:,1));
acf_y = autocorr(x(:,2));
acf_z = autocorr(x(:,3));
MI = abs(mean([abs(acf_x(2)) abs(acf_y(2)) abs(acf_z(2))]));
LoggedSignals.State(4) = MI;
LoggedSignals.State(5:7) = LoggedSignals.Xcur(end,:);
LoggedSignals.State(8:10) = LoggedSignals.dXcur(end,:);
% NextObs = [LoggedSignals.State(2) LoggedSignals.State(4)]';
% NextObs = LoggedSignals.State(2:4)';
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
