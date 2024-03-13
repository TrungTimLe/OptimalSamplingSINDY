function [InitialObservation, LoggedSignals] = ResetFunction_CVDPL(NSR)
n = 5;
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
% Initialize parameters
mu = 5;
c1 = 0.01;
c2 = 10;
F = 5;
Ts_init = 0.016;
sigma = 10; rho = 28; beta = 8/3;
tau_fast = 0.2;
tau_slow = F*tau_fast;
x0 = [2 0 -8 8 27];
par_vec = [mu,sigma,rho,beta,tau_fast,tau_slow,c1,c2];
polyorder = 3;

% Generate data for RL
dt = 0.004; % delta t
Tmax = 300;
tspan = [dt:dt:Tmax];
[t,x]=ode45(@(t,x) coupled_vdp_lorenz(t,x,par_vec),tspan,x0,options);
% Generate noisy x
% find the eta vale
for i = 1:n
    eta = sqrt(NSR*median(x(:,i).^2));
    nx(:,i) = x(:,i) + eta*randn(size(x,1),1);
end
% Filter noise                
T = dt;             % Sampling period       
L = size(x,1);             % Length of signal
t = (0:L-1)*T;        % Time vector
fl = 1:0.1:20;
Fs = 1/dt;
for dim = 1:n
    % find optimal cut-off frequency
    RMSE = [];
    for k = 1:length(fl)
        [b,a] = butter(5,fl(k)/(Fs/2),'low');
        clean_x_tmp = filtfilt(b,a,nx(:,dim));
        RMSE(k) = sqrt(mean((clean_x_tmp-x(:,dim)).^2));
    end

    [~,idx] = min(RMSE);
    [b,a] = butter(5,fl(idx)/(Fs/2),'low');
    cx(:,dim) = filtfilt(b,a,nx(:,dim));
end
if NSR == 0
    cx = x; % Noise-free
end
% Compute derivative using fourth order central difference
V = cx;
dV = zeros(length(V)-5,n);
for i = 3:length(V)-3
    for k = 1:n
        dV(i-2,k) = (1/(12*dt))*(-V(i+2,k)+8*V(i+1,k)-8*V(i-1,k)+V(i-2,k));
    end
end  
dx(3:length(V)-3,:) = dV;

% Randomize the initial state.
% x0 = [-8*2*rand() 8*2*rand() 27*2*rand()];  % Random initial condition of Lorenz system;
% Return initial environment state variables 

% Sample k more intial data points
tnow = dt;
k = 1;
Xtmp = x0;
dXtmp = coupled_vdp_lorenz(0,Xtmp,par_vec)';
for i = 1:k
    tnow = tnow + Ts_init;
    Xtmp = [Xtmp; cx(ceil(tnow/dt),:)];
    dXtmp = [dXtmp; dx(ceil(tnow/dt),:)];
end
% Generate full-period data with lowest Ts
% calculate F-norm
Theta = poolData(Xtmp,n,polyorder);
% Compute mutual information
acf_x = autocorr(Xtmp(:,1));
acf_y = autocorr(Xtmp(:,2));
acf_z = autocorr(Xtmp(:,3));
acf_u = autocorr(Xtmp(:,4));
acf_t = autocorr(Xtmp(:,5));
MI = abs(mean([abs(acf_x(2)) abs(acf_y(2)) abs(acf_z(2)) abs(acf_u(2)) abs(acf_t(2))]));
% Ts, Condition number, MI, Trace(I_mat)
LoggedSignals.State = [Ts_init log(cond(Theta)) real(log(real(trace(inv(Theta'*Theta))))) MI Xtmp(end,:) dXtmp(end,:)];
LoggedSignals.Xcur = Xtmp;
LoggedSignals.dXcur = dXtmp;
LoggedSignals.cx = cx;
LoggedSignals.dx = dx;
LoggedSignals.dt = dt;
LoggedSignals.Time = tnow;
% InitialObservation = LoggedSignals.State(2:4)';
% InitialObservation = [LoggedSignals.State(1:3) LoggedSignals.State(5:14)]';
InitialObservation = LoggedSignals.State';
end
