function [InitialObservation, LoggedSignals] = WendlingResetFunction(NSR,type)
% Simulation settings
Npart = 50; %Number of parts for fourier transform including intial part
Tend = 10*Npart; %Total simulation time
NTST = Tend*2000; %Number of Time Steps
Nout = 5; %Output every 5th time step
deltat = Tend/NTST; %Time step size
n = 8; % Number of states
% Parameter settings
% Initiate parameters
A = 5; %Average excitory synaptic gain
B = 25; %Average slow inhibitory synaptic gain
G = 15; %Average fast inhibitory synaptic gain

a = 100; %Inverse of excitory time constant
b = 50; %Inverse of slow inhibitory time constant
g = 500; %Inverse of fast inhibitory time constant

C = 135; %Connectivity parameter
C1 = C;
C2 = 0.8*C;
C3 = 0.25*C;
C4 = 0.25*C;
C5 = 0.3*C;
C6 = 0.1*C;
C7 = 0.8*C;

pf = 90; %Mean i n p u t
sd = 10; %Standard deviation input
e0 = 2.5;
r = 0.56;
v0 = 6;
% rng(type)
switch type
    case 1 % Noisy perturbation around an equilibrium
        B = 50;
    case 2 % Spikes Wave Discharges (SWD) 
        B = 40;
    case 3 % Sustained SWDs
        B = 25;
    case 4 % Slow rhythmic activity
        B = 10;
    case 5 % Rapid low-voltage activity
        B = 5;
        G = 25;
    case 6 % Quasi-sinusoidal activity
        B = 15;
        G = 0;
end

params = struct();
params.A = A;  % Average excitatory synaptic gain (mV)
params.B = B;    % Average slow inhibitory synaptic gain (mV)
params.G = G;    % Average fast inhibitory synaptic gain (mV)
params.a = a;   % Reciprocal of the excitatory time constant (Hz)
params.b = b;    % Reciprocal of the slow inhibitory time constant (Hz)
params.g = g;   % Reciprocal of the fast inhibitory time constant (Hz)
params.C = C;   % General connectivity constant
params.C1 = params.C;     % Connectivity: pyramidal to excitatory cells
params.C2 = 0.8 * params.C;  % Connectivity: excitatory to pyramidal cells
params.C3 = 0.25 * params.C; % Connectivity: pyramidal to slow inhibitory cells
params.C4 = 0.25 * params.C; % Connectivity: slow inhibitory to pyramidal cells
params.C5 = 0.3 * params.C;  % Connectivity: pyramidal to fast inhibitory cells
params.C6 = 0.1 * params.C;  % Connectivity: slow to fast inhibitory cells
params.C7 = 0.8 * params.C;  % Connectivity: fast inhibitory to pyramidal cells
params.e0 = e0;   % Half of the max firing rate (Hz)
params.r = r;   % Steepness parameter (mV^-1)
params.v0 = v0;     % Potential at half of the max firing rate (mV)
params.I = pf;     % External input to the cortical column (assumed constant part in Hz)

% Initial conditions
x = zeros(4,1);
y = zeros(4,1);
dx_true = zeros(8,ceil(NTST/Nout)+1);
x_true = zeros(8,ceil(NTST/Nout)+1);
S_term = zeros(4,ceil(NTST/Nout)+1);
cdx = zeros(ceil(NTST/Nout)+1,12);
% Initialize variables for computation
Im = eye(4,4);

Q1 = -deltat*diag([a^2, a^2 ,b^2, g^2]);
Q2 = Im-2*deltat*diag([a, a , b , g ]);
Q3 = deltat*diag([A*a ,A*a ,B*b ,G*g ]);

fd = [0 ; A*a*pf/C2; 0 ; 0]*deltat;
fs = [0 ; A*a/C2 ; 0 ; 0 ]*sqrt(deltat)*sd*randn(1,NTST) ;

P= [[0 , C2 , -C4,-C7 ] ;
    [C1 , 0 , 0 , 0];
    [C3 , 0 , 0 , 0];
    [C5, 0 , -C6 , 0]];

u = P*x;
uout = zeros(4,ceil(NTST/Nout)+1) ;
uout(:,1) = u;

xn = zeros(4,1);
yn = zeros(4,1);

tel = 0;
indexplot = 2;

for i = 1:NTST
    xn = x + deltat*y;
    yn = Q1*x + Q2*y + Q3*(2*e0./(1+exp(r.*(v0*ones(4,1)-u)))) + fd + fs(:,i);
    % yn = Q1*x + Q2*y + Q3*(2*e0./(1+exp(r.*(v0*ones(4,1)-u)))) + fd;
    x = xn;
    y = yn;
    u = P*x;
    x_vec = [x(1) y(1) x(2) y(2) x(3) y(3) x(4) y(4)]';
    S_term(:,indexplot) = 2*e0./(1+exp(r.*(v0*ones(4,1)-u)));
    dx_true(:,indexplot) = Wendling_dx(x_vec, params);
    x_true(:,indexplot) = x_vec;
    %Saves u for output every Nout iterations
    tel = tel + 1;
    if tel == Nout
        tel = 0;
        uout(:,indexplot) = u;
        indexplot = indexplot + 1;
    end
end

% Generate noisy x
% find the eta vale
cx = x_true';
dt = deltat;
full_dx = [dx_true; uout]';

for i = 1:size(full_dx,2)
    % eta = sqrt(NSR*median(full_dx(100:end,ii).^2));
    cdx(:,i) = full_dx(:,i) + NSR*randn(size(full_dx,1),1);
end
dXtmp = cdx(5:6,:);
Xtmp = cx(5:6,:);
sampled_data_idx = 5:6;
% Number of pre-sampled data points
k = 1;
tnow = 0;
for j = 1:k
    tnow = tnow + dt;
end
% Generate full-period data with lowest Ts
% calculate F-norm
polyorder = 1;
Theta = poolData(Xtmp,n,polyorder);
Theta = [Theta S_term(:,sampled_data_idx)'];
% Compute mutual information
acf_vec = zeros(size(Xtmp,2),1);
for i = 1:size(Xtmp,2)
    tmp = autocorr(Xtmp(:,i));
    acf_vec = abs(tmp(2));
end

MI = mean(acf_vec);
Ts_init = 1;
% Ts, Condition number, MI, Trace(I_mat)
LoggedSignals.State = [Ts_init real(log(cond(Theta))) real(log(real(trace(inv(Theta'*Theta))))) MI Xtmp(end,:) dXtmp(end,:)];
LoggedSignals.Xcur = Xtmp;
LoggedSignals.dXcur = dXtmp;
LoggedSignals.cx = cx;
LoggedSignals.dx = cdx;
LoggedSignals.S_term = S_term;
LoggedSignals.sampled_data_idx = sampled_data_idx;
LoggedSignals.dt = dt;
LoggedSignals.Time = tnow;
% InitialObservation = LoggedSignals.State(2:4)';
% InitialObservation = [LoggedSignals.State(1:3) LoggedSignals.State(5:14)]';
InitialObservation = LoggedSignals.State';

end
