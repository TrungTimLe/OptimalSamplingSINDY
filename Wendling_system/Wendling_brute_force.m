clear all, close all, clc
%% Generate Data
% Simulation settings
Npart = 50; %Number of parts for fourier transform including intial part
Tend = 10*Npart; %Total simulation time
NTST = Tend*10000; %Number of Time Steps
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

type = 4;
rng(type)
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
    otherwise
        disp('Invalid input!');
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


n_idx = 1000;
full_dx = [dx_true; uout]';
dx_true = full_dx;
dx_tmp = full_dx(5:n_idx,:);

x_noisy = x_true';
% for i = 1:size(x_noisy,2)
%     eta = sqrt(NSR*median(x_noisy(:,i).^2));
%     x_noisy(:,i) = x_noisy(:,i) + eta*randn(size(x_noisy,1),1);
% end
cx = x_noisy;
dt = deltat;

%% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 1;
x_tmp = x_true(:,5:n_idx)';
Theta = poolData(x_tmp,n,polyorder);
Theta = [Theta S_term(:,5:n_idx)'];
m = size(Theta,2);

% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx_tmp,lambda,12);
% poolDataLIST({'x0','y0','x1','y1','x2','y2','x3','y3'},Xi_true(1:9,1:8),n,polyorder);
%% Brute-force sampling
tic
MM = 1; % number of simulations
NSR = 0;


if NSR == 0
    tol = 0.001;
elseif NSR == 0.001
    tol = 0.4;
else
    tol = 8;
end
envConstants.Lambda_train = 0.01;
% Weight for log(condnum)
envConstants.Lambda_condnum = 0.01;
% Weight for log(trace)
envConstants.Lambda_trace = 0.01;


M = 10000;
max_step = 500;
T0 = 0.004;
p = [1/3 1/3 1/3];
best_sequence = zeros(MM,max_step);
tic
stat_mat = zeros(MM,5);
best_reward = zeros(MM,1);
% Generate noisy x
% find the eta vale
for i = 1:size(full_dx,2)
    % eta = sqrt(NSR*median(full_dx(100:end,ii).^2));
    cdx(:,i) = full_dx(:,i) + NSR*randn(size(full_dx,1),1);
end
dx_sample = cdx(5:6,:);
x_sample = cx(5:6,:);
% The lower and upper limit for Ts
Ts_high = 64;
Ts_low = 1;
for ii = 1:4
    tic
    best_sample_size = max_step;
    sample_size = zeros(M,1);
    train_time = zeros(M,1);
    condnum = zeros(M,1);
    Fnorm_error_last = zeros(M,1);
    trace_vec = zeros(M,1);
    reward = zeros(M,1);
    action_seq = zeros(MM,max_step);

    for i = 1:M
        XRL = x_sample;
        dXRL = dx_sample;
        sampled_data_idx = 5:6;
        tmp_seq = []; % action sequence   
        k = 1;
        tnow = 0;
        for j = 1:k
            tnow = tnow + dt;
        end
        idx = (tnow/dt);
        Fnorm_error_cur = 100;
        Ts = 1;
        while Fnorm_error_cur > tol && length(XRL) <= 500
            % Randomize the action
            r = mnrnd(1,p);
            action = find(r == 1);
            tmp = (action == 3 && Ts < Ts_high)*Ts*2 + (action == 2)*Ts + ...
                (action == 1 && Ts > Ts_low)*Ts/2;
            Ts = (tmp == 0)*Ts + (tmp ~= 0)*tmp;
            if action == 3 && tmp ~= 0 
                tmp_seq  = [tmp_seq  3];
            elseif action == 2 || tmp == 0
                tmp_seq  = [tmp_seq  2];
            else
               tmp_seq  = [tmp_seq  1];
            end
    
            % Add new data point
            tnow = tnow + Ts*dt;
            XRL = [XRL; cx(ceil(tnow/dt)+5,:)];
            dXRL = [dXRL; cdx(ceil(tnow/dt)+5,:)];
            sampled_data_idx(end+1) = ceil(tnow/dt)+5;
            Theta = poolData(XRL,n,polyorder);
            Theta = [Theta S_term(:,sampled_data_idx)'];
            Xi_hat = sparsifyDynamics(Theta,dXRL,lambda,12);
            Fnorm_error_cur = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
            % Update reward
            tmp1 = log(cond(Theta));
            tmp2 = real(log(real(trace(inv(Theta'*Theta)))));
            switch ii
                case 1
                    reward(i) = reward(i)-envConstants.Lambda_condnum*tmp1;
                case 2
                    reward(i) =  reward(i)-envConstants.Lambda_trace*tmp2;
                case 3
                    reward(i) =  reward(i)-envConstants.Lambda_train*tnow;
                case 4
                    reward(i) = reward(i)-envConstants.Lambda_trace*tmp2-envConstants.Lambda_condnum*tmp1-envConstants.Lambda_train*dt*idx;
            end 
        end
        tmp_seq(length(tmp_seq)+1:max_step) = 0;
        % Evaluate the policy
        action_seq(i,:) = tmp_seq;
        sample_size(i) = length(XRL);
        train_time(i) = tnow;
        Theta = poolData(XRL,n,polyorder);
        Theta = [Theta S_term(:,sampled_data_idx)'];
        condnum(i) = log(cond(Theta));
        trace_vec(i) = real(log(real(trace(inv(Theta'*Theta)))));
        Fnorm_error_last(i) = Fnorm_error_cur;
    end
    % Select the best policy
    selected_idx = find(~isnan(Fnorm_error_last));
    tmp = [sample_size(selected_idx) Fnorm_error_last(selected_idx) condnum(selected_idx) trace_vec(selected_idx) train_time(selected_idx)];
    switch ii
        case 1
            [max_t,idx] = min(condnum(selected_idx));
        case 2
            [max_t,idx] = max(trace_vec(selected_idx));
        case 3
            [max_t,idx] = min(train_time(selected_idx));
        case 4
            [max_t,idx] = max(reward(selected_idx));
    end
    reward1 = reward(selected_idx);
    action_seq1 = action_seq(selected_idx,:);
    best_reward(ii) = reward1(idx);
    best_sequence(ii,:) = action_seq1(idx,:);
    tmp_vec = tmp(idx,:);
    stat_mat(ii,:) = tmp_vec;
    elapsedTime(ii) = toc;
end
elapsedTime = elapsedTime';