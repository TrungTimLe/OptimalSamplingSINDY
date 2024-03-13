clear all, close all, clc
%% Generate Data
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

type = 4;
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
%% Part 3: RL Observation and Action Specifications
% ObservationInfo = rlNumericSpec([6 1]);
% ObservationInfo.Name = 'Lorenz States and Sampling period';
% ObservationInfo.Description = 'x, y, z, Ts, Fnorm error, Mutual information';

ObservationInfo = rlNumericSpec([24 1]);
ObservationInfo.Name = 'Observations';
ObservationInfo.Description = 'States, Ts, Condition number, Mutual information, Trace(I_mat)';

ActionInfo = rlFiniteSetSpec([-1 0 1]);
ActionInfo.Name = 'Sampling Action';
%% Create the environment using the custom function handles
% Define environment constants
% Lambda_SINDy for R_SINDy
% Lambda in [0,1]
envConstants.Lambda_SINDy = 0;
% Lambda_TVDiff for mutual information
envConstants.Lambda_MI = 0;
% Penalty for training time
envConstants.Lambda_train = 0.01;
% Weight for log(condnum)
envConstants.Lambda_condnum = 0.01;
% Weight for log(trace)
envConstants.Lambda_trace = 0.01;
% The lower limit for Ts
envConstants.Ts_low = 1;
% The upper limit for Ts
envConstants.Ts_high = 64;
% The Lorenz parameters
% envConstants.Beta = params;
NSR = 0.005;
% The stopping criteria
if NSR == 0
    envConstants.tol = 0.001;
elseif NSR == 0.001
    envConstants.tol = 0.4;
else
    envConstants.tol = 8;
end

% The true Xi vector 
envConstants.Xi_true = Xi_true;
% The true Xi vector 
envConstants.lambda = 0.1;

StepHandle = @(Action,LoggedSignals) WendlingStepFunction(Action,LoggedSignals,envConstants);
% Use the same reset function, specifying it as a function handle rather than by using its name.
ResetHandle = @() WendlingResetFunction(NSR,type);
% Create the environment using the custom function handles.
env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandle,ResetHandle);

%% Validate custom functions
% Inital sample time
InitialObs = reset(env);
n_steps = 20;
convergence_step = 0;
for i = 1:n_steps
    [NextObs,Reward,IsDone,LoggedSignals] = step(env,0); 
    convergence_step = convergence_step + (IsDone == 0);
end
disp(convergence_step)
%% Create DQN Agent
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
% Create a deep neural network
dnn = [
featureInputLayer(obsInfo.Dimension(1),'Normalization','none','Name','state')
fullyConnectedLayer(12,'Name','CriticStateFC1')
reluLayer('Name','CriticRelu1')
fullyConnectedLayer(12, 'Name','CriticStateFC2')
reluLayer('Name','CriticCommonRelu')
fullyConnectedLayer(length(actInfo.Elements),'Name','output')];
% View the network confguration
% figure
% plot(layerGraph(dnn))
% Specify some training options for the critic representation 
criticOpts = rlRepresentationOptions('LearnRate',0.001,'GradientThreshold',1);
% Create the critic representation using the specifed neural network and options
critic = rlQValueRepresentation(dnn,obsInfo,actInfo,'Observation',{'state'},criticOpts);
% Create the DQN agent
agentOpts = rlDQNAgentOptions(...
'UseDoubleDQN',false, ...
'TargetSmoothFactor',1, ...
'TargetUpdateFrequency',4, ...
'ExperienceBufferLength',100000, ...
'DiscountFactor',0.99, ...
'MiniBatchSize',256);
agent = rlDQNAgent(critic,agentOpts);
%% Train DQN Agent
max_step = 500;
trainOpts = rlTrainingOptions(...
'MaxEpisodes',500, ...
'MaxStepsPerEpisode',max_step, ...
'Verbose',false, ...
'Plots','training-progress',...
'StopTrainingCriteria','AverageReward',...
'StopTrainingValue',10000);

% Train the agent.
tic
trainingStats = train(agent,env,trainOpts); 
elapsedTime = toc;
%% Simulate DQN Agent
if NSR ~= 0
    M = 100; % number of simulations
else
    M = 10;
end
totalReward = zeros(M,1);
best_sample_size = max_step;
sample_size = zeros(M,1);
train_time = zeros(M,1);
condnum = zeros(M,1);
Fnorm_error_last = zeros(M,1);
trace_vec = zeros(M,1);
for i = 1:M
    sampled_data_idx = 5:6;
    simOptions = rlSimulationOptions('MaxSteps',max_step);
    experience = sim(env,agent,simOptions);
    totalReward(i) = sum(experience.Reward);
    % Plot action
%     experience.Action.SamplingAction.Data(:)'
%     plot(experience.Action.SamplingAction.Data(:))
    % tmp = experience.Observation.LorenzStatesAndSamplingPeriod.Data;
    % data = reshape(experience.Observation.LorenzStatesAndSamplingPeriod.Data,6,length(tmp));
    % plot(data(1,:))
    % Evaluate the policy
    RL = experience.Action.SamplingAction.Data(:)';
    cx = experience.Observation.Observations.Data(5:12,:,:);
    dx = experience.Observation.Observations.Data(13:end,:,:);
    XRL = reshape(cx,n,length(cx))';
    dXRL = reshape(dx,12,length(dx))';
    % % RL
    k = 1;
    tnow = dt;
    Ts_low = 1;
    Ts_init = 1;
    % The upper limit for Ts
    Ts_high = 64;
    Ts = Ts_init;
    for j = 1:k
        tnow = tnow + dt;
    end
    idx = (tnow/dt);

    for j = 1:length(RL)
        tmp = RL(j);
        Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
            || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
        idx = idx + Ts;
        tnow = tnow + Ts*dt;
        sampled_data_idx(end+1) = ceil(tnow/dt)+4;
    end
    % Evaluate the policy
    XRL = [x_tmp(1,:); XRL];
    dXRL = [dx_tmp(1,:); dXRL];
    sample_size(i) = length(XRL);
    train_time(i) = dt*idx;
    Theta = poolData(XRL,n,polyorder);
    Theta = [Theta S_term(:,sampled_data_idx)'];
    Xi_hat = sparsifyDynamics(Theta,dXRL,envConstants.lambda,12);
    condnum(i) = log(cond(Theta));
    trace_vec(i) = real(log(real(trace(inv(Theta'*Theta)))));
    Fnorm_error_last(i) = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
    if sample_size(i) < best_sample_size
        best_sample_size = sample_size(i);
        best_act_seq = experience.Action.SamplingAction.Data(:)';
    end
end
idx = find(sample_size < max_step);
stat_mat = [sample_size(idx) Fnorm_error_last(idx) condnum(idx) trace_vec(idx) train_time(idx)];
stat_mean = [mean(stat_mat) length(idx)/M elapsedTime]';
stat_std = std(stat_mat);
stat_std = stat_std';
mean(totalReward(idx))
std(totalReward(idx))