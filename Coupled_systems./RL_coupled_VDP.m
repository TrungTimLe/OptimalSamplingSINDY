clear all, close all, clc
%% Generate Data
n = 4;
% Initialize parameters
mu1 = 5;
mu2 = 4;
c1 = 0.005;
c2 = 1;
F = 5;
tau_fast = 0.2;
tau_slow = F*tau_fast;
x0 = [2,0,0,2];
par_vec = [mu1,mu2,tau_fast,tau_slow,c1,c2];

% Integrate
dt = 0.04;
tspan = [dt:dt:500*dt];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x] = ode45(@(t,x) coupled_vdp(t,x,par_vec),tspan,x0,options);
% Compute derivatives
dx = zeros(length(x),n);
for i = 1:length(x)
    dx(i,:) = coupled_vdp(0,x(i,:),par_vec);
end
% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx,lambda,n);
% poolDataLIST({'x1','x2','x3','x4'},Xi_true,n,polyorder);

%% Part 3: RL Observation and Action Specifications
ObservationInfo = rlNumericSpec([7 1]);
ObservationInfo.Name = 'System States and Sampling period';
ObservationInfo.Description = 'x1, x2, x3, x4, x5, Ts, Fnorm error, Mutual information';

ActionInfo = rlFiniteSetSpec([-1 0 1]);
ActionInfo.Name = 'Sampling Action';
%% Create the environment using the custom function handles
% Define environment constants
% Penalty for training time
envConstants.Lambda_train = 0;
% Lambda_SINDy for R_SINDy
envConstants.Lambda_SINDy = 0.01;
% Lambda_TVDiff for mutual information
envConstants.Lambda_MI = 0;
% The lower limit for Ts
envConstants.Ts_low = 0.001;
% The upper limit for Ts
envConstants.Ts_high = 0.64;
% The system parameters
F = 5;
envConstants.mu1 = 5;
envConstants.mu2 = 4;
envConstants.tau_fast = 0.2;
envConstants.tau_slow = envConstants.tau_fast*F;
envConstants.c1 = 0.005; 
envConstants.c2 = 1;
% The Lorenz process noise level eta
envConstants.eta = 0;
% The stopping criteria
envConstants.tol = 0.001;
% The true Xi vector 
envConstants.Xi_true = Xi_true;
% The true Xi vector 
envConstants.lambda = 0.1;

StepHandle = @(Action,LoggedSignals) CoupledVDPStepFunction(Action,LoggedSignals,envConstants);
% Use the same reset function, specifying it as a function handle rather than by using its name.
ResetHandle = @() CoupledVDPResetFunction(envConstants);
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
trainOpts = rlTrainingOptions(...
'MaxEpisodes',500, ...
'MaxStepsPerEpisode',500, ...
'Verbose',false, ...
'Plots','training-progress',...
'StopTrainingCriteria','AverageReward',...
'StopTrainingValue',10000);

% Train the agent.
trainingStats = train(agent,env,trainOpts); 

%% Simulate DQN Agent
M = 5; % number of simulations
totalReward = zeros(M,1);
max_reward = -100000;
for i = 1:M
    simOptions = rlSimulationOptions('MaxSteps',500);
    experience = sim(env,agent,simOptions);
    totalReward(i) = sum(experience.Reward);
    if totalReward(i) > max_reward
        max_reward = totalReward(i);
        best_act_seq = experience.Action.SamplingAction.Data(:)';
    end
%     % Plot action
%     experience.Action.SamplingAction.Data(:)'
%     plot(experience.Action.SamplingAction.Data(:))
    % tmp = experience.Observation.LorenzStatesAndSamplingPeriod.Data;
    % data = reshape(experience.Observation.LorenzStatesAndSamplingPeriod.Data,6,length(tmp));
    % plot(data(1,:))
end
[max_r,idx] = max(totalReward);
max_r
mean(totalReward)
std(totalReward)
best_act_seq
