clear all, close all, clc
%% Generate Data
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3; %number of state
x0 = [-8; 8; 27];  % Initial condition

% Integrate
dt = 0.004;
tspan = [dt:dt:40];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);

% Generate noisy data
noise_level = 0;
for i = 1:n
    for j = 1:N
        noisy_x(j,i) = x(j,i) + noise_level*rand()*var(x(:,i));
    end
end
%% Compute derivative using Total Variation Regularized Numerical Differentiation (TVDiff)
for i=1:length(x)
    dx_true(i,:) = lorenz(0,x(i,:),Beta);
end
% dx = TVRegDiff(noisy_x(:,1), 500, 0.2, [], 'small', 1e-6, 0.01, 0, 0);
% % Validate estimated derivative
% figure,
% plot(ds_true(:,1));
% hold on
% plot(dx)
% legend('True dx/dt','TVDiff estimates')
% xlabel('Time t'),ylabel('dx/dt')
% 
% dy = TVRegDiff(noisy_x(:,2), 500, 0.2, [], 'small', 1e-6, 0.01, 0, 0);
% dz = TVRegDiff(noisy_x(:,3), 500, 0.2, [], 'small', 1e-6, 0.01, 0, 0);
% dxx = [dx(2:end) dy(2:end) dz(2:end)];
%% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

%% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx_true,lambda,n);
poolDataLIST({'x','y','z'},Xi_true,n,polyorder);

%% Part 3: RL Observation and Action Specifications
ObservationInfo = rlNumericSpec([6 1]);
ObservationInfo.Name = 'Lorenz States and Sampling period';
ObservationInfo.Description = 'x, y, z, Ts, Fnorm error, Mutual information';

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
envConstants.Lambda_train = 0;
% The lower limit for Ts
envConstants.Ts_low = 0.004;
% The upper limit for Ts
envConstants.Ts_high = 0.16;
% The Lorenz parameters
envConstants.Beta = [10; 28; 8/3];
% The Lorenz process noise level eta
envConstants.eta = 0;
% The stopping criteria
envConstants.tol = 0.001;
% The true Xi vector 
envConstants.Xi_true = Xi_true;
% The true Xi vector 
envConstants.lambda = 0.1;

StepHandle = @(Action,LoggedSignals) LorenzStepFunction(Action,LoggedSignals,envConstants);
% Use the same reset function, specifying it as a function handle rather than by using its name.
ResetHandle = @() LorenzResetFunction(envConstants.eta,envConstants.Xi_true,envConstants.lambda);
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
% % Plot attractor with noisy states
% figure
% L = 1:length(x);
% plot3(x(L,1),x(L,2),x(L,3),'Color',[.1 .1 .1],'LineWidth',1.5)
% hold on
% plot3(LoggedSignals.CurrentNoisySample(:,1),LoggedSignals.CurrentNoisySample(:,2),...
%     LoggedSignals.CurrentNoisySample(:,3),'Color',[0.75 0 0],'LineWidth',1.5)
% legend('Full attractor','Noisy data')
% axis on
% view(-5,12)
% axis tight
% xlabel('x'), ylabel('y'), zlabel('z')
% set(gca,'FontSize',14)
% set(gcf,'Position',[100 100 600 400])
% set(gcf,'PaperPositionMode','auto')
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
'MaxStepsPerEpisode',200, ...
'Verbose',false, ...
'Plots','training-progress',...
'StopTrainingCriteria','AverageReward',...
'StopTrainingValue',10000);

% Train the agent.
trainingStats = train(agent,env,trainOpts); 

%% Simulate DQN Agent
M = 3; % number of simulations
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