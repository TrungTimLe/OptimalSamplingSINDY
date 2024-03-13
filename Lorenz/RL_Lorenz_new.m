clear all, close all, clc
%% Generate Data
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3; %number of state
x0 = [-8; 8; 27];  % Initial condition

% Integrate
dt = 0.04;
tspan = [dt:dt:40];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x] = ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
% Compute derivative using Total Variation Regularized Numerical Differentiation (TVDiff)
for i=1:length(x)
    dx_true(i,:) = lorenz(0,x(i,:),Beta);
end
% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 2;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx_true,lambda,n);
poolDataLIST({'x','y','z'},Xi_true,n,polyorder);

%% Generate data for RL
NSR = 0; % Noise-to-signal ratio
n = 3; %number of states
x0 =[-8; 8; 27];  % Initial condition
dt = 0.004; % delta t
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
Tmax = 200;
tspan = [dt:dt:Tmax];
N = length(tspan);
[t,x] = ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
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
fl = 1:0.1:10;
Fs = 1/dt;
for dim = 1:3
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
%% QR factorization
% p = 20; % Number of samples
% Theta = poolData(cx,n,polyorder);
% 
% [Q,R,pivot] = qr(Theta*Theta','vector');
% [~,idx] = sort(pivot);
% selected_idx = idx(1:p);
% % Plot the sampled data
% figure,
% % Plot the attractor
% L = 1:length(x);
% scatter3(x(1,1),x(1,2),x(1,3),'filled')
% hold on
% plot3(cx(L,1),cx(L,2),cx(L,3),'Color',[.1 .1 .1],'LineWidth',0.5)
% axis on
% grid on
% view(-5,12)
% axis tight
% xlabel('x'), ylabel('y'), zlabel('z')
% set(gca,'FontSize',10)
% set(gcf,'Position',[100 100 600 400])
% set(gcf,'PaperPositionMode','auto')
% plot3(cx(selected_idx,1),cx(selected_idx,2),cx(selected_idx,3),'o','MarkerSize',3,'Color',[0.75 0 0],...
%     'LineWidth',1.5)
%% Part 3: RL Observation and Action Specifications
% ObservationInfo = rlNumericSpec([6 1]);
% ObservationInfo.Name = 'Lorenz States and Sampling period';
% ObservationInfo.Description = 'x, y, z, Ts, Fnorm error, Mutual information';

ObservationInfo = rlNumericSpec([10 1]);
ObservationInfo.Name = 'Observations';
ObservationInfo.Description = 'Ts, Condition number, Mutual information, Trace(I_mat)';

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
envConstants.Ts_low = 0.004;
% The upper limit for Ts
envConstants.Ts_high = 0.32;
% The Lorenz parameters
envConstants.Beta = [10; 28; 8/3];
% The stopping criteria
if NSR == 0
    envConstants.tol = 0.001;
elseif NSR == 0.001
    envConstants.tol = 0.1;
else
    envConstants.tol = 1;
end
% The true Xi vector
envConstants.Xi_true = Xi_true;
% The true Xi vector
envConstants.lambda = 0.1;

StepHandle = @(Action,LoggedSignals) LorenzStepFunction_new(Action,LoggedSignals,envConstants);
% Use the same reset function, specifying it as a function handle rather than by using its name.
ResetHandle = @() LorenzResetFunction_new(NSR);
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
fullyConnectedLayer(6,'Name','CriticStateFC1')
reluLayer('Name','CriticRelu1')
fullyConnectedLayer(6, 'Name','CriticStateFC2')
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
'MaxEpisodes',100, ...
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
    cx = experience.Observation.Observations.Data(5:7,:,:);
    dx = experience.Observation.Observations.Data(8:end,:,:);
    XRL = reshape(cx,3,length(cx))';
    dXRL = reshape(dx,3,length(dx))';
    % % RL
    k = 1;
    tnow = dt;
    Ts_low = 0.004;
    Ts_init = 0.016;
    % The upper limit for Ts
    Ts_high = 0.32;
    Ts = Ts_init;
    for j = 1:k
        tnow = tnow + Ts_init;
    end
    idx = (tnow/dt);

    for j = 1:length(RL)
        tmp = RL(j);
        Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
            || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
        idx = idx + Ts/dt;
    end
    % Evaluate the policy
    XRL = [x0'; XRL];
    dXRL = [dx_true(1,:); dXRL];
    sample_size(i) = length(XRL);
    train_time(i) = dt*idx;
    Theta = poolData(XRL,n,polyorder);
    Xi_hat = sparsifyDynamics(Theta,dXRL,envConstants.lambda,n);
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

%% Plot the training process
% record = experience.Observation.Observations.Data;
% % Compute the Fnorm error
% % Sample k more intial data points
% k = 2;
% Xtmp = x0';
% tnow = dt;
% Ts_init = 0.016;
% dXtmp = lorenz(0,Xtmp,Beta)';
% for i = 1:k
%     tnow = tnow + Ts_init;
%     Xtmp = [Xtmp; cx(ceil(tnow/dt),:)];
%     dXtmp = [dXtmp; dx(ceil(tnow/dt),:)];
% end
% 
% for i = 2:size(record,3)
%     Xtmp = [Xtmp; record(5:7,1,i)'];
%     dXtmp = [dXtmp; record(8:10,1,i)'];
%     Theta = poolData(Xtmp,n,polyorder);
%     Xi_hat = sparsifyDynamics(Theta,dXtmp,envConstants.lambda,n);
%     Fnorm_error(i-1) = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
% end
% 
% figure('Position',[680,50,976,946])
% subplot(4,1,1)
% semilogy(Fnorm_error)
% ylabel('Fnorm error')
% grid on
% set(gca,'FontSize',14','FontWeight','bold')
% 
% subplot(4,1,2)
% plot(reshape(record(2,1,2:end),size(record,3)-1,1));
% grid on
% ylabel('Log(\kappa)')
% set(gca,'FontSize',14','FontWeight','bold')
% 
% subplot(4,1,3)
% plot(reshape(record(3,1,2:end),size(record,3)-1,1))
% grid on
% ylabel('Log(tr(I))')
% set(gca,'FontSize',14','FontWeight','bold')
% 
% subplot(4,1,4)
% plot(reshape(record(4,1,2:end),size(record,3)-1,1))
% ylabel('MI')
% grid on
% xlabel('Training step')
% set(gca,'FontSize',14','FontWeight','bold')