clear all, close all, clc
%% Generate Data
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3; %number of state
x0=[-8; 8; 27];  % Initial condition

% Integrate
dt = 0.04;
tspan=[dt:dt:1500*dt];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);

% Generate noisy data
noise_level = 0.001;
for i = 1:n
    for j = 1:N
        noisy_x(j,i) = x(j,i) + noise_level*rand()*var(x(:,i));
    end
end
%% Compute derivative using Total Variation Regularized Numerical Differentiation (TVDiff)
for i=1:length(x)
    ds_true(i,:) = lorenz(0,x(i,:),Beta);
end
dx = TVRegDiff(noisy_x(:,1), 500, 0.2, [], 'small', 1e-6, 0.03, 0, 0);
% Validate estimated derivative
figure,
plot(ds_true(:,1));
hold on
plot(dx)
legend('True dx/dt','TVDiff estimates')
xlabel('Time t'),ylabel('dx/dt')

dy = TVRegDiff(noisy_x(:,2), 500, 0.2, [], 'small', 1e-6, 0.03, 0, 0);
dz = TVRegDiff(noisy_x(:,3), 500, 0.2, [], 'small', 1e-6, 0.03, 0, 0);
dxx = [dx(2:end) dy(2:end) dz(2:end)];
% Compute noisy derivative
for i=1:length(x)
    dx_noisy(i,:) = lorenz(0,noisy_x(i,:),Beta);
end
%% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

%% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,ds_true,lambda,n);
poolDataLIST({'x','y','z'},Xi_true,n,polyorder);
%% Simulate error
n_init = 4;
dx1 = dx_noisy(1:n_init,:);
idx = n_init;
n_data = length(x);
count = 0;
Fnorm_error = 10000;
while (idx < length(x) && Fnorm_error(end) > 0.001)
    idx = idx + 1;
    count = count + 1;
    dx1 = [dx1; dx_noisy(idx,:)];
    Xi_hat = sparsifyDynamics(Theta(1:idx,:),dx1,lambda,n);
    poolDataLIST({'x','y','z'},Xi_hat,n,polyorder);
%     de_dSdot = 2*(pinv(Theta(1:idx,:)))'*(pinv(Theta(1:idx,:))*dx1-Xi_true);
%     de_ds(count,1:3) = [-Beta(1)*de_dSdot(end,1) -de_dSdot(end,2) -Beta(3)*de_dSdot(end,3)]';
    Fnorm_error(count) = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
end

figure,
plot(Fnorm_error)
ylim([0 1000])
xlabel('Time t to which data are sampled up')
ylabel('Frobenius-norm error')
% figure,
% plot(de_ds(:,1))
% hold on
% plot(de_ds(:,2))
% plot(de_ds(:,3))
% legend('x','y','z')
%% Plot attractor
%%  Part 1: Attractor with inital data
figure
L = 1:length(x);
plot3(x(L,1),x(L,2),x(L,3),'Color',[.1 .1 .1],'LineWidth',1.5)
hold on
L = 1:(n_init+count);
plot3(x(L,1),x(L,2),x(L,3),'Color','g','LineWidth',1.5)
L = 1:(n_init+count);
plot3(noisy_x(L,1),noisy_x(L,2),noisy_x(L,3),'Color',[0.75 0 0],'LineWidth',1.5)
legend('Full attractor','True data','Noisy data')
axis on
view(-5,12)
axis tight
xlabel('x'), ylabel('y'), zlabel('z')
set(gca,'FontSize',14)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')
% legend('Full data','Initial data')



