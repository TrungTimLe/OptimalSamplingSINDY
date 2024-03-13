clear all, close all, clc
%% Generate Data
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3; %number of state
x0 = [-8; 8; 27];  % Initial condition
rng(1)
% Integrate
dt = 0.01;
tspan = [dt:dt:40];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);

% Generate noisy data
noise_level = 10;
noisy_x = x + noise_level*randn(size(x,1),size(x,2));
%% Filter the noise
Fs = 1/0.01; % Sampling frequency
% Plot the frequency spectrum

% Fourier analysis                 
T = dt;             % Sampling period       
L = size(x,1);             % Length of signal
t = (0:L-1)*T;        % Time vector
Y = fft(noisy_x);
fl = 1:0.1:8;
RMSE = [];
dim = 1;
for k = 1:length(fl)
    [b,a] = butter(5,fl(k)/(Fs/2),'low');
    clean_x = filtfilt(b,a,noisy_x(:,dim));
    RMSE(k) = sqrt(mean((clean_x-x(:,dim)).^2));
end

[~,idx] = min(RMSE);
[b,a] = butter(5,fl(idx)/(Fs/2),'low');
clean_x = filtfilt(b,a,noisy_x(:,dim));

% figure,
% subplot(3,1,1)
% plot(fl,RMSE,'s-','LineWidth',1.5,'MarkerSize',2)
% title(['Optimal f = ' num2str(fl(idx)) ', \eta = ' num2str(noise_level)],'FontSize',12)
% xlabel('Cut-off frequency','FontSize',12)
% ylabel('RMSE','FontSize',12)
% grid on
% 
% subplot(3,1,2)
% plot_idx = 200;
% plot(x(1:plot_idx ,dim),'--','LineWidth',1.5)
% hold on
% plot(noisy_x(1:plot_idx,dim),'--','LineWidth',1)
% plot(clean_x(1:plot_idx),'-','LineWidth',1.5)
% legend('True x','Noisy x','Clean x');
% xlabel('Time t','FontSize',12)
% ylabel('x','FontSize',12)
% grid on
% % Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
% 
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% f = Fs*(0:(L/2))/L;
% subplot(3,1,3)
% plot(f,P1,'LineWidth',1.5) 
% title('Single-Sided Amplitude Spectrum of x','FontSize',12)
% hold on
% xline(fl(idx),'r','LineWidth',1.5)
% grid on
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% xlim([0, 20])

%% Fit a regression model for cut-off value
% eta = 0.1:0.2:5;
% dim = 1;
% rng(1)
% Fs = 1/0.01; % Sampling frequency
% % Fourier analysis                 
% T = dt;             % Sampling period       
% L = size(x,1);             % Length of signal
% t = (0:L-1)*T;        % Time vector
% fl = 1:0.1:8;
% 
% for k = 1:length(eta)
%     noise_level = eta(k);
%     noisy_x = x + noise_level*randn(size(x,1),size(x,2));
%     Y = fft(noisy_x);   
%     dim = 1;
%     RMSE1 = [];
%     RMSE2 = [];
%     RMSE3 = [];
%     for i = 1:length(fl)
%         [b,a] = butter(5,fl(i)/(Fs/2),'low');
%         clean_x = filtfilt(b,a,noisy_x(:,dim));
%         RMSE1(i) = sqrt(mean((clean_x-x(:,dim)).^2));
%     end
%     
%     dim = 2;
%     for i = 1:length(fl)
%         [b,a] = butter(5,fl(i)/(Fs/2),'low');
%         clean_x = filtfilt(b,a,noisy_x(:,dim));
%         RMSE2(i) = sqrt(mean((clean_x-x(:,dim)).^2));
%     end
%     
%     dim = 3;
%     for i = 1:length(fl)
%         [b,a] = butter(5,fl(i)/(Fs/2),'low');
%         clean_x = filtfilt(b,a,noisy_x(:,dim));
%         RMSE3(i) = sqrt(mean((clean_x-x(:,dim)).^2));
%     end
%     % Find the optimal value
%     [M,I] = min(RMSE1);
%     RMSE_opt1(k) = M;
%     fl_opt1(k) = fl(I);
%     
%     [M,I] = min(RMSE2);
%     RMSE_opt2(k) = M;
%     fl_opt2(k) = fl(I);
%     
%     [M,I] = min(RMSE3);
%     RMSE_opt3(k) = M;
%     fl_opt3(k) = fl(I);
% end
% subplot(3,2,1)
% plot(eta,fl_opt1,'LineWidth',1.5)
% xlabel('\eta'),ylabel('Optimal f (x)')
% grid on
% 
% subplot(3,2,2)
% plot(eta,RMSE_opt1,'LineWidth',1.5)
% xlabel('\eta'),ylabel('Min RMSE')
% grid on
% 
% subplot(3,2,3)
% plot(eta,fl_opt2,'LineWidth',1.5)
% xlabel('\eta'),ylabel('Optimal f (y)')
% grid on
% 
% subplot(3,2,4)
% plot(eta,RMSE_opt2,'LineWidth',1.5)
% xlabel('\eta'),ylabel('Min RMSE')
% grid on
% 
% subplot(3,2,5)
% plot(eta,fl_opt3,'LineWidth',1.5)
% xlabel('\eta'),ylabel('Optimal f (z)')
% grid on
% 
% subplot(3,2,6)
% plot(eta,RMSE_opt3,'LineWidth',1.5)
% xlabel('\eta'),ylabel('Min RMSE')
% grid on
%% Compute derivative using Total Variation Regularized Numerical Differentiation (TVDiff)
% for i=1:length(x)
%     ds_true(i,:) = lorenz(0,x(i,:),Beta);
% end
% 
% alpha = 0.024*noise_level^2+0.057*noise_level-0.0062;
% if alpha <= 0 
%     alpha = 0.01;
% end
% n_iter = 3;
% dx = TVRegDiff(noisy_x(:,1), n_iter, alpha, [], 'small', 1e-6, 0.01, 0, 0);
% alpha = 0.009088*noise_level^2+0.04166*noise_level+0.006757;
% if alpha <= 0 
%     alpha = 0.01;
% end
% dy = TVRegDiff(noisy_x(:,2), n_iter, alpha, [], 'small', 1e-6, 0.01, 0, 0);
% alpha = 0.009791*noise_level^2+0.05758*noise_level-0.01469;
% if alpha <= 0 
%     alpha = 0.01;
% end
% dz = TVRegDiff(noisy_x(:,3), n_iter, alpha, [], 'small', 1e-6, 0.01, 0, 0);
% TVDiff_dx = [dx(1:end-1) dy(1:end-1) dz(1:end-1)];
% 
% for i = 1:length(x)
%     noisy_dx(i,:) = lorenz(0,x(i,:),Beta)' + noise_level*randn(1,3);
% end
% TVDiff_dx = noisy_dx;
% %% Implement SINdy method
% % Pool Data  (i.e., build library of nonlinear time series)
% polyorder = 3;
% Theta = poolData(x,n,polyorder);
% m = size(Theta,2);
% 
% % Compute Sparse regression: sequential least squares
% lambda = 0.25;      % lambda is our sparsification knob.
% Xi_true = sparsifyDynamics(Theta,ds_true,lambda,n);
% 
% Ts = [0.02 0.04 0.08 0.16]; % sampling period
% max_iter = 100;
% Theta = poolData(x,n,polyorder);
% n_init = 10;
% for i = 1:length(Ts)
%     step = Ts(i)/0.01;
%     cur_idx = step*n_init;
%     x_cur = noisy_x(1:step:cur_idx,:);
%     dx_cur = TVDiff_dx(1:step:cur_idx,:);
%     for j = 1:max_iter
%         % Sample new data point
%         cur_idx = cur_idx + step;
%         x_cur = [x_cur; noisy_x(cur_idx,:)];
%         dx_cur = [dx_cur; TVDiff_dx(cur_idx,:)];
%         Theta = poolData(x_cur,n,polyorder);
%         Xi_hat = sparsifyDynamics(Theta,dx_cur,lambda,n);
%         Fnorm_error(i,j) = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
%         Cond_num(i,j) = cond(Theta);
%         I_mat = inv(Theta'*Theta);% information matrix
%         I_mat_trace(i,j) = trace(I_mat);
%         I_mat_det(i,j) = det(I_mat);
%         I_mat_eig(i,j) = min(eig(I_mat));
%     end
% end
% % Plot results
% figure,
% subplot(3,1,1)
% semilogy(Fnorm_error','-','LineWidth',1)
% xlabel('Iteration','FontSize',12),ylabel('Fnorm error','FontSize',12)
% legend('Ts = 0.02','Ts = 0.04','Ts = 0.08','Ts = 0.16')
% grid on
% title(['\eta = ' num2str(noise_level)],'FontSize',14)
% 
% subplot(3,1,2)
% semilogy(Cond_num','-','LineWidth',1)
% xlabel('Iteration','FontSize',12),ylabel('Cond number','FontSize',12)
% grid on
% 
% subplot(3,1,3)
% semilogy(I_mat_trace','LineWidth',1.5)
% grid on
% title('A-optimality')
% xlabel('Iteration','FontSize',12),ylabel('Value','FontSize',12)
%% Derivatives estimation methods comparison
eta = 0.1:0.2:10;
rng(1)
Fs = 1/0.01; % Sampling frequency
% Fourier analysis                 
T = dt;             % Sampling period       
L = size(x,1);             % Length of signal
t = (0:L-1)*T;        % Time vector
fl = 1:0.1:8;
for i=1:length(x)
    dx_true(i,:) = lorenz(0,x(i,:),Beta);
end
dim = 1;
for k = 1:length(eta)
    % Generate the data
    noise_level = eta(k);
    noisy_x = x + noise_level*randn(size(x,1),size(x,2));
    % Filter the data using lowpass filter
    Y = fft(noisy_x);   

    tmp_RMSE = [];
    for i = 1:length(fl)
        [b,a] = butter(5,fl(i)/(Fs/2),'low');
        clean_x = filtfilt(b,a,noisy_x(:,dim));
        tmp_RMSE(i) = sqrt(mean((clean_x-x(:,dim)).^2));
    end
    % Find the optimal value
    [M,I] = min(tmp_RMSE);
    [b,a] = butter(5,fl(I)/(Fs/2),'low');
    clean_x = filtfilt(b,a,noisy_x(:,dim));
    % compute derivative using fourth order central difference
    dx_CD = zeros(length(clean_x)-5,1);
    for i=3:length(x)-3
        dx_CD(i-2) = (1/(12*dt))*(-clean_x(i+2)+8*clean_x(i+1)-8*clean_x(i-1)+clean_x(i-2));
    end
    RMSE_CD(k) = sqrt(mean((dx_CD(1:end)-dx_true(1+2:size(dx_CD,1)+2,dim)).^2));
    if dim == 1
        alpha = 0.024*eta(k)^2+0.057*eta(k)-0.0062;
    elseif dim == 2
        alpha = 0.009088*eta(k)^2+0.04166*eta(k)+0.006757;
    else
        alpha = 0.009791*eta(k)^2+0.05758*eta(k)-0.01469;
    end
    if alpha <= 0
        alpha = 0.01;
    end
    TVDiff_dx = TVRegDiff(noisy_x(:,1), 3, alpha, [], 'small', 1e-6, 0.01, 0, 0);
    RMSE_TVDiffNoisy(k) = sqrt(mean((TVDiff_dx(1:size(dx_CD,1))-dx_true(1:size(dx_CD,1),dim)).^2));
    TVDiff_dxC = TVRegDiff(clean_x(:,1), 3, 0.02, [], 'small', 1e-6, 0.01, 0, 0);
    RMSE_TVDiffClean(k) = sqrt(mean((TVDiff_dxC(1:size(dx_CD,1))-dx_true(1:size(dx_CD,1),dim)).^2));
    if k == 50
        dx1 = dx_CD;
        dx2 = TVDiff_dx;
        dx3 = TVDiff_dxC;
    end
end
figure,
subplot(2,1,1)
plot(dx_true((3:500+2),1),'--','LineWidth',1)
title(['\eta = ' num2str(eta(13))])
hold on
plot(dx1(1:500),'LineWidth',1)
plot(dx2(3:500+2),'LineWidth',1)
plot(dx3(3:500+2),'LineWidth',1)
legend('True dx/dt','CD estimates','Noisy TVDiff estimates','Clean TVDiff estimates');
xlabel('Time t','FontSize',12)
ylabel('dx/dt','FontSize',12)
grid on

subplot(2,1,2)
plot(eta,RMSE_CD,'-','LineWidth',1)
hold on
plot(eta,RMSE_TVDiffNoisy,'LineWidth',1)
plot(eta,RMSE_TVDiffClean,'LineWidth',1)
legend('CD','Noisy TVDiff','Clean TVDiff');
xlabel('\eta','FontSize',12)
ylabel('RMSE','FontSize',12)
grid on
%% Number of iterations sensitivity
% n_iters = 10:20:100;
% RMSE_iter = size(n_iters,1);
% TVDiff_dx = zeros(length(n_iters),size(x,1)+1);
% for i = 1:length(n_iters)
%     TVDiff_dx(i,:) = TVRegDiff(noisy_x(:,1), n_iters(i), 0.2, [], 'small', 1e-6, 0.01, 0, 0);
%     RMSE_iter(i) = sqrt(mean((TVDiff_dx(i,1:end-1)'-ds_true(:,1)).^2));
% end
% % Validate estimated derivative
% figure,
% subplot(3,1,1)
% plot(x(:,1),'-','LineWidth',1.5)
% hold on
% plot(noisy_x(:,1),'LineWidth',0.5)
% legend('True x','Noisy x');
% xlabel('Time t','FontSize',12)
% ylabel('x','FontSize',12)
% 
% subplot(3,1,2)
% plot(ds_true(:,1),'LineWidth',1.5);
% hold on
% plot(ds_noisy(:,1),'--','LineWidth',0.5);
% for i = 1:length(n_iters)
%    plot(TVDiff_dx(i,1:end-1),'-','LineWidth',1); 
% end
% legend('True dx/dt','Noisy dx/dt',['TVDiff (iter = ' num2str(n_iters(1)) ')']...
%     ,['TVDiff (iter = ' num2str(n_iters(2)) ')'],['TVDiff (iter = ' num2str(n_iters(3)) ')'],...
%     ['TVDiff (iter = ' num2str(n_iters(4)) ')'],['TVDiff (iter = ' num2str(n_iters(5)) ')'])
% xlabel('Time t'),ylabel('dx/dt')
% 
% subplot(3,1,3)
% plot(n_iters,RMSE_iter,'-s','LineWidth',1.5)
% xlabel('Number of iterations','FontSize',12)
% ylabel('RMSE','FontSize',12)
%% alpha sensitivity
% level = 5;
% idx = (1:0.1:(level+1))-4;
% alpha = 10.^(idx); % Regularization parameter
% RMSE = size(alpha,1);
% N = length(alpha);
% TVDiff_dx = zeros(length(alpha),size(x,1)+1);
% for i = 1:N
%     TVDiff_dx(i,:) = TVRegDiff(noisy_x(:,1), 50, alpha(i), [], 'small', 1e-6, 0.01, 0, 0);
%     RMSE(i) = sqrt(mean((TVDiff_dx(i,1:end-1)'-ds_true(:,1)).^2));
% end
% % Validate estimated derivative
% figure,
% subplot(3,1,1)
% plot(x(:,1),'-','LineWidth',1.5)
% hold on
% plot(noisy_x(:,1),'LineWidth',0.5)
% title(['\eta = ' num2str(noise_level)])
% legend('True x','Noisy x');
% xlabel('Time t','FontSize',12)
% ylabel('x','FontSize',12)
% % Find the optimal value
% [M,I] = min(RMSE);
% subplot(3,1,2)
% plot(ds_true(:,1),'LineWidth',1.5);
% hold on
% plot(ds_noisy(:,1),'--','LineWidth',0.5);
% plot(TVDiff_dx(I,1:end-1),'-','LineWidth',1.5); 
% legend('True dx/dt','Noisy dx/dt',['TVDiff (alpha = ' num2str(alpha(I)) ')'])
% xlabel('Time t'),ylabel('dx/dt')
% 
% subplot(3,1,3)
% semilogx(alpha,RMSE,'-s','LineWidth',1.5,'MarkerSize',2)
% xlabel('\alpha','FontSize',12)
% ylabel('RMSE','FontSize',12)
%% alpha sensitivity for different noise level
% eta = 0:0.1:5;
% dim = 3;
% rng(1)
% for k = 1:length(eta)
%     noise_level = eta(k);
%     noisy_x = x + noise_level*randn(size(x,1),size(x,2));
%     level = 5;
%     idx = (1:0.1:(level+1))-4;
%     alpha = 10.^(idx); % Regularization parameter
%     N = length(alpha);
%     TVDiff_dx = zeros(length(alpha),size(x,1)+1);
%     for i = 1:N
%         TVDiff_dx(i,:) = TVRegDiff(noisy_x(:,dim), 3, alpha(i), [], 'small', 1e-12, 0.01, 0, 0);
%         RMSE(i) = sqrt(mean((TVDiff_dx(i,1:end-1)'-ds_true(1:end,dim)).^2));
%     end
%     % Find the optimal value
%     [M,I] = min(RMSE);
%     RMSE_opt(k) = M;
%     alpha_opt(k) = alpha(I);
% end
% subplot(2,1,1)
% plot(eta,alpha_opt,'LineWidth',1.5)
% xlabel('\eta'),ylabel('Optimal \alpha')
% grid on
% 
% subplot(2,1,2)
% plot(eta,RMSE_opt,'LineWidth',1.5)
% xlabel('\eta'),ylabel('Min RMSE')
% grid on
%% Epsilon sensitivity
% idx = (1:0.1:10)-11;
% ep = 10.^(idx); % Regularization parameter
% N = length(ep);
% % alpha = 0.025;
% alpha = 0.12589;
% TVDiff_dx = zeros(N,size(x,1)+1);
% for i = 1:N
%     TVDiff_dx(i,:) = TVRegDiff(noisy_x(:,1), 3, alpha, [], 'small', ep(i), 0.01, 0, 0);
%     RMSE(i) = sqrt(mean((TVDiff_dx(i,1:end-1)'-ds_true(:,1)).^2));
% end
% % Validate estimated derivative
% figure,
% subplot(3,1,1)
% plot(x(:,1),'-','LineWidth',1.5)
% hold on
% plot(noisy_x(:,1),'LineWidth',0.5)
% title(['\eta = ' num2str(noise_level)])
% legend('True x','Noisy x');
% xlabel('Time t','FontSize',12)
% ylabel('x','FontSize',12)
% % Find the optimal value
% [M,I] = min(RMSE);
% subplot(3,1,2)
% plot(ds_true(:,1),'LineWidth',1.5);
% hold on
% plot(ds_noisy(:,1),'--','LineWidth',0.5);
% plot(TVDiff_dx(I,1:end-1),'-','LineWidth',1.5); 
% legend('True dx/dt','Noisy dx/dt',['TVDiff (ep = ' num2str(ep(I)) ')'])
% xlabel('Time t'),ylabel('dx/dt')
% 
% subplot(3,1,3)
% semilogx(ep,RMSE,'-s','LineWidth',1.5,'MarkerSize',2)
% xlabel('\ep','FontSize',12)
% ylabel('RMSE','FontSize',12)
%% dx sensitivity
% idx = (1:0.1:5)-6;
% dx = 10.^(idx); % Regularization parameter
% N = length(dx);
% alpha = 0.025;
% % alpha = 0.12589;
% TVDiff_dx = zeros(N,size(x,1)+1);
% for i = 1:N
%     TVDiff_dx(i,:) = TVRegDiff(noisy_x(:,1), 3, alpha, [], 'small',  1e-6, dx(i), 0, 0);
%     RMSE(i) = sqrt(mean((TVDiff_dx(i,1:end-1)'-ds_true(:,1)).^2));
% end
% % Validate estimated derivative
% figure,
% subplot(3,1,1)
% plot(x(:,1),'-','LineWidth',1.5)
% hold on
% plot(noisy_x(:,1),'LineWidth',0.5)
% title(['\eta = ' num2str(noise_level)])
% legend('True x','Noisy x');
% xlabel('Time t','FontSize',12)
% ylabel('x','FontSize',12)
% % Find the optimal value
% [M,I] = min(RMSE);
% subplot(3,1,2)
% plot(ds_true(:,1),'LineWidth',1.5);
% hold on
% plot(ds_noisy(:,1),'--','LineWidth',0.5);
% plot(TVDiff_dx(I,1:end-1),'-','LineWidth',1.5); 
% legend('True dx/dt','Noisy dx/dt',['TVDiff (dx = ' num2str(dx(I)) ')'])
% xlabel('Time t'),ylabel('dx/dt')
% 
% subplot(3,1,3)
% semilogx(dx,RMSE,'-s','LineWidth',1.5,'MarkerSize',2)
% xlabel('dx','FontSize',12)
% ylabel('RMSE','FontSize',12)
%% alpha sensitivity for different noise level
% eta = 0.5:0.1:5;
% dim = 2;
% wind = 10:20:500;
% rng(1)
% for k = 1:length(eta)
%     for w = 1:length(wind)
%         noise_level = eta(k);
%         noisy_x = x + noise_level*randn(size(x,1),size(x,2));
%         if dim == 1
%             alpha = 0.024*eta(k)^2+0.057*eta(k)-0.0062;
%         elseif dim == 2
%             alpha = 0.009088*eta(k)^2+0.04166*eta(k)+0.006757;
%         else
%             alpha = 0.009791*eta(k)^2+0.05758*eta(k)-0.01469;
%         end
%         start_idx = 1;
%         end_idx = start_idx + wind(w)-1;
%         count = 0;
%         RMSE = [];
%         while end_idx <= size(x,1)
%             count = count + 1;
%             TVDiff_dx = TVRegDiff(noisy_x(start_idx:end_idx,dim), 3, alpha, [], 'small', 1e-12, 0.01, 0, 0);
%             RMSE(w,count) = sqrt(mean((TVDiff_dx(1:end-1)-ds_true(start_idx:end_idx,dim)).^2));
%             start_idx = start_idx + wind(w);
%             end_idx = end_idx + wind(w);
%         end
%         RMSE_all(w) = sum(RMSE(w,:));
%     end
%     % Find the optimal value
%     [M,I] = min(RMSE_all);
%     RMSE_opt(k) = M;
%     wind_opt(k) = wind(I);
% end
% subplot(2,1,1)
% plot(eta,wind_opt,'LineWidth',1.5)
% xlabel('\eta'),ylabel('Optimal window size')
% grid on
% 
% subplot(2,1,2)
% plot(eta,RMSE_opt,'LineWidth',1.5)
% xlabel('\eta'),ylabel('Min RMSE')
% grid on