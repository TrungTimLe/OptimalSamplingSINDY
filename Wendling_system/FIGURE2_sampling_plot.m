clear all; close all; clc;

% % Preallocate a cell array to store outputs for each type
% uout_results = cell(4, 1);
% 
% % Loop through each type
% for t = 1:4
%     fprintf('Running simulation for Type %d...\n', t);
%     uout_results{t} = simulate_neural_activity(t);
% end
load('R:\Research\SINDy_sampling_paper\Code\Wendling system\uout_results.mat')
% Greedy sampling indices for each type
greedy_samples = {
    [5, 6, 16, 17, 24, 44, 53, 59, 60, 61, 62, 63, 64], % Type 1
    [5, 6, 9, 24, 31, 51, 54, 58, 59, 60, 62, 63, 64],  % Type 2
    [5, 6, 15, 20, 21, 25, 45, 61, 62, 64, 65, 66, 67], % Type 3
    [5, 6, 8, 10, 16, 38, 50, 63, 81, 102, 103, 104, 105] % Type 4
};

% Brute-force search changes for each type
brute_force_changes = {
    [3, 3, 3, 2, 3, 3, 2, 3, 2, 2, 2], % Type 1
    [3, 3, 3, 3, 3, 2, 2, 2, 1, 3, 2], % Type 2
    [3, 3, 3, 3, 3, 3, 2, 1, 1, 1, 2], % Type 3
    [3, 3, 3, 3, 3, 2, 3, 2, 2, 2, 1]  % Type 4
};

% RL changes for each type
rl_changes = {
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], % Type 1
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, -1], % Type 2
    [1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],    % Type 3
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]  % Type 4
};

% Calculate the sampled indices
brute_force_samples = cellfun(@(c) calculate_indices(c, 1, 64, 'brute_force'), brute_force_changes, 'UniformOutput', false);
rl_samples = cellfun(@(c) calculate_indices(c, 1, 64, 'rl'), rl_changes, 'UniformOutput', false);

% Initialize figure
figure('Position', [100, 100, 1024, 768]);

% Plot results for each type
for t = 1:4
    subplot(2, 2, t);
    hold on;
    grid on;
    
    tmp = uout_results{t}(1,:);

    % Determine the maximum index among all sampling strategies for the current type
    max_index = max([max(greedy_samples{t}), max(brute_force_samples{t}), max(rl_samples{t})]);

    % Ensure max_index does not exceed the length of tmp
    max_index = min(max_index, length(tmp));
    
    % Full data
    plot(tmp(1:max_index+20), 'k-', 'LineWidth', 1);
    xlim([0 max_index+20])
    % Greedy samples
    M_size = 4;
    plot(greedy_samples{t}, tmp(greedy_samples{t}), '^', 'MarkerSize', M_size, 'LineWidth', 1, 'Color',[0 0.4470 0.7410]);

    % Brute-force samples
    plot(brute_force_samples{t}, tmp(brute_force_samples{t}), 'o', 'MarkerSize', M_size, 'LineWidth', 1, 'Color',[0.4660 0.6740 0.1880]);

    % RL samples
    plot(rl_samples{t}, tmp(rl_samples{t}), 's', 'MarkerSize', M_size, 'LineWidth', 1, 'Color',[0.75 0 0]);

    title(sprintf('Type %d', t));
    xlabel('Time step');
    ylabel('u_p_y');
    hold off;
    set(gca, 'FontSize', 14, 'FontWeight', 'bold');
end
legend('Full data', 'Greedy samples', 'RBFS samples', 'RL-based samples');
% print(gcf, 'FIGURE2_Wendling.png', '-dpng', '-r1200');

%-----------------------------------------------------
function indices = calculate_indices(changes, Ts_low, Ts_high, mode)
    indices = [5, 6]; % Initial samples
    period = 1; % Initial period
    for change = changes
        if strcmp(mode, 'brute_force')
            if change == 3
                period = min(period * 2, Ts_high);
            elseif change == 1
                period = max(period / 2, Ts_low);
            end
        elseif strcmp(mode, 'rl')
            if change == 1
                period = min(period * 2, Ts_high);
            elseif change == -1
                period = max(period / 2, Ts_low);
            end
        end
        indices(end+1) = indices(end) + period;
    end
end

function uout = simulate_neural_activity(type)
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

    % Set the type based on input
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
            return;
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

    % Return the final output
    uout = uout(:, 1:indexplot-1);
end