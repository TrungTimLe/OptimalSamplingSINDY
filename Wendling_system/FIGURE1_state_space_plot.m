clear all, close all, clc
%% Generate Data
% Simulation settings
Npart = 50; %Number of parts for fourier transform including intial part
Tend = 10*Npart; %Total simulation time
NTST = Tend*2000; %Number of Time Steps
Nout = 1; % Output every 5th time step
deltat = Tend/NTST; %Time step size

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
sd = 2; %Standard deviation input
e0 = 2.5;
r = 0.56;
v0 = 6;

%% Define the parameters for the Wendling model
n = 3;
switch n
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

% Initial conditions
x = zeros(4,1);
y = zeros(4,1);

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

    x = xn;
    y = yn;
    u = P*x;

    %Saves u for output every Nout iterations
    tel = tel + 1;
    if tel == Nout
        tel = 0;
        uout(:,indexplot) = u;
        indexplot = indexplot + 1;
    end
end
%% Plot time series:
% Plotting the first k data points of the simulated u variables
k = 10000;
figure('Position',[507,50,935,946]);
subplot(4,1,1);
plot([0:Nout:(Nout*(k-1))]*deltat, uout(1,1:k));
title('u_{py}');
grid on
xlabel('Time');
ylabel('Amplitude');

subplot(4,1,2);
plot([0:Nout:(Nout*(k-1))]*deltat, uout(2,1:k));
title('u_{ex}');
grid on

xlabel('Time');
ylabel('Amplitude');

subplot(4,1,3);
plot([0:Nout:(Nout*(k-1))]*deltat, uout(3,1:k));
title('u_{is}');
grid on

xlabel('Time');
ylabel('Amplitude');

subplot(4,1,4);
plot([0:Nout:(Nout*(k-1))]*deltat, uout(4,1:k));
title('u_{if}');
grid on

xlabel('Time');
ylabel('Amplitude');

% Define the descriptions for each type
typeDescriptions = ["Noisy perturbation around an equilibrium", ...
    "Spikes Wave Discharges (SWD)", ...
    "Sustained SWDs", ...
    "Slow rhythmic activity", ...
    "Rapid low-voltage activity", ...
    "Quasi-sinusoidal activity"];

% Create a title string that includes the type of simulation
titleStr = sprintf('Simulated u variables over time (Type %d: %s)', n, typeDescriptions(n));

% Set the title of the figure
sgtitle(titleStr);

